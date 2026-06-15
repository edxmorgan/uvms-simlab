"""RSL-RL VecEnv adapter for :class:`UvmsBatchEnv`."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

try:
    from rsl_rl.env import VecEnv
except Exception:  # pragma: no cover - exercised only when rsl_rl is not installed.
    class VecEnv:  # type: ignore[no-redef]
        """Fallback base so this module can still be imported without RSL-RL."""


from uvms_rl.config import load_experiment
from uvms_rl.rl_env import UvmsBatchEnv
from uvms_rl.tensor import is_torch_tensor


class RslRlUvmsEnv(VecEnv):
    """Wrap ``UvmsBatchEnv`` in the interface consumed by RSL-RL runners."""

    def __init__(
        self,
        env: UvmsBatchEnv,
        *,
        cfg: dict[str, Any] | None = None,
        clip_actions: float | None = None,
        action_scale: float | list[float] | tuple[float, ...] | None = None,
        residual_teacher: dict[str, Any] | None = None,
        residual_scale: float | list[float] | tuple[float, ...] | None = None,
        reset_on_init: bool = True,
    ):
        self.unwrapped = env
        self.num_envs = int(env.robot_count)
        self.num_actions = int(env.action_dim)
        self.max_episode_length = int(env.max_episode_steps)
        self.device = torch.device(env.device if env.device is not None else "cpu")
        self.cfg = dict(cfg or {})
        self.clip_actions = None if clip_actions is None else float(clip_actions)
        self.action_scale = self._make_action_scale(action_scale)
        self.residual_teacher = dict(residual_teacher or {})
        self.residual_scale = self._make_action_scale(residual_scale)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._obs = env.reset() if reset_on_init else env.observations()
        self._sync_episode_length_buf()

    @property
    def num_obs(self) -> int:
        return int(self.unwrapped.policy_observation_dim)

    @property
    def num_privileged_obs(self) -> int:
        return self.num_obs

    def get_observations(self) -> TensorDict:
        return self._observation_dict(self._obs)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        self._push_episode_length_buf()
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        if self.residual_teacher:
            base_actions = self.teacher_actions(self._observation_dict(self._obs), self.residual_teacher)
            residual_actions = actions
            if self.residual_scale is not None:
                residual_actions = residual_actions * self.residual_scale
            actions = base_actions + residual_actions
            if self.clip_actions is not None:
                actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        action_arg = actions.to(self.device)
        if self.action_scale is not None:
            action_arg = action_arg * self.action_scale
        if self.unwrapped.backend == "cpu":
            action_arg = action_arg.detach().cpu().numpy().astype(np.float32, copy=False)

        obs, rewards, dones, info = self.unwrapped.step(action_arg)
        timeout = self._timeout_tensor()
        dones_t = self._to_torch(dones, dtype=torch.bool)
        rewards_t = self._to_torch(rewards, dtype=torch.float32)
        if bool(torch.any(dones_t).item()):
            obs = self.unwrapped.reset_done(dones_t)
        self._obs = obs
        self._sync_episode_length_buf()
        extras = self._extras(info, timeout)
        return self._observation_dict(obs), rewards_t, dones_t, extras

    def reset(self) -> TensorDict:
        self._obs = self.unwrapped.reset()
        self._sync_episode_length_buf()
        return self.get_observations()

    @classmethod
    def from_experiment(
        cls,
        name_or_path: str,
        *,
        backend: str | None = None,
        robot_count: int | None = None,
        clip_actions: float | None = None,
        action_scale: float | list[float] | tuple[float, ...] | None = None,
    ) -> "RslRlUvmsEnv":
        experiment = load_experiment(name_or_path)
        env_cfg = dict(experiment.config.get("env", {}))
        task_cfg = dict(experiment.config.get("task", {}))
        if backend is not None:
            env_cfg["backend"] = backend
        if robot_count is not None:
            env_cfg["robot_count"] = int(robot_count)
        if "dynamics_profile" not in env_cfg:
            raise ValueError(f"experiment '{experiment.name}' is missing env.dynamics_profile")
        dynamics_profile = env_cfg["dynamics_profile"]
        print(
            "uvms_rl active profile:",
            f"experiment={experiment.name}",
            f"backend={env_cfg.get('backend', 'cpu')}",
            f"dynamics_profile={dynamics_profile}",
        )
        env = UvmsBatchEnv(
            robot_count=int(env_cfg.get("robot_count", 1024)),
            control_dt=float(env_cfg.get("control_dt", env_cfg.get("dt", 0.01))),
            sim_dt=float(env_cfg.get("sim_dt", env_cfg.get("control_dt", env_cfg.get("dt", 0.01)))),
            max_episode_steps=int(env_cfg.get("max_episode_steps", 500)),
            seed=env_cfg.get("seed"),
            task=experiment.task_cls,
            task_config=task_cfg,
            backend=str(env_cfg.get("backend", "cpu")),
            dynamics_profile=dynamics_profile,
        )
        return cls(env, cfg=experiment.config, clip_actions=clip_actions, action_scale=action_scale)

    def configure_residual_teacher(
        self,
        teacher: dict[str, Any] | None,
        residual_scale: float | list[float] | tuple[float, ...] | None = None,
    ) -> None:
        self.residual_teacher = dict(teacher or {})
        self.residual_scale = self._make_action_scale(residual_scale)

    def teacher_actions(self, obs: TensorDict, teacher: dict[str, Any] | None = None) -> torch.Tensor:
        teacher_cfg = dict(teacher or self.residual_teacher)
        name = str(teacher_cfg.get("name", "")).strip().lower()
        if name != "pd_hover":
            raise ValueError(f"unsupported residual teacher '{name}'")

        policy_obs = obs["policy"].to(self.device)
        actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)

        force_kp = float(teacher_cfg.get("force_kp", 120.0))
        force_kd = float(teacher_cfg.get("force_kd", 45.0))
        yaw_kp = float(teacher_cfg.get("yaw_kp", 8.0))
        angular_kd = float(teacher_cfg.get("angular_kd", 3.0))

        xyz_error = policy_obs[:, 0:3]
        yaw_error = policy_obs[:, 3]
        linear_velocity = policy_obs[:, 4:7]
        angular_velocity = policy_obs[:, 7:10]

        actions[:, 0:3] = force_kp * xyz_error - force_kd * linear_velocity
        actions[:, 3:6] = -angular_kd * angular_velocity
        actions[:, 5] += yaw_kp * yaw_error

        if self.action_scale is None:
            return actions

        scale = self.action_scale.to(self.device)
        finite_limits = scale > 0.0
        actions = torch.where(finite_limits, torch.clamp(actions, -scale, scale), torch.zeros_like(actions))
        normalized = torch.zeros_like(actions)
        normalized[:, finite_limits] = actions[:, finite_limits] / scale[finite_limits]
        return normalized

    def _observation_dict(self, obs) -> TensorDict:
        policy = self._to_torch(obs, dtype=torch.float32)
        return TensorDict({"policy": policy}, batch_size=[self.num_envs], device=self.device)

    def _to_torch(self, value, *, dtype: torch.dtype) -> torch.Tensor:
        if is_torch_tensor(value):
            return value.to(device=self.device, dtype=dtype)
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _make_action_scale(self, action_scale):
        if action_scale is None:
            return None
        if isinstance(action_scale, (int, float)):
            return torch.full((self.num_actions,), float(action_scale), dtype=torch.float32, device=self.device)
        scale = torch.as_tensor(action_scale, dtype=torch.float32, device=self.device)
        if tuple(scale.shape) != (self.num_actions,):
            raise ValueError(f"action_scale must be scalar or shape {(self.num_actions,)}, got {tuple(scale.shape)}")
        return scale

    def _timeout_tensor(self) -> torch.Tensor:
        return self._to_torch(self.unwrapped.episode_steps, dtype=torch.long) >= self.max_episode_length

    def _sync_episode_length_buf(self) -> None:
        with torch.inference_mode(False):
            self.episode_length_buf = self._to_torch(self.unwrapped.episode_steps, dtype=torch.long).clone()

    def _push_episode_length_buf(self) -> None:
        if self.unwrapped.backend == "gpu":
            with torch.inference_mode(False):
                self.unwrapped.episode_steps = self.episode_length_buf.to(device=self.device, dtype=torch.int32).clone()
        else:
            self.unwrapped.episode_steps = self.episode_length_buf.detach().cpu().numpy().astype(np.int32, copy=True)

    def _extras(self, info, timeout: torch.Tensor) -> dict[str, Any]:
        task_info = getattr(info, "task", {}) or {}
        log = {
            "/uvms_rl/sim_time": float(getattr(info, "sim_time", 0.0)),
            "/uvms_rl/control_dt": float(getattr(info, "control_dt", self.unwrapped.control_dt)),
            "/uvms_rl/sim_dt": float(getattr(info, "sim_dt", self.unwrapped.sim_dt)),
        }
        for key, value in task_info.items():
            log[f"/uvms_rl/{key}"] = value
        return {"time_outs": timeout, "log": log}
