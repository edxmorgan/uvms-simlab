"""Vehicle hover task for the batched UVMS environment."""

from __future__ import annotations

from typing import Any

import numpy as np

from uvms_rl.task_base import TaskBase
from uvms_rl.tensor import is_torch_tensor


def _range_pair(config: dict[str, Any], key: str, default: tuple[float, float]) -> tuple[float, float]:
    value = config.get(key, default)
    if isinstance(value, dict):
        raise ValueError(f"{key} must be a two-element range, not a mapping")
    if len(value) != 2:
        raise ValueError(f"{key} must contain [low, high]")
    return float(value[0]), float(value[1])


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _wrap_to_pi_torch(angle):
    import torch

    return torch.remainder(angle + torch.pi, 2.0 * torch.pi) - torch.pi


class HoverVehicleTask(TaskBase):
    """Reach and hold a target xyz/yaw with the vehicle base."""

    name = "hover_vehicle"

    def __init__(self, robot_count: int, config: dict[str, Any] | None = None, seed: int | None = None):
        super().__init__(robot_count=robot_count, config=config, seed=seed)
        self.action_dim = int(self.config.get("action_dim", 11))
        self.target_xyz_tensor = None
        self.target_yaw_tensor = None
        self.success_streak_tensor = None
        self.previous_tracking_cost_tensor = None

    @property
    def policy_observation_dim(self) -> int:
        # xyz error, yaw error, uvw, pqr, arm q, arm qd, previous action
        return 3 + 1 + 3 + 3 + 5 + 5 + self.action_dim

    def reset(self, env) -> np.ndarray:
        indices = np.arange(self.robot_count, dtype=np.int64)
        obs = self._reset_rows(env, indices)
        return obs

    def reset_indices(self, env, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("indices must be a one-dimensional array")
        return self._reset_rows(env, indices)

    def _reset_rows(self, env, indices: np.ndarray) -> np.ndarray:
        cfg = self.config
        count = int(indices.size)
        obs = np.zeros((count, env.observation_dim), dtype=np.float32)
        if count == 0:
            return obs
        x_low, x_high = _range_pair(cfg, "target_x", (-2.0, 2.0))
        y_low, y_high = _range_pair(cfg, "target_y", (-2.0, 2.0))
        z_low, z_high = _range_pair(cfg, "target_z", (-1.5, -0.2))
        yaw_low, yaw_high = _range_pair(cfg, "target_yaw", (-np.pi, np.pi))

        target_xyz = np.column_stack(
            [
                self.rng.uniform(x_low, x_high, size=count),
                self.rng.uniform(y_low, y_high, size=count),
                self.rng.uniform(z_low, z_high, size=count),
            ]
        ).astype(np.float32)
        target_yaw = self.rng.uniform(yaw_low, yaw_high, size=count).astype(np.float32)

        if not hasattr(self, "target_xyz") or len(getattr(self, "target_xyz", [])) != self.robot_count:
            self.target_xyz = np.zeros((self.robot_count, 3), dtype=np.float32)
            self.target_yaw = np.zeros(self.robot_count, dtype=np.float32)
            self.success_streak = np.zeros(self.robot_count, dtype=np.int32)
            self.previous_tracking_cost = np.zeros(self.robot_count, dtype=np.float32)
        self.target_xyz[indices] = target_xyz
        self.target_yaw[indices] = target_yaw
        self.success_streak[indices] = 0

        xyz_noise = float(cfg.get("initial_xyz_noise", 2.0))
        yaw_noise = float(cfg.get("initial_yaw_noise", np.pi))
        velocity_noise = float(cfg.get("initial_velocity_noise", 0.05))

        obs[:, 0:3] = target_xyz + self.rng.uniform(-xyz_noise, xyz_noise, size=(count, 3)).astype(np.float32)
        obs[:, 5] = target_yaw + self.rng.uniform(-yaw_noise, yaw_noise, size=count).astype(np.float32)
        obs[:, 6:12] = self.rng.uniform(-velocity_noise, velocity_noise, size=(count, 6)).astype(np.float32)
        self.previous_tracking_cost[indices] = self._tracking_cost_np(
            np.linalg.norm(obs[:, 0:3] - target_xyz, axis=1),
            np.abs(_wrap_to_pi(obs[:, 5] - target_yaw)),
        )
        if self.target_xyz_tensor is not None and getattr(env, "backend", "cpu") == "gpu":
            import torch

            with torch.inference_mode(False):
                torch_indices = torch.as_tensor(indices, dtype=torch.long, device=env.device)
                if self.target_xyz_tensor.is_inference():
                    self.target_xyz_tensor = self.target_xyz_tensor.clone()
                if self.target_yaw_tensor.is_inference():
                    self.target_yaw_tensor = self.target_yaw_tensor.clone()
                if self.success_streak_tensor.is_inference():
                    self.success_streak_tensor = self.success_streak_tensor.clone()
                if self.previous_tracking_cost_tensor is None:
                    self.previous_tracking_cost_tensor = torch.zeros(
                        self.robot_count, dtype=torch.float32, device=env.device
                    )
                if self.previous_tracking_cost_tensor.is_inference():
                    self.previous_tracking_cost_tensor = self.previous_tracking_cost_tensor.clone()
                self.target_xyz_tensor[torch_indices] = torch.as_tensor(target_xyz, dtype=torch.float32, device=env.device)
                self.target_yaw_tensor[torch_indices] = torch.as_tensor(target_yaw, dtype=torch.float32, device=env.device)
                self.success_streak_tensor[torch_indices] = 0
                self.previous_tracking_cost_tensor[torch_indices] = torch.as_tensor(
                    self.previous_tracking_cost[indices], dtype=torch.float32, device=env.device
                )
        return obs

    def prepare_backend(self, env) -> None:
        if getattr(env, "backend", "cpu") != "gpu":
            return
        import torch

        with torch.inference_mode(False):
            self.target_xyz_tensor = torch.as_tensor(self.target_xyz, dtype=torch.float32, device=env.device).clone()
            self.target_yaw_tensor = torch.as_tensor(self.target_yaw, dtype=torch.float32, device=env.device).clone()
            self.success_streak_tensor = torch.zeros(self.robot_count, dtype=torch.int32, device=env.device)
            self.previous_tracking_cost_tensor = torch.as_tensor(
                self.previous_tracking_cost, dtype=torch.float32, device=env.device
            ).clone()

    def sync_reset_observations(self, env, observations: np.ndarray) -> None:
        """Align progress shaping state after an explicit simulator-state reset."""
        obs = np.asarray(observations, dtype=np.float32)
        if obs.shape != (self.robot_count, env.observation_dim):
            raise ValueError(f"observations must have shape {(self.robot_count, env.observation_dim)}, got {obs.shape}")
        self.previous_tracking_cost = self._tracking_cost_np(
            np.linalg.norm(obs[:, 0:3] - self.target_xyz, axis=1),
            np.abs(_wrap_to_pi(obs[:, 5] - self.target_yaw)),
        )
        self.success_streak.fill(0)
        if getattr(env, "backend", "cpu") == "gpu" and self.previous_tracking_cost_tensor is not None:
            import torch

            with torch.inference_mode(False):
                if self.previous_tracking_cost_tensor.is_inference():
                    self.previous_tracking_cost_tensor = self.previous_tracking_cost_tensor.clone()
                if self.success_streak_tensor is not None and self.success_streak_tensor.is_inference():
                    self.success_streak_tensor = self.success_streak_tensor.clone()
                self.previous_tracking_cost_tensor[:] = torch.as_tensor(
                    self.previous_tracking_cost, dtype=torch.float32, device=env.device
                )
                if self.success_streak_tensor is not None:
                    self.success_streak_tensor.zero_()

    def policy_observation(self, env, sim_obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        if is_torch_tensor(sim_obs):
            import torch

            self._require_torch_targets(env)
            xyz = sim_obs[:, 0:3]
            yaw = sim_obs[:, 5]
            uvw = sim_obs[:, 6:9]
            pqr = sim_obs[:, 9:12]
            arm_q = sim_obs[:, 12:17]
            arm_qd = sim_obs[:, 17:22]
            xyz_error = self.target_xyz_tensor - xyz
            yaw_error = _wrap_to_pi_torch(self.target_yaw_tensor - yaw)[:, None]
            return torch.cat([xyz_error, yaw_error, uvw, pqr, arm_q, arm_qd, actions], dim=1)

        xyz = sim_obs[:, 0:3]
        yaw = sim_obs[:, 5]
        uvw = sim_obs[:, 6:9]
        pqr = sim_obs[:, 9:12]
        arm_q = sim_obs[:, 12:17]
        arm_qd = sim_obs[:, 17:22]

        xyz_error = self.target_xyz - xyz
        yaw_error = _wrap_to_pi(self.target_yaw - yaw)[:, None]
        return np.concatenate([xyz_error, yaw_error, uvw, pqr, arm_q, arm_qd, actions], axis=1).astype(np.float32)

    def reward_done(self, env, sim_obs: np.ndarray, actions: np.ndarray):
        if is_torch_tensor(sim_obs):
            return self._reward_done_torch(env, sim_obs, actions)

        cfg = self.config
        xyz = sim_obs[:, 0:3]
        yaw = sim_obs[:, 5]
        uvw = sim_obs[:, 6:9]
        pqr = sim_obs[:, 9:12]

        pos_err = np.linalg.norm(xyz - self.target_xyz, axis=1)
        yaw_err = np.abs(_wrap_to_pi(yaw - self.target_yaw))
        lin_speed = np.linalg.norm(uvw, axis=1)
        ang_speed = np.linalg.norm(pqr, axis=1)
        effort = np.linalg.norm(actions, axis=1)
        tracking_cost = self._tracking_cost_np(pos_err, yaw_err)
        progress = self.previous_tracking_cost - tracking_cost
        self.previous_tracking_cost = tracking_cost.astype(np.float32)
        tracking_reward = float(cfg.get("tracking_reward_weight", 0.0)) * np.exp(
            -float(cfg.get("tracking_position_scale", 2.0)) * np.square(pos_err)
            -float(cfg.get("tracking_yaw_scale", 1.0)) * np.square(yaw_err)
        )

        reward = (
            -tracking_cost
            + float(cfg.get("progress_weight", 0.0)) * progress
            + tracking_reward
            -float(cfg.get("linear_velocity_weight", 0.1)) * lin_speed
            -float(cfg.get("angular_velocity_weight", 0.05)) * ang_speed
            -float(cfg.get("action_weight", 0.01)) * effort
        ).astype(np.float32)

        pos_tol = float(cfg.get("success_position_tolerance", 0.15))
        yaw_tol = float(cfg.get("success_yaw_tolerance", 0.15))
        required_streak = int(cfg.get("success_streak_steps", 10))
        success_now = (pos_err < pos_tol) & (yaw_err < yaw_tol)
        self.success_streak = np.where(success_now, self.success_streak + 1, 0)
        success = self.success_streak >= required_streak

        timeout = env.episode_steps >= env.max_episode_steps
        out_of_bounds = np.linalg.norm(xyz, axis=1) > float(cfg.get("out_of_bounds_radius", 10.0))

        reward += success.astype(np.float32) * float(cfg.get("success_bonus", 10.0))
        reward -= out_of_bounds.astype(np.float32) * float(cfg.get("out_of_bounds_penalty", 10.0))
        done = success | timeout | out_of_bounds

        info = {
            "mean_position_error": float(np.mean(pos_err)),
            "mean_yaw_error": float(np.mean(yaw_err)),
            "success_rate": float(np.mean(success)),
            "timeout_rate": float(np.mean(timeout)),
            "out_of_bounds_rate": float(np.mean(out_of_bounds)),
            "mean_progress": float(np.mean(progress)),
            "mean_tracking_reward": float(np.mean(tracking_reward)),
        }
        return reward, done, info

    def _require_torch_targets(self, env) -> None:
        import torch

        def is_env_device(tensor) -> bool:
            if tensor is None:
                return False
            env_device = torch.device(env.device)
            return tensor.device.type == env_device.type and (
                env_device.index is None or tensor.device.index == env_device.index
            )

        if (
            not is_env_device(self.target_xyz_tensor)
            or not is_env_device(self.target_yaw_tensor)
            or not is_env_device(self.previous_tracking_cost_tensor)
        ):
            self.prepare_backend(env)

    def _tracking_cost_np(self, pos_err: np.ndarray, yaw_err: np.ndarray) -> np.ndarray:
        return (
            float(self.config.get("position_weight", 2.0)) * pos_err
            + float(self.config.get("yaw_weight", 0.5)) * yaw_err
        ).astype(np.float32)

    def _reward_done_torch(self, env, sim_obs, actions):
        import torch

        self._require_torch_targets(env)
        cfg = self.config
        xyz = sim_obs[:, 0:3]
        yaw = sim_obs[:, 5]
        uvw = sim_obs[:, 6:9]
        pqr = sim_obs[:, 9:12]

        pos_err = torch.linalg.norm(xyz - self.target_xyz_tensor, dim=1)
        yaw_err = torch.abs(_wrap_to_pi_torch(yaw - self.target_yaw_tensor))
        lin_speed = torch.linalg.norm(uvw, dim=1)
        ang_speed = torch.linalg.norm(pqr, dim=1)
        effort = torch.linalg.norm(actions, dim=1)
        tracking_cost = (
            float(cfg.get("position_weight", 2.0)) * pos_err
            + float(cfg.get("yaw_weight", 0.5)) * yaw_err
        ).to(torch.float32)
        progress = self.previous_tracking_cost_tensor - tracking_cost
        self.previous_tracking_cost_tensor = tracking_cost.clone()
        tracking_reward = float(cfg.get("tracking_reward_weight", 0.0)) * torch.exp(
            -float(cfg.get("tracking_position_scale", 2.0)) * torch.square(pos_err)
            -float(cfg.get("tracking_yaw_scale", 1.0)) * torch.square(yaw_err)
        )

        reward = (
            -tracking_cost
            + float(cfg.get("progress_weight", 0.0)) * progress
            + tracking_reward
            -float(cfg.get("linear_velocity_weight", 0.1)) * lin_speed
            -float(cfg.get("angular_velocity_weight", 0.05)) * ang_speed
            -float(cfg.get("action_weight", 0.01)) * effort
        ).to(torch.float32)

        pos_tol = float(cfg.get("success_position_tolerance", 0.15))
        yaw_tol = float(cfg.get("success_yaw_tolerance", 0.15))
        required_streak = int(cfg.get("success_streak_steps", 10))
        success_now = (pos_err < pos_tol) & (yaw_err < yaw_tol)
        self.success_streak_tensor = torch.where(
            success_now,
            self.success_streak_tensor + 1,
            torch.zeros_like(self.success_streak_tensor),
        )
        success = self.success_streak_tensor >= required_streak

        timeout = env.episode_steps >= env.max_episode_steps
        out_of_bounds = torch.linalg.norm(xyz, dim=1) > float(cfg.get("out_of_bounds_radius", 10.0))

        reward = reward + success.to(torch.float32) * float(cfg.get("success_bonus", 10.0))
        reward = reward - out_of_bounds.to(torch.float32) * float(cfg.get("out_of_bounds_penalty", 10.0))
        done = success | timeout | out_of_bounds

        info = {
            "mean_position_error": float(torch.mean(pos_err).detach().cpu().item()),
            "mean_yaw_error": float(torch.mean(yaw_err).detach().cpu().item()),
            "success_rate": float(torch.mean(success.to(torch.float32)).detach().cpu().item()),
            "timeout_rate": float(torch.mean(timeout.to(torch.float32)).detach().cpu().item()),
            "out_of_bounds_rate": float(torch.mean(out_of_bounds.to(torch.float32)).detach().cpu().item()),
            "mean_progress": float(torch.mean(progress).detach().cpu().item()),
            "mean_tracking_reward": float(torch.mean(tracking_reward).detach().cpu().item()),
        }
        return reward, done, info


Task = HoverVehicleTask
