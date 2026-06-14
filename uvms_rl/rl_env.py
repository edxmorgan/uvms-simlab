# Copyright (C) 2026 Edward Morgan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.

"""Vectorized Python wrapper for BatchUvmsCore."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ros2_control_blue_reach_5 import _batch_uvms_core
from simlab.dynamics_profiles import load_robot_dynamics_profile, required_float, required_vector
from uvms_rl.task_base import TaskBase
from uvms_rl.tensor import torch_available, torch_cuda_tensor_from_ptr


@dataclass(frozen=True)
class UvmsBatchInfo:
    tick_id: int
    sim_time: float
    step_count: float
    control_dt: float
    sim_dt: float
    substeps: int
    backend: str
    task: dict[str, Any] = field(default_factory=dict)


class UvmsBatchEnv:
    """Small Gym-like vector environment over the C++ batch UVMS core."""

    observation_dim = int(_batch_uvms_core.OBSERVATION_DIM)
    action_dim = int(_batch_uvms_core.ACTION_DIM)
    vehicle_state_dim = int(_batch_uvms_core.VEHICLE_STATE_DIM)
    vehicle_action_dim = int(_batch_uvms_core.VEHICLE_ACTION_DIM)
    arm_joint_count = int(_batch_uvms_core.ARM_JOINT_COUNT)
    arm_action_dim = int(_batch_uvms_core.ARM_ACTION_DIM)

    def __init__(
        self,
        robot_count: int,
        dt: float | None = 0.01,
        *,
        control_dt: float | None = None,
        sim_dt: float | None = None,
        max_episode_steps: int = 500,
        seed: int | None = None,
        task: type[TaskBase] | TaskBase | None = None,
        task_config: dict[str, Any] | None = None,
        backend: str = "cpu",
        dynamics_profile: str | dict[str, Any] | None = "dory_alpha",
    ):
        if robot_count < 1:
            raise ValueError("robot_count must be at least 1")
        if control_dt is None:
            control_dt = 0.01 if dt is None else dt
        if sim_dt is None:
            sim_dt = control_dt
        if control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        if sim_dt <= 0.0:
            raise ValueError("sim_dt must be positive")
        if sim_dt > control_dt:
            raise ValueError("sim_dt must be less than or equal to control_dt")
        substeps_float = control_dt / sim_dt
        substeps = int(round(substeps_float))
        if substeps < 1 or not np.isclose(substeps * sim_dt, control_dt, rtol=1e-7, atol=1e-12):
            raise ValueError("control_dt must be an integer multiple of sim_dt")
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be at least 1")

        self.robot_count = int(robot_count)
        self.control_dt = float(control_dt)
        self.sim_dt = float(sim_dt)
        self.substeps = int(substeps)
        self.dt = self.control_dt
        self.max_episode_steps = int(max_episode_steps)
        self.backend = self._normalize_backend(backend)
        self.device = None
        if self.backend == "gpu":
            if not torch_available():
                raise RuntimeError("uvms_rl backend='gpu' requires PyTorch with CUDA available")
            import torch

            self.device = torch.device("cuda")
        self.rng = np.random.default_rng(seed)
        if self.backend == "gpu":
            import torch

            self.episode_steps = torch.zeros(self.robot_count, dtype=torch.int32, device=self.device)
        else:
            self.episode_steps = np.zeros(self.robot_count, dtype=np.int32)
        self._tick_id = 0
        self._last_task_info: dict[str, Any] = {}
        self._core = self._make_core(self.backend)
        self._dynamics_profile = load_dynamics_profile(dynamics_profile)
        if self.backend == "gpu":
            import torch

            self._previous_actions = torch.zeros((self.robot_count, self.action_dim), dtype=torch.float32, device=self.device)
        else:
            self._previous_actions = np.zeros((self.robot_count, self.action_dim), dtype=np.float32)

        task_config_with_shape = dict(task_config or {})
        task_config_with_shape["action_dim"] = self.action_dim
        if isinstance(task, type) and issubclass(task, TaskBase):
            self.task = task(robot_count=self.robot_count, config=task_config_with_shape, seed=seed)
        elif isinstance(task, str):
            raise TypeError(
                "UvmsBatchEnv no longer loads tasks by registry name. "
                "Use uvms_rl.config.load_experiment(...) and pass experiment.task_cls."
            )
        else:
            self.task = task
        self.policy_observation_dim = (
            self.task.policy_observation_dim if self.task is not None else self.observation_dim
        )

    @staticmethod
    def _normalize_backend(backend: str) -> str:
        backend_name = str(backend).strip().lower()
        if backend_name in {"", "cpu"}:
            return "cpu"
        if backend_name == "gpu":
            return "gpu"
        raise ValueError(f"unknown uvms_rl backend '{backend}'")

    def _make_core(self, backend: str):
        if backend == "cpu":
            return _batch_uvms_core.BatchUvmsCore(self.robot_count)
        try:
            from ros2_control_blue_reach_5 import _batch_uvms_gpu_core
        except ImportError as exc:
            raise RuntimeError(
                "uvms_rl backend='gpu' requested, but the GPU batch core extension is not installed. "
                "Rebuild ros2_control_blue_reach_5 with UVMS_ENABLE_CUDA_DYNAMICS=ON."
            ) from exc
        return _batch_uvms_gpu_core.BatchUvmsCore(self.robot_count)

    def reset(self, *, hold_commands: bool = False, observations: np.ndarray | None = None) -> np.ndarray:
        if self.task is not None and observations is None:
            obs_arg = self.task.reset(self)
        else:
            obs_arg = None if observations is None else np.asarray(observations, dtype=np.float32)
            if self.task is not None:
                self.task.reset(self)

        self._core.reset(bool(hold_commands), obs_arg)
        self._apply_dynamics_profile()
        self._tick_id = int(self._core.tick_id)
        if self.backend == "gpu":
            self.episode_steps.zero_()
            self._previous_actions.zero_()
            if self.task is not None and hasattr(self.task, "prepare_backend"):
                self.task.prepare_backend(self)
        else:
            self.episode_steps.fill(0)
            self._previous_actions.fill(0.0)
        self._last_task_info = {}
        return self._policy_observations(self.sim_observations(), self._previous_actions)

    def _apply_dynamics_profile(self) -> None:
        if self._dynamics_profile is None:
            return
        vehicle_params = pack_vehicle_params(self._dynamics_profile)
        if vehicle_params is not None:
            self._core.set_vehicle_params(vehicle_params)
        arm_params = pack_arm_params(self._dynamics_profile)
        if arm_params is not None:
            self._core.set_arm_params(arm_params)
        env = arm_environment(self._dynamics_profile)
        if env is not None:
            self._core.set_arm_environment(*env)

    def step(self, actions):
        if self.backend == "gpu":
            import torch

            action_array = torch.as_tensor(actions, dtype=torch.float32, device=self.device).contiguous()
        else:
            action_array = np.asarray(actions, dtype=np.float32)
        if tuple(action_array.shape) != (self.robot_count, self.action_dim):
            raise ValueError(
                f"actions must have shape {(self.robot_count, self.action_dim)}, got {action_array.shape}"
            )
        self._tick_id += 1
        if self.backend == "gpu":
            self._core.set_actions_from_device(int(action_array.data_ptr()), self._tick_id)
        else:
            self._core.set_actions(action_array, self._tick_id)
        for _ in range(self.substeps):
            self._core.step(self.sim_dt)
        self.episode_steps += 1

        sim_obs = self.sim_observations()
        if self.task is None:
            rewards = self.rewards()
            dones = self.dones()
            policy_obs = sim_obs
            task_info = {}
        else:
            rewards, dones, task_info = self.task.reward_done(self, sim_obs, action_array)
            if self.backend != "gpu":
                rewards = np.asarray(rewards, dtype=np.float32)
                dones = np.asarray(dones, dtype=bool)
            policy_obs = self.task.policy_observation(self, sim_obs, action_array)

        self._previous_actions = action_array.clone() if self.backend == "gpu" else action_array.copy()
        self._last_task_info = dict(task_info)
        return policy_obs, rewards, dones, self.info(task_info)

    def _policy_observations(self, sim_obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        if self.task is None:
            return sim_obs
        return self.task.policy_observation(self, sim_obs, actions)

    def sim_observations(self) -> np.ndarray:
        if self.backend == "gpu":
            import torch

            vehicle = torch_cuda_tensor_from_ptr(
                int(self._core.vehicle_state_ptr),
                (self.robot_count, self.vehicle_state_dim),
                self._core,
            )
            arm = torch_cuda_tensor_from_ptr(
                int(self._core.arm_state_ptr),
                (self.robot_count, self.arm_joint_count * 2),
                self._core,
            )
            return torch.cat([vehicle, arm], dim=1)
        return np.asarray(self._core.observations(), dtype=np.float32)

    def observations(self) -> np.ndarray:
        return self._policy_observations(self.sim_observations(), self._previous_actions)

    def rewards(self) -> np.ndarray:
        if self.backend == "gpu":
            import torch

            return torch.as_tensor(np.asarray(self._core.rewards(), dtype=np.float32), device=self.device)
        return np.asarray(self._core.rewards(), dtype=np.float32)

    def dones(self) -> np.ndarray:
        if self.backend == "gpu":
            import torch

            return torch.as_tensor(np.asarray(self._core.dones(), dtype=np.uint8).astype(bool), device=self.device)
        return np.asarray(self._core.dones(), dtype=np.uint8).astype(bool)

    def actions(self) -> np.ndarray:
        if self.backend == "gpu":
            return torch_cuda_tensor_from_ptr(
                int(self._core.actions_ptr),
                (self.robot_count, self.action_dim),
                self._core,
            )
        return np.asarray(self._core.actions(), dtype=np.float32)

    def info(self, task_info: dict[str, Any] | None = None) -> UvmsBatchInfo:
        return UvmsBatchInfo(
            tick_id=int(self._core.tick_id),
            sim_time=float(self._core.sim_time),
            step_count=float(self._core.step_count),
            control_dt=self.control_dt,
            sim_dt=self.sim_dt,
            substeps=self.substeps,
            backend=self.backend,
            task=dict(self._last_task_info if task_info is None else task_info),
        )


def load_dynamics_profile(profile: str | dict[str, Any] | None) -> dict[str, Any] | None:
    if profile is None:
        return None
    if isinstance(profile, dict):
        return profile
    loaded = load_robot_dynamics_profile(str(profile))
    if not isinstance(loaded, dict) or not loaded:
        raise ValueError(f"failed to load UVMS dynamics profile '{profile}'")
    return loaded


def pack_vehicle_params(profile: dict[str, Any]) -> np.ndarray | None:
    vehicle = profile.get("vehicle")
    if not isinstance(vehicle, dict):
        return None
    values = [
        required_float(vehicle, "m_x_du"),
        required_float(vehicle, "m_y_dv"),
        required_float(vehicle, "m_z_dw"),
        required_float(vehicle, "mz_g_x_dq"),
        required_float(vehicle, "mz_g_y_dp"),
        required_float(vehicle, "mz_g_k_dv"),
        required_float(vehicle, "mz_g_m_du"),
        required_float(vehicle, "i_x_k_dp"),
        required_float(vehicle, "i_y_m_dq"),
        required_float(vehicle, "i_z_n_dr"),
        required_float(vehicle, "weight"),
        required_float(vehicle, "buoyancy"),
        required_float(vehicle, "x_g_weight_minus_x_b_buoyancy"),
        required_float(vehicle, "y_g_weight_minus_y_b_buoyancy"),
        required_float(vehicle, "z_g_weight_minus_z_b_buoyancy"),
        required_float(vehicle, "x_u"),
        required_float(vehicle, "y_v"),
        required_float(vehicle, "z_w"),
        required_float(vehicle, "k_p"),
        required_float(vehicle, "m_q"),
        required_float(vehicle, "n_r"),
        required_float(vehicle, "x_uu"),
        required_float(vehicle, "y_vv"),
        required_float(vehicle, "z_ww"),
        required_float(vehicle, "k_pp"),
        required_float(vehicle, "m_qq"),
        required_float(vehicle, "n_rr"),
        *required_vector(vehicle, "current_velocity", 6),
    ]
    return np.asarray(values, dtype=np.float32)


def pack_arm_params(profile: dict[str, Any]) -> np.ndarray | None:
    manipulator = profile.get("manipulator")
    if not isinstance(manipulator, dict):
        return None
    values = [
        *required_vector(manipulator, "link_masses", 4),
        *required_vector(manipulator, "link_first_moments", 12),
        *required_vector(manipulator, "link_inertias", 24),
        *required_vector(manipulator, "viscous_friction", 4),
        *required_vector(manipulator, "coulomb_friction", 4),
        *required_vector(manipulator, "static_friction", 4),
        *required_vector(manipulator, "stribeck_velocity", 4),
        *required_vector(manipulator, "gravity_vector", 3),
        *required_vector(manipulator, "payload_com", 3),
        required_float(manipulator, "payload_mass"),
        *required_vector(manipulator, "base_pose", 6),
        *required_vector(manipulator, "world_pose", 6),
        *required_vector(manipulator, "tip_offset_pose", 6),
    ]
    if len(values) != 81:
        raise ValueError(f"packed manipulator dynamics must contain 81 values, got {len(values)}")
    return np.asarray(values, dtype=np.float32)


def arm_environment(profile: dict[str, Any]) -> tuple[float, float, float, float] | None:
    manipulator = profile.get("manipulator")
    if not isinstance(manipulator, dict):
        return None
    return (
        required_float(manipulator, "endeffector_mass"),
        required_float(manipulator, "endeffector_damping"),
        required_float(manipulator, "endeffector_stiffness"),
        required_float(manipulator, "baumgarte_alpha"),
    )
