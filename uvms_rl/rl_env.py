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
from uvms_rl.tasks.base import TaskBase
from uvms_rl.tasks.registry import make_task


@dataclass(frozen=True)
class UvmsBatchInfo:
    tick_id: int
    sim_time: float
    step_count: float
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
        dt: float = 0.01,
        *,
        max_episode_steps: int = 500,
        seed: int | None = None,
        task: str | TaskBase | None = None,
        task_config: dict[str, Any] | None = None,
    ):
        if robot_count < 1:
            raise ValueError("robot_count must be at least 1")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be at least 1")

        self.robot_count = int(robot_count)
        self.dt = float(dt)
        self.max_episode_steps = int(max_episode_steps)
        self.rng = np.random.default_rng(seed)
        self.episode_steps = np.zeros(self.robot_count, dtype=np.int32)
        self._tick_id = 0
        self._last_task_info: dict[str, Any] = {}
        self._core = _batch_uvms_core.BatchUvmsCore(self.robot_count)
        self._previous_actions = np.zeros((self.robot_count, self.action_dim), dtype=np.float32)

        if isinstance(task, str):
            self.task = make_task(task, robot_count=self.robot_count, config=task_config, seed=seed)
        else:
            self.task = task
        self.policy_observation_dim = (
            self.task.policy_observation_dim if self.task is not None else self.observation_dim
        )

    def reset(self, *, hold_commands: bool = False, observations: np.ndarray | None = None) -> np.ndarray:
        if self.task is not None and observations is None:
            obs_arg = self.task.reset(self)
        else:
            obs_arg = None if observations is None else np.asarray(observations, dtype=np.float32)
            if self.task is not None:
                self.task.reset(self)

        self._core.reset(bool(hold_commands), obs_arg)
        self._tick_id = int(self._core.tick_id)
        self.episode_steps.fill(0)
        self._previous_actions.fill(0.0)
        self._last_task_info = {}
        return self._policy_observations(self.sim_observations(), self._previous_actions)

    def step(self, actions: np.ndarray):
        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.shape != (self.robot_count, self.action_dim):
            raise ValueError(
                f"actions must have shape {(self.robot_count, self.action_dim)}, got {action_array.shape}"
            )
        self._tick_id += 1
        self._core.set_actions(action_array, self._tick_id)
        self._core.step(self.dt)
        self.episode_steps += 1

        sim_obs = self.sim_observations()
        if self.task is None:
            rewards = self.rewards()
            dones = self.dones()
            policy_obs = sim_obs
            task_info = {}
        else:
            rewards, dones, task_info = self.task.reward_done(self, sim_obs, action_array)
            rewards = np.asarray(rewards, dtype=np.float32)
            dones = np.asarray(dones, dtype=bool)
            policy_obs = self.task.policy_observation(self, sim_obs, action_array)

        self._previous_actions = action_array.copy()
        self._last_task_info = dict(task_info)
        return policy_obs, rewards, dones, self.info(task_info)

    def _policy_observations(self, sim_obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        if self.task is None:
            return sim_obs
        return self.task.policy_observation(self, sim_obs, actions)

    def sim_observations(self) -> np.ndarray:
        return np.asarray(self._core.observations(), dtype=np.float32)

    def observations(self) -> np.ndarray:
        return self._policy_observations(self.sim_observations(), self._previous_actions)

    def rewards(self) -> np.ndarray:
        return np.asarray(self._core.rewards(), dtype=np.float32)

    def dones(self) -> np.ndarray:
        return np.asarray(self._core.dones(), dtype=np.uint8).astype(bool)

    def actions(self) -> np.ndarray:
        return np.asarray(self._core.actions(), dtype=np.float32)

    def info(self, task_info: dict[str, Any] | None = None) -> UvmsBatchInfo:
        return UvmsBatchInfo(
            tick_id=int(self._core.tick_id),
            sim_time=float(self._core.sim_time),
            step_count=float(self._core.step_count),
            task=dict(self._last_task_info if task_info is None else task_info),
        )
