"""Task interface for vectorized UVMS RL experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class TaskBase(ABC):
    """Defines reset, policy observation, reward, and done logic."""

    name = "base"

    def __init__(self, robot_count: int, config: dict[str, Any] | None = None, seed: int | None = None):
        self.robot_count = int(robot_count)
        self.config = dict(config or {})
        self.rng = np.random.default_rng(seed)

    @property
    @abstractmethod
    def policy_observation_dim(self) -> int:
        """Number of values returned to the policy per environment."""

    @abstractmethod
    def reset(self, env) -> np.ndarray:
        """Return initial raw simulator observations with shape [N, 22]."""

    def reset_indices(self, env, indices: np.ndarray) -> np.ndarray:
        """Return reset raw simulator observations for selected environment rows."""

        raise NotImplementedError(f"{type(self).__name__} does not implement partial reset")

    @abstractmethod
    def policy_observation(self, env, sim_obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Map raw simulator observations to task-relative policy observations."""

    @abstractmethod
    def reward_done(self, env, sim_obs: np.ndarray, actions: np.ndarray):
        """Return reward [N], done [N], and task info dict."""
