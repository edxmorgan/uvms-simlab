from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class VehicleTrajectoryGeneratorTemplate(ABC):
    """Base interface for vehicle Cartesian trajectory generators."""

    registry_name = ""
    visible = True

    active: bool

    @abstractmethod
    def current_reference(self):
        """Return owned XYZ position, velocity, acceleration arrays, or None.

        This read-only snapshot uses the execution clock without advancing it.
        It is available immediately after start, and absent when inactive.
        """
        pass

    def preview_samples(self, *, horizon=None, sample_dt=0.05):
        """Return (seconds-from-now, xyz) for the actual generated trajectory.

        Unsupported generators fail closed at the execution validation boundary.
        Preview must not advance the execution clock or consume a control step.
        """
        raise NotImplementedError('trajectory generator does not implement timed preview')

    @abstractmethod
    def start_from_path(
        self,
        current_position: Sequence[float],
        path_xyz: np.ndarray,
        max_vel: Sequence[float],
        max_acc: Sequence[float],
        max_jerk: Sequence[float],
        current_velocity: Sequence[float] | None = None,
    ) -> None:
        pass

    @abstractmethod
    def update(self, yaw_blend_factor: float):
        pass

    @abstractmethod
    def close(self) -> None:
        pass
