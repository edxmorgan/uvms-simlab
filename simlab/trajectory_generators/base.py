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
    def start_from_path(
        self,
        current_position: Sequence[float],
        path_xyz: np.ndarray,
        max_vel: Sequence[float],
        max_acc: Sequence[float],
        max_jerk: Sequence[float],
    ) -> None:
        pass

    @abstractmethod
    def update(self, yaw_blend_factor: float):
        pass

    @abstractmethod
    def close(self) -> None:
        pass
