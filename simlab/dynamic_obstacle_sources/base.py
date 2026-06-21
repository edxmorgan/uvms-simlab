from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ros2_control_blue_reach_5.msg import DynamicObstacle, DynamicObstacleArray


@dataclass(frozen=True)
class DynamicObstacleSourceRequest:
    robot: Any
    existing_obstacles: DynamicObstacleArray
    world_frame: str
    name: str = ""
    distance_ahead: float = 4.0
    radius: float = 0.8


@dataclass(frozen=True)
class DynamicObstacleSourceResult:
    obstacle: DynamicObstacle
    center_world: np.ndarray
    detail_fields: dict[str, Any] = field(default_factory=dict)


class DynamicObstacleSourceTemplate(ABC):
    """Base interface for dynamic obstacle creation sources."""

    registry_name = ""
    visible = True

    @abstractmethod
    def create(self, request: DynamicObstacleSourceRequest) -> DynamicObstacleSourceResult | None:
        pass
