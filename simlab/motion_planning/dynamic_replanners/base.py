from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simlab.robot import Robot
    from simlab.uvms_backend import UVMSBackendCore
    from simlab.vehicle_waypoint_mission import VehicleWaypointMission


@dataclass(frozen=True)
class ReplanDecision:
    action: str
    reason: str = ""
    obstacle_id: str = ""
    clearance_m: float | None = None
    path_signature: str = ""
    t_offset_s: float | None = None

    @property
    def should_replan(self) -> bool:
        return self.action == "replan"


@dataclass(frozen=True)
class TimedPathSample:
    xyz: np.ndarray
    t_offset: float


class DynamicReplannerTemplate(ABC):
    """Base interface for per-robot dynamic replanning supervisors."""

    registry_name = ""
    visible = True

    @abstractmethod
    def configure(
        self,
        *,
        cooldown_s: float | None = None,
        lookahead_time_s: float | None = None,
        safety_margin_m: float | None = None,
        collision_stop_enabled: bool | None = None,
        collision_stop_margin_m: float | None = None,
        max_samples: int | None = None,
        replan_hysteresis_m: float | None = None,
    ) -> None:
        pass

    @abstractmethod
    def tick(self) -> None:
        pass

    @abstractmethod
    def evaluate(self) -> ReplanDecision:
        pass

    @abstractmethod
    def status_summary(self) -> str:
        pass

    @abstractmethod
    def reset_history(self, *, reset_count: bool = False) -> None:
        pass
