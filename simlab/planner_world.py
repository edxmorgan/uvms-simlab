from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simlab.dynamic_world import DynamicClearance, DynamicWorldModel
from simlab.fcl_checker import FCLWorld


@dataclass(frozen=True)
class PlannerWorldClearance:
    source: str
    distance_m: float
    obstacle_id: str = ""


class PlannerWorld:
    """Common static/dynamic world query API for planner implementations."""

    def __init__(
        self,
        *,
        fcl_world: FCLWorld,
        dynamic_world: DynamicWorldModel | None = None,
    ):
        self.fcl_world = fcl_world
        self.dynamic_world = dynamic_world

    def min_clearance_xyz(
        self,
        xyz,
        *,
        t_offset: float = 0.0,
    ) -> PlannerWorldClearance:
        point = np.asarray(xyz, dtype=float).reshape(3)
        static_distance = float(self.fcl_world.min_distance_xyz(point))
        best = PlannerWorldClearance(source="static", distance_m=static_distance)

        dynamic_clearance = self._dynamic_clearance(point, t_offset=t_offset)
        if dynamic_clearance is not None and dynamic_clearance.distance_m < best.distance_m:
            return PlannerWorldClearance(
                source="dynamic",
                distance_m=float(dynamic_clearance.distance_m),
                obstacle_id=dynamic_clearance.obstacle_id,
            )
        return best

    def in_collision_xyz(self, xyz, *, t_offset: float = 0.0) -> bool:
        point = np.asarray(xyz, dtype=float).reshape(3)
        if self.fcl_world.planner_in_collision_at_xyz(point):
            return True
        dynamic_clearance = self._dynamic_clearance(point, t_offset=t_offset)
        return dynamic_clearance is not None and dynamic_clearance.distance_m <= 0.0

    def is_state_valid_xyz(
        self,
        xyz,
        *,
        safety_margin: float = 0.0,
        t_offset: float = 0.0,
    ) -> bool:
        margin = max(0.0, float(safety_margin))
        if margin > 0.0:
            return self.min_clearance_xyz(xyz, t_offset=t_offset).distance_m >= margin
        return not self.in_collision_xyz(xyz, t_offset=t_offset)

    def _dynamic_clearance(self, xyz: np.ndarray, *, t_offset: float) -> DynamicClearance | None:
        if self.dynamic_world is None:
            return None
        return self.dynamic_world.min_clearance_xyz(xyz, t_offset=t_offset)
