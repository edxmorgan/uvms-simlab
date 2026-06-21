from __future__ import annotations

from simlab.dynamic_obstacle_sources.base import (
    DynamicObstacleSourceRequest,
    DynamicObstacleSourceResult,
    DynamicObstacleSourceTemplate,
)
from simlab.utils.path_obstacles import make_path_obstacle


class PathSphereObstacleSource(DynamicObstacleSourceTemplate):
    """Place a spherical obstacle ahead of the selected robot on its active path."""

    registry_name = "path_sphere"

    def create(self, request: DynamicObstacleSourceRequest) -> DynamicObstacleSourceResult | None:
        placement = make_path_obstacle(
            robot=request.robot,
            existing_obstacles=request.existing_obstacles,
            world_frame=request.world_frame,
            distance_ahead=max(0.5, float(request.distance_ahead or 4.0)),
            radius=max(0.05, float(request.radius or 0.8)),
            name=request.name,
        )
        if placement is None:
            return None
        return DynamicObstacleSourceResult(
            obstacle=placement.obstacle,
            center_world=placement.center_world,
            detail_fields={
                "path_ahead_m": placement.distance_along_path_m,
                "euclidean_from_robot_m": placement.distance_from_robot_m,
                "remaining_path_m": placement.remaining_path_m,
                "nearest_path_index": placement.nearest_path_index,
            },
        )
