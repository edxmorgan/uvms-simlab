from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ros2_control_blue_reach_5.msg import DynamicObstacle, DynamicObstacleArray


@dataclass(frozen=True)
class PathObstaclePlacement:
    obstacle: DynamicObstacle
    center_world: np.ndarray
    distance_along_path_m: float
    distance_from_robot_m: float
    remaining_path_m: float
    nearest_path_index: int


@dataclass(frozen=True)
class PathPointAhead:
    center_world: np.ndarray
    distance_along_path_m: float
    distance_from_robot_m: float
    remaining_path_m: float
    nearest_path_index: int


def make_path_obstacle(
    *,
    robot,
    existing_obstacles: DynamicObstacleArray,
    world_frame: str,
    distance_ahead: float,
    radius: float,
    name: str = "",
) -> PathObstaclePlacement | None:
    point = path_point_ahead(robot, world_frame=world_frame, distance_ahead=max(0.5, float(distance_ahead)))
    if point is None:
        return None

    obstacle_id = str(name or "").strip() or next_path_obstacle_id(existing_obstacles)
    if dynamic_obstacle_id_exists(existing_obstacles, obstacle_id):
        raise ValueError(f"dynamic obstacle id '{obstacle_id}' already exists")

    obstacle_radius = max(0.05, float(radius))
    obstacle = DynamicObstacle()
    obstacle.id = obstacle_id
    obstacle.pose.position.x = float(point.center_world[0])
    obstacle.pose.position.y = float(point.center_world[1])
    obstacle.pose.position.z = float(point.center_world[2])
    obstacle.pose.orientation.w = 1.0
    obstacle.collision_type = DynamicObstacle.GEOMETRY_SPHERE
    obstacle.collision_dimensions = [obstacle_radius]
    obstacle.visual_type = DynamicObstacle.GEOMETRY_SPHERE
    obstacle.visual_dimensions = [obstacle_radius]
    obstacle.color.r = 1.0
    obstacle.color.g = 0.35
    obstacle.color.b = 0.05
    obstacle.color.a = 0.80
    return PathObstaclePlacement(
        obstacle=obstacle,
        center_world=point.center_world,
        distance_along_path_m=point.distance_along_path_m,
        distance_from_robot_m=point.distance_from_robot_m,
        remaining_path_m=point.remaining_path_m,
        nearest_path_index=point.nearest_path_index,
    )


def path_point_ahead(robot, *, world_frame: str, distance_ahead: float) -> PathPointAhead | None:
    planned = getattr(robot.planner, "planned_result", None)
    if not planned or not planned.get("is_success", False):
        return None
    try:
        path_xyz = np.asarray(planned.get("xyz", []), dtype=float).reshape(-1, 3)
    except Exception:
        return None
    if path_xyz.shape[0] < 2:
        return None

    pose_now = robot._pose_from_state_in_frame(world_frame)
    if pose_now is None:
        return None
    current = np.array([pose_now.position.x, pose_now.position.y, pose_now.position.z], dtype=float)
    nearest_idx = int(np.argmin(np.linalg.norm(path_xyz - current, axis=1)))
    remaining = path_xyz[max(0, nearest_idx):]
    if remaining.shape[0] < 2:
        return None

    cursor = current
    distance_left = max(0.0, float(distance_ahead))
    remaining_path_m = float(
        np.linalg.norm(remaining[0] - current)
        + np.sum(np.linalg.norm(np.diff(remaining, axis=0), axis=1))
    )
    requested_distance_m = distance_left
    for target in remaining[1:]:
        segment = target - cursor
        segment_length = float(np.linalg.norm(segment))
        if segment_length < 1e-9:
            cursor = target
            continue
        if segment_length >= distance_left:
            center = cursor + (segment / segment_length) * distance_left
            distance_along_path = min(requested_distance_m, remaining_path_m)
            return PathPointAhead(
                center_world=center,
                distance_along_path_m=distance_along_path,
                distance_from_robot_m=float(np.linalg.norm(center - current)),
                remaining_path_m=remaining_path_m,
                nearest_path_index=nearest_idx,
            )
        distance_left -= segment_length
        cursor = target
    center = remaining[-1].copy()
    return PathPointAhead(
        center_world=center,
        distance_along_path_m=remaining_path_m,
        distance_from_robot_m=float(np.linalg.norm(center - current)),
        remaining_path_m=remaining_path_m,
        nearest_path_index=nearest_idx,
    )


def dynamic_obstacle_id_exists(obstacles: DynamicObstacleArray, obstacle_id: str) -> bool:
    return any(obstacle.id == obstacle_id for obstacle in obstacles.obstacles)


def next_path_obstacle_id(obstacles: DynamicObstacleArray) -> str:
    existing = {obstacle.id for obstacle in obstacles.obstacles}
    index = 1
    while f"path_obstacle_{index}" in existing:
        index += 1
    return f"path_obstacle_{index}"
