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
    goal_clearance_m: float
    robot_clearance_m: float
    robot_clearance_margin_m: float


@dataclass(frozen=True)
class PathPointAhead:
    center_world: np.ndarray
    distance_along_path_m: float
    distance_from_robot_m: float
    distance_from_goal_m: float
    remaining_path_m: float
    nearest_path_index: int
    goal_clearance_m: float


def make_path_obstacle(
    *,
    robot,
    existing_obstacles: DynamicObstacleArray,
    world_frame: str,
    distance_ahead: float,
    radius: float,
    robot_collision_radius: float = 0.4,
    robot_clearance_margin: float = 0.0,
    name: str = "",
) -> PathObstaclePlacement | None:
    min_obstacle_radius = 0.05
    requested_obstacle_radius = max(min_obstacle_radius, float(radius))
    robot_radius = max(0.0, float(robot_collision_radius))
    robot_clearance_margin_m = max(0.0, float(robot_clearance_margin))
    effective_planner_radius = robot_radius + robot_clearance_margin_m
    distance_ahead_m = max(0.5, float(distance_ahead))
    goal_clearance_m = requested_obstacle_radius + effective_planner_radius
    point = path_point_ahead(
        robot,
        world_frame=world_frame,
        distance_ahead=distance_ahead_m,
        start_clearance_m=goal_clearance_m,
        goal_clearance_m=goal_clearance_m,
    )
    obstacle_radius = requested_obstacle_radius
    if point is None:
        min_goal_clearance_m = effective_planner_radius + min_obstacle_radius
        point = path_point_ahead(
            robot,
            world_frame=world_frame,
            distance_ahead=distance_ahead_m,
            start_clearance_m=min_goal_clearance_m,
            goal_clearance_m=min_goal_clearance_m,
        )
        if point is None:
            return None

    max_radius_from_robot = point.distance_from_robot_m - effective_planner_radius
    max_radius_from_goal = point.distance_from_goal_m - effective_planner_radius
    obstacle_radius = min(requested_obstacle_radius, max_radius_from_robot, max_radius_from_goal)
    if obstacle_radius < min_obstacle_radius - 1e-9:
        return None
    obstacle_radius = max(min_obstacle_radius, obstacle_radius)
    goal_clearance_m = obstacle_radius + effective_planner_radius

    obstacle_id = str(name or "").strip() or next_path_obstacle_id(existing_obstacles)
    if dynamic_obstacle_id_exists(existing_obstacles, obstacle_id):
        raise ValueError(f"dynamic obstacle id '{obstacle_id}' already exists")

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
        goal_clearance_m=goal_clearance_m,
        robot_clearance_m=point.distance_from_robot_m - obstacle_radius - robot_radius,
        robot_clearance_margin_m=robot_clearance_margin_m,
    )


def path_point_ahead(
    robot,
    *,
    world_frame: str,
    distance_ahead: float,
    start_clearance_m: float = 0.0,
    goal_clearance_m: float = 0.0,
) -> PathPointAhead | None:
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

    polyline = np.vstack([current, remaining])
    segment_lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    remaining_path_m = float(np.sum(segment_lengths))
    if remaining_path_m < 1e-9:
        return None

    start_clearance_m = max(0.0, float(start_clearance_m))
    goal_clearance_m = max(0.0, float(goal_clearance_m))
    target_distance_m = min(max(0.5, float(distance_ahead)), remaining_path_m)
    goal = remaining[-1]

    best = None
    distance_along_path_m = 0.0
    for segment_index, segment_length_raw in enumerate(segment_lengths):
        segment_length = float(segment_length_raw)
        if segment_length < 1e-9:
            continue
        start = polyline[segment_index]
        end = polyline[segment_index + 1]
        feasible = _outside_sphere_intervals(start, end, current, start_clearance_m)
        feasible = _intersect_intervals(
            feasible,
            _outside_sphere_intervals(start, end, goal, goal_clearance_m),
        )
        for t_low, t_high in feasible:
            t_low = max(t_low, (0.5 - distance_along_path_m) / segment_length)
            t_high = min(t_high, 1.0)
            if t_low > t_high + 1e-12:
                continue
            interval_low = distance_along_path_m + max(0.0, t_low) * segment_length
            interval_high = distance_along_path_m + t_high * segment_length
            candidate_distance = float(np.clip(target_distance_m, interval_low, interval_high))
            candidate_error = abs(candidate_distance - target_distance_m)
            if best is None or candidate_error < best[0] - 1e-12:
                t = (candidate_distance - distance_along_path_m) / segment_length
                center = start + t * (end - start)
                best = (candidate_error, candidate_distance, center)
        distance_along_path_m += segment_length

    if best is None:
        return None

    _, distance_along_path_m, center = best
    return PathPointAhead(
        center_world=center,
        distance_along_path_m=distance_along_path_m,
        distance_from_robot_m=float(np.linalg.norm(center - current)),
        distance_from_goal_m=float(np.linalg.norm(center - goal)),
        remaining_path_m=remaining_path_m,
        nearest_path_index=nearest_idx,
        goal_clearance_m=goal_clearance_m,
    )


def _outside_sphere_intervals(start, end, center, radius) -> list[tuple[float, float]]:
    """Return segment parameters whose Euclidean distance is at least radius."""
    if radius <= 0.0:
        return [(0.0, 1.0)]

    delta = end - start
    offset = start - center
    a = float(np.dot(delta, delta))
    b = 2.0 * float(np.dot(offset, delta))
    c = float(np.dot(offset, offset)) - radius * radius
    discriminant = b * b - 4.0 * a * c
    if discriminant <= 1e-12:
        return [(0.0, 1.0)] if c >= -1e-12 else []

    root = float(np.sqrt(discriminant))
    first = (-b - root) / (2.0 * a)
    second = (-b + root) / (2.0 * a)
    intervals = []
    if first >= 0.0:
        intervals.append((0.0, min(1.0, first)))
    if second <= 1.0:
        intervals.append((max(0.0, second), 1.0))
    return [(low, high) for low, high in intervals if low <= high + 1e-12]


def _intersect_intervals(left, right) -> list[tuple[float, float]]:
    intersections = []
    for left_low, left_high in left:
        for right_low, right_high in right:
            low = max(left_low, right_low)
            high = min(left_high, right_high)
            if low <= high + 1e-12:
                intersections.append((low, high))
    return intersections


def dynamic_obstacle_id_exists(obstacles: DynamicObstacleArray, obstacle_id: str) -> bool:
    return any(obstacle.id == obstacle_id for obstacle in obstacles.obstacles)


def next_path_obstacle_id(obstacles: DynamicObstacleArray) -> str:
    existing = {obstacle.id for obstacle in obstacles.obstacles}
    index = 1
    while f"path_obstacle_{index}" in existing:
        index += 1
    return f"path_obstacle_{index}"
