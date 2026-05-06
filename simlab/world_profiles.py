import json
import os
from pathlib import Path

import ament_index_python
from rclpy.node import Node
from ros2_control_blue_reach_5.msg import DynamicObstacle, DynamicObstacleArray


def world_profiles_root() -> Path:
    return Path(ament_index_python.get_package_share_directory("simlab")) / "world_profiles"


def list_world_profiles() -> list[str]:
    root = world_profiles_root()
    if not root.exists():
        return []
    return sorted(path.stem for path in root.glob("*.json") if path.is_file())


def resolve_world_profile(profile_name: str) -> Path:
    profile = str(profile_name or "").strip()
    if not profile:
        raise ValueError("world profile name is required")
    path = Path(os.path.expanduser(profile))
    if path.is_absolute():
        return path
    if path.suffix:
        return world_profiles_root() / path
    return world_profiles_root() / f"{profile}.json"


def load_world_profile(profile_name: str, node: Node | None = None) -> dict:
    try:
        profile_path = resolve_world_profile(profile_name)
        loaded = json.loads(profile_path.read_text())
    except Exception as exc:
        if node is not None:
            node.get_logger().error(f"Failed to load world profile '{profile_name}': {exc}")
        return {}
    if not isinstance(loaded, dict):
        if node is not None:
            node.get_logger().error(f"World profile must be a JSON object: {profile_path}")
        return {}
    return loaded


def dynamic_obstacles_from_world_profile(profile: dict, default_frame_id: str = "world") -> DynamicObstacleArray:
    if not isinstance(profile, dict):
        raise ValueError("world profile must be a JSON object")
    frame_id = str(profile.get("frame_id", default_frame_id) or default_frame_id)
    items = profile.get("obstacles", [])
    if not isinstance(items, list):
        raise ValueError("world profile obstacles must be a list")

    msg = DynamicObstacleArray()
    msg.header.frame_id = frame_id
    msg.obstacles = [_dynamic_obstacle_from_config(item, index) for index, item in enumerate(items)]
    _validate_unique_dynamic_obstacle_ids(msg.obstacles)
    return msg


def _effective_dynamic_obstacle_ids(obstacles) -> list[str]:
    return [(obstacle.id.strip() or f"obstacle_{index}") for index, obstacle in enumerate(obstacles)]


def _validate_unique_dynamic_obstacle_ids(obstacles) -> None:
    seen = set()
    duplicates = set()
    for obstacle_id in _effective_dynamic_obstacle_ids(obstacles):
        if obstacle_id in seen:
            duplicates.add(obstacle_id)
        seen.add(obstacle_id)
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(f"duplicate dynamic obstacle id(s): {duplicate_list}")


def _dynamic_obstacle_from_config(config: dict, index: int) -> DynamicObstacle:
    if not isinstance(config, dict):
        raise ValueError(f"obstacles[{index}] must be an object")

    obstacle = DynamicObstacle()
    obstacle.id = str(config.get("id", f"obstacle_{index}"))
    obstacle.collision_type = _dynamic_obstacle_geometry_type(config.get("type", config.get("collision_type", "sphere")))
    obstacle.collision_dimensions = _dynamic_obstacle_dimensions(config, "dimensions", "collision_dimensions")
    obstacle.visual_type = _dynamic_obstacle_geometry_type(config.get("visual_type", obstacle.collision_type))
    obstacle.visual_dimensions = _dynamic_obstacle_dimensions(config, "visual_dimensions", "dimensions", allow_empty=True)
    obstacle.visual_mesh_resource = str(config.get("visual_mesh_resource", ""))

    position = _dynamic_obstacle_vector(config, "position", 3, [0.0, 0.0, 0.0])
    orientation = _dynamic_obstacle_vector(config, "orientation", 4, [0.0, 0.0, 0.0, 1.0])
    obstacle.pose.position.x, obstacle.pose.position.y, obstacle.pose.position.z = position
    obstacle.pose.orientation.x, obstacle.pose.orientation.y, obstacle.pose.orientation.z, obstacle.pose.orientation.w = orientation

    linear = _dynamic_obstacle_vector(config, "linear_velocity", 3, [0.0, 0.0, 0.0])
    angular = _dynamic_obstacle_vector(config, "angular_velocity", 3, [0.0, 0.0, 0.0])
    obstacle.twist.linear.x, obstacle.twist.linear.y, obstacle.twist.linear.z = linear
    obstacle.twist.angular.x, obstacle.twist.angular.y, obstacle.twist.angular.z = angular

    color = _dynamic_obstacle_vector(config, "color", 4, [0.95, 0.42, 0.12, 0.75])
    obstacle.color.r, obstacle.color.g, obstacle.color.b, obstacle.color.a = color
    return obstacle


def _dynamic_obstacle_geometry_type(value) -> int:
    if isinstance(value, int):
        return int(value)
    mapping = {
        "none": DynamicObstacle.GEOMETRY_NONE,
        "sphere": DynamicObstacle.GEOMETRY_SPHERE,
        "box": DynamicObstacle.GEOMETRY_BOX,
        "cylinder": DynamicObstacle.GEOMETRY_CYLINDER,
        "mesh": DynamicObstacle.GEOMETRY_MESH,
    }
    normalized = str(value).strip().lower()
    if normalized not in mapping:
        raise ValueError(f"unknown dynamic obstacle geometry type '{value}'")
    return mapping[normalized]


def _dynamic_obstacle_dimensions(
    config: dict,
    primary_key: str,
    fallback_key: str,
    *,
    allow_empty: bool = False,
) -> list[float]:
    values = config.get(primary_key, config.get(fallback_key, []))
    if values is None:
        values = []
    result = [float(v) for v in list(values)]
    if not allow_empty and not result:
        raise ValueError(f"dynamic obstacle '{config.get('id', '<unnamed>')}' is missing dimensions")
    return result


def _dynamic_obstacle_vector(config: dict, key: str, size: int, default: list[float]) -> list[float]:
    values = [float(v) for v in list(config.get(key, default))]
    if len(values) != size:
        raise ValueError(f"dynamic obstacle '{key}' must contain exactly {size} values")
    return values
