import json
import os
from pathlib import Path

import ament_index_python
from rclpy.node import Node
from ros2_control_blue_reach_5.msg import DynamicObstacleArray
from bringup.obstacle_description import (
    obstacle_from_config as _dynamic_obstacle_from_config,
    validate_unique_obstacle_ids as _validate_unique_dynamic_obstacle_ids,
)


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
