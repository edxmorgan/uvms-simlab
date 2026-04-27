from abc import ABC, abstractmethod
from typing import Any, Dict

from rclpy.node import Node


class PlannerTemplate(ABC):
    registry_name: str = ""
    visible: bool = True

    def __init__(self, node: Node):
        self.node = node

    @abstractmethod
    def plan_vehicle(
        self,
        *,
        start_xyz,
        start_quat_wxyz,
        goal_xyz,
        goal_quat_wxyz,
        time_limit: float,
        robot_collision_radius: float,
    ) -> Dict[str, Any]:
        """Return the planner result dict consumed by PlannerActionServer."""
