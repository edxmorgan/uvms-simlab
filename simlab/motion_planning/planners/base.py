from abc import ABC, abstractmethod
from simlab.motion_planning.result import MotionPlanResult

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
        dynamic_obstacle_prediction_speed: float = 0.0,
    ) -> MotionPlanResult:
        """Return a typed motion-planning result."""
