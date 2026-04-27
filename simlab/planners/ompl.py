from typing import Any, Dict

from rclpy.node import Node

from simlab.planners.base import PlannerTemplate
from simlab.planners.se3_ompl import OmplPlanner


class Se3OmplPlannerBase(PlannerTemplate):
    ompl_planner_type: str = ""

    def __init__(self, node: Node):
        super().__init__(node)
        self._planner = OmplPlanner(
            node,
            safety_margin=float(node.safety_margin),
            env_bounds=tuple(float(v) for v in node.env_bounds),
        )

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
        return self._planner.plan_se3_path(
            start_xyz=start_xyz,
            start_quat_wxyz=start_quat_wxyz,
            goal_xyz=goal_xyz,
            goal_quat_wxyz=goal_quat_wxyz,
            time_limit=float(time_limit),
            planner_type=str(self.ompl_planner_type),
        )


class RrtStarPlanner(Se3OmplPlannerBase):
    registry_name = "RRTstar"
    ompl_planner_type = "RRTstar"


class BitStarPlanner(Se3OmplPlannerBase):
    registry_name = "Bitstar"
    ompl_planner_type = "Bitstar"


class RrtConnectPlanner(Se3OmplPlannerBase):
    registry_name = "RRTConnect"
    ompl_planner_type = "RRTConnect"
