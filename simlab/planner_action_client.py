from rclpy.action import ActionClient
from rclpy.node import Node
from simlab_msgs.action import PlanVehicle
import numpy as np
from typing import Any, Callable, Dict, Optional, Sequence


class PlannerActionClient:
    """Reusable action client wrapper for vehicle planning requests."""

    def __init__(
        self,
        node: Node,
        action_name: str = "planner",
        on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self._node = node
        self._tag = "[PlannerClient]"
        self._action_client = ActionClient(self._node, PlanVehicle, action_name)
        self._goal_handle = None
        self._busy = False
        self._on_result = on_result
        self.last_result: Optional[Dict[str, Any]] = None

    @property
    def busy(self) -> bool:
        return self._busy

    def send_goal(
        self,
        *,
        start_xyz: Sequence[float],
        start_quat_wxyz: Sequence[float],
        goal_xyz: Sequence[float],
        goal_quat_wxyz: Sequence[float],
        planner_name: str,
        time_limit: float,
        robot_collision_radius: float,
    ) -> bool:
        if self._busy:
            self._node.get_logger().warn(f"{self._tag} action already running; ignoring request.")
            return False
        if not self._action_client.server_is_ready():
            self._node.get_logger().warn(f"{self._tag} action server is not ready.")
            return False

        goal_msg = PlanVehicle.Goal()
        goal_msg.start_xyz = [float(v) for v in start_xyz]
        goal_msg.start_quat_wxyz = [float(v) for v in start_quat_wxyz]
        goal_msg.goal_xyz = [float(v) for v in goal_xyz]
        goal_msg.goal_quat_wxyz = [float(v) for v in goal_quat_wxyz]
        goal_msg.planner_name = str(planner_name)
        goal_msg.time_limit = float(time_limit)
        goal_msg.robot_collision_radius = float(robot_collision_radius)
        self._busy = True
        self._node.get_logger().info(
            f"{self._tag} sending planner request "
            f"planner={goal_msg.planner_name} "
            f"radius={goal_msg.robot_collision_radius:.3f} "
            f"start={goal_msg.start_xyz} goal={goal_msg.goal_xyz}"
        )
        send_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_callback,
        )
        send_future.add_done_callback(self._goal_response_callback)
        return True

    def _goal_response_callback(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._busy = False
            self._node.get_logger().error(f"{self._tag} goal submission failed: {exc}")
            return

        if not goal_handle.accepted:
            self._busy = False
            self._node.get_logger().warn(f"{self._tag} goal rejected by action server.")
            return

        self._goal_handle = goal_handle
        self._node.get_logger().info(f"{self._tag} goal accepted by action server.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future) -> None:
        try:
            wrapped_result = future.result()
            status = int(wrapped_result.status)
            result = wrapped_result.result
            is_success = bool(getattr(result, "is_success", bool(getattr(result, "success", False))))

            xyz_flat = np.asarray(getattr(result, "xyz", []), dtype=float).reshape(-1)
            quat_flat = np.asarray(getattr(result, "quat_wxyz", []), dtype=float).reshape(-1)
            xyz_count = int(xyz_flat.size // 3)
            quat_count = int(quat_flat.size // 4)
            count = int(getattr(result, "count", min(xyz_count, quat_count)))
            if count < 0:
                count = 0
            count = min(count, xyz_count, quat_count)

            parsed_result: Dict[str, Any] = {
                "is_success": is_success,
                "count": count,
                "xyz": xyz_flat[: count * 3].reshape(count, 3),
                "quat_wxyz": quat_flat[: count * 4].reshape(count, 4),
                "path_length_cost": float(getattr(result, "path_length_cost", float("nan"))),
                "geom_length": float(getattr(result, "geom_length", float("nan"))),
                "message": str(getattr(result, "message", "")),
            }
            self.last_result = parsed_result

            self._node.get_logger().info(
                f"{self._tag} action finished status={status}, "
                f"success={is_success}, count={count}, "
                f"path_length_cost={parsed_result['path_length_cost']:.4f}, "
                f"geom_length={parsed_result['geom_length']:.4f}, "
                f"message='{parsed_result['message']}'"
            )
            if self._on_result is not None:
                self._on_result(parsed_result)
        except Exception as exc:
            self._node.get_logger().error(f"{self._tag} result handling failed: {exc}")
        finally:
            self._goal_handle = None
            self._busy = False

    def _feedback_callback(self, feedback_msg) -> None:
        feedback = feedback_msg.feedback
        self._node.get_logger().info(
            f"{self._tag} feedback stage='{feedback.stage}'",
            throttle_duration_sec=1.0,
        )
