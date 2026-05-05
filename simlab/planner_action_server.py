import rclpy
import threading
import numpy as np
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from simlab.action import PlanVehicle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import tf2_ros
from simlab.fcl_checker import FCLWorld
from simlab.shutdown import install_signal_shutdown_handler, spin_until_shutdown
from simlab.planners import DEFAULT_PLANNER_CLASSES

class PlannerActionServer(Node):

    def __init__(self):
        super().__init__('planner_action_server')
        self._tag = "[PlannerServer]"
        self.get_logger().info(f"{self._tag} starting planner_action_server")

        self.declare_parameter('robot_description', '')
        self.declare_parameter('world_frame', 'world')
        urdf_string = self.get_parameter('robot_description').get_parameter_value().string_value
        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        if not urdf_string:
            raise RuntimeError(
                "robot_description is empty on planner action server. "
                "Start server with robot description parameter."
            )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.fcl_world = FCLWorld(urdf_string=urdf_string, world_frame=self.world_frame, vehicle_radius=0.4)
        self._fcl_ok = False

        self.get_logger().info(
            f"{self._tag} default robot collision radius is {self.fcl_world.vehicle_radius:.3f} m "
            f"(overridden per action goal)."
        )

        self.safety_margin: float = 1e-2
        self.env_bounds = tuple(float(v) for v in self.fcl_world.env_xyz_bounds)
        self._planners = {
            planner_cls.registry_name: planner_cls(self)
            for planner_cls in DEFAULT_PLANNER_CLASSES
        }

        self._goal_handle = None
        self._goal_lock = threading.Lock()
        self._action_server = ActionServer(
            self,
            PlanVehicle,
            'planner',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            handle_accepted_callback=self.handle_accepted_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup())

    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""
        valid = (
            len(goal_request.start_xyz) == 3
            and len(goal_request.start_quat_wxyz) == 4
            and len(goal_request.goal_xyz) == 3
            and len(goal_request.goal_quat_wxyz) == 4
            and goal_request.robot_collision_radius > 0.0
            and goal_request.planner_name in self._planners
        )
        if not valid:
            self.get_logger().warn(
                f"{self._tag} rejecting goal, invalid dimensions "
                f"start_xyz={len(goal_request.start_xyz)} "
                f"start_quat={len(goal_request.start_quat_wxyz)} "
                f"goal_xyz={len(goal_request.goal_xyz)} "
                f"goal_quat={len(goal_request.goal_quat_wxyz)} "
                f"radius={goal_request.robot_collision_radius:.4f} "
                f"planner='{goal_request.planner_name}'"
            )
            return GoalResponse.REJECT

        self.get_logger().info(
            f"{self._tag} received goal "
            f"planner={goal_request.planner_name} "
            f"radius={goal_request.robot_collision_radius:.3f} "
            f"start={list(goal_request.start_xyz)} goal={list(goal_request.goal_xyz)} "
            f"time_limit={goal_request.time_limit:.3f}s "
        )
        return GoalResponse.ACCEPT

    def handle_accepted_callback(self, goal_handle):
        with self._goal_lock:
            # This server only allows one goal at a time
            if self._goal_handle is not None and self._goal_handle.is_active:
                self.get_logger().info(f"{self._tag} aborting previous active goal")
                # Abort the existing goal
                self._goal_handle.abort()
            self._goal_handle = goal_handle
            self.get_logger().info(f"{self._tag} accepted planner goal")

        goal_handle.execute()

    def cancel_callback(self, goal_handle):
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info(f"{self._tag} received cancel request")
        return CancelResponse.ACCEPT

    def _fcl_update_callback(self):
        self._fcl_ok = self.fcl_world.update_from_tf(self.tf_buffer, rclpy.time.Time())

    def execute_callback(self, goal_handle):
        req = goal_handle.request
        self.get_logger().info(f"{self._tag} executing planner goal")

        feedback_msg = PlanVehicle.Feedback()
        feedback_msg.stage = "accepted_request"
        goal_handle.publish_feedback(feedback_msg)

        feedback_msg.stage = "planner_input_received"
        goal_handle.publish_feedback(feedback_msg)

        # Ensure we have at least one up-to-date TF/FCL snapshot before solving.
        self._fcl_update_callback()
        self.fcl_world.set_robot_collision_radius(float(req.robot_collision_radius))
        if not self._fcl_ok:
            goal_handle.abort()
            result = PlanVehicle.Result()
            result.success = False
            result.is_success = False
            result.xyz = []
            result.quat_wxyz = []
            result.count = 0
            result.path_length_cost = float("nan")
            result.geom_length = float("nan")
            result.message = "Planner blocked because TF/FCL state is not ready."
            self.get_logger().warn(f"{self._tag} {result.message}")
            return result
        

        self.get_logger().info(
            f"{self._tag} input summary "
            f"planner={req.planner_name} "
            f"radius={req.robot_collision_radius:.3f} "
            f"start_xyz={list(req.start_xyz)} goal_xyz={list(req.goal_xyz)} "
            f"env_bounds={list(self.env_bounds)} "
            f"safety_margin={self.safety_margin:.4f}"
        )

        result = PlanVehicle.Result()
        result.success = False
        result.is_success = False
        result.xyz = []
        result.quat_wxyz = []
        result.count = 0
        result.path_length_cost = float("nan")
        result.geom_length = float("nan")
        result.message = "Planner did not run."
        planner = self._planners[req.planner_name]
        try:
            feedback_msg.stage = "planning"
            goal_handle.publish_feedback(feedback_msg)

            plan = planner.plan_vehicle(
                start_xyz=req.start_xyz,
                start_quat_wxyz=req.start_quat_wxyz,
                goal_xyz=req.goal_xyz,
                goal_quat_wxyz=req.goal_quat_wxyz,
                time_limit=float(req.time_limit),
                robot_collision_radius=float(req.robot_collision_radius),
            )
        except Exception as ex:
            goal_handle.abort()
            result.success = False
            result.is_success = False
            result.message = f"Planner exception, {ex}"
            self.get_logger().error(f"{self._tag} {result.message}")
            return result

        success = bool(plan.get("is_success", False))
        message = str(plan.get("message", "Planner finished."))
        xyz_flat = np.asarray(plan.get("xyz", []), dtype=float).reshape(-1)
        quat_flat = np.asarray(plan.get("quat_wxyz", []), dtype=float).reshape(-1)
        if xyz_flat.size % 3 != 0:
            self.get_logger().warn(
                f"{self._tag} planner returned xyz of invalid size {xyz_flat.size}, expected multiple of 3"
            )
            xyz_flat = np.array([], dtype=float)
        if quat_flat.size % 4 != 0:
            self.get_logger().warn(
                f"{self._tag} planner returned quat of invalid size {quat_flat.size}, expected multiple of 4"
            )
            quat_flat = np.array([], dtype=float)

        max_xyz_count = int(xyz_flat.size // 3)
        max_quat_count = int(quat_flat.size // 4)
        count = int(plan.get("count", max_xyz_count))
        if count < 0:
            count = 0
        count = min(count, max_xyz_count, max_quat_count)

        result.success = success
        result.is_success = success
        result.xyz = xyz_flat[: count * 3].tolist()
        result.quat_wxyz = quat_flat[: count * 4].tolist()
        result.count = count
        try:
            result.path_length_cost = float(plan.get("path_length_cost", float("nan")))
        except Exception:
            result.path_length_cost = float("nan")
        try:
            result.geom_length = float(plan.get("geom_length", float("nan")))
        except Exception:
            result.geom_length = float("nan")
        result.message = message

        if success:
            goal_handle.succeed()
            feedback_msg.stage = "succeeded"
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f"{self._tag} goal succeeded, {message}")
        else:
            goal_handle.abort()
            feedback_msg.stage = "failed"
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().warn(f"{self._tag} goal failed, {message}")

        return result

    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    install_signal_shutdown_handler()
    planner_action_server = PlannerActionServer()

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor()

    try:
        spin_until_shutdown(planner_action_server, executor=executor)
    finally:
        executor.shutdown()
        try:
            planner_action_server.destroy_node()
        except (KeyboardInterrupt, Exception):
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass

if __name__ == '__main__':
    main()
