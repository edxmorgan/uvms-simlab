# Copyright (C) 2025 Edward Morgan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

#!/usr/bin/env python3
import copy
from datetime import datetime
from simlab.uvms_parameters import ReachParams
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from simlab.fcl_checker import FCLWorld
import numpy as np
from scipy.spatial import ConvexHull
import os
import ament_index_python
from simlab.utils import geometry
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from visualization_msgs.msg import Marker
import tf2_ros
from typing import Dict, List
from simlab.robot import Robot, ControlMode
from geometry_msgs.msg import Pose
from std_msgs.msg import Header, String
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from control_msgs.msg import DynamicJointState
from rviz_2d_overlay_msgs.msg import OverlayText
from std_srvs.srv import Trigger
from simlab.srv import (
    BackendPoseCommand,
    BackendRobotCommand,
    BackendWorldCommand,
    BackendWaypointCommand,
)
from ros2_control_blue_reach_5.msg import DynamicObstacleArray
from ros2_control_blue_reach_5.srv import ResetSimUvms, SetDynamicObstacles
from simlab.planner_markers import PathPlanner
from simlab.cartesian_ruckig import VehicleCartesianRuckig
from simlab.dynamic_replanner import DynamicReplanner
from simlab.dynamic_world import DynamicWorldModel
from simlab.utils.frames import PoseX
from simlab.vehicle_waypoint_mission import (
    VehicleWaypointMission,
    VehicleWaypointViz,
    pose_position_distance,
)
from simlab.utils.path_obstacles import make_path_obstacle
from simlab.world_profiles import (
    dynamic_obstacles_from_world_profile,
    list_world_profiles,
    load_world_profile,
)


def parameter_or_default(node: Node, name: str, default):
    parameter = node.get_parameter_or(name, default)
    return getattr(parameter, "value", parameter)


class UVMSBackendCore:
    def __init__(self, node: Node,
                  urdf_string: str,
                  arm_base_wrt_vehicle_center_Pose: Pose, 
                  vehicle_target_frame: str, 
                  arm_base_target_frame: str, 
                  world_frame: str, 
                  world_target_endeffector_frame: str, 
                  robot_model):
        package_share_directory = ament_index_python.get_package_share_directory('simlab')
        self.node = node
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self.node)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)
        self.start_recording_client = self.node.create_client(Trigger, "/bag_recorder_node/start_recording")
        self.stop_recording_client = self.node.create_client(Trigger, "/bag_recorder_node/stop_recording")
        self.camera_parameters_client = self.node.create_client(
            SetParameters,
            "/gstreamer_camera_node/set_parameters",
        )
        self.sim_camera_parameters_client = self.node.create_client(
            SetParameters,
            "/sim_camera_renderer_node/set_parameters",
        )
        self.dynamic_obstacles_client = self.node.create_client(
            SetDynamicObstacles,
            "/dynamic_obstacle_sim_node/set_dynamic_obstacles",
        )
        self.mcap_recording_active = False
        self._pending_camera_robot: Robot | None = None
        self.use_vehicle_hardware = bool(self.node.get_parameter_or("use_vehicle_hardware", False).value)
        self.camera_source = str(self.node.get_parameter_or("camera_source", "auto").value)
        self.urdf_string = urdf_string
        self.vehicle_target_frame = vehicle_target_frame
        self.world_frame = world_frame
        self.arm_base_target_frame = arm_base_target_frame
        self.arm_base_wrt_vehicle_center_Pose = arm_base_wrt_vehicle_center_Pose
        self.world_endeffector_target_frame = world_target_endeffector_frame
        self.robot_model = robot_model
        self.fcl_world = FCLWorld(urdf_string=urdf_string, world_frame=self.world_frame, vehicle_radius=0.4)
        self.node.get_logger().info(f"self.fcl_world.floor_depth: {self.fcl_world.floor_depth:.3f} m")
        self.node.get_logger().info(f"Minimum coordinates (min_x, min_y, min_z): {self.fcl_world.min_coords}")
        self.node.get_logger().info(f"Maximum coordinates (max_x, max_y, max_z): {self.fcl_world.max_coords}")
        self.node.get_logger().info(f"Oriented Bounding Box corners: {self.fcl_world.obb_corners}")

         # Load workspace point cloud and hull
        workspace_pts_path = os.path.join(package_share_directory, 'model_functions/arm/workspace.npy')
        self.workspace_pts = np.load(workspace_pts_path)
        self.workspace_hull = ConvexHull(self.workspace_pts)

        # ROV ellipsoid point cloud and hull
        self.rov_ellipsoid_cl_pts = geometry.generate_rov_ellipsoid(a=0.3, b=0.3, c=0.2, num_points=10000)
        self.vehicle_body_hull = ConvexHull(self.rov_ellipsoid_cl_pts)

        pointcloud_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, durability=QoSDurabilityPolicy.VOLATILE,
                                    reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )

        self.taskspace_pc_publisher_ = self.node.create_publisher(PointCloud2,'workspace_pointcloud',pointcloud_qos)
        self.rov_pc_publisher_ = self.node.create_publisher(PointCloud2, 'base_pointcloud', pointcloud_qos)
        self.robot_metrics_overlay_pub = self.node.create_publisher(
            OverlayText,
            'robot_metrics_overlay_text',
            10,
        )
        self.research_overlay_pub = self.node.create_publisher(
            String,
            'chatter',
            10,
        )

        # stack clouds that represent the vehicle occupied volume
        all_pts = np.vstack([
            np.asarray(self.rov_ellipsoid_cl_pts, dtype=float),
            np.asarray(self.workspace_pts, dtype=float)
        ])

        robot_collision_radius = geometry.compute_bounding_sphere_radius(all_pts, quantile=0.995, pad=0.03)
        self.node.get_logger().info(f"Planner robot approximation sphere radius set to {robot_collision_radius:.3f} m")
        self.fcl_world.set_robot_collision_radius(robot_collision_radius)
        self.dynamic_world = None
        self.dynamic_obstacle_snapshot = DynamicObstacleArray()
        self.dynamic_obstacle_snapshot.header.frame_id = self.world_frame

        viz_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST,depth=1,durability=QoSDurabilityPolicy.VOLATILE,
                    reliability=QoSReliabilityPolicy.RELIABLE)
        planner_viz_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        # publisher for FCL environment AABB
        self.env_aabb_pub = self.node.create_publisher(Marker, "fcl_environment_aabb", viz_qos)
        self.env_bounds_timer = self.node.create_timer(1.0 / 1.0, self.publish_fcl_environment_aabb_callback)
        self.fcl_update_rate = float(self.node.get_parameter_or("interactive_fcl_update_rate", 10.0).value)
        if self.fcl_update_rate > 0.0:
            self.fcl_update_timer = self.node.create_timer(1.0 / self.fcl_update_rate, self.fcl_update_callback)
        else:
            self.fcl_update_timer = None
        self.target_vehicle_marker_in_world_tf_timer = self.node.create_timer(1.0 / 20.0, self.target_vehicle_in_world_tf_timer_callback)
        self.target_arm_base_marker_tf_timer = self.node.create_timer(1.0 / 20.0, self.target_arm_base_tf_timer_callback)
        self.target_endeffector_in_world_tf_timer = self.node.create_timer(1.0 / 20.0, self.target_endeffector_in_world_tf_timer_callback)
        self.vehicle_target_cloud_timer = self.node.create_timer(1.0 / 2.0, self.vehicle_target_cloud_timer_callback)
        self.task_on_vehicle_solve_timer = self.node.create_timer(1.0 / 5.0, self.plan_and_execute_task_trajectory_wrt_vehicle)
        self.robot_metrics_overlay_timer = self.node.create_timer(1.0 / 5.0, self.publish_robot_metrics_overlay_callback)
        self.research_overlay_timer = self.node.create_timer(1.0, self.publish_research_overlay_callback)
        
        self.planner_marker_publisher = self.node.create_publisher(Marker, "planned_waypoints_marker", planner_viz_qos)
        self.robots:List[Robot] = []
        self.robot_selected = None
        self.vehicle_waypoint_missions: Dict[int, VehicleWaypointMission] = {}
        self.vehicle_waypoint_viz: Dict[int, VehicleWaypointViz] = {}
        self.max_cartesian_waypoints = 500
        self.robots_prefix = self.node.get_parameter('robots_prefix').value
        for k, prefix in enumerate(self.robots_prefix):
            planner = PathPlanner(self.planner_marker_publisher, ns=f"planner/{prefix}", base_id=k)
            vehicle_cart_traj = VehicleCartesianRuckig(
                self.node,
                dofs=3,
                control_dt=1.0 / 60.0,
                max_waypoints=self.max_cartesian_waypoints,
            )

            robot_k = Robot(
                self.node,
                self.tf_buffer,
                k,
                4,
                prefix,
                planner,
                vehicle_cart_traj,
                world_frame=self.world_frame,
                create_subscriptions=False,
            )
            robot_k.set_control_mode(ControlMode.PLANNER)


            self.robots.append(robot_k)
            self.vehicle_waypoint_missions[k] = VehicleWaypointMission(robot_prefix=prefix)
            self.vehicle_waypoint_viz[k] = VehicleWaypointViz(
                self.planner_marker_publisher,
                ns=f"vehicle_waypoints/{prefix}",
                base_id=2000 + 10 * k,
            )

        self.dynamics_states_sub = self.node.create_subscription(
            DynamicJointState,
            'dynamic_joint_states',
            self._dynamic_joint_states_cb,
            10,
        )
        self.mocap_pose_sub = self.node.create_subscription(
            PoseStamped,
            'mocap_pose',
            self._mocap_pose_cb,
            10,
        )
        self.initialise_target_Poses()
        self.camera_selection_retry_timer = self.node.create_timer(
            0.5,
            self._retry_pending_camera_selection,
        )
        self.set_robot_selected(self.robots[0].k_robot)
        self.vehicle_waypoint_execution_timer = self.node.create_timer(
            1.0 / 10.0,
            self.vehicle_waypoint_execution_callback,
        )
        self.vehicle_waypoint_viz_timer = self.node.create_timer(
            1.0 / 2.0,
            self.vehicle_waypoint_viz_callback,
        )
        self.dynamic_replanners: Dict[int, DynamicReplanner] = {}
        self.dynamic_replanner_timer = None
        self.dynamic_replanning_enabled = False
        self.dynamic_replanning_rate = float(parameter_or_default(self.node, "dynamic_replanning_rate", 3.0))
        self.dynamic_replanning_cooldown = float(parameter_or_default(self.node, "dynamic_replanning_cooldown", 1.0))
        self.dynamic_replanning_lookahead_time = float(
            parameter_or_default(self.node, "dynamic_replanning_lookahead_time", 4.0)
        )
        self.dynamic_replanning_safety_margin = float(
            parameter_or_default(self.node, "dynamic_replanning_safety_margin", 0.25)
        )
        self.dynamic_replanning_hysteresis = float(
            parameter_or_default(self.node, "dynamic_replanning_hysteresis", 0.10)
        )
        self.dynamic_collision_stop_enabled = bool(
            parameter_or_default(self.node, "dynamic_collision_stop_enabled", True)
        )
        self.dynamic_collision_stop_margin = float(
            parameter_or_default(self.node, "dynamic_collision_stop_margin", 0.0)
        )
        self.dynamic_replanning_enabled = bool(parameter_or_default(self.node, "dynamic_replanning_enabled", False))
        if self.dynamic_replanning_enabled:
            self._enable_dynamic_replanning()
        self._create_backend_api_services()

    def close(self) -> None:
        self._disable_dynamic_replanning()
        for robot in self.robots:
            robot.close()
        self.robots.clear()
        self.robot_selected = None

    def dynamic_replanning_callback(self) -> None:
        for replanner in list(self.dynamic_replanners.values()):
            replanner.tick()

    def _enable_dynamic_replanning(self) -> None:
        if self.dynamic_replanning_rate <= 0.0:
            raise RuntimeError("dynamic_replanning_rate must be > 0 when dynamic replanning is enabled.")
        if self.dynamic_world is None:
            self.dynamic_world = DynamicWorldModel(
                self.node,
                world_frame=self.world_frame,
                robot_radius_provider=lambda: float(self.fcl_world.vehicle_radius),
            )

        self.dynamic_replanners = {
            robot.k_robot: DynamicReplanner(
                self,
                robot,
                self.vehicle_waypoint_missions[robot.k_robot],
                cooldown_s=self.dynamic_replanning_cooldown,
                lookahead_time_s=self.dynamic_replanning_lookahead_time,
                safety_margin_m=self.dynamic_replanning_safety_margin,
                collision_stop_enabled=self.dynamic_collision_stop_enabled,
                collision_stop_margin_m=self.dynamic_collision_stop_margin,
                replan_hysteresis_m=self.dynamic_replanning_hysteresis,
            )
            for robot in self.robots
        }
        self._recreate_dynamic_replanner_timer()
        self.dynamic_replanning_enabled = True
        self.node.get_logger().info(self._dynamic_replanning_status_message())

    def _disable_dynamic_replanning(self) -> None:
        self.dynamic_replanning_enabled = False
        self.dynamic_replanners.clear()
        if self.dynamic_replanner_timer is not None:
            self.dynamic_replanner_timer.cancel()
            self.node.destroy_timer(self.dynamic_replanner_timer)
            self.dynamic_replanner_timer = None
        if self.dynamic_world is not None:
            self.dynamic_world.close()
            self.dynamic_world = None
        self.node.get_logger().info("Dynamic replanning disabled.")

    def _recreate_dynamic_replanner_timer(self) -> None:
        if self.dynamic_replanner_timer is not None:
            self.dynamic_replanner_timer.cancel()
            self.node.destroy_timer(self.dynamic_replanner_timer)
            self.dynamic_replanner_timer = None
        if self.dynamic_replanning_rate <= 0.0:
            raise RuntimeError("dynamic_replanning_rate must be > 0 when dynamic replanning is enabled.")
        self.dynamic_replanner_timer = self.node.create_timer(
            1.0 / self.dynamic_replanning_rate,
            self.dynamic_replanning_callback,
        )

    def _dynamic_replanning_status_message(self) -> str:
        return (
            "Dynamic replanning "
            f"{'enabled' if self.dynamic_replanning_enabled else 'disabled'}: "
            f"rate={self.dynamic_replanning_rate:.2f} Hz, "
            f"cooldown={self.dynamic_replanning_cooldown:.2f} s, "
            f"lookahead={self.dynamic_replanning_lookahead_time:.2f} s, "
            f"safety_margin={self.dynamic_replanning_safety_margin:.3f} m, "
            f"replan_hysteresis={self.dynamic_replanning_hysteresis:.3f} m, "
            f"collision_stop={'enabled' if self.dynamic_collision_stop_enabled else 'disabled'}, "
            f"collision_stop_margin={self.dynamic_collision_stop_margin:.3f} m"
            + self._dynamic_replanner_status_suffix()
        )

    def _dynamic_replanner_status_suffix(self) -> str:
        if not self.dynamic_replanners:
            return ""
        summaries = "; ".join(
            replanner.status_summary()
            for _, replanner in sorted(self.dynamic_replanners.items())
        )
        return f"; {summaries}"

    def set_dynamic_replanning(
        self,
        *,
        enabled: bool | None = None,
        rate: float | None = None,
        cooldown: float | None = None,
        lookahead_time: float | None = None,
        safety_margin: float | None = None,
        replan_hysteresis: float | None = None,
    ) -> tuple[bool, str]:
        if rate is not None and rate > 0.0:
            self.dynamic_replanning_rate = float(rate)
        if cooldown is not None and cooldown >= 0.0:
            self.dynamic_replanning_cooldown = float(cooldown)
        if lookahead_time is not None and lookahead_time >= 0.0:
            self.dynamic_replanning_lookahead_time = float(lookahead_time)
        if safety_margin is not None and safety_margin >= 0.0:
            self.dynamic_replanning_safety_margin = float(safety_margin)
        if replan_hysteresis is not None and replan_hysteresis >= 0.0:
            self.dynamic_replanning_hysteresis = float(replan_hysteresis)

        for replanner in self.dynamic_replanners.values():
            replanner.configure(
                cooldown_s=self.dynamic_replanning_cooldown,
                lookahead_time_s=self.dynamic_replanning_lookahead_time,
                safety_margin_m=self.dynamic_replanning_safety_margin,
                collision_stop_enabled=self.dynamic_collision_stop_enabled,
                collision_stop_margin_m=self.dynamic_collision_stop_margin,
                replan_hysteresis_m=self.dynamic_replanning_hysteresis,
            )

        if self.dynamic_replanning_enabled:
            self._recreate_dynamic_replanner_timer()

        if enabled is True and not self.dynamic_replanning_enabled:
            self._enable_dynamic_replanning()
        elif enabled is False and self.dynamic_replanning_enabled:
            self._disable_dynamic_replanning()

        return True, self._dynamic_replanning_status_message()

    def _create_backend_api_services(self) -> None:
        self.node.create_service(
            BackendRobotCommand,
            "/backend/robot_command",
            self._backend_robot_command_callback,
        )
        self.node.create_service(
            BackendWorldCommand,
            "/backend/world_command",
            self._backend_world_command_callback,
        )
        self.node.create_service(
            BackendPoseCommand,
            "/backend/pose_command",
            self._backend_pose_command_callback,
        )
        self.node.create_service(
            BackendWaypointCommand,
            "/backend/waypoint_command",
            self._backend_waypoint_command_callback,
        )
        self.node.get_logger().info(
            "Backend API ready: /backend/robot_command, /backend/world_command, "
            "/backend/pose_command, /backend/waypoint_command"
        )

    def _robot_for_api(self, robot_index: int) -> Robot | None:
        if 0 <= int(robot_index) < len(self.robots):
            return self.robots[int(robot_index)]
        self.node.get_logger().warn(f"Backend API rejected invalid robot_index={robot_index}.")
        return None

    @staticmethod
    def _api_response(response, success: bool, message: str):
        response.success = bool(success)
        response.message = str(message)
        return response

    def _select_robot_for_api(self, robot_index: int) -> tuple[bool, str]:
        robot = self._robot_for_api(robot_index)
        if robot is None:
            return False, f"invalid robot_index={robot_index}"
        self.set_robot_selected(robot.k_robot)
        return True, f"selected {robot.prefix}"

    def _cmd_replay_controller(self, robot: Robot):
        controller = robot.controller_instance("CmdReplay")
        if controller is None or not hasattr(controller, "load_profile"):
            return None
        return controller

    def select_robot(self, robot_index: int) -> tuple[bool, str]:
        return self._select_robot_for_api(robot_index)

    def set_robot_controller(self, robot: Robot, controller_name: str) -> tuple[bool, str]:
        if controller_name not in robot.list_controllers():
            return False, f"unknown controller '{controller_name}'"
        if not robot.set_controller(controller_name):
            return False, f"controller {controller_name} rejected for {robot.prefix}"
        return True, f"controller set to {controller_name} for {robot.prefix}"

    def set_robot_planner(self, robot: Robot, planner_name: str) -> tuple[bool, str]:
        if planner_name not in robot.list_planners():
            return False, f"unknown planner '{planner_name}'"
        robot.set_planner(planner_name)
        return True, f"planner set to {planner_name} for {robot.prefix}"

    def set_robot_control_space(self, robot: Robot, control_space_name: str) -> tuple[bool, str]:
        if control_space_name not in robot.list_control_spaces():
            return False, f"unknown control space '{control_space_name}'"
        if robot.control_mode in (ControlMode.REPLAY, ControlMode.REPLAY_SETTLE):
            return False, f"control-space switch rejected for {robot.prefix}; CmdReplay is active"
        robot.set_control_space(control_space_name)
        return True, f"control space set to {control_space_name} for {robot.prefix}"

    def set_robot_dynamics_profile(self, robot: Robot, profile_name: str, on_success=None) -> tuple[bool, str]:
        if "real" in robot.prefix:
            return False, f"dynamics profiles are disabled for real robot {robot.prefix}"
        if profile_name not in robot.list_dynamics_profiles():
            return False, f"unknown dynamics profile '{profile_name}'"
        robot.apply_dynamics_profile(profile_name, on_success=on_success)
        return True, f"dynamics profile request sent for {robot.prefix}: {profile_name}"

    def set_world_profile(self, profile_name: str) -> tuple[bool, str]:
        profile_name = str(profile_name or "").strip()
        if profile_name not in list_world_profiles():
            return False, f"unknown world profile '{profile_name}'"
        profile = load_world_profile(profile_name, self.node)
        if not profile:
            return False, f"failed to load world profile '{profile_name}'"
        try:
            obstacle_msg = dynamic_obstacles_from_world_profile(profile, self.world_frame)
        except Exception as exc:
            return False, f"invalid world profile '{profile_name}': {exc}"
        if not self._apply_dynamic_obstacles(obstacle_msg, f"world profile '{profile_name}'"):
            return False, f"dynamic obstacle simulator is not ready for world profile '{profile_name}'"
        self._reset_dynamic_replanner_history(reset_count=True)
        return True, f"world profile request sent: {profile_name}"

    def spawn_path_obstacle(
        self,
        robot_index: int,
        *,
        name: str = "",
        distance_ahead: float = 4.0,
        radius: float = 0.8,
    ) -> tuple[bool, str]:
        robot = self._robot_for_api(int(robot_index))
        if robot is None:
            return False, f"invalid robot_index={robot_index}"

        try:
            placement = make_path_obstacle(
                robot=robot,
                existing_obstacles=self.dynamic_obstacle_snapshot,
                world_frame=self.world_frame,
                distance_ahead=max(0.5, float(distance_ahead or 4.0)),
                radius=max(0.05, float(radius or 0.8)),
                name=name,
            )
        except ValueError as exc:
            return False, str(exc)
        if placement is None:
            return False, f"no active planned path available for {robot.prefix}"

        obstacle_msg = copy.deepcopy(self.dynamic_obstacle_snapshot)
        obstacle_msg.header.frame_id = obstacle_msg.header.frame_id or self.world_frame
        obstacle_msg.obstacles.append(placement.obstacle)
        if not self._apply_dynamic_obstacles(obstacle_msg, f"path obstacle '{placement.obstacle.id}'"):
            return False, "dynamic obstacle simulator is not ready"

        xyz = np.asarray(placement.center_world, dtype=float).round(3).tolist()
        radius_m = float(placement.obstacle.collision_dimensions[0])
        return (
            True,
            f"path obstacle '{placement.obstacle.id}' requested at {xyz}, "
            f"radius={radius_m:.3f} m, "
            f"path_ahead={placement.distance_along_path_m:.3f} m, "
            f"euclidean_from_robot={placement.distance_from_robot_m:.3f} m, "
            f"remaining_path={placement.remaining_path_m:.3f} m, "
            f"nearest_path_index={placement.nearest_path_index}",
        )

    def clear_dynamic_obstacles(self) -> tuple[bool, str]:
        obstacle_msg = dynamic_obstacles_from_world_profile({"frame_id": self.world_frame, "obstacles": []}, self.world_frame)
        if not self._apply_dynamic_obstacles(obstacle_msg, "clear dynamic obstacles"):
            return False, "dynamic obstacle simulator is not ready"
        self._reset_dynamic_replanner_history(reset_count=True)
        return True, "dynamic obstacle clear request sent"

    def _reset_dynamic_replanner_history(self, *, reset_count: bool = False) -> None:
        for replanner in self.dynamic_replanners.values():
            replanner.reset_history(reset_count=reset_count)

    def _apply_dynamic_obstacles(self, obstacle_msg, label: str) -> bool:
        if obstacle_msg is None:
            return True
        service_name = "/dynamic_obstacle_sim_node/set_dynamic_obstacles"
        if not self.dynamic_obstacles_client.wait_for_service(timeout_sec=0.2):
            self.node.get_logger().warn(f"dynamic obstacle service {service_name} is not ready.")
            return False
        self.dynamic_obstacle_snapshot = copy.deepcopy(obstacle_msg)

        request = SetDynamicObstacles.Request()
        request.obstacles = obstacle_msg
        future = self.dynamic_obstacles_client.call_async(request)

        def _done_callback(done_future) -> None:
            try:
                response = done_future.result()
            except Exception as exc:
                self.node.get_logger().warn(f"{label} failed: {exc}")
                return
            if response is not None and response.success:
                self.node.get_logger().info(f"{label} applied: {response.message}")
            else:
                message = "" if response is None else response.message
                self.node.get_logger().warn(f"{label} rejected: {message}")

        future.add_done_callback(_done_callback)
        return True

    def select_replay_profile(self, robot: Robot, profile_name: str) -> tuple[bool, str]:
        controller = self._cmd_replay_controller(robot)
        if controller is None:
            return False, f"{robot.prefix} has no CmdReplay controller"
        if hasattr(robot, "cancel_replay_settle"):
            robot.cancel_replay_settle(mark_failed=False)
        if hasattr(robot, "_stop_replay_session_recording"):
            robot._stop_replay_session_recording("profile_changed")
        ok = controller.load_profile(profile_name)
        if not ok:
            return False, "replay profile rejected"
        if robot.controller_name != "CmdReplay":
            self.node.get_logger().info(
                f"Choose the CmdReplay controller for {robot.prefix} before starting playback."
            )
        return True, f"replay profile {profile_name} selected for {robot.prefix}"

    def start_replay(self, robot: Robot) -> tuple[bool, str]:
        controller = self._cmd_replay_controller(robot)
        if controller is None:
            return False, f"{robot.prefix} has no CmdReplay controller"
        if robot.controller_name != "CmdReplay":
            return False, f"choose CmdReplay controller for {robot.prefix} before replay"
        if hasattr(controller, "has_valid_playback") and not controller.has_valid_playback():
            return False, f"CmdReplay profile is not selected or has no valid samples for {robot.prefix}"

        if not robot.activate_cmd_replay_controller():
            return False, f"CmdReplay activation failed for {robot.prefix}"
        request = controller.build_reset_request()
        if not controller.begin_sequence(request.hold_commands):
            return False, f"CmdReplay reset rejected for {robot.prefix}"

        def _fail_replay_reset():
            controller.mark_reset_failed()
            robot.publish_commands([0.0] * 6, [0.0] * 5)

        if hasattr(controller, "reset_mode") and controller.reset_mode() == "controller_settle":
            robot.apply_sim_dynamics_from_reset_request(
                request,
                on_success=lambda: robot.start_replay_controller_settle(controller),
                on_failure=_fail_replay_reset,
            )
            return True, f"CmdReplay controller-settle reset requested for {robot.prefix}"

        def _start_after_reset():
            robot.set_control_mode(ControlMode.REPLAY)
            controller.mark_reset_succeeded()

        robot.reset_simulation_with_state(
            request,
            on_success=_start_after_reset,
            on_failure=_fail_replay_reset,
        )
        return True, f"CmdReplay reset requested for {robot.prefix}"

    def stop_replay(self, robot: Robot) -> tuple[bool, str]:
        controller = self._cmd_replay_controller(robot)
        if controller is None:
            return False, f"{robot.prefix} has no CmdReplay controller"
        if hasattr(robot, "cancel_replay_settle"):
            robot.cancel_replay_settle(mark_failed=False)
        controller.stop_playback()
        if hasattr(robot, "_stop_replay_session_recording"):
            robot._stop_replay_session_recording("stopped")
        robot.publish_commands([0.0] * 6, [0.0] * 5)
        return True, f"stopped CmdReplay for {robot.prefix}"

    def command_grasper(self, robot: Robot, action: str) -> tuple[bool, str]:
        if action not in ("open", "close"):
            return False, "grasper action must be 'open' or 'close'"
        ok = robot.command_grasper_from_menu(action)
        return ok, f"grasper {action} requested for {robot.prefix}" if ok else f"grasper {action} rejected for {robot.prefix}"

    def set_robot_ik_tool_axis(self, robot: Robot, axis) -> tuple[bool, str]:
        robot.ik_tool_axis = np.asarray(axis, dtype=float)
        return True, f"IK tool axis set for {robot.prefix}"

    def set_robot_ik_base_align_weight(self, robot: Robot, weight: float) -> tuple[bool, str]:
        robot.ik_base_align_w = float(weight)
        return True, f"IK base align weight set for {robot.prefix}"

    def _clip_pose_to_environment_bounds(self, pose: Pose) -> tuple[Pose, bool]:
        clipped_pose = copy.deepcopy(pose)
        p = clipped_pose.position
        clipped_xyz = self.fcl_world.enforce_bounds([p.x, p.y, p.z])
        clipped = not np.allclose([p.x, p.y, p.z], clipped_xyz)
        p.x, p.y, p.z = clipped_xyz
        return clipped_pose, clipped

    def set_vehicle_target(self, robot: Robot, pose: Pose) -> tuple[bool, str]:
        self.set_robot_selected(robot.k_robot)
        self.target_vehicle_pose, clipped = self._clip_pose_to_environment_bounds(pose)
        suffix = " within environment bounds" if clipped else ""
        return True, f"vehicle target set for {robot.prefix}{suffix}"

    def set_task_target_world(self, robot: Robot, pose: Pose) -> tuple[bool, str]:
        self.set_robot_selected(robot.k_robot)
        self.target_world_endeffector_pose, clipped = self._clip_pose_to_environment_bounds(pose)
        suffix = " within environment bounds" if clipped else ""
        return True, f"world task target set for {robot.prefix}{suffix}"

    def set_task_target_arm_base(self, robot: Robot, pose: Pose) -> tuple[bool, str]:
        self.set_robot_selected(robot.k_robot)
        if not self.is_valid_arm_base_task(pose):
            return False, f"arm-base task target rejected for {robot.prefix}: outside reachable workspace"
        self.target_arm_base_endeffector_pose = pose
        return True, f"arm-base task target set for {robot.prefix}"

    def add_vehicle_waypoint_for_robot(
        self,
        robot: Robot,
        pose: Pose | None = None,
        use_current_target: bool = True,
    ) -> tuple[bool, str]:
        if robot.task_based_controller:
            return False, "vehicle waypoints are only available for vehicle path planning"
        self.set_robot_selected(robot.k_robot)
        if not use_current_target and pose is not None:
            ok, message = self.set_vehicle_target(robot, pose)
            if not ok:
                return False, message
        count = self.add_selected_vehicle_waypoint()
        return True, f"added waypoint {count} for {robot.prefix}"

    def reset_vehicle_world(self, robot: Robot, pose: Pose) -> tuple[bool, str]:
        if "real" in robot.prefix:
            return False, f"world-frame reset is disabled for real robot {robot.prefix}"

        xyz_world = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
        clipped_xyz = self.fcl_world.enforce_bounds(xyz_world)
        if not np.allclose(xyz_world, clipped_xyz):
            return (
                False,
                f"world-frame reset rejected for {robot.prefix}: "
                f"pose {xyz_world.round(3).tolist()} is outside environment bounds",
            )

        quat_world_wxyz = np.array(
            [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z],
            dtype=float,
        )
        pose_map_ned = robot.world_nwu_to_map_ned(
            xyz_world_nwu=xyz_world,
            quat_world_wxyz=quat_world_wxyz,
            warn_context=f"reset_vehicle_world({robot.prefix})",
        )
        if pose_map_ned is None:
            return False, f"world-frame reset rejected for {robot.prefix}: TF is not ready"

        p_ned, rpy_ned = pose_map_ned
        request = ResetSimUvms.Request()
        request.reset_vehicle = True
        request.reset_manipulator = False
        request.hold_commands = True
        request.use_vehicle_state = True
        request.vehicle_pose = [
            float(p_ned[0]),
            float(p_ned[1]),
            float(p_ned[2]),
            float(rpy_ned[0]),
            float(rpy_ned[1]),
            float(rpy_ned[2]),
        ]
        request.vehicle_twist = [0.0] * 6
        request.vehicle_wrench = [0.0] * 6
        request.use_manipulator_state = False

        self.set_robot_selected(robot.k_robot)
        self.clear_vehicle_waypoints_for_robot(robot.k_robot)

        def _start_feedback_hold_after_reset() -> None:
            robot.set_control_mode(ControlMode.PLANNER)
            robot.hold_pose_with_feedback(request.vehicle_pose)
            robot.release_simulation()

        robot.reset_simulation_with_state(request, on_success=_start_feedback_hold_after_reset)
        return (
            True,
            f"world-frame vehicle reset requested for {robot.prefix} at "
            f"{xyz_world.round(3).tolist()}; feedback hold will start after reset",
        )

    def _backend_robot_command_callback(self, request, response):
        command = str(request.command).strip()
        if command == "start_mcap_recording":
            ok = self.start_mcap_recording()
            message = "MCAP recording start requested" if ok else "MCAP recording start rejected"
            return self._api_response(response, ok, message)
        if command == "stop_mcap_recording":
            ok = self.stop_mcap_recording()
            message = "MCAP recording stop requested" if ok else "MCAP recording stop rejected"
            return self._api_response(response, ok, message)

        robot = self._robot_for_api(request.robot_index)
        if command != "select_robot" and robot is None:
            return self._api_response(response, False, f"invalid robot_index={request.robot_index}")

        try:
            if command == "select_robot":
                return self._api_response(response, *self.select_robot(request.robot_index))
            if command == "set_controller":
                return self._api_response(response, *self.set_robot_controller(robot, request.name))
            if command == "set_planner":
                return self._api_response(response, *self.set_robot_planner(robot, request.name))
            if command == "set_control_space":
                return self._api_response(response, *self.set_robot_control_space(robot, request.name))
            if command == "set_dynamics_profile":
                return self._api_response(response, *self.set_robot_dynamics_profile(robot, request.name))
            if command == "plan_execute":
                self.set_robot_selected(robot.k_robot)
                return self._api_response(response, self.plan_execute_selected(), f"plan_execute requested for {robot.prefix}")
            if command == "reset_simulation":
                self.set_robot_selected(robot.k_robot)
                return self._api_response(response, self.reset_selected_simulation(), f"reset requested for {robot.prefix}")
            if command == "release_simulation":
                self.set_robot_selected(robot.k_robot)
                return self._api_response(
                    response,
                    self.release_selected_simulation(feedback_hold=True),
                    f"release with feedback hold requested for {robot.prefix}",
                )
            if command == "release_simulation_raw":
                self.set_robot_selected(robot.k_robot)
                return self._api_response(response, self.release_selected_simulation(), f"raw release requested for {robot.prefix}")
            if command in ("hold_current_state", "hold_current_vehicle_pose"):
                return self._api_response(response, *self.hold_robot_current_state(robot))
            if command == "release_and_hold":
                return self._api_response(response, *self.hold_robot_current_state(robot, release_if_held=True))
            if command == "replay_select_profile":
                return self._api_response(response, *self.select_replay_profile(robot, request.name))
            if command == "replay_start":
                return self._api_response(response, *self.start_replay(robot))
            if command == "replay_stop":
                return self._api_response(response, *self.stop_replay(robot))
            if command == "grasper":
                return self._api_response(response, *self.command_grasper(robot, request.name))
            if command == "set_ik_tool_axis":
                return self._api_response(response, *self.set_robot_ik_tool_axis(robot, request.vector3))
            if command == "set_ik_base_align_weight":
                return self._api_response(response, *self.set_robot_ik_base_align_weight(robot, request.scalar))
        except Exception as exc:
            return self._api_response(response, False, f"{command} failed: {exc}")

        return self._api_response(response, False, f"unknown backend robot command '{command}'")

    def _backend_world_command_callback(self, request, response):
        command = str(request.command).strip()
        try:
            if command == "set_world_profile":
                return self._api_response(response, *self.set_world_profile(request.name))
            if command == "clear_dynamic_obstacles":
                return self._api_response(response, *self.clear_dynamic_obstacles())
            if command == "spawn_path_obstacle":
                return self._api_response(
                    response,
                    *self.spawn_path_obstacle(
                        int(request.robot_index),
                        name=request.name,
                        distance_ahead=request.distance_ahead if request.distance_ahead > 0.0 else 4.0,
                        radius=request.radius if request.radius > 0.0 else 0.8,
                    ),
                )
            if command == "enable_dynamic_replanning":
                return self._api_response(
                    response,
                    *self.set_dynamic_replanning(
                        enabled=True,
                        rate=request.rate if request.rate > 0.0 else None,
                        cooldown=request.cooldown if request.cooldown > 0.0 else None,
                        lookahead_time=request.lookahead_time if request.lookahead_time > 0.0 else None,
                        safety_margin=request.safety_margin if request.safety_margin > 0.0 else None,
                        replan_hysteresis=request.replan_hysteresis if request.replan_hysteresis > 0.0 else None,
                    ),
                )
            if command == "disable_dynamic_replanning":
                return self._api_response(response, *self.set_dynamic_replanning(enabled=False))
            if command == "set_dynamic_replanning":
                return self._api_response(
                    response,
                    *self.set_dynamic_replanning(
                        enabled=bool(request.enabled),
                        rate=request.rate if request.rate > 0.0 else None,
                        cooldown=request.cooldown if request.cooldown > 0.0 else None,
                        lookahead_time=request.lookahead_time if request.lookahead_time > 0.0 else None,
                        safety_margin=request.safety_margin if request.safety_margin > 0.0 else None,
                        replan_hysteresis=request.replan_hysteresis if request.replan_hysteresis > 0.0 else None,
                    ),
                )
            if command == "dynamic_replanning_status":
                return self._api_response(response, True, self._dynamic_replanning_status_message())
        except Exception as exc:
            return self._api_response(response, False, f"{command} failed: {exc}")

        return self._api_response(response, False, f"unknown backend world command '{command}'")

    def _backend_pose_command_callback(self, request, response):
        robot = self._robot_for_api(request.robot_index)
        if robot is None:
            return self._api_response(response, False, f"invalid robot_index={request.robot_index}")
        command = str(request.command).strip()
        try:
            if command == "set_vehicle_target":
                return self._api_response(response, *self.set_vehicle_target(robot, request.pose))
            if command == "set_task_target_world":
                return self._api_response(response, *self.set_task_target_world(robot, request.pose))
            if command == "set_task_target_arm_base":
                return self._api_response(response, *self.set_task_target_arm_base(robot, request.pose))
            if command == "add_waypoint":
                return self._api_response(
                    response,
                    *self.add_vehicle_waypoint_for_robot(
                        robot,
                        pose=request.pose,
                        use_current_target=request.use_current_target,
                    ),
                )
            if command == "reset_vehicle_world":
                return self._api_response(response, *self.reset_vehicle_world(robot, request.pose))
        except Exception as exc:
            return self._api_response(response, False, f"{command} failed: {exc}")
        return self._api_response(response, False, f"unknown backend pose command '{command}'")

    def _backend_waypoint_command_callback(self, request, response):
        robot = self._robot_for_api(request.robot_index)
        if robot is None:
            return self._api_response(response, False, f"invalid robot_index={request.robot_index}")
        command = str(request.command).strip()
        try:
            if command == "delete":
                ok = self.remove_vehicle_waypoint_for_robot(robot.k_robot, int(request.waypoint_index))
                return self._api_response(response, ok, f"delete waypoint {request.waypoint_index} for {robot.prefix}")
            if command == "clear":
                self.clear_vehicle_waypoints_for_robot(robot.k_robot)
                return self._api_response(response, True, f"cleared waypoints for {robot.prefix}")
            if command == "stop":
                self.set_robot_selected(robot.k_robot)
                self.stop_selected_vehicle_waypoints()
                return self._api_response(response, True, f"stopped waypoint mission for {robot.prefix}")
            if command == "execute":
                self.set_robot_selected(robot.k_robot)
                ok = self.execute_selected_vehicle_waypoints()
                return self._api_response(response, ok, f"execute waypoint mission for {robot.prefix}")
        except Exception as exc:
            return self._api_response(response, False, f"{command} failed: {exc}")
        return self._api_response(response, False, f"unknown backend waypoint command '{command}'")

    def _call_recording_service(self, client, action_name: str, on_success=None) -> bool:
        if not client.wait_for_service(timeout_sec=0.2):
            self.node.get_logger().warn(
                f"MCAP recording {action_name} rejected: bag_recorder_node service is not ready."
            )
            return False

        future = client.call_async(Trigger.Request())

        def _done_callback(done_future) -> None:
            try:
                response = done_future.result()
            except Exception as exc:
                self.node.get_logger().error(f"MCAP recording {action_name} failed: {exc}")
                return

            if response.success:
                self.node.get_logger().info(f"MCAP recording {action_name}: {response.message}")
                if on_success is not None:
                    on_success()
            else:
                self.node.get_logger().warn(f"MCAP recording {action_name} rejected: {response.message}")

        future.add_done_callback(_done_callback)
        return True

    def start_mcap_recording(self, on_success=None) -> bool:
        if self.mcap_recording_active:
            self.node.get_logger().warn("MCAP recording start rejected: recording is already active.")
            return False

        def _mark_active() -> None:
            self.mcap_recording_active = True
            if on_success is not None:
                on_success()

        return self._call_recording_service(self.start_recording_client, "start", on_success=_mark_active)

    def stop_mcap_recording(self, on_success=None) -> bool:
        if not self.mcap_recording_active:
            self.node.get_logger().warn("MCAP recording stop rejected: recording is not active.")
            return False

        def _mark_inactive() -> None:
            self.mcap_recording_active = False
            if on_success is not None:
                on_success()

        return self._call_recording_service(self.stop_recording_client, "stop", on_success=_mark_inactive)

    def reset_selected_simulation(self) -> bool:
        robot = self.robot_selected
        if "real" in robot.prefix:
            self.node.get_logger().warn(f"Reset Manager is disabled for real robot {robot.prefix}.")
            return False
        self.clear_vehicle_waypoints_for_robot(robot.k_robot)
        robot.reset_simulation()
        return True

    def hold_robot_current_state(self, robot: Robot, *, release_if_held: bool = False) -> tuple[bool, str]:
        if robot.control_mode in (ControlMode.REPLAY, ControlMode.REPLAY_SETTLE):
            return False, f"feedback hold rejected for {robot.prefix}; CmdReplay is active"
        self.set_robot_selected(robot.k_robot)
        robot.set_control_mode(ControlMode.PLANNER)
        robot.hold_current_state_with_feedback()
        if release_if_held and robot.sim_reset_hold:
            robot.release_simulation()
            return True, f"feedback hold and release requested for {robot.prefix}"
        return True, f"feedback hold requested for {robot.prefix}"

    def release_selected_simulation(self, *, feedback_hold: bool = False) -> bool:
        robot = self.robot_selected
        if "real" in robot.prefix:
            self.node.get_logger().warn(f"Reset Manager is disabled for real robot {robot.prefix}.")
            return False
        if feedback_hold:
            ok, message = self.hold_robot_current_state(robot, release_if_held=False)
            if not ok:
                self.node.get_logger().warn(message)
                return False
        robot.release_simulation()
        return True

    def plan_execute_selected(self) -> bool:
        robot = self.robot_selected
        if robot.controller_name == "CmdReplay":
            self.node.get_logger().warn(
                "Plan & Execute requires a feedback controller; choose PID/InvDyn/OGES first."
            )
            return False
        if robot.control_mode != ControlMode.PLANNER:
            robot.set_controller(robot.controller_name)

        if robot.task_based_controller:
            self.plan_task_trajectory()
            return True
        if self.selected_vehicle_waypoint_mission().waypoints:
            return self.execute_selected_vehicle_waypoints()
        self.plan_vehicle_trajectory()
        return True

    def selected_vehicle_waypoint_mission(self) -> VehicleWaypointMission:
        return self.vehicle_waypoint_missions[self.robot_selected.k_robot]

    def add_selected_vehicle_waypoint(self) -> int:
        if self.robot_selected.task_based_controller:
            self.node.get_logger().warn("Vehicle waypoints are only available for vehicle path planning.")
            return 0
        mission = self.selected_vehicle_waypoint_mission()
        count = mission.add_waypoint(self.target_vehicle_pose)
        self.node.get_logger().info(
            f"Added vehicle waypoint {count} for {self.robot_selected.prefix} at "
            f"[{self.target_vehicle_pose.position.x:.2f}, {self.target_vehicle_pose.position.y:.2f}, {self.target_vehicle_pose.position.z:.2f}]"
        )
        if mission.executing:
            self.node.get_logger().info(
                f"Extended active vehicle waypoint mission for {self.robot_selected.prefix}; "
                f"{len(mission.waypoints)} total waypoints queued."
            )
        return count

    def remove_last_selected_vehicle_waypoint(self) -> bool:
        mission = self.selected_vehicle_waypoint_mission()
        removed = mission.pop_last_waypoint()
        if removed is None:
            self.node.get_logger().warn(f"No vehicle waypoints to remove for {self.robot_selected.prefix}.")
            return False
        self.node.get_logger().info(
            f"Removed last vehicle waypoint for {self.robot_selected.prefix}; {len(mission.waypoints)} remaining."
        )
        return True

    def remove_vehicle_waypoint_for_robot(self, robot_k: int, waypoint_index: int) -> bool:
        robot = self.robots[robot_k]
        mission = self.vehicle_waypoint_missions[robot_k]
        was_executing = mission.executing
        removed_active_waypoint = mission.active_index == waypoint_index
        removed = mission.pop_waypoint(waypoint_index)
        if removed is None:
            self.node.get_logger().warn(
                f"No vehicle waypoint {waypoint_index + 1} to remove for {robot.prefix}."
            )
            return False
        if was_executing and removed_active_waypoint:
            mission.stop()
            robot.abrupt_planner_stop()
            self.node.get_logger().info(
                f"Removed active vehicle waypoint {waypoint_index + 1} for {robot.prefix}; mission stopped."
            )
        else:
            self.node.get_logger().info(
                f"Removed vehicle waypoint {waypoint_index + 1} for {robot.prefix}; {len(mission.waypoints)} remaining."
            )
        return True

    def remove_selected_vehicle_waypoint_at(self, waypoint_index: int) -> bool:
        return self.remove_vehicle_waypoint_for_robot(self.robot_selected.k_robot, waypoint_index)

    def clear_selected_vehicle_waypoints(self) -> None:
        mission = self.selected_vehicle_waypoint_mission()
        mission.clear()
        self.robot_selected.abrupt_planner_stop()
        self._clear_vehicle_waypoint_viz(self.robot_selected)
        self.node.get_logger().info(f"Cleared vehicle waypoints for {self.robot_selected.prefix}.")

    def clear_vehicle_waypoints_for_robot(self, robot_k: int) -> None:
        robot = self.robots[robot_k]
        mission = self.vehicle_waypoint_missions[robot_k]
        mission.clear()
        robot.abrupt_planner_stop()
        self._clear_vehicle_waypoint_viz(robot)
        self.node.get_logger().info(f"Cleared vehicle waypoints for {robot.prefix}.")

    def _clear_vehicle_waypoint_viz(self, robot: Robot) -> None:
        stamp_now = self.node.get_clock().now().to_msg()
        self.vehicle_waypoint_viz[robot.k_robot].clear(stamp_now, self.world_frame)
        if robot.planner is not None:
            robot.planner.clear_path(stamp_now, robot.world_frame)
            robot.planner.clear_target(stamp_now, robot.world_frame)

    def stop_selected_vehicle_waypoints(self) -> None:
        mission = self.selected_vehicle_waypoint_mission()
        if not mission.executing:
            self.node.get_logger().info(f"No active vehicle waypoint mission for {self.robot_selected.prefix}.")
            return
        mission.stop()
        self.robot_selected.abrupt_planner_stop(publish_zero=False)
        self.robot_selected.hold_current_state_with_feedback()
        self.node.get_logger().info(f"Stopped vehicle waypoint mission for {self.robot_selected.prefix}.")

    def execute_selected_vehicle_waypoints(self) -> bool:
        robot = self.robot_selected
        if robot.control_mode in (ControlMode.REPLAY, ControlMode.REPLAY_SETTLE):
            self.node.get_logger().warn(
                f"Vehicle waypoint mission ignored for {robot.prefix}; CmdReplay is active."
            )
            return False
        if robot.task_based_controller:
            self.node.get_logger().warn("Vehicle waypoint missions are available only in joint-space vehicle planning mode.")
            return False

        mission = self.selected_vehicle_waypoint_mission()
        if mission.executing:
            self.node.get_logger().info(
                f"Vehicle waypoint mission already running for {robot.prefix}; "
                f"{len(mission.waypoints) - mission.current_index} waypoint(s) remaining."
            )
            return False
        if robot.sim_reset_hold:
            if not mission.waypoints:
                self.node.get_logger().warn(f"No saved vehicle waypoints for {robot.prefix}.")
                return False

            robot.set_control_mode(ControlMode.PLANNER)
            robot.hold_current_state_with_feedback()

            def _start_after_release() -> None:
                if mission.executing:
                    return
                if not mission.start():
                    self.node.get_logger().warn(f"No saved vehicle waypoints for {robot.prefix}.")
                    return
                self.node.get_logger().info(
                    f"Executing {len(mission.waypoints)} vehicle waypoints for {robot.prefix} after release."
                )
                self._dispatch_vehicle_waypoint_if_ready(robot, mission)

            robot.release_simulation(on_success=_start_after_release)
            self.node.get_logger().info(
                f"Vehicle waypoint mission for {robot.prefix} queued until reset hold is released."
            )
            return True
        if not mission.start():
            self.node.get_logger().warn(f"No saved vehicle waypoints for {robot.prefix}.")
            return False

        self.node.get_logger().info(
            f"Executing {len(mission.waypoints)} vehicle waypoints for {robot.prefix}."
        )
        robot.set_control_mode(ControlMode.PLANNER)
        robot.abrupt_planner_stop()
        return self._dispatch_vehicle_waypoint_if_ready(robot, mission)

    def publish_fcl_environment_aabb_callback(self):
        stamp_now = self.node.get_clock().now().to_msg()
        min_marker, max_marker = geometry.visualize_min_max_coords(self.fcl_world.min_coords,
                                                                         self.fcl_world.max_coords,
                                                                           self.fcl_world.floor_depth, self.world_frame)
        min_marker.header.stamp = stamp_now
        max_marker.header.stamp = stamp_now
        self.env_aabb_pub.publish(min_marker)
        self.env_aabb_pub.publish(max_marker)

    def format_robot_metrics_overlay_text(self) -> str:
        lines = [
            'Robot Control Status',
            '',
        ]
        for index, robot in enumerate(self.robots):
            metrics = robot.get_energy_metrics()
            tracking = robot.get_controller_performance_metrics()
            mission = self.vehicle_waypoint_missions[robot.k_robot]
            state = robot.get_state()
            ned_vel = np.asarray(state.get("ned_vel", [0.0] * 6), dtype=float).reshape(-1)
            linear_speed_mps = float(np.linalg.norm(ned_vel[:3])) if ned_vel.size >= 3 else 0.0
            controller_in_use = robot.controller_name
            selected = ' *' if robot == self.robot_selected else ''
            hold_state = 'HELD' if robot.sim_reset_hold else 'RELEASED'
            total_energy = (
                metrics['vehicle_control_energy_abs'] +
                metrics['arm_control_energy_abs']
            )
            total_power = (
                metrics['vehicle_control_power_abs'] +
                metrics['arm_control_power_abs']
            )
            if index > 0:
                lines.append('-' * 72)
            if mission.executing:
                active_idx = mission.active_display_index()
                active_label = active_idx + 1 if active_idx is not None else mission.current_index + 1
                waypoint_info = f"WP {active_label}/{len(mission.waypoints)} {mission.state.upper()}"
            elif mission.waypoints:
                waypoint_info = f"WP queued {len(mission.waypoints)}"
            else:
                waypoint_info = "WP none"
            lines.append(
                f"{metrics['prefix']}{selected} | {hold_state} | {controller_in_use} | "
                f"v {linear_speed_mps:.3f} m/s | payload {metrics['arm_payload_mass']:.3f} kg | "
                f"g {metrics['arm_gravity']:.3f} m/s^2 | {waypoint_info} | "
                f"score {tracking.get('tracking_score', 0.0):.3f} "
                f"(rms {tracking.get('tracking_score_rms', 0.0):.3f}) | "
                f"nXTE {tracking.get('vehicle_n_cross_track', 0.0):.3f} | "
                f"nATE {tracking.get('vehicle_n_along_track', 0.0):.3f} | "
                f"nVel {tracking.get('vehicle_n_linear_velocity', 0.0):.3f} | "
                f"nAtt {tracking.get('vehicle_n_attitude', 0.0):.3f} | "
                f"nArm {tracking.get('arm_n_position', 0.0):.3f} | "
                f"E {total_energy:.2f} J | dE/dt {total_power:.2f} W"
            )
        return '\n'.join(lines)

    def format_research_overlay_text(self) -> str:
        return f"© {datetime.now().year} Louisiana State University. Research use."

    def publish_robot_metrics_overlay_callback(self) -> None:
        text = self.format_robot_metrics_overlay_text()
        lines = text.splitlines() or [""]
        longest_line = max(len(line) for line in lines)
        msg = OverlayText()
        msg.action = OverlayText.ADD
        msg.text_size = 14.0
        char_width = msg.text_size * 0.92
        robot_count = sum(1 for line in lines if line.startswith("robot_"))
        msg.width = min(1800, max(900, int(longest_line * char_width) + 96))
        msg.height = max(356, int(238 + 118 * max(robot_count, 1)))
        msg.horizontal_distance = 28
        msg.vertical_distance = 170
        msg.horizontal_alignment = OverlayText.RIGHT
        msg.vertical_alignment = OverlayText.TOP
        msg.bg_color.a = 0.45
        msg.line_width = 2
        msg.font = 'DejaVu Sans Mono'
        msg.fg_color.r = 1.0
        msg.fg_color.g = 0.9
        msg.fg_color.b = 0.2
        msg.fg_color.a = 0.95
        msg.text = text
        self.robot_metrics_overlay_pub.publish(msg)

    def publish_research_overlay_callback(self) -> None:
        msg = String()
        msg.data = self.format_research_overlay_text()
        self.research_overlay_pub.publish(msg)

    def _dynamic_joint_states_cb(self, msg: DynamicJointState) -> None:
        for robot in self.robots:
            robot.listener_callback(msg)

    def _mocap_pose_cb(self, msg: PoseStamped) -> None:
        for robot in self.robots:
            robot._mocap_pose_cb(msg)

    def fcl_update_callback(self):
        self.fcl_world.update_from_tf(self.tf_buffer, rclpy.time.Time())

    def set_robot_selected(self, robot_k):
        for r in self.robots:
            if r.k_robot == robot_k:
                if getattr(self, "robot_selected", None) is r:
                    return
                self.robot_selected = r
                self.node.get_logger().info(f"Robot {self.robot_selected.prefix} selected for planning.")
                self._set_camera_frame_for_robot(self.robot_selected)
                return
        self.node.get_logger().error(f"No robot with k_robot={robot_k}")
        raise ValueError(f"No robot with k_robot={robot_k}")

    def _set_camera_frame_for_robot(self, robot: Robot) -> None:
        self._pending_camera_robot = robot
        if self._try_set_camera_frame_for_robot(robot):
            self._pending_camera_robot = None

    def _retry_pending_camera_selection(self) -> None:
        robot = self._pending_camera_robot
        if robot is None:
            return
        if self._try_set_camera_frame_for_robot(robot):
            self._pending_camera_robot = None

    def _try_set_camera_frame_for_robot(self, robot: Robot) -> bool:
        robot_has_sim_camera = robot.prefix != "robot_real_" or not self.use_vehicle_hardware
        if self.camera_source == "mixed":
            return self._set_sim_camera_selected_robot(robot)
        if self.camera_source == "sim" and robot_has_sim_camera:
            return self._set_sim_camera_selected_robot(robot)

        if self.use_vehicle_hardware and self.camera_source != "sim" and robot.prefix != "robot_real_":
            self.node.get_logger().debug(
                "Camera frame update skipped for simulated robot because the active camera stream is the real vehicle camera."
            )
            return True

        if not self.camera_parameters_client.service_is_ready():
            self.node.get_logger().debug(
                "Camera frame update skipped because /gstreamer_camera_node/set_parameters is not ready."
            )
            return False

        frame_id = f"{robot.prefix}camera_link"
        request = SetParameters.Request()
        request.parameters = [
            Parameter(
                name="frame_id",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_STRING,
                    string_value=frame_id,
                ),
            )
        ]
        future = self.camera_parameters_client.call_async(request)

        def _log_camera_frame_result(done_future):
            try:
                response = done_future.result()
            except Exception as exc:
                self.node.get_logger().warn(f"Camera frame update failed for {robot.prefix}: {exc}")
                return
            if not response.results or not response.results[0].successful:
                reason = response.results[0].reason if response.results else "empty response"
                self.node.get_logger().warn(f"Camera frame update rejected for {robot.prefix}: {reason}")

        future.add_done_callback(_log_camera_frame_result)
        return True

    def _set_sim_camera_selected_robot(self, robot: Robot) -> bool:
        if not self.sim_camera_parameters_client.service_is_ready():
            self.node.get_logger().debug(
                "Sim camera selection skipped because /sim_camera_renderer_node/set_parameters is not ready."
            )
            return False

        request = SetParameters.Request()
        request.parameters = [
            Parameter(
                name="selected_prefix",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_STRING,
                    string_value=robot.prefix,
                ),
            )
        ]
        future = self.sim_camera_parameters_client.call_async(request)

        def _log_sim_camera_result(done_future):
            try:
                response = done_future.result()
            except Exception as exc:
                self.node.get_logger().warn(f"Sim camera selection failed for {robot.prefix}: {exc}")
                return
            if not response.results or not response.results[0].successful:
                reason = response.results[0].reason if response.results else "empty response"
                self.node.get_logger().warn(f"Sim camera selection rejected for {robot.prefix}: {reason}")

        future.add_done_callback(_log_sim_camera_result)
        return True
    
    def target_vehicle_in_world_tf_timer_callback(self):
        stamp_now = self.node.get_clock().now().to_msg()
        vehicle_target_t = geometry.get_broadcast_tf(stamp=stamp_now,
                                                           pose=self.target_vehicle_pose,
                                                             parent_frame=self.world_frame,
                                                               child_frame=self.vehicle_target_frame)
        self.tf_broadcaster.sendTransform(vehicle_target_t)

    def target_arm_base_tf_timer_callback(self):
        stamp_now = self.node.get_clock().now().to_msg()
        arm_base_t = geometry.get_broadcast_tf(stamp=stamp_now,
                                                    pose=self.arm_base_wrt_vehicle_center_Pose,
                                                      parent_frame=self.vehicle_target_frame,
                                                        child_frame=self.arm_base_target_frame)
        self.tf_broadcaster.sendTransform(arm_base_t)

    def target_endeffector_in_world_tf_timer_callback(self):
        if self.target_world_endeffector_pose is None:
            return
        stamp_now = self.node.get_clock().now().to_msg()
        endeffector_t = geometry.get_broadcast_tf(
            stamp=stamp_now,
            pose=self.target_world_endeffector_pose,
            parent_frame=self.world_frame,
            child_frame=self.world_endeffector_target_frame,
        )
        self.tf_broadcaster.sendTransform(endeffector_t)

    def vehicle_target_cloud_timer_callback(self):
        if not self.robot_selected.task_based_controller:
            header = Header()
            header.frame_id = self.vehicle_target_frame
            # These are interactive target-frame clouds. A zero stamp lets RViz
            # render them with the latest target TF instead of waiting for an
            # exact timestamp match from a separate TF timer.
            header.stamp = rclpy.time.Time().to_msg()

            if self.taskspace_pc_publisher_.get_subscription_count() > 0:
                rov_cloud_msg = pc2.create_cloud_xyz32(header, self.workspace_pts)
                self.taskspace_pc_publisher_.publish(rov_cloud_msg)

            if self.rov_pc_publisher_.get_subscription_count() > 0:
                cloud_msg = pc2.create_cloud_xyz32(header, self.rov_ellipsoid_cl_pts)
                self.rov_pc_publisher_.publish(cloud_msg)

    def is_valid_arm_base_task(self, target_arm_base_endeffector_pose: Pose) -> bool:
        pose_v = self.robot_selected.try_transform_pose(
            target_arm_base_endeffector_pose,
            target_frame=self.vehicle_target_frame,
            source_frame=self.arm_base_target_frame, 
            warn_context="is_valid_arm_base_task",
        )
        if pose_v is None:
            return False

        p = pose_v.position
        xyz = np.array([p.x, p.y, p.z], dtype=float)
        return geometry.is_point_valid(self.workspace_hull, self.vehicle_body_hull, xyz)

    def initialise_target_Poses(self):
        self.target_vehicle_pose = Pose()
        self.target_vehicle_pose.orientation.w = 1.0
        self.target_arm_base_endeffector_pose = Pose()
        self.target_arm_base_endeffector_pose.position.x = ReachParams.endeffector_wrt_base_home[0]
        self.target_arm_base_endeffector_pose.position.y = ReachParams.endeffector_wrt_base_home[1]
        self.target_arm_base_endeffector_pose.position.z = ReachParams.endeffector_wrt_base_home[2]
        self.target_arm_base_endeffector_pose.orientation.w = 1.0
        self.target_world_endeffector_pose = Pose()
        self.target_world_endeffector_pose.orientation.w = 1.0

    def reset_selected_robot_targets(self):
        self.initialise_target_Poses()
        self.node.get_logger().info(
            f"Reset planner and marker targets for {self.robot_selected.prefix}."
        )

    def plan_vehicle_trajectory(self):
        if self.robot_selected.control_mode in (ControlMode.REPLAY, ControlMode.REPLAY_SETTLE):
            self.robot_selected.abrupt_planner_stop()
            self.node.get_logger().warn(
                f"Planner request ignored for {self.robot_selected.prefix}; CmdReplay is active."
            )
            return False
        self.robot_selected.set_control_mode(ControlMode.PLANNER)
        goal_pose = self.target_vehicle_pose
        return self.robot_selected.plan_vehicle_trajectory_action(
            goal_pose=goal_pose,
            time_limit=1.0,
            robot_collision_radius=float(self.fcl_world.vehicle_radius),
        )

    def _dispatch_vehicle_waypoint_if_ready(
        self,
        robot: Robot,
        mission: VehicleWaypointMission,
    ) -> bool:
        if not mission.executing:
            return False
        if robot.control_mode in (ControlMode.REPLAY, ControlMode.REPLAY_SETTLE):
            mission.stop()
            robot.abrupt_planner_stop()
            self.node.get_logger().info(
                f"Waypoint dispatch stopped for {robot.prefix}; CmdReplay is active."
            )
            return False
        if mission.active_index is not None:
            self.node.get_logger().debug(
                f"Waypoint dispatch skipped for {robot.prefix}; waypoint {mission.active_index + 1} is already active."
            )
            return False
        if robot.sim_reset_hold or robot.task_based_controller:
            reason = "simulation is held after reset" if robot.sim_reset_hold else "robot is in task-based controller mode"
            self.node.get_logger().info(
                f"Waypoint dispatch blocked for {robot.prefix}: {reason}.",
                throttle_duration_sec=1.0,
            )
            return False
        if robot.planner_action_client.busy:
            self.node.get_logger().info(
                f"Waypoint dispatch blocked for {robot.prefix}: planner action is still busy.",
                throttle_duration_sec=1.0,
            )
            return False
        if robot.vehicle_cart_traj is not None and robot.vehicle_cart_traj.active:
            self.node.get_logger().info(
                f"Waypoint dispatch blocked for {robot.prefix}: vehicle trajectory is still active.",
                throttle_duration_sec=1.0,
            )
            return False

        goal_pose = mission.current_waypoint()
        if goal_pose is None:
            mission.stop()
            self.node.get_logger().warn(
                f"Waypoint dispatch stopped for {robot.prefix}: no current waypoint is available."
            )
            return False

        robot.set_control_mode(ControlMode.PLANNER)
        sent = robot.plan_vehicle_trajectory_action(
            goal_pose=goal_pose,
            time_limit=1.0,
            robot_collision_radius=float(self.fcl_world.vehicle_radius),
        )
        if sent:
            mission.mark_planning()
            self.node.get_logger().info(
                f"Dispatched vehicle waypoint {mission.active_index + 1}/{len(mission.waypoints)} for {robot.prefix}."
            )
            return True

        mission.stop()
        self.node.get_logger().warn(f"Failed to dispatch vehicle waypoint mission for {robot.prefix}.")
        return False

    def _is_robot_at_waypoint(self, robot: Robot, goal_pose: Pose, tolerance_m: float) -> bool:
        pose_now = robot._pose_from_state_in_frame(self.world_frame)
        if pose_now is None:
            return False
        return pose_position_distance(pose_now, goal_pose) <= float(tolerance_m)

    def _robot_waypoint_tracking_metrics(self, robot: Robot, goal_pose: Pose) -> dict | None:
        pose_now = robot._pose_from_state_in_frame(self.world_frame)
        if pose_now is None:
            return None

        distance_m = pose_position_distance(pose_now, goal_pose)
        state = robot.get_state()
        ned_vel = np.asarray(state.get("ned_vel", [0.0] * 6), dtype=float).reshape(-1)
        linear_speed_mps = float(np.linalg.norm(ned_vel[:3])) if ned_vel.size >= 3 else 0.0
        yaw_blend = float(getattr(robot, "yaw_blend_factor", 0.0))
        yaw_blend_threshold = float(
            getattr(robot.vehicle_cart_traj, "yaw_finish_threshold", 0.98)
        ) if robot.vehicle_cart_traj is not None else 0.98

        return {
            "distance_m": distance_m,
            "linear_speed_mps": linear_speed_mps,
            "yaw_blend": yaw_blend,
            "yaw_blend_threshold": yaw_blend_threshold,
        }

    def _has_robot_reached_vehicle_waypoint(
        self,
        robot: Robot,
        goal_pose: Pose,
        *,
        position_tolerance_m: float,
    ) -> tuple[bool, dict | None]:
        metrics = self._robot_waypoint_tracking_metrics(robot, goal_pose)
        if metrics is None:
            return False, None

        reached = (
            metrics["distance_m"] <= float(position_tolerance_m)
            and metrics["yaw_blend"] >= metrics["yaw_blend_threshold"]
        )
        return reached, metrics

    def vehicle_waypoint_execution_callback(self) -> None:
        for robot in self.robots:
            mission = self.vehicle_waypoint_missions[robot.k_robot]
            if not mission.executing:
                continue
            if robot.control_mode in (ControlMode.REPLAY, ControlMode.REPLAY_SETTLE):
                mission.stop()
                robot.abrupt_planner_stop()
                self.node.get_logger().info(
                    f"Stopped vehicle waypoint mission for {robot.prefix}; CmdReplay is active."
                )
                continue
            if robot.sim_reset_hold:
                mission.stop()
                continue

            if mission.active_index is None:
                self._dispatch_vehicle_waypoint_if_ready(robot, mission)
                continue

            goal_pose = mission.active_waypoint()
            if goal_pose is None:
                mission.stop()
                continue

            goal_reached_by_state, tracking_metrics = self._has_robot_reached_vehicle_waypoint(
                robot,
                goal_pose,
                position_tolerance_m=mission.position_tolerance_m,
            )

            if robot.planner_action_client.busy:
                continue

            if robot.vehicle_cart_traj is not None and robot.vehicle_cart_traj.active and not goal_reached_by_state:
                mission.mark_tracking()
                continue

            if goal_reached_by_state:
                if robot.vehicle_cart_traj is not None:
                    robot.vehicle_cart_traj.active = False
                reached_index = mission.active_index
                has_more = mission.advance()
                self.node.get_logger().info(
                    f"Reached vehicle waypoint {reached_index + 1}/{len(mission.waypoints)} for {robot.prefix}."
                )
                if has_more:
                    self._dispatch_vehicle_waypoint_if_ready(robot, mission)
                else:
                    self.node.get_logger().info(f"Completed vehicle waypoint mission for {robot.prefix}.")
            else:
                mission.stop()
                robot.abrupt_planner_stop(publish_zero=False)
                robot.hold_current_state_with_feedback()
                if tracking_metrics is not None:
                    self.node.get_logger().warn(
                        f"Vehicle waypoint mission stopped for {robot.prefix}; "
                        f"planner/trajectory ended before the waypoint was reached "
                        f"(pos_err={tracking_metrics['distance_m']:.3f} m, "
                        f"pos_tol={mission.position_tolerance_m:.3f} m, "
                        f"speed={tracking_metrics['linear_speed_mps']:.3f} m/s, "
                        f"yaw_blend={tracking_metrics['yaw_blend']:.3f}, "
                        f"yaw_blend_threshold={tracking_metrics['yaw_blend_threshold']:.3f})."
                    )
                else:
                    self.node.get_logger().warn(
                        f"Vehicle waypoint mission stopped for {robot.prefix}; planner/trajectory ended before the waypoint was reached."
                    )

    def vehicle_waypoint_viz_callback(self) -> None:
        if self.robot_selected is None:
            return
        stamp_now = self.node.get_clock().now().to_msg()
        selected_robot = self.robot_selected
        for robot in self.robots:
            viz = self.vehicle_waypoint_viz[robot.k_robot]
            if robot.k_robot != selected_robot.k_robot:
                viz.clear(stamp_now, self.world_frame)
                continue
            mission = self.vehicle_waypoint_missions[robot.k_robot]
            viz.update(
                stamp=stamp_now,
                frame_id=self.world_frame,
                waypoints=mission.waypoints,
                active_index=mission.active_display_index(),
            )

    def solve_execute_inverse_kinematics_wrt_vehicle_frame(self, task_pose:Pose):
        msg = {'is_success':False,'result':None}
        if self.is_valid_arm_base_task(task_pose):
            q_ik_sol = self.robot_selected.manipulator_inverse_kinematics(
                np.array([task_pose.position.x, task_pose.position.y, task_pose.position.z]))
            msg['is_success'] = True
            msg['result'] = q_ik_sol
        return msg

    def plan_and_execute_task_trajectory_wrt_vehicle(self):
        if self.robot_selected.sim_reset_hold:
            return
        if not self.robot_selected.task_based_controller:
            msg = self.solve_execute_inverse_kinematics_wrt_vehicle_frame(self.target_arm_base_endeffector_pose)
            if msg['is_success']:
                self.robot_selected.arm.q_command = msg['result']

    def plan_task_trajectory(self):
        if self.robot_selected.sim_reset_hold:
            self.node.get_logger().warn(
                f"Task trajectory request ignored for {self.robot_selected.prefix}; simulation is held after reset."
            )
            return
        if self.robot_selected.task_based_controller and self.robot_selected.task_pose_in_world:
            self.robot_selected.enable_planner_output()
            self.robot_selected.solve_inverse_kinematics_wrt_world_frame(self.target_world_endeffector_pose)
            self.node.get_logger().info(f"task trajectory plan & control", throttle_duration_sec=2.0)
            self.node.get_logger().info(f"{self.robot_selected.task_pose_in_world} robot.", throttle_duration_sec=2.0)
            self.node.get_logger().info(f"{self.target_world_endeffector_pose} target.", throttle_duration_sec=2.0)
            self.node.get_logger().info(f"{self.target_arm_base_endeffector_pose} base target.", throttle_duration_sec=2.0)
        return
