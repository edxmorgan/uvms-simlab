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
from simlab.uvms_parameters import ReachParams
import rclpy
from rclpy.node import Node
from simlab.fcl_checker import FCLWorld
import numpy as np
from scipy.spatial import ConvexHull
import os
import ament_index_python
from simlab import backend_utils
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from visualization_msgs.msg import Marker
import tf2_ros
from typing import Dict, List
from simlab.robot import Robot, ControlMode
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from control_msgs.msg import DynamicJointState
from std_srvs.srv import Trigger
from simlab.planner_markers import PathPlanner
from simlab.cartesian_ruckig import VehicleCartesianRuckig
from simlab.frame_utils import PoseX
from simlab.vehicle_waypoint_mission import (
    VehicleWaypointMission,
    VehicleWaypointViz,
    pose_position_distance,
)

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
        self.rov_ellipsoid_cl_pts = backend_utils.generate_rov_ellipsoid(a=0.3, b=0.3, c=0.2, num_points=10000)
        self.vehicle_body_hull = ConvexHull(self.rov_ellipsoid_cl_pts)

        pointcloud_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, durability=QoSDurabilityPolicy.VOLATILE,
                                    reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.taskspace_pc_publisher_ = self.node.create_publisher(PointCloud2,'workspace_pointcloud',pointcloud_qos)
        self.rov_pc_publisher_ = self.node.create_publisher(PointCloud2, 'base_pointcloud', pointcloud_qos)

        # stack clouds that represent the vehicle occupied volume
        all_pts = np.vstack([
            np.asarray(self.rov_ellipsoid_cl_pts, dtype=float),
            np.asarray(self.workspace_pts, dtype=float)
        ])

        robot_collision_radius = backend_utils.compute_bounding_sphere_radius(all_pts, quantile=0.995, pad=0.03)
        self.node.get_logger().info(f"Planner robot approximation sphere radius set to {robot_collision_radius:.3f} m")
        self.fcl_world.set_robot_collision_radius(robot_collision_radius)

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
        self.fcl_update_timer = self.node.create_timer(1.0 / 50.0, self.fcl_update_callback)
        self.target_vehicle_marker_in_world_tf_timer = self.node.create_timer(1.0 / 60.0, self.target_vehicle_in_world_tf_timer_callback)
        self.target_arm_base_marker_tf_timer = self.node.create_timer(1.0 / 60.0, self.target_arm_base_tf_timer_callback)
        self.target_endeffector_in_world_tf_timer = self.node.create_timer(1.0 / 60.0, self.target_endeffector_in_world_tf_timer_callback)
        self.vehicle_target_cloud_timer = self.node.create_timer(1.0 / 100.0, self.vehicle_target_cloud_timer_callback)
        self.task_on_vehicle_solve_timer = self.node.create_timer(1.0 / 10.0, self.plan_and_execute_task_trajectory_wrt_vehicle)
        
        self.planner_marker_publisher = self.node.create_publisher(Marker, "planned_waypoints_marker", planner_viz_qos)
        self.robots:List[Robot] = []
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
        self.set_robot_selected(self.robots[0].k_robot)
        self.vehicle_waypoint_execution_timer = self.node.create_timer(
            1.0 / 20.0,
            self.vehicle_waypoint_execution_callback,
        )
        self.vehicle_waypoint_viz_timer = self.node.create_timer(
            1.0 / 10.0,
            self.vehicle_waypoint_viz_callback,
        )

    def close(self) -> None:
        for robot in self.robots:
            robot.close()
        self.robots.clear()
        self.robot_selected = None

    def _call_recording_service(self, client, action_name: str, on_success=None) -> None:
        if not client.wait_for_service(timeout_sec=0.2):
            self.node.get_logger().warn(
                f"MCAP recording {action_name} rejected: bag_recorder_node service is not ready."
            )
            return

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

    def start_mcap_recording(self, on_success=None) -> None:
        self._call_recording_service(self.start_recording_client, "start", on_success=on_success)

    def stop_mcap_recording(self, on_success=None) -> None:
        self._call_recording_service(self.stop_recording_client, "stop", on_success=on_success)

    def reset_selected_simulation(self) -> bool:
        robot = self.robot_selected
        if "real" in robot.prefix:
            self.node.get_logger().warn(f"Reset Manager is disabled for real robot {robot.prefix}.")
            return False
        self.clear_vehicle_waypoints_for_robot(robot.k_robot)
        robot.reset_simulation()
        return True

    def release_selected_simulation(self) -> bool:
        robot = self.robot_selected
        if "real" in robot.prefix:
            self.node.get_logger().warn(f"Reset Manager is disabled for real robot {robot.prefix}.")
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
        self.robot_selected.abrupt_planner_stop()
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
        min_marker, max_marker = backend_utils.visualize_min_max_coords(self.fcl_world.min_coords,
                                                                         self.fcl_world.max_coords,
                                                                           self.fcl_world.floor_depth, self.world_frame)
        min_marker.header.stamp = stamp_now
        max_marker.header.stamp = stamp_now
        self.env_aabb_pub.publish(min_marker)
        self.env_aabb_pub.publish(max_marker)

    def format_robot_metrics_overlay_text(self) -> str:
        lines = ['Robot Control Status']
        for index, robot in enumerate(self.robots):
            metrics = robot.get_energy_metrics()
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
                f"E {total_energy:.2f} J | dE/dt {total_power:.2f} W"
            )
        return '\n'.join(lines)

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
                self.robot_selected = r
                self.node.get_logger().info(f"Robot {self.robot_selected.prefix} selected for planning.")
                return
        self.node.get_logger().error(f"No robot with k_robot={robot_k}")
        raise ValueError(f"No robot with k_robot={robot_k}")
    
    def target_vehicle_in_world_tf_timer_callback(self):
        stamp_now = self.node.get_clock().now().to_msg()
        vehicle_target_t = backend_utils.get_broadcast_tf(stamp=stamp_now,
                                                           pose=self.target_vehicle_pose,
                                                             parent_frame=self.world_frame,
                                                               child_frame=self.vehicle_target_frame)
        self.tf_broadcaster.sendTransform(vehicle_target_t)

    def target_arm_base_tf_timer_callback(self):
        stamp_now = self.node.get_clock().now().to_msg()
        arm_base_t = backend_utils.get_broadcast_tf(stamp=stamp_now,
                                                    pose=self.arm_base_wrt_vehicle_center_Pose,
                                                      parent_frame=self.vehicle_target_frame,
                                                        child_frame=self.arm_base_target_frame)
        self.tf_broadcaster.sendTransform(arm_base_t)

    def target_endeffector_in_world_tf_timer_callback(self):
        if self.target_world_endeffector_pose is None:
            return
        stamp_now = self.node.get_clock().now().to_msg()
        endeffector_t = backend_utils.get_broadcast_tf(
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
            header.stamp = self.node.get_clock().now().to_msg()

            rov_cloud_msg = pc2.create_cloud_xyz32(header, self.workspace_pts)
            self.taskspace_pc_publisher_.publish(rov_cloud_msg)

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
        return backend_utils.is_point_valid(self.workspace_hull, self.vehicle_body_hull, xyz)

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
                robot.abrupt_planner_stop()
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
