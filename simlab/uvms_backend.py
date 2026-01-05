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
from alpha_reach import Params as alpha_params 
import rclpy
from rclpy.node import Node
from fcl_checker import FCLWorld
import numpy as np
from scipy.spatial import ConvexHull
import os
import ament_index_python
import backend_utils
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from visualization_msgs.msg import Marker
import tf2_ros
from typing import List
from robot import Robot
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from se3_ompl_planner import plan_se3_path
from planner_markers import PathPlanner
from cartesian_ruckig import VehicleCartesianRuckig, EndeffectorCartesianRuckig
from frame_utils import PoseX

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
        workspace_pts_path = os.path.join(package_share_directory, 'manipulator/workspace.npy')
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

        planner_radius = backend_utils.compute_bounding_sphere_radius(all_pts, quantile=0.995, pad=0.03)
        self.node.get_logger().info(f"Planner robot approximation sphere radius set to {planner_radius:.3f} m")
        self.fcl_world.set_planner_radius(planner_radius)

        viz_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST,depth=1,durability=QoSDurabilityPolicy.VOLATILE,
                    reliability=QoSReliabilityPolicy.RELIABLE)
        # publisher for FCL environment AABB
        self.env_aabb_pub = self.node.create_publisher(Marker, "fcl_environment_aabb", viz_qos)
        self.env_bounds_timer = self.node.create_timer(1.0 / 1.0, self.publish_fcl_environment_aabb_callback)
        self.fcl_update_timer = self.node.create_timer(1.0 / 50.0, self.fcl_update_callback)
        self.target_vehicle_marker_in_world_tf_timer = self.node.create_timer(1.0 / 60.0, self.target_vehicle_in_world_tf_timer_callback)
        self.target_arm_base_marker_tf_timer = self.node.create_timer(1.0 / 60.0, self.target_arm_base_tf_timer_callback)
        self.target_endeffector_in_world_tf_timer = self.node.create_timer(1.0 / 60.0, self.target_endeffector_in_world_tf_timer_callback)
        self.vehicle_target_cloud_timer = self.node.create_timer(1.0 / 100.0, self.vehicle_target_cloud_timer_callback)
        self.task_on_vehicle_solve_timer = self.node.create_timer(1.0 / 10.0, self.plan_and_execute_task_trajectory_wrt_vehicle)
        self.task_on_world_solve_timer = self.node.create_timer(1.0 / 10.0, self.plan_and_execute_task_trajectory_wrt_world)

        
        self.planner_marker_publisher = self.node.create_publisher(Marker, "planned_waypoints_marker", viz_qos)
        self.robots:List[Robot] = []
        self.max_cartesian_waypoints = 500
        self.robots_prefix = self.node.get_parameter('robots_prefix').value
        self.controllers = self.node.get_parameter('controllers').value
        for k, (prefix, controller) in enumerate(zip(self.robots_prefix, self.controllers)):
            robot_k = Robot(self.node, self.tf_buffer, k, 4, prefix, controller)
            robot_k.vehicle_cart_traj = VehicleCartesianRuckig(
                self.node,
                dofs=3,
                control_dt=1.0 / robot_k.control_frequency,
                max_waypoints=self.max_cartesian_waypoints,
            )
            robot_k.planner = PathPlanner(self.planner_marker_publisher, ns=f"planner/{prefix}", base_id=k)

            self.robots.append(robot_k)

        self.robot_selected = self.robots[0]

        self.initialise_target_Poses()
    
        self.max_vel = np.array([0.15, 0.15, 0.10], dtype=float)
        self.max_acc = np.array([0.15, 0.15, 0.12], dtype=float)
        self.max_jerk = np.array([0.5, 0.5, 0.4], dtype=float)

        self.max_end_vel = np.array([1.15, 1.15, 1.10], dtype=float)
        self.max_end_acc = np.array([1.15, 1.15, 1.12], dtype=float)
        self.max_end_jerk = np.array([0.5, 0.5, 0.4], dtype=float)

    def publish_fcl_environment_aabb_callback(self):
        stamp_now = self.node.get_clock().now().to_msg()
        min_marker, max_marker = backend_utils.visualize_min_max_coords(self.fcl_world.min_coords,
                                                                         self.fcl_world.max_coords,
                                                                           self.fcl_world.floor_depth, self.world_frame)
        min_marker.header.stamp = stamp_now
        max_marker.header.stamp = stamp_now
        self.env_aabb_pub.publish(min_marker)
        self.env_aabb_pub.publish(max_marker)

    def fcl_update_callback(self):
        self.fcl_world.update_from_tf(self.tf_buffer, rclpy.time.Time())

    def set_robot_selected(self, robot_k):
        for r in self.robots:
            if r.k_robot == robot_k:
                self.robot_selected = r
                self.node.get_logger().info(f"Robot {self.robot_selected.prefix} selected for planning.")
                return
        raise self.node.get_logger().error(f"No robot with k_robot={robot_k}")
    
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
        stamp_now = self.node.get_clock().now().to_msg()
        endeffector_t = backend_utils.get_broadcast_tf(
            stamp=stamp_now,
            pose=self.target_world_endeffector_pose,
            parent_frame=self.world_frame,
            child_frame=self.world_endeffector_target_frame,
        )
        self.tf_broadcaster.sendTransform(endeffector_t)

    def vehicle_target_cloud_timer_callback(self):
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
        self.target_arm_base_endeffector_pose = Pose()
        self.target_arm_base_endeffector_pose.position.x = alpha_params.endeffector_wrt_base_home[0]
        self.target_arm_base_endeffector_pose.position.y = alpha_params.endeffector_wrt_base_home[1]
        self.target_arm_base_endeffector_pose.position.z = alpha_params.endeffector_wrt_base_home[2]
        self.target_world_endeffector_pose = Pose()

    def _get_robot_pose_now_world(self) -> Pose:
        return self.robot_selected._pose_from_state_in_frame(self.world_frame)

    def _pose_to_xyz_quat_wxyz(self, pose: Pose):
        xyz = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
        quat_wxyz = np.array(
            [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z],
            dtype=float
        )
        return xyz, quat_wxyz

    def _get_vehicle_goal_from_marker(self):
        goal_pose = self.target_vehicle_pose
        goal_xyz = np.array([goal_pose.position.x, goal_pose.position.y, goal_pose.position.z], dtype=float)
        goal_quat_wxyz = np.array(
            [goal_pose.orientation.w, goal_pose.orientation.x, goal_pose.orientation.y, goal_pose.orientation.z],
            dtype=float
        )
        return goal_xyz, goal_quat_wxyz

    def _plan_se3(self, start_xyz, start_quat_wxyz, goal_xyz, goal_quat_wxyz):
        return plan_se3_path(
            self.node,
            start_xyz=start_xyz,
            start_quat_wxyz=start_quat_wxyz,
            goal_xyz=goal_xyz,
            goal_quat_wxyz=goal_quat_wxyz,
            time_limit=1.0,
            safety_margin=1e-2,
            env_bounds=self.fcl_world.env_xyz_bounds,
        )

    def _start_vehicle_cartesian_ruckig(self, start_xyz, path_xyz: np.ndarray) -> None:
        self.robot_selected.vehicle_cart_traj.start_from_path(
            current_position=list(start_xyz),
            path_xyz=path_xyz,
            max_vel=self.max_vel,
            max_acc=self.max_acc,
            max_jerk=self.max_jerk,
        )

    def _log_plan_context(self, start_xyz, start_quat_wxyz, goal_xyz, goal_quat_wxyz) -> None:
        self.node.get_logger().info(
            f"Planning for {self.robot_selected.prefix}, "
            f"start_xyz={np.array(start_xyz, float).round(3).tolist()}, "
            f"goal_xyz={np.array(goal_xyz, float).round(3).tolist()}"
        )

    def _save_vehicle_goal_from_target(self):
        goal_xyz_world_nwu, goal_quat_wxyz_world = self._get_vehicle_goal_from_marker()

        # Convert world (NWU) -> map (NED)
        res = self.robot_selected.world_nwu_to_map_ned(
            xyz_world_nwu=goal_xyz_world_nwu,
            quat_world_wxyz=goal_quat_wxyz_world,
            warn_context=f"save goal world->map ({self.robot_selected.prefix})",
        )
        if res is None:
            self.robot_selected.final_goal_map_ned_6 = None
            return

        p_goal_ned, rpy_goal_ned = res
        self.robot_selected.final_goal_map_ned_6 = np.array(
            [p_goal_ned[0], p_goal_ned[1], p_goal_ned[2],
            rpy_goal_ned[0], rpy_goal_ned[1], rpy_goal_ned[2]],
            dtype=float,
        )


    def plan_vehicle_trajectory(self):
        self.node.get_logger().info(
            f"Planning and executing motion for {self.robot_selected.prefix} to target pose..."
        )

        pose_now = self._get_robot_pose_now_world()
        start_xyz, start_quat_wxyz = self._pose_to_xyz_quat_wxyz(pose_now)

        goal_xyz, goal_quat_wxyz = self._get_vehicle_goal_from_marker()

        self._log_plan_context(start_xyz, start_quat_wxyz, goal_xyz, goal_quat_wxyz)

        self._save_vehicle_goal_from_target()

        k_planner = self.robot_selected.planner
        try:
            k_planner.planned_result = self._plan_se3(
                start_xyz=start_xyz,
                start_quat_wxyz=start_quat_wxyz,
                goal_xyz=goal_xyz,
                goal_quat_wxyz=goal_quat_wxyz,
            )

            self.node.get_logger().info(k_planner.planned_result.get("message", "Planner finished."))

            if not k_planner.planned_result.get("is_success", False):
                return k_planner.planned_result

            path_xyz = np.asarray(k_planner.planned_result["xyz"], dtype=float)
            self._start_vehicle_cartesian_ruckig(start_xyz, path_xyz)

            self.node.get_logger().info(
                f"{self.robot_selected.prefix} started Ruckig trajectory with {path_xyz.shape[0]} waypoints"
            )
            return k_planner.planned_result

        except Exception as e:
            self.node.get_logger().error(f"Planner failed, {e}")
            k_planner.planned_result = {"is_success": False, "message": "Planner did not find a solution"}
            return k_planner.planned_result
        
    def solve_execute_inverse_kinematics_wrt_vehicle_frame(self, task_pose:Pose):
        msg = {'is_success':False,'result':None}
        if self.is_valid_arm_base_task(task_pose):
            q_ik_sol = self.robot_selected.manipulator_inverse_kinematics(
                np.array([task_pose.position.x, task_pose.position.y, task_pose.position.z]))
            msg['is_success'] = True
            msg['result'] = q_ik_sol
        return msg

    def plan_and_execute_task_trajectory_wrt_vehicle(self):
        msg = self.solve_execute_inverse_kinematics_wrt_vehicle_frame(self.target_arm_base_endeffector_pose)
        if msg['is_success']:
            self.robot_selected.arm.joint_desired = msg['result']

    def plan_and_execute_task_trajectory_wrt_world(self):
        if self.robot_selected.joint_4_in_world is None:
            return
        self.node.get_logger().info(f"{self.robot_selected.joint_4_in_world} robot.", throttle_duration_sec=2.0)
        self.node.get_logger().info(f"{self.target_world_endeffector_pose} target.", throttle_duration_sec=2.0)