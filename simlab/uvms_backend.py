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
from robot import Robot, ControlMode
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from planner_markers import PathPlanner
from cartesian_ruckig import VehicleCartesianRuckig
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

            robot_k = Robot(self.node, self.tf_buffer, k, 4, prefix, planner, vehicle_cart_traj)
            robot_k.set_control_mode(ControlMode.PLANNER)


            self.robots.append(robot_k)

        self.robot_selected = self.robots[0]

        self.initialise_target_Poses()


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
        self.target_arm_base_endeffector_pose = Pose()
        self.target_arm_base_endeffector_pose.position.x = alpha_params.endeffector_wrt_base_home[0]
        self.target_arm_base_endeffector_pose.position.y = alpha_params.endeffector_wrt_base_home[1]
        self.target_arm_base_endeffector_pose.position.z = alpha_params.endeffector_wrt_base_home[2]
        self.target_world_endeffector_pose = Pose()

    def plan_vehicle_trajectory(self):
        goal_pose = self.target_vehicle_pose
        self.robot_selected.plan_vehicle_trajectory_action(
            goal_pose=goal_pose,
            time_limit=1.0,
            robot_collision_radius=float(self.fcl_world.vehicle_radius),
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
        if not self.robot_selected.task_based_controller:
            msg = self.solve_execute_inverse_kinematics_wrt_vehicle_frame(self.target_arm_base_endeffector_pose)
            if msg['is_success']:
                self.robot_selected.arm.q_command = msg['result']

    def plan_task_trajectory(self):
        if self.robot_selected.task_based_controller and self.robot_selected.task_pose_in_world:
            self.robot_selected.solve_inverse_kinematics_wrt_world_frame(self.target_world_endeffector_pose)
            self.node.get_logger().info(f"task trajectory plan & control", throttle_duration_sec=2.0)
            self.node.get_logger().info(f"{self.robot_selected.task_pose_in_world} robot.", throttle_duration_sec=2.0)
            self.node.get_logger().info(f"{self.target_world_endeffector_pose} target.", throttle_duration_sec=2.0)
            self.node.get_logger().info(f"{self.target_arm_base_endeffector_pose} base target.", throttle_duration_sec=2.0)
        return
