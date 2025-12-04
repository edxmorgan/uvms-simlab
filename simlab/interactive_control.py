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
import numpy as np
np.float = float  # Patch NumPy to satisfy tf_transformations' use of np.float

import copy
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import casadi as ca

from visualization_msgs.msg import Marker, InteractiveMarkerControl, InteractiveMarkerFeedback
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from robot import Robot
import tf2_ros

from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2

from sensor_msgs.msg import PointCloud2


from alpha_reach import Params as alpha
from se3_ompl_planner import plan_se3_path

from ruckig import Result

from interactive_utils import *

from frame_utils import PoseX


from uvms_backend import UVMSBackend

class BasicControlsNode(Node):
    def __init__(self):
        super().__init__('uvms_interactive_controls',
                         automatically_declare_parameters_from_overrides=True)

        # FCL for planning, env in world frame
        urdf_string = self.get_parameter('robot_description').get_parameter_value().string_value


        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)


        self.backend = UVMSBackend(self, self.tf_buffer, urdf_string)
        self.backend.robot_selected = self.backend.robots[0]

        pointcloud_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.taskspace_pc_publisher_ = self.create_publisher(PointCloud2,'workspace_pointcloud',pointcloud_qos)
        self.rov_pc_publisher_ = self.create_publisher(PointCloud2, 'base_pointcloud', pointcloud_qos)        

        # Create marker server, menu handler
        self.server = InteractiveMarkerServer(self, "uvms_interactive_controls")

        self.menu_handler = MenuHandler()
        self.menu_id_to_robot_index = {}

        self.execute_handle = self.menu_handler.insert("Plan & execute", callback=self.processFeedback)
        robot_select_menu_handle = self.menu_handler.insert("Robots")

        # add a menu item for this robot and remember which handle maps to which index
        for robot in self.backend.robots:
            h = self.menu_handler.insert(f"Use {robot.prefix}", parent=robot_select_menu_handle, callback=self.processFeedback)
            self.menu_id_to_robot_index[h] = robot.k_robot

        pick_handle = self.menu_handler.insert("Mark pick target")
        place_handle = self.menu_handler.insert("Mark place target")
        run_pick_place = self.menu_handler.insert("Run pick & place")
        # phases: MOVE_TO_PICK_APPROACH ->LOWER_AND_GRASP -> RETRACT -> MOVE_TO_PLACE_APPROACH -> LOWER_AND_RELEASE -> RETRACT.

        self.vehicle_marker_frame = "vehicle_marker_frame"
        self.endeffector_marker_frame = "endeffector_marker_frame"

        # Create markers
        self.uv_marker = make_UVMS_Dof_Marker(
            name='uv_marker',
            description='interactive marker for controlling vehicle',
            frame_id=self.backend.base_frame,
            control_frame='uv',
            fixed=False,
            interaction_mode=InteractiveMarkerControl.MOVE_ROTATE_3D,
            initial_pose=self.backend.current_target_vehicle_marker_pose,
            scale=1.0,
            arm_base_pose=self.backend.arm_base_pose,
            show_6dof=True,
            ignore_dof=['roll','pitch']
        )

        self.server.insert(self.uv_marker)
        self.server.setCallback(self.uv_marker.name, self.processFeedback)


        self.task_marker = make_UVMS_Dof_Marker(
            name='task_marker',
            description='interactive marker for controlling endeffector',
            frame_id=self.vehicle_marker_frame,
            control_frame='task',
            fixed=False,
            interaction_mode=InteractiveMarkerControl.MOVE_ROTATE_3D,
            initial_pose=self.backend.current_target_task_pose,
            scale=0.2,
            arm_base_pose=None,
            show_6dof=True,
            ignore_dof=[]
        )

        self.server.insert(self.task_marker)
        self.server.setCallback(self.task_marker.name, self.processFeedback)

        # Add menu control
        menu_control = make_menu_control()
        self.uv_marker.controls.append(copy.deepcopy(menu_control))
        self.menu_handler.apply(self.server, self.uv_marker.name)
        self.server.applyChanges()

        self.control_frequency = 500.0  # Hz
        self.control_timer = self.create_timer(1.0 / self.control_frequency, self.marker_tf_timer_callback)
        self.cloud_frequency = 100.0     # Hz
        self.cloud_timer = self.create_timer(1.0 / self.cloud_frequency, self.cloud_timer_callback)

    def marker_tf_timer_callback(self):
        stamp_now = self.get_clock().now().to_msg()
        t = get_broadcast_tf(stamp_now, self.backend.current_target_vehicle_marker_pose, self.backend.base_frame, self.vehicle_marker_frame)
        self.tf_broadcaster.sendTransform(t)

    def cloud_timer_callback(self):
        header = Header()
        header.frame_id = self.vehicle_marker_frame
        header.stamp = self.get_clock().now().to_msg()

        rov_cloud_msg = pc2.create_cloud_xyz32(header, self.backend.workspace_pts)
        self.taskspace_pc_publisher_.publish(rov_cloud_msg)

        cloud_msg = pc2.create_cloud_xyz32(header, self.backend.rov_ellipsoid_cl_pts)
        self.rov_pc_publisher_.publish(cloud_msg)
        
    def processFeedback(self, feedback):
        # For uv_marker
        if feedback.marker_name == "uv_marker":
            if feedback.pose:
                env_xyz_bounds = self.backend.fcl_world._compute_env_bounds_from_fcl(z_min=self.backend.bottom_z, pad_xy=0.0, pad_z=0.0)
                x_min, x_max, y_min, y_max, z_min, z_max = env_xyz_bounds
                pos = feedback.pose.position

                xyz = np.array([pos.x, pos.y, pos.z], dtype=float)
                mins = np.array([x_min, y_min, z_min], dtype=float)
                maxs = np.array([x_max, y_max, z_max], dtype=float)

                clipped_xyz = np.clip(xyz, mins, maxs)

                if not np.array_equal(xyz, clipped_xyz):
                    pos.x, pos.y, pos.z = clipped_xyz.tolist()
                    self.server.setPose(feedback.marker_name, feedback.pose)
                    self.server.applyChanges()

                self.backend.current_target_vehicle_marker_pose = feedback.pose
            if feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
                if feedback.menu_entry_id == self.execute_handle:
                    if self.backend.robot_selected is None or self.backend.current_target_vehicle_marker_pose is None:
                        self.get_logger().warn("Execute clicked but robot selection or planned pose is missing.")
                        return
                    planner_result = self.backend.plan_vehicle_trajectory()
                else:
                    # otherwise, a robot menu item was clicked
                    if feedback.menu_entry_id in self.menu_id_to_robot_index:
                        selected_robot_index = self.menu_id_to_robot_index[feedback.menu_entry_id]
                        self.backend.set_robot_selected(self.robots[selected_robot_index])
                        self.get_logger().info(f"Robot {self.robots_prefix[selected_robot_index]} selected for planning.")

            elif feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
                pass

        elif feedback.marker_name == "task_marker" and feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:            
            ik_msg = self.backend.solve_inverse_kinematics_wrt_vehicle_frame(feedback.pose)
            if ik_msg['is_success']:
                self.backend.current_target_task_pose = feedback.pose
                [self.backend.q0_des, self.backend.q1_des, self.backend.q2_des, _] = ik_msg['result']
                
                self.get_logger().debug(
                    f"Task marker updated with IK: {self.backend.q0_des, self.backend.q1_des, self.backend.q2_des, self.backend.q3_des}"
                )
            else:
                # The task marker is at the boundary; compute the displacement since the last valid pose.
                dx = feedback.pose.position.x - self.backend.current_target_task_pose.position.x
                dy = feedback.pose.position.y - self.backend.current_target_task_pose.position.y
                dz = feedback.pose.position.z - self.backend.current_target_task_pose.position.z

                # Shift the uv_marker by this delta so that the task marker remains at the boundary.
                self.backend.current_target_vehicle_marker_pose.position.x += dx
                self.backend.current_target_vehicle_marker_pose.position.y += dy
                self.backend.current_target_vehicle_marker_pose.position.z += dz

                # Update the uv_marker pose on the server.
                self.server.setPose("uv_marker", self.backend.current_target_vehicle_marker_pose)
                self.server.applyChanges()

                # Reset the task marker back to the last valid pose (i.e. at the boundary).
                self.server.setPose("task_marker", self.backend.current_target_task_pose)
                self.server.applyChanges()


def main(args=None):
    rclpy.init(args=args)
    node = BasicControlsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
