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
from visualization_msgs.msg import InteractiveMarkerControl, InteractiveMarkerFeedback
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
import interactive_utils as marker_util
from uvms_backend import UVMSBackend
import rclpy.time
from geometry_msgs.msg import Pose
from tf2_ros import TransformException
from tf2_geometry_msgs import do_transform_pose


class BasicControlsNode(Node):
    def __init__(self):
        super().__init__('uvms_interactive_controls',
                         automatically_declare_parameters_from_overrides=True)

        # FCL for planning, env in world frame
        urdf_string = self.get_parameter('robot_description').get_parameter_value().string_value
        self.vehicle_marker_frame = "vehicle_marker_frame"
        self.task_target_frame = "task_marker_frame"
        # self.backend = UVMSBackend(self, urdf_string, self.vehicle_marker_frame, self.task_target_frame)
 
        # Create marker server, menu handler
        self.server = InteractiveMarkerServer(self, "uvms_interactive_controls")

        self.menu_handler = MenuHandler()
        self.menu_id_to_robot_index = {}

        self.execute_handle = self.menu_handler.insert("Plan & execute", callback=self.plan_execute)
        robot_select_menu_handle = self.menu_handler.insert("Robots")

        # add a menu item for this robot and remember which handle maps to which index
        for robot in self.backend.robots:
            h = self.menu_handler.insert(f"Use {robot.prefix}", parent=robot_select_menu_handle, callback=self.switch_robot_in_use)
            self.menu_id_to_robot_index[h] = robot.k_robot
            self.menu_handler.setCheckState(h, MenuHandler.UNCHECKED)
            if self.backend.robot_selected.k_robot == robot.k_robot:
                self.menu_handler.setCheckState(h, MenuHandler.CHECKED)

        self.control_handle = self.menu_handler.insert('Control space')
        self.task_space_handle = self.menu_handler.insert('Task space', parent=self.control_handle, callback=self.switch_control_Type)
        self.menu_handler.setCheckState(self.task_space_handle, MenuHandler.UNCHECKED)
        self.joint_space_handle = self.menu_handler.insert('Joint space', parent=self.control_handle,callback=self.switch_control_Type)
        self.menu_handler.setCheckState(self.joint_space_handle, MenuHandler.CHECKED)


        task_handle = self.menu_handler.insert('tasks')
        pick_handle = self.menu_handler.insert("Mark pick target", parent=task_handle)
        place_handle = self.menu_handler.insert("Mark place target", parent=task_handle)
        run_pick_place = self.menu_handler.insert("Run pick & place", parent=task_handle)
        # phases: MOVE_TO_PICK_APPROACH ->LOWER_AND_GRASP -> RETRACT -> MOVE_TO_PLACE_APPROACH -> LOWER_AND_RELEASE -> RETRACT.

        # Create markers
        self.uv_marker = marker_util.make_UVMS_Dof_Marker(
            name='uv_marker',
            description='interactive marker for controlling vehicle',
            frame_id=self.backend.world_origin,
            control_frame='uv',
            fixed=False,
            interaction_mode=InteractiveMarkerControl.MOVE_ROTATE_3D,
            initial_pose=Pose(),
            scale=1.0,
            # arm_base_pose=self.backend.arm_base_pose,
            arm_base_pose=Pose(),
            show_6dof=True,
            ignore_dof=['roll','pitch']
        )

        # self.vehicle_task_marker = marker_util.make_UVMS_Dof_Marker(
        #     name='vehicle_task_marker',
        #     description='interactive marker for controlling endeffector',
        #     frame_id=self.vehicle_marker_frame,
        #     control_frame='vehicle_task',
        #     fixed=False,
        #     interaction_mode=InteractiveMarkerControl.MOVE_ROTATE_3D,
        #     initial_pose=self.backend.current_task_pose,
        #     scale=0.2,
        #     arm_base_pose=None,
        #     show_6dof=True,
        #     ignore_dof=[]
        # )

        # self.world_task_marker = marker_util.make_UVMS_Dof_Marker(
        #     name='world_task_marker',
        #     description='interactive marker for controlling endeffector',
        #     frame_id=self.backend.world_origin,
        #     control_frame='world_task',
        #     fixed=False,
        #     interaction_mode=InteractiveMarkerControl.MOVE_ROTATE_3D,
        #     initial_pose=self.backend.current_task_pose,
        #     scale=0.2,
        #     arm_base_pose=None,
        #     show_6dof=True,
        #     ignore_dof=[]
        # )

        self.server.insert(self.uv_marker)
        self.server.setCallback(self.uv_marker.name, self.processFeedback)

        # self.server.insert(self.vehicle_task_marker)
        # self.server.setCallback(self.vehicle_task_marker.name, self.processFeedback)

        menu_control = marker_util.make_menu_control()
        self.uv_marker.controls.append(copy.deepcopy(menu_control))

        self.menu_handler.apply(self.server, self.uv_marker.name)
        # self.menu_handler.apply(self.server, self.vehicle_task_marker.name)
        self.server.applyChanges()

    def switch_robot_in_use(self, feedback: InteractiveMarkerFeedback):
        if feedback.menu_entry_id in self.menu_id_to_robot_index:
            selected_robot_k = self.menu_id_to_robot_index[feedback.menu_entry_id]

            # uncheck all other robot menu entries
            for mid in self.menu_id_to_robot_index.keys():
                if mid != feedback.menu_entry_id:
                    self.menu_handler.setCheckState(mid, MenuHandler.UNCHECKED)

            # check the one that was clicked
            self.menu_handler.setCheckState(feedback.menu_entry_id, MenuHandler.CHECKED)

            self.menu_handler.apply(self.server, feedback.marker_name)
            self.server.applyChanges()

            self.backend.set_robot_selected(selected_robot_k)

    def switch_control_Type(self, feedback: InteractiveMarkerFeedback):
        if feedback.menu_entry_id == self.task_space_handle:
            # Switch to task space control
            self.current_space = "task"

            # Take the current end effector pose in vehicle_marker_frame
            pose_in_vehicle = copy.deepcopy(self.vehicle_task_marker.pose)

            try:
                # Transform vehicle_task pose into base_frame
                transform = self.backend.tf_buffer.lookup_transform(
                    self.backend.base_frame,         # target frame
                    self.vehicle_marker_frame,       # source frame
                    rclpy.time.Time())              # latest available
                pose_in_base = do_transform_pose(pose_in_vehicle, transform)
                self.world_task_marker.pose = pose_in_base
            except TransformException as ex:
                self.get_logger().warn(
                    f"Could not transform vehicle_task pose to base frame, {ex}"
                )
                return

            # Show task marker, hide uv marker
            self.backend.solve_whole_body_inverse_kinematics_wrt_world_frame(self.world_task_marker.pose)
            self.server.insert(self.world_task_marker)
            self.server.setCallback(self.world_task_marker.name, self.processFeedback)

            self.server.erase(self.uv_marker.name)
            self.server.erase(self.vehicle_task_marker.name)

            self.menu_handler.setCheckState(self.task_space_handle, MenuHandler.CHECKED)
            self.menu_handler.setCheckState(self.joint_space_handle, MenuHandler.UNCHECKED)

            self.menu_handler.apply(self.server, self.world_task_marker.name)
            self.server.applyChanges()
            self.get_logger().info("Switched control to task space")

        elif feedback.menu_entry_id == self.joint_space_handle:
            # Switch to joint or vehicle space control
            self.current_space = "joint"
            
            self.server.insert(self.uv_marker)
            self.server.setCallback(self.uv_marker.name, self.processFeedback)

            self.server.insert(self.vehicle_task_marker)
            self.server.setCallback(self.vehicle_task_marker.name, self.processFeedback)

            self.server.erase(self.world_task_marker.name)

            self.menu_handler.setCheckState(self.task_space_handle, MenuHandler.UNCHECKED)
            self.menu_handler.setCheckState(self.joint_space_handle, MenuHandler.CHECKED)

            self.menu_handler.apply(self.server, self.uv_marker.name)
            self.menu_handler.apply(self.server, self.vehicle_task_marker.name)
            self.server.applyChanges()
            self.get_logger().info("Switched control to joint space")

    def plan_execute(self, feedback: InteractiveMarkerFeedback):
        if self.backend.robot_selected is None or self.backend.target_vehicle_pose is None:
            self.get_logger().warn("Execute clicked but robot selection or planned pose is missing.")
            return
        self.backend.plan_vehicle_trajectory()


    def processFeedback(self, feedback: InteractiveMarkerFeedback):
        env_xyz_bounds = self.backend.fcl_world._compute_env_bounds_from_fcl(z_min=self.backend.bottom_z, pad_xy=0.0, pad_z=0.0)
        x_min, x_max, y_min, y_max, z_min, z_max = env_xyz_bounds
        mins = np.array([x_min, y_min, z_min], dtype=float)
        maxs = np.array([x_max, y_max, z_max], dtype=float)
        # For uv_marker
        if feedback.marker_name == self.uv_marker.name:
            if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
                pos = feedback.pose.position

                xyz = np.array([pos.x, pos.y, pos.z], dtype=float)
                clipped_xyz = np.clip(xyz, mins, maxs)

                if not np.array_equal(xyz, clipped_xyz):
                    pos.x, pos.y, pos.z = clipped_xyz.tolist()
                    self.server.setPose(feedback.marker_name, feedback.pose)
                    self.server.applyChanges()

                self.backend.target_vehicle_pose = feedback.pose


        elif feedback.marker_name == self.vehicle_task_marker.name:
            if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
                if self.backend.is_valid_task(feedback.pose):
                    self.backend.target_task_pose_wrt_vehicle = feedback.pose
                    # self.backend.solve_execute_inverse_kinematics_wrt_vehicle_frame(feedback.pose)
                    self.backend.plan_task_trajectory_wrt_vehicle()
                else:
                    # The task marker is at the boundary; compute the displacement since the last valid pose.
                    dx = feedback.pose.position.x - self.backend.target_task_pose_wrt_vehicle.position.x
                    dy = feedback.pose.position.y - self.backend.target_task_pose_wrt_vehicle.position.y
                    dz = feedback.pose.position.z - self.backend.target_task_pose_wrt_vehicle.position.z

                    # Shift the uv_marker by this delta so that the task marker remains at the boundary.
                    self.backend.target_vehicle_pose.position.x += dx
                    self.backend.target_vehicle_pose.position.y += dy
                    self.backend.target_vehicle_pose.position.z += dz

                    # Update the uv_marker pose on the server.
                    self.server.setPose(self.uv_marker.name, self.backend.target_vehicle_pose)
                    self.server.applyChanges()

                    # Reset the task marker back to the last valid pose (i.e. at the boundary).
                    self.server.setPose(self.vehicle_task_marker.name, self.backend.target_task_pose_wrt_vehicle)
                    self.server.applyChanges()

        elif feedback.marker_name == self.world_task_marker.name:
            if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
                pos = feedback.pose.position
                xyz = np.array([pos.x, pos.y, pos.z], dtype=float)

                clipped_xyz = np.clip(xyz, mins, maxs)

                if not np.array_equal(xyz, clipped_xyz):
                    pos.x, pos.y, pos.z = clipped_xyz.tolist()
                    self.server.setPose(feedback.marker_name, feedback.pose)
                    self.server.applyChanges()


            self.backend.solve_whole_body_inverse_kinematics_wrt_world_frame(feedback.pose)


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