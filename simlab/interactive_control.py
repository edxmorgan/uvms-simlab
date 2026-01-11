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
import rclpy
from rclpy.node import Node
from uvms_backend import UVMSBackendCore
from visualization_msgs.msg import InteractiveMarkerControl, InteractiveMarkerFeedback
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
import interactive_utils as marker_util
from geometry_msgs.msg import Pose
import numpy as np
from alpha_reach import Params as alpha
from frame_utils import PoseX

class InteractiveControlsNode(Node):
    def __init__(self):
        super().__init__('uvms_interactive_controls',
                         automatically_declare_parameters_from_overrides=True)
        urdf_string = self.get_parameter('robot_description').get_parameter_value().string_value
        self.world_frame = "world"
        self.vehicle_target_frame = "vehicle_marker_frame"
        self.arm_base_target_frame = "arm_base_marker_frame"
        self.world_endeffector_target_frame = "world_endeffector_marker_frame"

        # Arm base pose wrt vehicle center
        self.arm_base_wrt_vehicle_center_Pose = PoseX.from_pose(
                xyz=alpha.base_T0_new[0:3],
                rot=alpha.base_T0_new[3:6],
                rot_rep="euler_xyz",
                frame="NWU").get_pose_as_Pose_msg()
        
        self.uvms_backend: UVMSBackendCore = UVMSBackendCore(self, urdf_string,
                                                              self.arm_base_wrt_vehicle_center_Pose,
                                                              self.vehicle_target_frame, self.arm_base_target_frame, 
                                                              self.world_frame,
                                                              self.world_endeffector_target_frame, alpha)

        # Create marker server, menu handler
        self.server = InteractiveMarkerServer(self, "uvms_interactive_controls")

        self.menu_handler = MenuHandler()
        self.menu_id_to_robot_index = {}

        self.execute_handle = self.menu_handler.insert("Plan & Execute", callback=self.plan_execute)
        robot_select_menu_handle = self.menu_handler.insert("Robots")

        # add a menu item for this robot and remember which handle maps to which index
        for robot in self.uvms_backend.robots:
            h = self.menu_handler.insert(f"Use {robot.prefix}", parent=robot_select_menu_handle, callback=self.switch_robot_in_use)
            self.menu_id_to_robot_index[h] = robot.k_robot
            self.menu_handler.setCheckState(h, MenuHandler.UNCHECKED)
            if self.uvms_backend.robot_selected.k_robot == robot.k_robot:
                self.menu_handler.setCheckState(h, MenuHandler.CHECKED)

        self.control_handle = self.menu_handler.insert('Control Space')
        self.task_space_handle = self.menu_handler.insert('Task Space', parent=self.control_handle)
        self.menu_handler.setCheckState(self.task_space_handle, MenuHandler.UNCHECKED)

        self.x_axis_align_target_task_space_handle = self.menu_handler.insert(
            'x-axis align',
            parent=self.task_space_handle,
            callback=self.switch_control_Type
        )
        self.y_axis_align_target_task_space_handle = self.menu_handler.insert(
            'y-axis align',
            parent=self.task_space_handle,
            callback=self.switch_control_Type
        )
        self.z_axis_align_target_task_space_handle = self.menu_handler.insert(
            'z-axis align',
            parent=self.task_space_handle,
            callback=self.switch_control_Type
        )
        self.align_tool_axis_handle = self.menu_handler.insert(
            'align arm with base',
            parent=self.task_space_handle,
            callback=self.toggle_align_weight,
        )
        self.axis_menu_map = {}
        self.axis_menu_map[self.x_axis_align_target_task_space_handle] = np.array([1, 0, 0], dtype=int)
        self.axis_menu_map[self.y_axis_align_target_task_space_handle] = np.array([0, 1, 0], dtype=int)
        self.axis_menu_map[self.z_axis_align_target_task_space_handle] = np.array([0, 0, 1], dtype=int)
        self.menu_handler.setCheckState(self.x_axis_align_target_task_space_handle, MenuHandler.UNCHECKED)
        self.menu_handler.setCheckState(self.y_axis_align_target_task_space_handle, MenuHandler.UNCHECKED)
        self.menu_handler.setCheckState(self.z_axis_align_target_task_space_handle, MenuHandler.UNCHECKED)
        self.menu_handler.setCheckState(self.align_tool_axis_handle, MenuHandler.UNCHECKED)
        self.align_tool_axis_enabled = False


        self.joint_space_handle = self.menu_handler.insert('Joint Space', parent=self.control_handle,callback=self.switch_control_Type)
        self.menu_handler.setCheckState(self.joint_space_handle, MenuHandler.CHECKED)

        grasper_handle = self.menu_handler.insert('Grasper')
        self.open_grasper_handle = self.menu_handler.insert('Open', parent=grasper_handle, callback=self.grasper_callback)
        self.close_grasper_handle = self.menu_handler.insert("Close", parent=grasper_handle, callback=self.grasper_callback)


        task_handle = self.menu_handler.insert('Tasks')
        pick_place = self.menu_handler.insert('Pick & Place', parent=task_handle)
        pick_handle = self.menu_handler.insert("Pick target", parent=pick_place)
        place_handle = self.menu_handler.insert("Place target", parent=pick_place)
        run_pick_place = self.menu_handler.insert("Run", parent=pick_place)
        # phases: MOVE_TO_PICK_APPROACH ->LOWER_AND_GRASP -> RETRACT -> MOVE_TO_PLACE_APPROACH -> LOWER_AND_RELEASE -> RETRACT.
        
        # Create markers
        self.uv_marker = marker_util.make_UVMS_Dof_Marker(
            name='uv_marker',
            description='interactive marker for controlling vehicle',
            frame_id=self.uvms_backend.world_frame,
            control_frame='uv',
            fixed=False,
            interaction_mode=InteractiveMarkerControl.MOVE_ROTATE_3D,
            initial_pose=Pose(),
            scale=1.0,
            arm_base_pose=self.arm_base_wrt_vehicle_center_Pose,
            show_6dof=True,
            ignore_dof=['roll','pitch']
        )

        self.arm_base_task_marker = marker_util.make_UVMS_Dof_Marker(
            name='arm_base_task_marker',
            description='interactive marker for controlling endeffector wrt arm base mounted on vehicle',
            frame_id=self.arm_base_target_frame,
            control_frame='arm_base_task',
            fixed=False,
            interaction_mode=InteractiveMarkerControl.MOVE_ROTATE_3D,
            initial_pose=self.uvms_backend.target_arm_base_endeffector_pose, #change to appropriate initial pose
            scale=0.2,
            show_6dof=True,
            ignore_dof=[]
        )
        # Create menu control
        self.menu_control = marker_util.make_menu_control()
        # Start in joint control
        self._apply_joint_control_mode()

    def grasper_callback(self, feedback: InteractiveMarkerFeedback):
        if feedback.menu_entry_id == self.open_grasper_handle:
            self.uvms_backend.open_grasper()
        elif feedback.menu_entry_id == self.close_grasper_handle:
            self.uvms_backend.close_grasper()
    
    def plan_execute(self, feedback: InteractiveMarkerFeedback):
        if self.uvms_backend.task_based_controller:
            pass
        else:
            self.uvms_backend.plan_vehicle_trajectory()


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

            self.uvms_backend.set_robot_selected(selected_robot_k)

    def _clear_markers(self) -> None:
        # erase by name, not by object
        try:
            self.server.erase(self.uv_marker.name)
        except Exception:
            pass
        try:
            self.server.erase(self.arm_base_task_marker.name)
        except Exception:
            pass
        self.server.applyChanges()

    def _apply_joint_control_mode(self) -> None:
        self._clear_markers()

        # uv marker (vehicle)
        self.server.insert(self.uv_marker)
        self.server.setCallback(self.uv_marker.name, self.vehicle_marker_processFeedback)

        # endeffector marker (arm base frame)
        self.server.insert(self.arm_base_task_marker)
        self.server.setCallback(self.arm_base_task_marker.name, self.arm_base_task_marker_processFeedback)

        # attach menu control to the marker before applyChanges
        self.uv_marker.controls.append(copy.deepcopy(self.menu_control))
        self.server.insert(self.uv_marker)  # re-insert so server sees updated controls

        self.menu_handler.apply(self.server, self.uv_marker.name)
        self.menu_handler.apply(self.server, self.arm_base_task_marker.name)
        self.server.applyChanges()

    def _apply_task_control_mode(self) -> None:
        self._clear_markers()

        self.server.insert(self.arm_base_task_marker)
        self.server.setCallback(self.arm_base_task_marker.name, self.world_task_marker_processFeedback)

        self.arm_base_task_marker.controls.append(copy.deepcopy(self.menu_control))
        self.server.insert(self.arm_base_task_marker)  # re-insert to update controls

        self.menu_handler.apply(self.server, self.arm_base_task_marker.name)
        self.server.applyChanges()

    def switch_control_Type(self, feedback: InteractiveMarkerFeedback):
        self.menu_handler.setCheckState(self.task_space_handle, MenuHandler.UNCHECKED)
        self.menu_handler.setCheckState(self.joint_space_handle, MenuHandler.UNCHECKED)
        for mid in self.axis_menu_map.keys():
            self.menu_handler.setCheckState(mid, MenuHandler.UNCHECKED)

        if feedback.menu_entry_id in self.axis_menu_map:
            self.menu_handler.setCheckState(feedback.menu_entry_id, MenuHandler.CHECKED)
            self.menu_handler.apply(self.server, feedback.marker_name)
            self.server.applyChanges()

            axis = self.axis_menu_map.get(feedback.menu_entry_id)
            self.uvms_backend.tool_axis = axis
            self.active_axis_handle = feedback.menu_entry_id
            self.get_logger().info(f"Tool axis align set to {axis.tolist()}")
    
            self.menu_handler.setCheckState(self.task_space_handle, MenuHandler.CHECKED)
            self._apply_task_control_mode()
            self.uvms_backend.task_based_controller = True
            self.get_logger().info("Switched to TASK space control.")

        elif feedback.menu_entry_id == self.joint_space_handle:
            self.menu_handler.setCheckState(self.joint_space_handle, MenuHandler.CHECKED)
            self._apply_joint_control_mode()
            self.uvms_backend.task_based_controller = False
            self.get_logger().info("Switched to JOINT space control.")

        self.menu_handler.apply(self.server, feedback.marker_name)
        self.server.applyChanges()

    def toggle_align_weight(self, feedback: InteractiveMarkerFeedback):
        self.align_tool_axis_enabled = not self.align_tool_axis_enabled
        state = MenuHandler.CHECKED if self.align_tool_axis_enabled else MenuHandler.UNCHECKED
        self.menu_handler.setCheckState(self.align_tool_axis_handle, state)
        self.uvms_backend.w_align = 1.0 if self.align_tool_axis_enabled else 0.0
        self.menu_handler.apply(self.server, feedback.marker_name)
        self.server.applyChanges()
        self.get_logger().info(f"Align arm with base weight set to {self.uvms_backend.w_align:.1f}")
    
    def vehicle_marker_processFeedback(self, feedback: InteractiveMarkerFeedback):
        pos = feedback.pose.position
        clipped_xyz = self.uvms_backend.fcl_world.enforce_bounds([pos.x, pos.y, pos.z])
        pos.x = clipped_xyz[0]
        pos.y = clipped_xyz[1]
        pos.z = clipped_xyz[2]

        self.server.setPose(feedback.marker_name, feedback.pose)
        self.server.applyChanges()
        self.get_logger().debug("Clipped uv_marker position to stay within environment bounds.")
        self.uvms_backend.target_vehicle_pose = feedback.pose
        self.set_endeffector_world_marker_pose(self.uvms_backend.target_arm_base_endeffector_pose, self.uvms_backend.arm_base_target_frame)

    def set_endeffector_world_marker_pose(self, new_pose: Pose, source_frame: str) -> None:
        if source_frame == self.uvms_backend.world_frame:
            self.uvms_backend.target_world_endeffector_pose = new_pose
            return

        task_pose_world_new = self.uvms_backend.robot_selected.try_transform_pose(
            new_pose,
            target_frame=self.uvms_backend.world_frame,
            source_frame=source_frame,
            warn_context="set_endeffector_world_marker_pose",
        )
        self.uvms_backend.target_world_endeffector_pose = task_pose_world_new

    def arm_base_task_marker_processFeedback(self, feedback: InteractiveMarkerFeedback):
        if self.uvms_backend.is_valid_arm_base_task(feedback.pose):
            self.uvms_backend.target_arm_base_endeffector_pose = feedback.pose
            self.set_endeffector_world_marker_pose(self.uvms_backend.target_arm_base_endeffector_pose, self.uvms_backend.arm_base_target_frame)
            self.get_logger().debug("Updated arm base task marker pose.")
            return

        self.server.setPose(self.arm_base_task_marker.name, self.uvms_backend.target_arm_base_endeffector_pose)
        self.server.applyChanges()


    def world_task_marker_processFeedback(self, feedback: InteractiveMarkerFeedback):
        pose_world = self.uvms_backend.robot_selected.try_transform_pose(
            feedback.pose,
            target_frame=self.uvms_backend.world_frame,
            source_frame=self.uvms_backend.arm_base_target_frame,
            warn_context="world_task_marker_processFeedback,to_world",
        )

        p = pose_world.position
        p.x, p.y, p.z = self.uvms_backend.fcl_world.enforce_bounds([p.x, p.y, p.z])

        pose_arm = self.uvms_backend.robot_selected.try_transform_pose(
            pose_world,
            target_frame=self.uvms_backend.arm_base_target_frame,
            source_frame=self.uvms_backend.world_frame,
            warn_context="world_task_marker_processFeedback,to_arm_base",
        )

        self.server.setPose(feedback.marker_name, pose_arm)
        self.server.applyChanges()

        # no duplicate transform now, we already have pose_world
        self.set_endeffector_world_marker_pose(pose_world, self.uvms_backend.world_frame)



def main(args=None):
    rclpy.init(args=args)
    node = InteractiveControlsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
