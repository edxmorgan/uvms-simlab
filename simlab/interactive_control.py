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
from robot import Robot, ControlSpace

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

        self.execute_handle = self.menu_handler.insert("Plan & Execute", callback=self.plan_execute)

        self.control_space_menu_map = {}  # mid -> (k_robot, control_space_name)
        self.axis_menu_map = {}
        self.grasp_menu_map = {}
        self.controller_menu_map = {}
        self.planner_menu_map = {}
        self.robot_menu_parents = {}  # k_robot -> dict of submenu parent handles
        robot_select_menu_handle = self.menu_handler.insert("Robots")

        for robot in self.uvms_backend.robots:
            robot_handle = self.menu_handler.insert(
                f"{robot.prefix}",
                parent=robot_select_menu_handle,
                callback=self.switch_robot_in_use,
            )
            robot.user_id = robot_handle

            self.menu_handler.setCheckState(robot_handle,
                                             MenuHandler.CHECKED if self.uvms_backend.robot_selected.k_robot == robot.k_robot else MenuHandler.UNCHECKED)

            # control space submenu per robot
            cs_parent = self.menu_handler.insert("Control Space")
            for cs_name in robot.list_control_spaces():
                mid = self.menu_handler.insert(
                    cs_name,
                    parent=cs_parent,
                    callback=self.switch_control_space_type,
                )
                self.control_space_menu_map[mid] = (robot, cs_name)
                self.menu_handler.setCheckState(mid, MenuHandler.CHECKED if robot.control_space == cs_name else MenuHandler.UNCHECKED)
                
            controller_parent = self.menu_handler.insert('Controller')

            for controller_name in robot.list_controllers():
                controller_handle = self.menu_handler.insert(
                    f"{controller_name}",
                    parent=controller_parent,
                    callback=self.switch_controller_type,
                )
                self.controller_menu_map[controller_handle] = (robot, controller_name)
                self.menu_handler.setCheckState(controller_handle,
                                                 MenuHandler.CHECKED if robot.controller_name == controller_name else MenuHandler.UNCHECKED)
                
            path_planner_parent = self.menu_handler.insert('Path Planner')
            for path_planner_name in robot.list_planners():
                path_planner_handle = self.menu_handler.insert(
                    f"{path_planner_name}",
                    parent=path_planner_parent,
                    callback=self.switch_planner_type,
                )
                self.planner_menu_map[path_planner_handle] = (robot, path_planner_name)
                self.menu_handler.setCheckState(path_planner_handle,
                                                 MenuHandler.CHECKED if robot.planner_name == path_planner_name else MenuHandler.UNCHECKED)
                
            ik_settings_parent = self.menu_handler.insert("IK Settings")
            x_axis_align_target_task_space_handle = self.menu_handler.insert(
                'x-axis align',
                parent=ik_settings_parent,
                callback=self.toggle_endeffector_axis_align
            )
            y_axis_align_target_task_space_handle = self.menu_handler.insert(
                'y-axis align',
                parent=ik_settings_parent,
                callback=self.toggle_endeffector_axis_align
            )
            z_axis_align_target_task_space_handle = self.menu_handler.insert(
                'z-axis align',
                parent=ik_settings_parent,
                callback=self.toggle_endeffector_axis_align
            )
            align_tool_axis_handle = self.menu_handler.insert(
                'align arm with base',
                parent=ik_settings_parent,
                callback=self.toggle_align_with_base_weight,
            )
            
            self.axis_menu_map[x_axis_align_target_task_space_handle] = (robot, np.array([1, 0, 0], dtype=int))
            self.axis_menu_map[y_axis_align_target_task_space_handle] = (robot, np.array([0, 1, 0], dtype=int))
            self.axis_menu_map[z_axis_align_target_task_space_handle] = (robot, np.array([0, 0, 1], dtype=int))
            self.axis_menu_map[align_tool_axis_handle] = (robot, None)

            self.menu_handler.setCheckState(x_axis_align_target_task_space_handle, MenuHandler.UNCHECKED)
            self.menu_handler.setCheckState(y_axis_align_target_task_space_handle, MenuHandler.UNCHECKED)
            self.menu_handler.setCheckState(z_axis_align_target_task_space_handle, MenuHandler.CHECKED)
            _, robot.ik_tool_axis = self.axis_menu_map.get(z_axis_align_target_task_space_handle)
            
            self.menu_handler.setCheckState(align_tool_axis_handle, MenuHandler.CHECKED)
            robot.ik_base_align_w = 1

            grasper_parent = self.menu_handler.insert('Grasper')
            self.open_grasper_handle = self.menu_handler.insert('Open', parent=grasper_parent, callback=self.grasper_callback)
            self.close_grasper_handle = self.menu_handler.insert('Close', parent=grasper_parent, callback=self.grasper_callback)
            self.grasp_menu_map[self.open_grasper_handle] = (robot, 'open')
            self.grasp_menu_map[self.close_grasper_handle] = (robot, 'close')
            self.menu_handler.setCheckState(self.open_grasper_handle, MenuHandler.UNCHECKED)
            self.menu_handler.setCheckState(self.close_grasper_handle, MenuHandler.CHECKED)

            # remember parents so visibility can be toggled per robot
            self.robot_menu_parents[robot.k_robot] = {
                "cs": cs_parent,
                "controller": controller_parent,
                "ik": ik_settings_parent,
                "grasper": grasper_parent,
                "planner": path_planner_parent,
            }

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
            initial_pose=self.uvms_backend.target_arm_base_endeffector_pose,
            scale=0.2,
            show_6dof=True,
            ignore_dof=[]
        )
        # Initial application of menu and markers
        self._apply_joint_control_mode(robot = self.uvms_backend.robot_selected)
        self._set_robot_submenus_visible(self.uvms_backend.robot_selected.k_robot)


    def plan_execute(self, feedback: InteractiveMarkerFeedback):
        if self.uvms_backend.robot_selected.task_based_controller:
            self.uvms_backend.plan_task_trajectory()
            return
        self.uvms_backend.plan_vehicle_trajectory()

    def _set_robot_submenus_visible(self, selected_k_robot: int) -> None:
        for r in self.uvms_backend.robots:
            visible = (r.k_robot == selected_k_robot)
            parents = self.robot_menu_parents.get(r.k_robot, {})
            for h in parents.values():
                self.menu_handler.setVisible(h, visible)

        # push updated menu state to all markers that have menus
        self.menu_handler.reApply(self.server)
        self.server.applyChanges()


    # switch to robot i
    def switch_robot_in_use(self, feedback: InteractiveMarkerFeedback):
        for r in self.uvms_backend.robots:
            self.menu_handler.setCheckState(r.user_id, MenuHandler.UNCHECKED)
            if r.user_id == feedback.menu_entry_id:
                selected_robot = r

        # check the one that was clicked
        self.menu_handler.setCheckState(feedback.menu_entry_id, MenuHandler.CHECKED)

        self.menu_handler.apply(self.server, feedback.marker_name)
        self.server.applyChanges()

        self.uvms_backend.set_robot_selected(selected_robot.k_robot)
        self._set_robot_submenus_visible(selected_robot.k_robot)

        self.get_logger().info(f"Switched to another robot is task based {selected_robot.task_based_controller} for robot {selected_robot.prefix}.")

        if selected_robot.control_space == ControlSpace.TASK_SPACE:
            self._apply_task_control_mode(robot=selected_robot, abort_motion=False)
        else:
            self._apply_joint_control_mode(robot=selected_robot, abort_motion=False)
            
    def _clear_markers(self) -> None:
        # erase by name, not by object
        self.server.erase(self.uv_marker.name)
        self.server.erase(self.arm_base_task_marker.name)
        self.server.applyChanges()

    def _apply_joint_control_mode(self, robot:Robot, abort_motion=True, keep_now_target=False) -> None:
        if robot != self.uvms_backend.robot_selected:
            return
        if abort_motion:
            robot.planner.planned_result = None
        self._clear_markers()

        # uv marker (vehicle)
        self.uv_marker.pose = getattr(self.uvms_backend, "_vehicle_desired_pose_from_ik_", self.uvms_backend.target_vehicle_pose)
        self.server.insert(self.uv_marker)
        self.server.setCallback(self.uv_marker.name, self.vehicle_marker_processFeedback)

        # endeffector marker (arm base frame)
        self.arm_base_task_marker.pose = self.uvms_backend.target_arm_base_endeffector_pose
        self.server.insert(self.arm_base_task_marker)
        self.server.setCallback(self.arm_base_task_marker.name, self.arm_base_task_marker_processFeedback)

        # attach menu control to the marker before applyChanges
        self.uv_marker.controls.append(marker_util.make_menu_control())
        self.server.insert(self.uv_marker)  # re-insert so server sees updated controls

        self.menu_handler.apply(self.server, self.uv_marker.name)
        self.menu_handler.apply(self.server, self.arm_base_task_marker.name)
        
        self.server.applyChanges()

        if keep_now_target:
            self.uvms_backend.target_vehicle_pose = self.uv_marker.pose
            arm_pose_world_new = robot.try_transform_pose(
                self.uvms_backend.target_world_endeffector_pose,
                target_frame=self.uvms_backend.arm_base_target_frame,
                source_frame=self.uvms_backend.world_frame,
                warn_context="set_endeffector_base_marker_pose",
            )
            if arm_pose_world_new is not None:
                self.uvms_backend.target_arm_base_endeffector_pose = arm_pose_world_new
        self.get_logger().info("Switched to JOINT space control.")


    def _apply_task_control_mode(self, robot:Robot, abort_motion=True) -> None:
        if robot != self.uvms_backend.robot_selected:
            return
        if abort_motion:
            robot.planner.planned_result = None
        self._clear_markers()

        self.server.insert(self.arm_base_task_marker)
        self.server.setCallback(self.arm_base_task_marker.name, self.world_task_marker_processFeedback)

        self.arm_base_task_marker.controls.append(marker_util.make_menu_control())
        self.server.insert(self.arm_base_task_marker)  # re-insert to update controls

        self.menu_handler.apply(self.server, self.arm_base_task_marker.name)
        self.server.applyChanges()

    def switch_controller_type(self, feedback: InteractiveMarkerFeedback):
        robot: Robot
        robot, controller_name = self.controller_menu_map.get(feedback.menu_entry_id)

        robot.set_controller(controller_name)

        # update checkmarks
        r_i: Robot
        for mid, (r_i, gsn) in self.controller_menu_map.items():
            if r_i.k_robot != robot.k_robot:
                continue
            self.menu_handler.setCheckState(mid, 
                                            MenuHandler.CHECKED if feedback.menu_entry_id == mid else MenuHandler.UNCHECKED)
        self.menu_handler.apply(self.server, feedback.marker_name)
        self.server.applyChanges()

    def switch_planner_type(self, feedback: InteractiveMarkerFeedback):
        robot: Robot
        robot, planner_name = self.planner_menu_map.get(feedback.menu_entry_id)

        robot.set_planner(planner_name)
        # update checkmarks
        r_i: Robot
        for mid, (r_i, gsn) in self.planner_menu_map.items():
            if r_i.k_robot != robot.k_robot:
                continue
            self.menu_handler.setCheckState(mid, 
                                            MenuHandler.CHECKED if feedback.menu_entry_id == mid else MenuHandler.UNCHECKED)
        self.menu_handler.apply(self.server, feedback.marker_name)
        self.server.applyChanges()

    def grasper_callback(self, feedback: InteractiveMarkerFeedback):
        robot: Robot
        robot, grasp_state_name = self.grasp_menu_map.get(feedback.menu_entry_id)

        if grasp_state_name in ['open','close']:
            if grasp_state_name == 'open':
                robot.arm.open_grasper()
            elif grasp_state_name == 'close':
                robot.arm.close_grasper()

            r_i: Robot
            for mid, (r_i, gsn) in self.grasp_menu_map.items():
                if r_i.k_robot != robot.k_robot:
                    continue
                self.menu_handler.setCheckState(mid, 
                                                MenuHandler.CHECKED if feedback.menu_entry_id == mid else MenuHandler.UNCHECKED)
            self.menu_handler.apply(self.server, feedback.marker_name)
            self.server.applyChanges()

    def switch_control_space_type(self, feedback: InteractiveMarkerFeedback):
        robot: Robot
        robot, control_space_name = self.control_space_menu_map.get(feedback.menu_entry_id)

        # set on the robot that owns this menu entry
        robot.set_control_space(control_space_name)

        # update checkmarks only for that robot's control-space entries
        r_i: Robot
        for mid, (r_i, csn) in self.control_space_menu_map.items():
            if r_i.k_robot != robot.k_robot:
                continue
            self.menu_handler.setCheckState(mid, 
                                            MenuHandler.CHECKED if feedback.menu_entry_id == mid else MenuHandler.UNCHECKED)

        self.menu_handler.apply(self.server, feedback.marker_name)
        self.server.applyChanges()

        if control_space_name == ControlSpace.TASK_SPACE:
            self._apply_task_control_mode(robot = robot)
        elif control_space_name == ControlSpace.JOINT_SPACE:
            self._apply_joint_control_mode(robot = robot, keep_now_target=True)

        self.get_logger().info(f"Control space set to {control_space_name} for {robot.prefix}")


    def toggle_endeffector_axis_align(self, feedback: InteractiveMarkerFeedback):
        robot: Robot
        robot, lookup_axis  = self.axis_menu_map.get(feedback.menu_entry_id)

        robot.ik_tool_axis = lookup_axis

        r_i: Robot
        for axis_target_handle, (r_i, axis) in self.axis_menu_map.items():
            if r_i.k_robot != robot.k_robot:
                continue
            self.menu_handler.setCheckState(axis_target_handle, 
                                            MenuHandler.CHECKED if feedback.menu_entry_id == axis_target_handle else MenuHandler.UNCHECKED)

        self.menu_handler.apply(self.server, feedback.marker_name)
        self.server.applyChanges()
        self.get_logger().info(f"Tool axis align set to {lookup_axis.tolist()} for {robot.prefix}")

    def toggle_align_with_base_weight(self, feedback: InteractiveMarkerFeedback):
        robot: Robot
        robot, _  = self.axis_menu_map.get(feedback.menu_entry_id)

        enabled = self.menu_handler.getCheckState(feedback.menu_entry_id) == MenuHandler.CHECKED
        self.menu_handler.setCheckState(feedback.menu_entry_id, MenuHandler.UNCHECKED if enabled else MenuHandler.CHECKED)

        robot.ik_base_align_w = self.menu_handler.getCheckState(feedback.menu_entry_id) == MenuHandler.CHECKED
        self.menu_handler.apply(self.server, feedback.marker_name)
        self.server.applyChanges()
        self.get_logger().info(f"Align arm with base weight set to {robot.ik_base_align_w:.1f}")
    
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
        self.sync_endeffector_world_marker_pose(self.uvms_backend.target_arm_base_endeffector_pose, self.uvms_backend.arm_base_target_frame)

    def arm_base_task_marker_processFeedback(self, feedback: InteractiveMarkerFeedback):
        if self.uvms_backend.is_valid_arm_base_task(feedback.pose):
            self.uvms_backend.target_arm_base_endeffector_pose = feedback.pose
            self.sync_endeffector_world_marker_pose(self.uvms_backend.target_arm_base_endeffector_pose, self.uvms_backend.arm_base_target_frame)
            self.get_logger().debug("Updated arm base task marker pose.")
            return

        self.server.setPose(self.arm_base_task_marker.name, self.uvms_backend.target_arm_base_endeffector_pose)
        self.server.applyChanges()

    def sync_endeffector_world_marker_pose(self, new_pose: Pose, source_frame: str) -> None:
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
        self.sync_endeffector_world_marker_pose(pose_world, self.uvms_backend.world_frame)

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
