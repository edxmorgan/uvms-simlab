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
import time
import gc

import rclpy
from rclpy.node import Node
from simlab.shutdown import install_signal_shutdown_handler, shutdown_node, spin_until_shutdown
from simlab.uvms_backend import UVMSBackendCore
from visualization_msgs.msg import InteractiveMarkerControl, InteractiveMarkerFeedback
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from simlab import interactive_utils as marker_util
from geometry_msgs.msg import Pose
import numpy as np
from simlab.uvms_parameters import ReachParams
from simlab.frame_utils import PoseX
from simlab.robot import Robot, ControlSpace

class InteractiveControlsNode(Node):
    def __init__(self):
        super().__init__('uvms_interactive_controls',
                         automatically_declare_parameters_from_overrides=True)
        urdf_string = self.get_parameter('robot_description').get_parameter_value().string_value
        self.world_frame = str(self.get_parameter_or('world_frame', 'world').value)
        self.vehicle_target_frame = "vehicle_marker_frame"
        self.arm_base_target_frame = "arm_base_marker_frame"
        self.world_endeffector_target_frame = "world_endeffector_marker_frame"

        # Arm base pose wrt vehicle center
        self.arm_base_wrt_vehicle_center_Pose = PoseX.from_pose(
                xyz=ReachParams.base_T0_new[0:3],
                rot=ReachParams.base_T0_new[3:6],
                rot_rep="euler_xyz",
                frame="NWU").get_pose_as_Pose_msg()
        

        self.uvms_backend: UVMSBackendCore = UVMSBackendCore(self, urdf_string,
                                                              self.arm_base_wrt_vehicle_center_Pose,
                                                              self.vehicle_target_frame, self.arm_base_target_frame, 
                                                              self.world_frame,
                                                              self.world_endeffector_target_frame, ReachParams)
        # Create marker server, menu handler
        self.server = InteractiveMarkerServer(self, "uvms_interactive_controls")

        self.menu_handler = MenuHandler()
        self.execute_handle = self.menu_handler.insert(
            "Plan & Execute",
            callback=self.plan_execute,
        )
        robot_select_menu_handle = self.menu_handler.insert("Robots", callback=self.noop_menu_callback)
        waypoints_parent = self.menu_handler.insert("Waypoints", callback=self.noop_menu_callback)
        path_planner_root = self.menu_handler.insert("Path Planner", callback=self.noop_menu_callback)
        csv_playback_root = self.menu_handler.insert("Cmd Replay", callback=self.noop_menu_callback)
        self.dynamics_profile_root = self.menu_handler.insert("Dynamics Profile", callback=self.noop_menu_callback)
        recording_root = self.menu_handler.insert("Data Recording", callback=self.noop_menu_callback)
        grasper_root = self.menu_handler.insert("Grasper", callback=self.noop_menu_callback)
        self.reset_manager_parent = self.menu_handler.insert("Reset Manager", callback=self.noop_menu_callback)
        robot_control_parent = self.menu_handler.insert("Robot Control", callback=self.noop_menu_callback)

        self.add_vehicle_waypoint_handle = self.menu_handler.insert(
            "Add",
            parent=waypoints_parent,
            callback=self.add_vehicle_waypoint,
        )
        self.delete_vehicle_waypoint_parent_handle = self.menu_handler.insert(
            "Delete",
            parent=waypoints_parent,
            callback=self.noop_menu_callback,
        )
        self.delete_vehicle_waypoint_handles = []
        self.delete_vehicle_waypoint_menu_map = {}
        self.clear_vehicle_waypoints_handle = self.menu_handler.insert(
            "Clear",
            parent=waypoints_parent,
            callback=self.clear_vehicle_waypoints,
        )
        self.stop_vehicle_waypoints_handle = self.menu_handler.insert(
            "Stop",
            parent=waypoints_parent,
            callback=self.stop_vehicle_waypoints,
        )
        self.reset_sim_handle = self.menu_handler.insert(
            "Reset",
            parent=self.reset_manager_parent,
            callback=self.reset_simulation,
        )
        self.release_sim_handle = self.menu_handler.insert(
            "Release",
            parent=self.reset_manager_parent,
            callback=self.release_simulation,
        )
        self.start_recording_handle = self.menu_handler.insert(
            "Start MCAP",
            parent=recording_root,
            callback=self.start_mcap_recording,
        )
        self.stop_recording_handle = self.menu_handler.insert(
            "Stop MCAP",
            parent=recording_root,
            callback=self.stop_mcap_recording,
        )
        self.mcap_recording_active = False

        self.control_space_menu_map = {}
        self.axis_menu_map = {}
        self.grasp_menu_map = {}
        self.controller_menu_map = {}
        self.planner_menu_map = {}
        self.csv_profile_menu_map = {}
        self.dynamics_profile_menu_map = {}

        control_space_names = []
        controller_names = []
        planner_names = []
        replay_profile_names = []
        dynamics_profile_names = []
        for robot in self.uvms_backend.robots:
            for cs_name in robot.list_control_spaces():
                if cs_name not in control_space_names:
                    control_space_names.append(cs_name)

            for controller_name in robot.list_controllers():
                if controller_name not in controller_names:
                    controller_names.append(controller_name)

            for planner_name in robot.list_planners():
                if planner_name not in planner_names:
                    planner_names.append(planner_name)

            cmd_replay_controller = robot.controller_instance("CmdReplay")
            if cmd_replay_controller is not None and hasattr(cmd_replay_controller, "list_profiles"):
                for profile_name in cmd_replay_controller.list_profiles():
                    if profile_name not in replay_profile_names:
                        replay_profile_names.append(profile_name)

            if "real" not in robot.prefix:
                for profile_name in robot.list_dynamics_profiles():
                    if profile_name not in dynamics_profile_names:
                        dynamics_profile_names.append(profile_name)

        for path_planner_name in planner_names:
            path_planner_handle = self.menu_handler.insert(
                f"{path_planner_name}",
                parent=path_planner_root,
                callback=self.switch_planner_type,
            )
            self.planner_menu_map[path_planner_handle] = path_planner_name
            self.menu_handler.setCheckState(path_planner_handle, MenuHandler.UNCHECKED)

        self.csv_profiles_parent = self.menu_handler.insert(
            "Profiles",
            parent=csv_playback_root,
            callback=self.noop_menu_callback,
        )
        for profile_name in replay_profile_names:
            profile_handle = self.menu_handler.insert(
                profile_name,
                parent=self.csv_profiles_parent,
                callback=self.switch_csv_playback_profile,
            )
            self.csv_profile_menu_map[profile_handle] = profile_name
            self.menu_handler.setCheckState(profile_handle, MenuHandler.UNCHECKED)

        self.csv_reset_handle = self.menu_handler.insert(
            "Reset Robot + Playback",
            parent=csv_playback_root,
            callback=self.reset_csv_playback,
        )
        self.csv_stop_handle = self.menu_handler.insert(
            "Stop",
            parent=csv_playback_root,
            callback=self.stop_csv_playback,
        )
        for profile_name in dynamics_profile_names:
            profile_handle = self.menu_handler.insert(
                profile_name,
                parent=self.dynamics_profile_root,
                callback=self.switch_dynamics_profile,
            )
            self.dynamics_profile_menu_map[profile_handle] = profile_name
            self.menu_handler.setCheckState(profile_handle, MenuHandler.UNCHECKED)

        cs_parent = self.menu_handler.insert(
            "Control Space",
            parent=robot_control_parent,
            callback=self.noop_menu_callback,
        )
        for cs_name in control_space_names:
            mid = self.menu_handler.insert(
                cs_name,
                parent=cs_parent,
                callback=self.switch_control_space_type,
            )
            self.control_space_menu_map[mid] = cs_name
            self.menu_handler.setCheckState(mid, MenuHandler.UNCHECKED)

        controller_parent = self.menu_handler.insert(
            "Controller",
            parent=robot_control_parent,
            callback=self.noop_menu_callback,
        )
        for controller_name in controller_names:
            controller_handle = self.menu_handler.insert(
                f"{controller_name}",
                parent=controller_parent,
                callback=self.switch_controller_type,
            )
            self.controller_menu_map[controller_handle] = controller_name
            self.menu_handler.setCheckState(controller_handle, MenuHandler.UNCHECKED)

        ik_settings_parent = self.menu_handler.insert(
            "IK Settings",
            parent=robot_control_parent,
            callback=self.noop_menu_callback,
        )
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
        self.axis_menu_map[x_axis_align_target_task_space_handle] = np.array([1, 0, 0], dtype=int)
        self.axis_menu_map[y_axis_align_target_task_space_handle] = np.array([0, 1, 0], dtype=int)
        self.axis_menu_map[z_axis_align_target_task_space_handle] = np.array([0, 0, 1], dtype=int)
        self.axis_menu_map[align_tool_axis_handle] = None
        self.menu_handler.setCheckState(x_axis_align_target_task_space_handle, MenuHandler.UNCHECKED)
        self.menu_handler.setCheckState(y_axis_align_target_task_space_handle, MenuHandler.UNCHECKED)
        self.menu_handler.setCheckState(z_axis_align_target_task_space_handle, MenuHandler.CHECKED)
        self.menu_handler.setCheckState(align_tool_axis_handle, MenuHandler.CHECKED)
        for robot in self.uvms_backend.robots:
            robot.ik_tool_axis = np.array([0, 0, 1], dtype=int)
            robot.ik_base_align_w = 1

        self.open_grasper_handle = self.menu_handler.insert('Open', parent=grasper_root, callback=self.grasper_callback)
        self.close_grasper_handle = self.menu_handler.insert('Close', parent=grasper_root, callback=self.grasper_callback)
        self.grasp_menu_map[self.open_grasper_handle] = 'open'
        self.grasp_menu_map[self.close_grasper_handle] = 'close'
        self.menu_handler.setCheckState(self.open_grasper_handle, MenuHandler.UNCHECKED)
        self.menu_handler.setCheckState(self.close_grasper_handle, MenuHandler.CHECKED)

        for robot in self.uvms_backend.robots:
            robot_handle = self.menu_handler.insert(
                f"{robot.prefix}",
                parent=robot_select_menu_handle,
                callback=self.switch_robot_in_use,
            )
            robot.user_id = robot_handle

            self.menu_handler.setCheckState(robot_handle,
                                             MenuHandler.CHECKED if self.uvms_backend.robot_selected.k_robot == robot.k_robot else MenuHandler.UNCHECKED)

        # Create markers
        self.uv_marker = marker_util.make_UVMS_Dof_Marker(
            name='uv_marker',
            description='interactive marker for controlling vehicle',
            frame_id=self.uvms_backend.world_frame,
            control_frame='uv',
            fixed=False,
            interaction_mode=InteractiveMarkerControl.MOVE_ROTATE_3D,
            initial_pose=self.uvms_backend.target_vehicle_pose,
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
        self._refresh_robot_menu_state(self.uvms_backend.robot_selected.k_robot)
        self._refresh_vehicle_waypoint_delete_menu()

    def reset_simulation(self, feedback: InteractiveMarkerFeedback):
        if self.uvms_backend.reset_selected_simulation():
            self._refresh_vehicle_waypoint_delete_menu()

    def release_simulation(self, feedback: InteractiveMarkerFeedback):
        self.uvms_backend.release_selected_simulation()

    def plan_execute(self, feedback: InteractiveMarkerFeedback):
        self.uvms_backend.plan_execute_selected()

    def add_vehicle_waypoint(self, feedback: InteractiveMarkerFeedback):
        ok, message = self.uvms_backend.add_vehicle_waypoint_for_robot(self.uvms_backend.robot_selected)
        if not ok:
            self.get_logger().warn(message)
            return
        self._refresh_vehicle_waypoint_delete_menu()

    def delete_vehicle_waypoint(self, feedback: InteractiveMarkerFeedback):
        robot_k, waypoint_index = self.delete_vehicle_waypoint_menu_map.get(
            feedback.menu_entry_id,
            (None, None),
        )
        if robot_k is None or waypoint_index is None:
            self.get_logger().warn(
                f"No waypoint is associated with menu entry id {feedback.menu_entry_id}."
            )
            return
        if not self.uvms_backend.remove_vehicle_waypoint_for_robot(robot_k, waypoint_index):
            return
        self._refresh_vehicle_waypoint_delete_menu()

    def clear_vehicle_waypoints(self, feedback: InteractiveMarkerFeedback):
        self.uvms_backend.clear_selected_vehicle_waypoints()
        self._refresh_vehicle_waypoint_delete_menu()

    def stop_vehicle_waypoints(self, feedback: InteractiveMarkerFeedback):
        self.uvms_backend.stop_selected_vehicle_waypoints()

    def _refresh_robot_menu_state(self, selected_k_robot: int) -> None:
        selected_robot = self.uvms_backend.robot_selected
        for mid, control_space_name in self.control_space_menu_map.items():
            self.menu_handler.setCheckState(
                mid,
                MenuHandler.CHECKED if selected_robot.control_space == control_space_name else MenuHandler.UNCHECKED,
            )

        for mid, controller_name in self.controller_menu_map.items():
            self.menu_handler.setCheckState(
                mid,
                MenuHandler.CHECKED if selected_robot.controller_name == controller_name else MenuHandler.UNCHECKED,
            )

        for mid, planner_name in self.planner_menu_map.items():
            self.menu_handler.setCheckState(
                mid,
                MenuHandler.CHECKED if selected_robot.planner_name == planner_name else MenuHandler.UNCHECKED,
            )

        cmd_replay_controller = selected_robot.controller_instance("CmdReplay")
        selected_profile = (
            getattr(cmd_replay_controller, "profile_name", None)
            if cmd_replay_controller is not None
            else None
        )
        for mid, profile_name in self.csv_profile_menu_map.items():
            self.menu_handler.setCheckState(
                mid,
                MenuHandler.CHECKED if selected_profile == profile_name else MenuHandler.UNCHECKED,
            )

        active_dynamics_profile = getattr(selected_robot, "active_dynamics_profile", "")
        for mid, profile_name in self.dynamics_profile_menu_map.items():
            self.menu_handler.setCheckState(
                mid,
                MenuHandler.CHECKED if active_dynamics_profile == profile_name else MenuHandler.UNCHECKED,
            )

        del selected_k_robot
        self.menu_handler.reApply(self.server)
        self.server.applyChanges()

    def _refresh_vehicle_waypoint_delete_menu(self) -> None:
        for handle in self.delete_vehicle_waypoint_handles:
            self.menu_handler.setVisible(handle, False)
        self.delete_vehicle_waypoint_handles = []
        self.delete_vehicle_waypoint_menu_map = {}

        mission = self.uvms_backend.selected_vehicle_waypoint_mission()
        robot = self.uvms_backend.robot_selected
        for idx, pose in enumerate(mission.waypoints):
            handle = self.menu_handler.insert(
                f"Waypoint {idx + 1} [{pose.position.x:.2f}, {pose.position.y:.2f}, {pose.position.z:.2f}]",
                parent=self.delete_vehicle_waypoint_parent_handle,
                callback=self.delete_vehicle_waypoint,
            )
            self.delete_vehicle_waypoint_handles.append(handle)
            self.delete_vehicle_waypoint_menu_map[handle] = (robot.k_robot, idx)

        self.menu_handler.reApply(self.server)
        self.server.applyChanges()

    def noop_menu_callback(self, feedback: InteractiveMarkerFeedback):
        return

    def _set_mcap_recording_menu_state(self, active: bool) -> None:
        self.mcap_recording_active = active
        self.menu_handler.setCheckState(
            self.start_recording_handle,
            MenuHandler.UNCHECKED if active else MenuHandler.CHECKED,
        )
        self.menu_handler.setCheckState(
            self.stop_recording_handle,
            MenuHandler.CHECKED if active else MenuHandler.UNCHECKED,
        )
        self.menu_handler.reApply(self.server)
        self.server.applyChanges()

    def start_mcap_recording(self, feedback: InteractiveMarkerFeedback):
        del feedback
        self.uvms_backend.start_mcap_recording(
            on_success=lambda: self._set_mcap_recording_menu_state(True)
        )

    def stop_mcap_recording(self, feedback: InteractiveMarkerFeedback):
        del feedback
        self.uvms_backend.stop_mcap_recording(
            on_success=lambda: self._set_mcap_recording_menu_state(False)
        )

    # switch to robot i
    def switch_robot_in_use(self, feedback: InteractiveMarkerFeedback):
        selected_robot = None
        for robot in self.uvms_backend.robots:
            if robot.user_id == feedback.menu_entry_id:
                selected_robot = robot
                break

        if selected_robot is None:
            self.get_logger().error(
                f"No robot is associated with menu entry id {feedback.menu_entry_id}."
            )
            return

        self._set_selected_robot_menu_state(selected_robot, feedback.marker_name)
        self.configure_selected_robot(selected_robot)

    def _set_selected_robot_menu_state(self, selected_robot: Robot, marker_name: str) -> None:
        for robot in self.uvms_backend.robots:
            state = MenuHandler.CHECKED if robot.k_robot == selected_robot.k_robot else MenuHandler.UNCHECKED
            self.menu_handler.setCheckState(robot.user_id, state)

        self.menu_handler.apply(self.server, marker_name)
        self.server.applyChanges()

    def configure_selected_robot(self, selected_robot: Robot):
        ok, message = self.uvms_backend.select_robot(selected_robot.k_robot)
        if not ok:
            self.get_logger().warn(message)
            return
        self._refresh_robot_menu_state(selected_robot.k_robot)
        self._refresh_vehicle_waypoint_delete_menu()

        self.get_logger().info(f"""Switched to another robot is task based 
                               {selected_robot.task_based_controller} for robot {selected_robot.prefix}.""")

        if selected_robot.control_space == ControlSpace.TASK_SPACE:
            self._apply_task_control_mode(robot=selected_robot, abort_motion=False)
        else:
            self._apply_joint_control_mode(robot=selected_robot, abort_motion=False)
            
    def _clear_markers(self) -> None:
        # erase by name, not by object
        self.server.erase(self.uv_marker.name)
        self.server.erase(self.arm_base_task_marker.name)
        self.server.applyChanges()

    def _ensure_menu_control(self, marker) -> None:
        if any(control.interaction_mode == InteractiveMarkerControl.MENU for control in marker.controls):
            return
        marker.controls.append(marker_util.make_menu_control())

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
        self._ensure_menu_control(self.uv_marker)
        self.server.insert(self.uv_marker)  # re-insert so server sees updated controls

        self.menu_handler.apply(self.server, self.uv_marker.name)
        self.menu_handler.apply(self.server, self.arm_base_task_marker.name)
        
        self.server.applyChanges()

        if keep_now_target:
            self.uvms_backend.set_vehicle_target(robot, self.uv_marker.pose)
            arm_pose_world_new = robot.try_transform_pose(
                self.uvms_backend.target_world_endeffector_pose,
                target_frame=self.uvms_backend.arm_base_target_frame,
                source_frame=self.uvms_backend.world_frame,
                warn_context="set_endeffector_base_marker_pose",
            )
            if arm_pose_world_new is not None:
                self.uvms_backend.set_task_target_arm_base(robot, arm_pose_world_new)
        self.get_logger().info("Switched to JOINT space control.")


    def _apply_task_control_mode(self, robot:Robot, abort_motion=True) -> None:
        if robot != self.uvms_backend.robot_selected:
            return
        if abort_motion:
            robot.planner.planned_result = None
        self._clear_markers()

        self.server.insert(self.arm_base_task_marker)
        self.server.setCallback(self.arm_base_task_marker.name, self.world_task_marker_processFeedback)

        self._ensure_menu_control(self.arm_base_task_marker)
        self.server.insert(self.arm_base_task_marker)  # re-insert to update controls

        self.menu_handler.apply(self.server, self.arm_base_task_marker.name)
        self.server.applyChanges()

    def switch_controller_type(self, feedback: InteractiveMarkerFeedback):
        robot = self.uvms_backend.robot_selected
        controller_name = self.controller_menu_map.get(feedback.menu_entry_id)

        ok, message = self.uvms_backend.set_robot_controller(robot, controller_name)
        if not ok:
            self.get_logger().warn(message)
            return

        # update checkmarks
        for mid, candidate_controller in self.controller_menu_map.items():
            self.menu_handler.setCheckState(mid, 
                                            MenuHandler.CHECKED if candidate_controller == controller_name else MenuHandler.UNCHECKED)
        self.menu_handler.apply(self.server, feedback.marker_name)
        self.server.applyChanges()

    def switch_planner_type(self, feedback: InteractiveMarkerFeedback):
        robot = self.uvms_backend.robot_selected
        planner_name = self.planner_menu_map.get(feedback.menu_entry_id)

        ok, message = self.uvms_backend.set_robot_planner(robot, planner_name)
        if not ok:
            self.get_logger().warn(message)
            return
        # update checkmarks
        for mid, candidate_planner in self.planner_menu_map.items():
            self.menu_handler.setCheckState(mid, 
                                            MenuHandler.CHECKED if candidate_planner == planner_name else MenuHandler.UNCHECKED)
        self.menu_handler.apply(self.server, feedback.marker_name)
        self.server.applyChanges()

    def switch_dynamics_profile(self, feedback: InteractiveMarkerFeedback):
        robot = self.uvms_backend.robot_selected
        profile_name = self.dynamics_profile_menu_map.get(feedback.menu_entry_id)

        def _on_success():
            for mid, candidate_profile in self.dynamics_profile_menu_map.items():
                self.menu_handler.setCheckState(
                    mid,
                    MenuHandler.CHECKED if candidate_profile == profile_name else MenuHandler.UNCHECKED,
                )
            self.menu_handler.apply(self.server, feedback.marker_name)
            self.server.applyChanges()

        ok, message = self.uvms_backend.set_robot_dynamics_profile(
            robot,
            profile_name,
            on_success=_on_success,
        )
        if not ok:
            self.get_logger().warn(message)

    def stop_csv_playback(self, feedback: InteractiveMarkerFeedback):
        robot = self.uvms_backend.robot_selected
        ok, message = self.uvms_backend.stop_replay(robot)
        if ok:
            self.get_logger().info(message)
        else:
            self.get_logger().warn(message)

    def switch_csv_playback_profile(self, feedback: InteractiveMarkerFeedback):
        robot = self.uvms_backend.robot_selected
        profile_name = self.csv_profile_menu_map.get(feedback.menu_entry_id)
        ok, message = self.uvms_backend.select_replay_profile(robot, profile_name)
        if not ok:
            self.get_logger().warn(message)
            return

        for mid, candidate_profile in self.csv_profile_menu_map.items():
            self.menu_handler.setCheckState(
                mid,
                MenuHandler.CHECKED if candidate_profile == profile_name else MenuHandler.UNCHECKED,
            )
        self.menu_handler.apply(self.server, feedback.marker_name)
        self.server.applyChanges()
        self.get_logger().info(message)

    def reset_csv_playback(self, feedback: InteractiveMarkerFeedback):
        robot = self.uvms_backend.robot_selected
        ok, message = self.uvms_backend.start_replay(robot)
        if ok:
            self.get_logger().info(message)
        else:
            self.get_logger().warn(message)

    def grasper_callback(self, feedback: InteractiveMarkerFeedback):
        robot = self.uvms_backend.robot_selected
        grasp_state_name = self.grasp_menu_map.get(feedback.menu_entry_id)

        if grasp_state_name in ['open','close']:
            ok, message = self.uvms_backend.command_grasper(robot, grasp_state_name)
            if not ok:
                self.get_logger().warn(message)
                return

            for mid, gsn in self.grasp_menu_map.items():
                self.menu_handler.setCheckState(mid, 
                                                MenuHandler.CHECKED if feedback.menu_entry_id == mid else MenuHandler.UNCHECKED)
            self.menu_handler.apply(self.server, feedback.marker_name)
            self.server.applyChanges()

    def switch_control_space_type(self, feedback: InteractiveMarkerFeedback):
        robot = self.uvms_backend.robot_selected
        control_space_name = self.control_space_menu_map.get(feedback.menu_entry_id)

        ok, message = self.uvms_backend.set_robot_control_space(robot, control_space_name)
        if not ok:
            self.get_logger().warn(message)
            return

        for mid, candidate_control_space in self.control_space_menu_map.items():
            self.menu_handler.setCheckState(mid, 
                                            MenuHandler.CHECKED if candidate_control_space == control_space_name else MenuHandler.UNCHECKED)

        self.menu_handler.apply(self.server, feedback.marker_name)
        self.server.applyChanges()

        if control_space_name == ControlSpace.TASK_SPACE:
            self._apply_task_control_mode(robot = robot)
        elif control_space_name == ControlSpace.JOINT_SPACE:
            self._apply_joint_control_mode(robot = robot, keep_now_target=True)

        self.get_logger().info(message)


    def toggle_endeffector_axis_align(self, feedback: InteractiveMarkerFeedback):
        robot = self.uvms_backend.robot_selected
        lookup_axis  = self.axis_menu_map.get(feedback.menu_entry_id)

        ok, message = self.uvms_backend.set_robot_ik_tool_axis(robot, lookup_axis)
        if not ok:
            self.get_logger().warn(message)
            return

        for axis_target_handle, axis in self.axis_menu_map.items():
            if axis is None:
                continue
            self.menu_handler.setCheckState(axis_target_handle, 
                                            MenuHandler.CHECKED if feedback.menu_entry_id == axis_target_handle else MenuHandler.UNCHECKED)

        self.menu_handler.apply(self.server, feedback.marker_name)
        self.server.applyChanges()
        self.get_logger().info(message)

    def toggle_align_with_base_weight(self, feedback: InteractiveMarkerFeedback):
        robot = self.uvms_backend.robot_selected

        enabled = self.menu_handler.getCheckState(feedback.menu_entry_id) == MenuHandler.CHECKED
        self.menu_handler.setCheckState(feedback.menu_entry_id, MenuHandler.UNCHECKED if enabled else MenuHandler.CHECKED)

        weight = 1.0 if self.menu_handler.getCheckState(feedback.menu_entry_id) == MenuHandler.CHECKED else 0.0
        ok, message = self.uvms_backend.set_robot_ik_base_align_weight(robot, weight)
        if not ok:
            self.get_logger().warn(message)
            return
        self.menu_handler.apply(self.server, feedback.marker_name)
        self.server.applyChanges()
        self.get_logger().info(message)
    
    def vehicle_marker_processFeedback(self, feedback: InteractiveMarkerFeedback):
        ok, message = self.uvms_backend.set_vehicle_target(self.uvms_backend.robot_selected, feedback.pose)
        if not ok:
            self.get_logger().warn(message)
            return

        self.server.setPose(feedback.marker_name, self.uvms_backend.target_vehicle_pose)
        self.server.applyChanges()
        self.get_logger().debug(message)
        self.sync_endeffector_world_marker_pose(self.uvms_backend.target_arm_base_endeffector_pose, self.uvms_backend.arm_base_target_frame)

    def arm_base_task_marker_processFeedback(self, feedback: InteractiveMarkerFeedback):
        ok, message = self.uvms_backend.set_task_target_arm_base(self.uvms_backend.robot_selected, feedback.pose)
        if ok:
            self.sync_endeffector_world_marker_pose(self.uvms_backend.target_arm_base_endeffector_pose, self.uvms_backend.arm_base_target_frame)
            self.get_logger().debug(message)
            return

        self.get_logger().debug(message)
        self.server.setPose(self.arm_base_task_marker.name, self.uvms_backend.target_arm_base_endeffector_pose)
        self.server.applyChanges()

    def sync_endeffector_world_marker_pose(self, new_pose: Pose, source_frame: str) -> bool:
        if source_frame == self.uvms_backend.world_frame:
            ok, message = self.uvms_backend.set_task_target_world(self.uvms_backend.robot_selected, new_pose)
            if not ok:
                self.get_logger().warn(message)
            return ok

        task_pose_world_new = self.uvms_backend.robot_selected.try_transform_pose(
            new_pose,
            target_frame=self.uvms_backend.world_frame,
            source_frame=source_frame,
            warn_context="set_endeffector_world_marker_pose",
        )
        if task_pose_world_new is None:
            return False

        ok, message = self.uvms_backend.set_task_target_world(self.uvms_backend.robot_selected, task_pose_world_new)
        if not ok:
            self.get_logger().warn(message)
        return ok

    def world_task_marker_processFeedback(self, feedback: InteractiveMarkerFeedback):
        pose_world = self.uvms_backend.robot_selected.try_transform_pose(
            feedback.pose,
            target_frame=self.uvms_backend.world_frame,
            source_frame=self.uvms_backend.arm_base_target_frame,
            warn_context="world_task_marker_processFeedback,to_world",
        )
        if pose_world is None:
            return

        ok, message = self.uvms_backend.set_task_target_world(self.uvms_backend.robot_selected, pose_world)
        if not ok:
            self.get_logger().warn(message)
            return

        pose_arm = self.uvms_backend.robot_selected.try_transform_pose(
            self.uvms_backend.target_world_endeffector_pose,
            target_frame=self.uvms_backend.arm_base_target_frame,
            source_frame=self.uvms_backend.world_frame,
            warn_context="world_task_marker_processFeedback,to_arm_base",
        )
        if pose_arm is None:
            return

        self.server.setPose(feedback.marker_name, pose_arm)
        self.server.applyChanges()

    def destroy_node(self):
        if getattr(self, "uvms_backend", None) is not None:
            self.uvms_backend.close()
            self.uvms_backend = None
        gc.collect()
        return super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    install_signal_shutdown_handler()
    node = InteractiveControlsNode()
    try:
        spin_until_shutdown(node)
    finally:
        shutdown_node(node)

if __name__ == '__main__':
    main()
