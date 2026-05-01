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

import csv
from datetime import datetime
from pathlib import Path as FilePath

import numpy as np
from typing import Dict
from control_msgs.msg import DynamicJointState
from scipy.spatial.transform import Rotation as R
import ament_index_python
import os
import rclpy
import casadi as ca
from nav_msgs.msg import Path
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, TwistStamped, AccelStamped
from rclpy.qos import QoSProfile, QoSHistoryPolicy
import copy
from std_msgs.msg import Float32
from ros2_control_blue_reach_5.srv import ResetSimUvms, SetSimDynamics
from std_srvs.srv import Trigger
from pyPS4Controller.controller import Controller
import threading
import glob
from typing import Sequence, Dict, Callable, Any, Optional
from control_msgs.msg import DynamicInterfaceGroupValues
from std_msgs.msg import Float64MultiArray
from simlab.controller_msg import FullRobotMsg
from simlab.controllers import DEFAULT_CONTROLLER_CLASSES
from simlab.dynamics_profiles import (
    is_valid_robot_dynamics_profile,
    list_robot_dynamics_profiles,
    load_robot_dynamics_profile,
    set_dynamics_request_from_profile,
)
from simlab.planner_markers import PathPlanner
from simlab.cartesian_ruckig import VehicleCartesianRuckig
from ruckig import Result
from simlab.uvms_parameters import ReachParams
from simlab.frame_utils import PoseX
from tf2_ros import TransformException, Buffer
from tf2_geometry_msgs import do_transform_pose, do_transform_vector3
from typing import Optional
from geometry_msgs.msg import Pose
from typing import Optional, Tuple, Sequence
import numpy as np
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Vector3Stamped
from enum import Enum
from dataclasses import dataclass
from simlab_msgs.msg import ControllerPerformance
from simlab.reference_targets import ReferenceTargetPublisher
from simlab.planner_action_client import PlannerActionClient
from simlab.planners import visible_planner_names
from simlab.performance_metrics import ControllerPerformanceMetrics

class ControlSpace(str, Enum):
    JOINT_SPACE = "joint_space"
    TASK_SPACE  = "task_space"


class ControlMode(str, Enum):
    TELEOP = "teleop"
    PLANNER = "planner"
    REPLAY = "replay"
    REPLAY_SETTLE = "replay_settle"

@dataclass
class ControllerSpec:
    name: str
    vehicle_fn: Callable[..., Any]
    arm_fn: Callable[..., Any]

@dataclass
class PlannerSpec:
    name: str

class PS4Controller(Controller):
    def __init__(self, ros_node: Node, prefix, **kwargs):
        super().__init__(**kwargs)
        self.ros_node: Node = ros_node
        
        # mode flag: False = joint control, True = light & mount control
        self.options_mode = False

        # running values
        self.light_value = 0.0
        self.mount_value = 0.0
        
        sim_gain = 5.0
        real_gain = 5.0
        self.gain = sim_gain
        self.gain = real_gain if 'real' in prefix else sim_gain

        # Gains for different DOFs
        self.max_torque = self.gain * 2.0             # for surge/sway
        self.heave_max_torque = self.gain * 5.0         # for heave (L2/R2)
        self.orient_max_torque = self.gain * 0.8        # for roll, pitch,
        self.yaw_max_torque = self.gain * 0.4 # for yaw

    def on_share_press(self):
        # toggle teleop vs planner
        new_mode = ControlMode.PLANNER if self.ros_node.control_mode == ControlMode.TELEOP else ControlMode.TELEOP
        self.ros_node.set_control_mode(new_mode)

   # —— Options toggles between modes ——    
    def on_options_press(self):
        self.options_mode = not self.options_mode
        # if returning to joint mode, zero out any light/mount commands
        if not self.options_mode:
            self.ros_node.light_publisher_.publish(Float32(data=0.0))
            self.ros_node.mountPitch_publisher_.publish(Float32(data=0.0))

    # —— Heave (unchanged) ——    
    def on_L2_press(self, value):
        scaled = self.heave_max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_z = -scaled

    def on_L2_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_z = 0.0

    def on_R2_press(self, value):
        scaled = self.heave_max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_z = scaled

    def on_R2_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_z = 0.0

    # —— Surge & Sway (unchanged) ——    
    def on_L3_up(self, value):
        scaled = self.max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_surge = -scaled

    def on_L3_down(self, value):
        scaled = self.max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_surge = -scaled

    def on_L3_right(self, value):
        scaled = self.max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_sway = scaled

    def on_L3_left(self, value):
        scaled = self.max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_sway = scaled

    def on_L3_x_at_rest(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_sway = 0.0

    def on_L3_y_at_rest(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_surge = 0.0

    # —— Roll control (unchanged) ——    
    def on_R1_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_roll =  self.orient_max_torque

    def on_L1_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_roll = -self.orient_max_torque

    def on_R1_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_roll = 0.0

    def on_L1_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_roll = 0.0

    # —— Pitch & Yaw (unchanged) ——    
    def on_R3_up(self, value):
        scaled = self.orient_max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_pitch = scaled

    def on_R3_down(self, value):
        scaled = self.orient_max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_pitch = scaled

    def on_R3_left(self, value):
        scaled = self.yaw_max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_yaw = scaled

    def on_R3_right(self, value):
        scaled = self.yaw_max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_yaw = scaled

    def on_R3_x_at_rest(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_yaw = 0.0

    def on_R3_y_at_rest(self):
        with self.ros_node.controller_lock:
            self.ros_node.rov_pitch = 0.0

    # —— D‑pad Left/Right ——    
    def on_left_arrow_press(self):
        if self.options_mode:
            self.ros_node.light_publisher_.publish(Float32(data=-10.0))
        else:
            with self.ros_node.controller_lock:
                self.ros_node.jointe = -3.0

    def on_right_arrow_press(self):
        if self.options_mode:
            self.ros_node.light_publisher_.publish(Float32(data=10.0))
        else:
            with self.ros_node.controller_lock:
                self.ros_node.jointe = 3.0

    def on_left_right_arrow_release(self):
        if self.options_mode:
            self.ros_node.light_publisher_.publish(Float32(data=0.0))
        else:
            with self.ros_node.controller_lock:
                self.ros_node.jointe = 0.0

    # —— D‑pad Up/Down ——    
    def on_up_arrow_press(self):
        if self.options_mode:
            self.ros_node.mountPitch_publisher_.publish(Float32(data=-10.0))
        else:
            with self.ros_node.controller_lock:
                self.ros_node.jointd = 2.0

    def on_down_arrow_press(self):
        if self.options_mode:
            self.ros_node.mountPitch_publisher_.publish(Float32(data=10.0))
        else:
            with self.ros_node.controller_lock:
                self.ros_node.jointd = -2.0

    def on_up_down_arrow_release(self):
        if self.options_mode:
            self.ros_node.mountPitch_publisher_.publish(Float32(data=0.0))
        else:
            with self.ros_node.controller_lock:
                self.ros_node.jointd = 0.0

    # —— Manipulator buttons (unchanged) ——    
    def on_triangle_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointc = 2.0

    def on_triangle_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointc = 0.0

    def on_x_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointc = -2.0

    def on_x_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointc = 0.0

    def on_square_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointb = 1.0

    def on_square_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointb = 0.0

    def on_circle_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointb = -1.0

    def on_circle_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointb = 0.0

    def on_R3_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointa = 1.0

    def on_R3_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointa = 0.0

    def on_L3_press(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointa = -1.0

    def on_L3_release(self):
        with self.ros_node.controller_lock:
            self.ros_node.jointa = 0.0

class Base:
    def get_interface_value(self, msg: DynamicJointState, dof_names: list, interface_names: list):
        names = msg.joint_names
        return [
            msg.interface_values[names.index(joint_name)].values[
                msg.interface_values[names.index(joint_name)].interface_names.index(interface_name)
            ]
            for joint_name, interface_name in zip(dof_names, interface_names)
        ]

class Axis_Interface_names:
    manipulator_position = 'position'
    manipulator_filtered_position = 'filtered_position'
    manipulator_velocity = 'velocity'
    manipulator_filtered_velocity = 'filtered_velocity'
    manipulator_estimation_acceleration = "estimated_acceleration"
    manipulator_effort = 'effort'
    
    floating_base_x = 'position.x'
    floating_base_y = 'position.y'
    floating_base_z = 'position.z'

    floating_base_roll = 'roll'
    floating_base_pitch = 'pitch'
    floating_base_yaw = 'yaw'

    floating_dx = 'velocity.x'
    floating_dy = 'velocity.y'
    floating_dz = 'velocity.z'

    floating_roll_vel = 'angular_velocity.x'
    floating_pitch_vel = 'angular_velocity.y'
    floating_yaw_vel = 'angular_velocity.z'

    floating_du = 'linear_acceleration.x'
    floating_dv = 'linear_acceleration.y'
    floating_dw = 'linear_acceleration.z'

    floating_dp = 'angular_acceleration.x'
    floating_dq = 'angular_acceleration.y'
    floating_dr = 'angular_acceleration.z'

    floating_force_x = 'force.x'
    floating_force_y = 'force.y'
    floating_force_z = 'force.z'
    floating_torque_x = 'torque.x'
    floating_torque_y = 'torque.y'
    floating_torque_z = 'torque.z'
    floating_control_power_abs = 'control_power_abs'
    floating_control_energy_abs = 'control_energy_abs'

    sim_time = 'sim_time'
    sim_period = 'sim_period'
    arm_payload_mass = 'payload.mass'
    arm_gravity = 'gravity'
    arm_control_power_abs = 'control_power_abs'
    arm_control_energy_abs = 'control_energy_abs'
    
class Manipulator(Base):
    def __init__(self, node: Node, n_joint, prefix):
        self.node = node
        self.n_joint = n_joint
        self.q = [0]*n_joint
        self.dq = [0]*n_joint
        self.ddq = [0]*n_joint
        self.sim_period = [0.0]
        self.effort = [0]*n_joint
        self.alpha_axis_a = f'{prefix}_axis_a'
        self.alpha_axis_b = f'{prefix}_axis_b'
        self.alpha_axis_c = f'{prefix}_axis_c'
        self.alpha_axis_d = f'{prefix}_axis_d'
        self.alpha_axis_e = f'{prefix}_axis_e'

        self.joints = [self.alpha_axis_e, self.alpha_axis_d, self.alpha_axis_c, self.alpha_axis_b]
        self.grasper = [self.alpha_axis_a]

        self.q_command = ReachParams.joint_home.tolist()
        self.dq_command = np.zeros((4,)).tolist()
        self.ddq_command = np.zeros((4,)).tolist()

        # Initialize grasper state so get_state() is safe before first update.
        self.grasper_q = [0.0]
        self.grasper_q_dot = [0.0]
        self.grasper_q_ddot = [0.0]
        self.close_grasper()

    def open_grasper(self):
        self.grasp_command = ReachParams.grasper_open

    def close_grasper(self):
        self.grasp_command = ReachParams.grasper_close

    def update_state(self, msg: DynamicJointState):
        self.q = self.get_interface_value(
            msg,
            self.joints,
            [Axis_Interface_names.manipulator_position] * 4
        )
        self.dq = self.get_interface_value(
            msg,
            self.joints,
            [Axis_Interface_names.manipulator_velocity] * 4
        )
        self.ddq = self.get_interface_value(
            msg,
            self.joints,
            [Axis_Interface_names.manipulator_estimation_acceleration] * 4
        )
        self.effort = self.get_interface_value(
            msg,
            self.joints,
            [Axis_Interface_names.manipulator_effort] * 4
        )
        self.sim_period = self.get_interface_value(
            msg,
            [self.alpha_axis_e],
            [Axis_Interface_names.sim_period]
        )
        self.grasper_q = self.get_interface_value(
            msg,
            self.grasper,
            [Axis_Interface_names.manipulator_position]
        )
        self.grasper_q_dot = self.get_interface_value(
            msg,
            self.grasper,
            [Axis_Interface_names.manipulator_velocity]
        )
        self.grasper_q_ddot = self.get_interface_value(
            msg,
            self.grasper,
            [Axis_Interface_names.manipulator_estimation_acceleration]
        )
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            'arm_effort':self.effort,
            'grasper_q': self.grasper_q,
            'grasper_qdot': self.grasper_q_dot,
            'grasper_qddot': self.grasper_q_ddot,
            'q':self.q,
            'dq':self.dq,
            'ddq':self.ddq,
            'dt':self.sim_period[0]
        }

class Robot(Base):
    def __init__(self, node: Node,
                 tf_buffer: Buffer,
                  k_robot, 
                  n_joint, 
                  prefix,
                  planner=None,
                  vehicle_cart_traj=None,
                  create_subscriptions: bool = True):
        self.planner: PathPlanner = planner
        self.vehicle_cart_traj: VehicleCartesianRuckig = vehicle_cart_traj
        self.menu_handle = None
        self.final_goal_map_ned_6 = None
        self.yaw_blend_factor = 0.0
        self.tf_buffer = tf_buffer
        self.task_based_controller = False

        self.dynamics_states_sub = None
        
        # Latest mocap pose [x, y, z, qw, qx, qy, qz]
        self.mocap_latest = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        self.v_c = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.mocap_pose_sub = None
        if create_subscriptions:
            self.dynamics_states_sub = node.create_subscription(
                    DynamicJointState,
                    'dynamic_joint_states',
                    self.listener_callback,
                    10
                )

            # Subscribe to the ENU, origin offset pose from MocapPathBuilder
            # Topic name must match MocapPathBuilder.mocap_pose_topic, default 'mocap_pose'
            self.mocap_pose_sub = node.create_subscription(
                PoseStamped,
                'mocap_pose',
                self._mocap_pose_cb,
                10
            )
        
        self.k_robot = k_robot
        self.user_id = None
        self.robot_name = f'uvms {prefix}: {k_robot}'
    
        package_share_directory = ament_index_python.get_package_share_directory(
                'simlab')
        fk_path = os.path.join(package_share_directory, 'model_functions/arm/fk_eval.casadi')
        ik_path = os.path.join(package_share_directory, 'model_functions/arm/ik_eval.casadi')

        vehicle_J_path = os.path.join(package_share_directory, 'model_functions/vehicle/J_uv.casadi')
        vehicle_ned2body_acc_path = os.path.join(package_share_directory, 'model_functions/vehicle/ned2body_acc.casadi')
        vehicle_ned2body_vel_path = os.path.join(package_share_directory, 'model_functions/vehicle/ned2body_vel.casadi')
        ik_wb_path = os.path.join(package_share_directory, 'whole_body/ik.so')

        self.fk_eval = ca.Function.load(fk_path) #  forward kinematics
        # also set a class attribute fk_eval so it can be shared
        if not hasattr(Robot, "fk_eval_cls"):
            Robot.fk_eval_cls = self.fk_eval

        self.ik_eval = ca.Function.load(ik_path) #  inverse kinematics
        # also set a class attribute ik_eval so it can be shared
        if not hasattr(Robot, "ik_eval_cls"):
            Robot.ik_eval_cls = self.ik_eval


        self.ik_wb_eval = ca.external('mapacc_task_based_ik',ik_wb_path) #  inverse kinematics

        # also set a class attribute ik_eval so it can be shared
        if not hasattr(Robot, "ik_wb_eval_cls"):
            Robot.ik_wb_eval_cls = self.ik_wb_eval

        self.vehicle_J = ca.Function.load(vehicle_J_path)
        self.vehicle_ned2body_acc = ca.Function.load(vehicle_ned2body_acc_path)
        self.vehicle_ned2body_vel = ca.Function.load(vehicle_ned2body_vel_path)

        self.node = node

        self.n_joint = n_joint
        self.floating_base_IOs = f'{prefix}IOs'
        self.arm_IOs = f'{prefix}_arm_IOs'
        self.map_frame = f"{prefix}map" 
        self.arm = Manipulator(node, n_joint, prefix)
        self.ned_pose = [0] * 6
        self.body_vel = [0] * 6
        self.body_acc = [0] * 6
        self.ned_vel = [0] * 6
        self.body_forces = [0] * 6
        self.vehicle_control_power = 0.0
        self.vehicle_control_energy = 0.0
        self.arm_control_power = 0.0
        self.arm_control_energy = 0.0
        self.arm_payload_mass = 0.0
        self.arm_gravity = 0.0
        self.prefix = prefix
        self.status = 'inactive'
        self.sim_time = 0.0
        self.start_time = 0.0
        self.joint4_frame = f"{self.prefix}joint_4"
        self.pose_command = [0.0]*6
        self.body_vel_command = [0.0]*6
        self.body_acc_command = [0.0]*6
        self.performance_metrics = ControllerPerformanceMetrics()
        self.performance_publisher = self.node.create_publisher(
            ControllerPerformance,
            f"/{self.prefix}/performance/controller",
            10,
        )
        self.controller_instances = [
            controller_class(self.node, self.n_joint, self.prefix)
            for controller_class in DEFAULT_CONTROLLER_CLASSES
        ]
        self.planner_action_client = PlannerActionClient(
            self.node,
            action_name="planner",
            on_result=self._on_planner_action_result,
        )
        self._accept_planner_results = True
        self.reference_pub = ReferenceTargetPublisher(self.node, self.prefix)
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.trajectory_path_publisher = self.node.create_publisher(Path, f'/{self.prefix}robotPath', qos_profile)

        self.mountPitch_publisher_ = self.node.create_publisher(Float32, '/alpha/cameraMountPitch', 10)
        self.light_publisher_ = self.node.create_publisher(Float32, '/alpha/lights', 10)

        self.vehicle_effort_command_publisher = self.node.create_publisher(
            DynamicInterfaceGroupValues,
            f"vehicle_effort_controller_{prefix}/commands",
            qos_profile
        )
        self.vehicle_pwm_command_publisher = self.node.create_publisher(
            Float64MultiArray,
            f'vehicle_thrusters_pwm_controller_{prefix}/commands',
            qos_profile
        )    
        self.manipulator_effort_command_publisher = self.node.create_publisher(
            Float64MultiArray,
            f"manipulation_effort_controller_{prefix}/commands",
            qos_profile
        )
        self.reset_sim_uvms_client = self.node.create_client(
            ResetSimUvms,
            f"/{self.prefix}reset_sim_uvms",
        )
        self.set_sim_dynamics_client = self.node.create_client(
            SetSimDynamics,
            f"/{self.prefix}set_sim_uvms_dynamics",
        )
        self.release_sim_uvms_client = self.node.create_client(
            Trigger,
            f"/{self.prefix}release_sim_uvms",
        )
        self.active_dynamics_profile = ""

        self.control_mode = ControlMode.TELEOP
        self._planner_output_enabled = False
        self._mode_before_replay = ControlMode.TELEOP
        self._replay_settle_controller = None
        self._replay_settle_started_sim_time = None
        self._replay_settle_config = {}
        self._replay_settle_arm_target = None
        self._replay_settle_vehicle_target = None
        self.sim_reset_hold = False
        if not self.node.has_parameter("cmd_replay_record_dir"):
            self.node.declare_parameter("cmd_replay_record_dir", "~/ros_ws/recordings/replay_sessions")
        self.cmd_replay_record_dir = FilePath(
            os.path.expanduser(str(self.node.get_parameter("cmd_replay_record_dir").value))
        )
        self._replay_record_handle = None
        self._replay_record_writer = None
        self._replay_record_path = None
        self._replay_record_controller = None
        self._replay_last_cmd_body_wrench = [0.0] * 6
        self._replay_last_cmd_arm_tau = [0.0] * 5
        self._replay_last_recorded_sim_time = None
        if not self.node.has_parameter("grasper_menu_open_effort"):
            self.node.declare_parameter("grasper_menu_open_effort", 1.0)
        if not self.node.has_parameter("grasper_menu_close_effort"):
            self.node.declare_parameter("grasper_menu_close_effort", -1.0)
        if not self.node.has_parameter("grasper_menu_effort_duration"):
            self.node.declare_parameter("grasper_menu_effort_duration", 1.0)
        self.grasper_menu_open_effort = float(self.node.get_parameter("grasper_menu_open_effort").value)
        self.grasper_menu_close_effort = float(self.node.get_parameter("grasper_menu_close_effort").value)
        self.grasper_menu_effort_duration = float(self.node.get_parameter("grasper_menu_effort_duration").value)
        self._menu_grasper_effort = 0.0
        self._menu_grasper_until_sec = 0.0

        # inverse IK tool axis and alignment weight CONFIGURATIONS
        self.ik_tool_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        self.ik_base_align_w = 0.0

        self.traj_path_poses = []
        self.max_traj_pose_count = 2000  # cap RViz message size
        self.path_publish_period = 0.1  # seconds between stored poses
        self._last_path_pub_time = None
        self._path_recording_enabled = True
        self.task_pose_in_world = None

        self.max_traj_vel = np.array([0.15, 0.15, 0.10], dtype=float)
        self.max_traj_acc = np.array([0.1, 0.1, 0.1], dtype=float)
        self.max_traj_jerk = np.array([0.05, 0.05, 0.05], dtype=float)

        self.node_name = node.get_name()
        # Search for joystick device in /dev/input
        device_interface = f"/dev/input/js{self.k_robot}"
        self.has_joystick_interface = False
        joystick_device = glob.glob(device_interface)

        if device_interface in joystick_device:
            self.node.get_logger().info(f"Found joystick device: {device_interface}")
            self.start_joystick(device_interface)
            self.has_joystick_interface = True
        else:
            self.node.get_logger().info(f"No joystick device found for robot {self.k_robot}.")
        self.robot_path_pub_timer = self.node.create_timer(1.0 / 60.0, self.publish_robot_path_callback)

        # Always visualize the plan if planner exists
        if self.planner is not None:
            self.planner_viz_timer = self.node.create_timer(1.0 / 60.0, self.planner_viz_callback)

        # Trajectory-related timers only if both exist
        if self.planner is not None and self.vehicle_cart_traj is not None:
            self.traj_sampler_timer = self.node.create_timer(1.0 / 60.0, self.com_trajectory_sampler_callback)
            self.trajectory_viz_timer = self.node.create_timer(1.0 / 60.0, self.trajectory_viz_callback)

        # one loop publishes
        self.control_loop_timer = self.node.create_timer(1.0 / 150.0, self.control_loop_callback)

        # joystick updates memory only (no publish inside)
        self.joystick_read_timer = self.node.create_timer(1.0/60.0, self.joystick_read_callback)

        # Define a threshold error at which we start yaw blending.
        self.pos_blend_threshold = 1.1
        self.world_task_pose_timer = self.node.create_timer(1.0 / 60.0, self.world_robot_task_pose_callback)

        # ---------------- Control Space ----------------
        self.control_space = ControlSpace.JOINT_SPACE

        # ---------------- Controller registry ----------------
        self._controllers: Dict[str, ControllerSpec] = {}

        # ---------------- Planner registry ----------------
        self._planners: Dict[str, PlannerSpec] = {}

        # Active function pointers used by control loop
        self.vehicle_controller_fn = None
        self.arm_controller_fn = None
        self.controller_name = None

        self.world_frame = 'world'

        for controller in self.controller_instances:
            self.register_controller(
                name=controller.registry_name,
                vehicle_fn=controller.vehicle_controller,
                arm_fn=controller.arm_controller,
            )

        # Bind the default controller without activating closed-loop publishing.
        self.set_controller("PID", activate=False)

        # Active path planner
        self.planner_name = None

        for planner_name in visible_planner_names():
            self.register_planner(name=planner_name)

        self.set_planner("Bitstar")

    def register_controller(
                self,
                name: str,
                vehicle_fn,
                arm_fn,
            ) -> None:
        self._controllers[name] = ControllerSpec(
            name=name,
            vehicle_fn=vehicle_fn,
            arm_fn=arm_fn,
        )

    def set_controller(self, name: str, activate: bool = True) -> bool:
        previous_controller = self.active_controller_instance()
        if activate and self.control_mode == ControlMode.REPLAY_SETTLE and name != "CmdReplay":
            self.cancel_replay_settle(mark_failed=True)
            self.control_mode = ControlMode.PLANNER
        spec = self._controllers[name]  # raises KeyError if missing, good fail-fast
        next_controller = getattr(spec.arm_fn, "__self__", None)
        self.vehicle_controller_fn = spec.vehicle_fn
        self.arm_controller_fn = spec.arm_fn
        self.controller_name = name
        if not activate:
            self.node.get_logger().info(f"Controller set to {name} for {self.prefix} (idle)")
            return True
        if self._active_controller_is_replay():
            self.set_control_mode(ControlMode.REPLAY)
        else:
            if previous_controller is not None and hasattr(previous_controller, "stop_playback"):
                previous_controller.stop_playback()
            self.set_control_mode(ControlMode.PLANNER)
        self.node.get_logger().info(f"Controller set to {name} for {self.prefix}")
        return True

    def active_controller_instance(self):
        return getattr(self.arm_controller_fn, "__self__", None)

    def controller_instance(self, name: str):
        spec = self._controllers.get(name)
        if spec is None:
            return None
        return getattr(spec.arm_fn, "__self__", None)

    def _controller_is_replay(self, controller) -> bool:
        return bool(controller is not None and hasattr(controller, "start_playback"))

    def _active_controller_is_replay(self) -> bool:
        return self._controller_is_replay(self.active_controller_instance())

    def _replay_is_active(self) -> bool:
        return self.control_mode in (ControlMode.REPLAY, ControlMode.REPLAY_SETTLE)

    def start_replay_controller_settle(self, replay_controller) -> None:
        feedback_controller_names = [name for name in self.list_controllers() if name != "CmdReplay"]
        if not feedback_controller_names:
            self.node.get_logger().error(f"No feedback controller available for replay settle on {self.prefix}.")
            replay_controller.mark_reset_failed()
            return

        self._replay_settle_config = replay_controller.controller_settle_config()
        requested_controller = str(self._replay_settle_config.get("controller", "PID"))
        if requested_controller in feedback_controller_names:
            settle_controller_name = requested_controller
        else:
            settle_controller_name = "PID" if "PID" in feedback_controller_names else feedback_controller_names[0]
            self.node.get_logger().warn(
                f"Replay settle controller '{requested_controller}' is not available for {self.prefix}; "
                f"using {settle_controller_name}."
            )
        spec = self._controllers[settle_controller_name]
        self.vehicle_controller_fn = spec.vehicle_fn
        self.arm_controller_fn = spec.arm_fn
        self.controller_name = settle_controller_name
        settle_controller = self.active_controller_instance()
        if settle_controller is not None and hasattr(settle_controller, "reset_controller_state"):
            settle_controller.reset_controller_state()
        self._replay_settle_controller = replay_controller
        self._replay_settle_started_sim_time = None
        self._replay_settle_arm_target = np.asarray(replay_controller.initial_manipulator_position(), dtype=float)
        self._replay_settle_vehicle_target = np.asarray(replay_controller.initial_vehicle_pose(), dtype=float)

        self.set_control_mode(ControlMode.REPLAY_SETTLE)
        self.arm.q_command = self._replay_settle_arm_target[:4].tolist()
        self.arm.dq_command = np.zeros((4,), dtype=float).tolist()
        self.arm.ddq_command = np.zeros((4,), dtype=float).tolist()
        self.pose_command = self._replay_settle_vehicle_target.tolist()
        self.body_vel_command = [0.0] * 6
        self.body_acc_command = [0.0] * 6
        self.node.get_logger().info(
            f"Replay controller settle started for {self.prefix} using {settle_controller_name}; "
            f"arm_pos_tol={self._replay_settle_config['position_tolerance']:.3f}, "
            f"arm_vel_tol={self._replay_settle_config['velocity_tolerance']:.3f}, "
            f"timeout={self._replay_settle_config['timeout_sec']:.1f}s."
        )

    def _clear_replay_settle(self) -> None:
        self._replay_settle_controller = None
        self._replay_settle_started_sim_time = None
        self._replay_settle_config = {}
        self._replay_settle_arm_target = None
        self._replay_settle_vehicle_target = None

    def cancel_replay_settle(self, mark_failed: bool = False) -> None:
        replay_controller = self._replay_settle_controller
        if mark_failed and replay_controller is not None and hasattr(replay_controller, "mark_reset_failed"):
            replay_controller.mark_reset_failed()
        self._clear_replay_settle()

    def _return_to_replay_after_settle_failure(self, replay_controller=None) -> None:
        spec = self._controllers.get("CmdReplay")
        if spec is not None:
            self.vehicle_controller_fn = spec.vehicle_fn
            self.arm_controller_fn = spec.arm_fn
            self.controller_name = "CmdReplay"
            self.set_control_mode(ControlMode.REPLAY)
        if replay_controller is not None and hasattr(replay_controller, "mark_reset_failed"):
            replay_controller.mark_reset_failed()

    def _replay_settle_is_done(self, state: Dict) -> tuple[bool, str]:
        cfg = self._replay_settle_config
        arm_target = self._replay_settle_arm_target
        vehicle_target = self._replay_settle_vehicle_target
        if arm_target is None or vehicle_target is None:
            return False, "missing settle target"

        q = np.asarray(list(state["q"]) + list(state["grasper_q"]), dtype=float)
        dq = np.asarray(list(state["dq"]) + list(state["grasper_qdot"]), dtype=float)
        pose = np.asarray(state["pose"], dtype=float)
        body_vel = np.asarray(state["body_vel"], dtype=float)

        arm_errors = np.abs(q[:4] - arm_target[:4])
        arm_position_error = float(np.max(arm_errors))
        arm_error_index = int(np.argmax(arm_errors))
        arm_velocity = float(np.max(np.abs(dq[:4])))
        vehicle_position_error = float(np.linalg.norm(pose[:3] - vehicle_target[:3]))
        vehicle_velocity = float(np.linalg.norm(body_vel[:3]))

        settled = (
            arm_position_error <= cfg["position_tolerance"]
            and arm_velocity <= cfg["velocity_tolerance"]
            and vehicle_position_error <= cfg["vehicle_position_tolerance"]
            and vehicle_velocity <= cfg["vehicle_velocity_tolerance"]
        )
        detail = (
            f"arm_err={arm_position_error:.4f} at joint {arm_error_index}, "
            f"arm_errors={np.round(arm_errors, 4).tolist()}, arm_vel={arm_velocity:.4f}, "
            f"vehicle_err={vehicle_position_error:.4f}, vehicle_vel={vehicle_velocity:.4f}"
        )
        return settled, detail

    def list_controllers(self) -> list:
        return sorted(list(self._controllers.keys()))
    
    def register_planner(
            self,
            name: str,
            ) -> None:
        self._planners[name] = PlannerSpec(
            name=name
        )
    
    def set_planner(self, name: str) -> None:
        spec = self._planners[name]  # raises KeyError if missing, good fail-fast
        self.planner_name = spec.name
        self.node.get_logger().info(f"Planner set to {name} for {self.prefix}")

    def list_planners(self) -> list:
        return sorted(list(self._planners.keys()))

    def set_control_space(self, control_space_name: str) -> None:
        self.control_space = control_space_name
        self.task_based_controller = (control_space_name == ControlSpace.TASK_SPACE)

    def list_control_spaces(self) -> list:
        return [m.value for m in ControlSpace]
    
    @classmethod
    def uvms_Forward_kinematics(cls, joint_qx, base_T0, world_pose, tipOffset):
        return cls.fk_eval_cls(joint_qx, base_T0, world_pose, tipOffset)

    @classmethod
    def manipulator_inverse_kinematics(cls, target_position):
        return cls.ik_eval_cls(target_position).full().flatten().tolist()
    
    @classmethod
    def manipulator_whole_body_inverse_kinematics(
        cls,
        q,
        world_pose,
        kp,
        p_des,
        w_rp,
        w_reg,
        k_rp,
        a_des_x, a_des_z,
        k_axis,
        w_axis,
        w_align, k_align,
        dt,
        base_T0,
        tipOffset,
    ):
        x_world_next, q_next, e_p_task_star_new, e_axis_task_star_new = cls.ik_wb_eval_cls(
            q,
            world_pose,
            kp,
            p_des,
            w_rp,
            w_reg,
            k_rp,
            a_des_x, a_des_z,
            k_axis,
            w_axis,
            w_align, k_align,
            dt,
            base_T0,
            tipOffset,
        )
        return x_world_next, q_next, e_p_task_star_new, e_axis_task_star_new

    def _mocap_pose_cb(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        # Order matches your CSV header: x, y, z, qw, qx, qy, qz
        self.mocap_latest = [float(p.x), float(p.y), float(p.z),
                            float(q.w), float(q.x), float(q.y), float(q.z)]


    def set_final_goal_in_world(self, goal_xyz_world_nwu, goal_quat_world_wxyz) -> None:
        self.final_goal_in_world = (goal_xyz_world_nwu, goal_quat_world_wxyz)

        res_map_ned = self.world_nwu_to_map_ned(
            xyz_world_nwu=goal_xyz_world_nwu,
            quat_world_wxyz=goal_quat_world_wxyz,
            warn_context=f"final_goal world->map ({self.prefix})",
        )
        if res_map_ned is None:
            self.final_goal_in_map_ned = None
            return

        p_goal_ned, rpy_goal_ned = res_map_ned
        # store goal in the same 6D format as your state['pose'] (NED euler_xyz)
        self.final_goal_in_map_ned = (
            np.asarray([p_goal_ned[0], p_goal_ned[1], p_goal_ned[2]], dtype=float),
            np.asarray([rpy_goal_ned[0], rpy_goal_ned[1], rpy_goal_ned[2]], dtype=float),
        )

    def compute_errors(self):
        st = self.get_state()

        X_curr = np.asarray(st["pose"], dtype=float)          # 6D NED in map frame
        X_wp_des = np.asarray(self.pose_command, dtype=float) # 6D NED in map frame

        err_wp = X_wp_des - X_curr
        err_wp_trans = np.abs(err_wp[:3])
        err_wp_rotation = np.abs(err_wp[3:])

        # Goal error (only if goal exists)
        if self.final_goal_map_ned_6 is None:
            err_goal_trans = np.zeros(3)
            err_goal_rotation = np.zeros(3)
        else:
            X_goal_des = np.asarray(self.final_goal_map_ned_6, dtype=float)
            err_goal = X_goal_des - X_curr
            err_goal_trans = np.abs(err_goal[:3])
            err_goal_rotation = np.abs(err_goal[3:])

        q_curr = np.asarray(st["q"], dtype=float).tolist()
        q_des  = np.asarray(self.arm.q_command, dtype=float).tolist()
        err_joints = [np.abs((Xd - Xc)) for Xd, Xc in zip(q_des, q_curr)]

        return err_wp_trans, err_wp_rotation, err_joints, err_goal_trans, err_goal_rotation


    def start_joystick(self, device_interface):
        # Shared variables updated by the PS4 controller callbacks.
        self.controller_lock = threading.Lock()
        self.rov_surge = 0.0      # Left stick horizontal (sway)
        self.rov_sway = 0.0      # Left stick vertical (surge)
        self.rov_z = 0.0      # Heave from triggers
        self.rov_roll = 0.0   # roll
        self.rov_pitch = 0.0  # Right stick vertical (pitch)
        self.rov_yaw = 0.0    # Right stick horizontal (yaw)

        self.jointe = 0.0
        self.jointd = 0.0
        self.jointc = 0.0
        self.jointb = 0.0
        self.jointa = 0.0

        # Instantiate the PS4 controller.
        # If you are not receiving analog stick events, try adjusting the event_format.
        self.ps4_controller = PS4Controller(
            ros_node=self,
            prefix=self.prefix,
            interface=device_interface,
            connecting_using_ds4drv=False,
            event_format="3Bh2b"  # Try "LhBB" if you experience mapping issues.
        )
        # Enable debug mode to print raw event data.
        self.ps4_controller.debug = True

        # Start the PS4 controller listener in a separate (daemon) thread.
        self.controller_thread = threading.Thread(target=self.ps4_controller.listen, daemon=True)
        self.controller_thread.start()

        self.node.get_logger().info(f"PS4 Teleop node initialized for robot {self.k_robot} to be control with js{self.k_robot}.")


    def update_state(self, msg: DynamicJointState):
        self.arm.update_state(msg)
        self.ned_pose = self.get_interface_value(
            msg,
            [self.floating_base_IOs] * 6,
            [
                Axis_Interface_names.floating_base_x,
                Axis_Interface_names.floating_base_y,
                Axis_Interface_names.floating_base_z,
                Axis_Interface_names.floating_base_roll,
                Axis_Interface_names.floating_base_pitch,
                Axis_Interface_names.floating_base_yaw
            ]
        )


        self.body_vel = self.get_interface_value(
            msg,
            [self.floating_base_IOs] * 6,
            [
                Axis_Interface_names.floating_dx,
                Axis_Interface_names.floating_dy,
                Axis_Interface_names.floating_dz,
                Axis_Interface_names.floating_roll_vel,
                Axis_Interface_names.floating_pitch_vel,
                Axis_Interface_names.floating_yaw_vel
            ]
        )

        self.body_acc = self.get_interface_value(
            msg,
            [self.floating_base_IOs] * 6,
            [
                Axis_Interface_names.floating_du,
                Axis_Interface_names.floating_dv,
                Axis_Interface_names.floating_dw,
                Axis_Interface_names.floating_dp,
                Axis_Interface_names.floating_dq,
                Axis_Interface_names.floating_dr
            ]
        )

        self.J_UV = self.vehicle_J(self.ned_pose[3:6]).full()

        self.ned_vel = self.to_ned_velocity(self.J_UV, self.body_vel)

        self.body_forces = self.get_interface_value(
            msg,
            [self.floating_base_IOs] * 6,
            [
            Axis_Interface_names.floating_force_x,
            Axis_Interface_names.floating_force_y, 
            Axis_Interface_names.floating_force_z,
            Axis_Interface_names.floating_torque_x,
            Axis_Interface_names.floating_torque_y,
            Axis_Interface_names.floating_torque_z
            ]
        )

        self.vehicle_control_power = self.get_interface_value(
            msg,
            [self.floating_base_IOs],
            [Axis_Interface_names.floating_control_power_abs]
        )[0]
        self.vehicle_control_energy = self.get_interface_value(
            msg,
            [self.floating_base_IOs],
            [Axis_Interface_names.floating_control_energy_abs]
        )[0]
        self.arm_control_power = self.get_interface_value(
            msg,
            [self.arm_IOs],
            [Axis_Interface_names.arm_control_power_abs]
        )[0]
        self.arm_control_energy = self.get_interface_value(
            msg,
            [self.arm_IOs],
            [Axis_Interface_names.arm_control_energy_abs]
        )[0]
        self.arm_payload_mass = self.get_interface_value(
            msg,
            [self.arm_IOs],
            [Axis_Interface_names.arm_payload_mass]
        )[0]
        self.arm_gravity = self.get_interface_value(
            msg,
            [self.arm_IOs],
            [Axis_Interface_names.arm_gravity]
        )[0]
   
        dynamics_sim_time = self.get_interface_value(msg,[self.floating_base_IOs],[Axis_Interface_names.sim_time])[0]
        if self.status == 'inactive':
            self.start_time = copy.copy(dynamics_sim_time)
            self.status = 'active'
        elif self.status == 'active':
            self.sim_time = dynamics_sim_time - self.start_time

    def get_state(self) -> Dict:
        xq = self.arm.get_state()
        xq['name'] = self.prefix
        xq['pose'] = self.ned_pose
        xq['body_vel'] = self.body_vel
        xq['body_acc'] = self.body_acc
        xq['ned_vel'] = self.ned_vel
        xq['body_forces'] = self.body_forces
        xq['vehicle_control_power_abs'] = self.vehicle_control_power
        xq['vehicle_control_energy_abs'] = self.vehicle_control_energy
        xq['arm_control_power_abs'] = self.arm_control_power
        xq['arm_control_energy_abs'] = self.arm_control_energy
        xq['arm_payload_mass'] = self.arm_payload_mass
        xq['arm_gravity'] = self.arm_gravity
        xq['status'] = self.status
        xq['sim_time'] = self.sim_time
        xq['prefix'] = self.prefix
        xq['mocap'] = self.mocap_latest
        return xq

    def get_energy_metrics(self) -> Dict:
        return {
            'prefix': self.prefix,
            'vehicle_control_power_abs': float(self.vehicle_control_power),
            'vehicle_control_energy_abs': float(self.vehicle_control_energy),
            'arm_control_power_abs': float(self.arm_control_power),
            'arm_control_energy_abs': float(self.arm_control_energy),
            'arm_payload_mass': float(self.arm_payload_mass),
            'arm_gravity': float(self.arm_gravity),
        }

    def _reference_trajectory_direction_ned(self) -> np.ndarray:
        body_vel_ref = ControllerPerformanceMetrics.fixed_array(self.body_vel_command, 6)
        try:
            pose_ref = ControllerPerformanceMetrics.fixed_array(self.pose_command, 6)
            jacobian = self.vehicle_J(pose_ref[3:6]).full()
            ned_vel_ref = np.asarray(
                self.to_ned_velocity(jacobian, body_vel_ref),
                dtype=float,
            ).reshape(-1)
            return ControllerPerformanceMetrics.fixed_array(ned_vel_ref, 6)[:3]
        except Exception:
            return np.zeros(3, dtype=float)

    def _update_controller_performance_metrics(self) -> Dict:
        state = self.get_state()
        active = (
            state.get("status") == "active"
            and not self.sim_reset_hold
            and self.control_mode in (ControlMode.PLANNER, ControlMode.REPLAY, ControlMode.REPLAY_SETTLE)
        )
        reference = {
            "pose": self.pose_command,
            "body_vel": self.body_vel_command,
            "body_acc": self.body_acc_command,
            "q": self.arm.q_command,
            "dq": self.arm.dq_command,
            "ddq": self.arm.ddq_command,
            "trajectory_direction": self._reference_trajectory_direction_ned(),
        }
        return self.performance_metrics.update(
            current=state,
            reference=reference,
            active=active,
        )

    def get_controller_performance_metrics(self) -> Dict:
        return self.performance_metrics.snapshot()

    def _publish_controller_performance_metrics(self) -> None:
        metrics = self._update_controller_performance_metrics()
        msg = ControllerPerformance()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.robot_prefix = self.prefix
        msg.active = bool(metrics.get("active", 0.0))
        msg.vehicle_cross_track_m = float(metrics.get("vehicle_cross_track_m", 0.0))
        msg.vehicle_along_track_m = float(metrics.get("vehicle_along_track_m", 0.0))
        msg.vehicle_n_cross_track = float(metrics.get("vehicle_n_cross_track", 0.0))
        msg.vehicle_n_along_track = float(metrics.get("vehicle_n_along_track", 0.0))
        msg.vehicle_n_position = float(metrics.get("vehicle_n_position", 0.0))
        msg.vehicle_n_attitude = float(metrics.get("vehicle_n_attitude", 0.0))
        msg.vehicle_n_linear_velocity = float(metrics.get("vehicle_n_linear_velocity", 0.0))
        msg.vehicle_n_angular_velocity = float(metrics.get("vehicle_n_angular_velocity", 0.0))
        msg.vehicle_n_linear_acceleration = float(metrics.get("vehicle_n_linear_acceleration", 0.0))
        msg.vehicle_n_angular_acceleration = float(metrics.get("vehicle_n_angular_acceleration", 0.0))
        msg.arm_n_position = float(metrics.get("arm_n_position", 0.0))
        msg.arm_n_velocity = float(metrics.get("arm_n_velocity", 0.0))
        msg.arm_n_acceleration = float(metrics.get("arm_n_acceleration", 0.0))
        msg.tracking_score = float(metrics.get("tracking_score", 0.0))
        msg.tracking_score_rms = float(metrics.get("tracking_score_rms", 0.0))
        msg.normalized_control_effort = float(metrics.get("normalized_control_effort", 0.0))
        msg.effort_per_tracking_score = float(metrics.get("effort_per_tracking_score", 0.0))
        msg.energy_per_meter = float(metrics.get("energy_per_meter", 0.0))
        msg.energy_per_second = float(metrics.get("energy_per_second", 0.0))
        msg.time_to_tolerance_sec = float(metrics.get("time_to_tolerance_sec", 0.0))
        msg.peak_tracking_score = float(metrics.get("peak_tracking_score", 0.0))
        msg.sample_count = float(metrics.get("sample_count", 0.0))
        self.performance_publisher.publish(msg)

    def try_transform_pose(
        self,
        pose_in_source: Pose,
        target_frame: str,
        source_frame: str,
        *,
        warn_context: str = "",
    ) -> Optional[Pose]:
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
            )
        except TransformException as ex:
            msg = f"TF not ready: {target_frame} <- {source_frame}, {ex}"
            if warn_context:
                msg = f"{warn_context}, {msg}"
            self.node.get_logger().warn(msg)
            return None

        return do_transform_pose(pose_in_source, tf)

    def get_frame_pose_in_frame(self, source_frame: str, target_frame: str) -> Optional[Pose]:
        """
        Return the pose of source_frame expressed in target_frame, as geometry_msgs/Pose.
        Uses your existing try_transform_pose helper.
        """
        identity = Pose()
        identity.position.x = 0.0
        identity.position.y = 0.0
        identity.position.z = 0.0
        identity.orientation.w = 1.0
        identity.orientation.x = 0.0
        identity.orientation.y = 0.0
        identity.orientation.z = 0.0

        return self.try_transform_pose(
            pose_in_source=identity,
            target_frame=target_frame,
            source_frame=source_frame,
            warn_context=f"get_frame_pose_in_frame({self.prefix})",
        )

    def _pose_from_state_in_frame(self, dst_frame: str) -> Optional[Pose]:
        """
        Returns the robot base pose expressed in dst_frame, or None if TF is unavailable.
        Source pose is constructed from the robot NED state, expressed in self.map_frame.
        """
        # Build Pose in the source frame that TF actually knows about: self.map_frame
        # PoseX: NED (internal) -> NWU (ROS-ish), and we treat that as being in map_frame.
        pose_src = PoseX.from_pose(
            xyz=np.array(self.ned_pose[0:3], float),
            rot=np.array(self.ned_pose[3:6], float),
            rot_rep="euler_xyz",
            frame="NED",
        ).get_pose_as_Pose_msg(frame="NWU")

        # Use shared helper for TF lookup + transform + logging
        pose_dst = self.try_transform_pose(
            pose_in_source=pose_src,
            target_frame=dst_frame,
            source_frame=self.map_frame,
            warn_context=f"_pose_from_state_in_frame({self.prefix})",
        )
        return pose_dst

    def _pose_msg_from_xyz_quat_wxyz_nwu(
        self,
        xyz: Sequence[float],
        quat_wxyz: Sequence[float],
    ) -> Pose:
        """Build geometry_msgs/Pose from NWU xyz and quaternion (wxyz)."""
        p = Pose()
        p.position.x = float(xyz[0])
        p.position.y = float(xyz[1])
        p.position.z = float(xyz[2])
        p.orientation.w = float(quat_wxyz[0])
        p.orientation.x = float(quat_wxyz[1])
        p.orientation.y = float(quat_wxyz[2])
        p.orientation.z = float(quat_wxyz[3])
        return p

    def world_nwu_to_map_ned(
        self,
        xyz_world_nwu: Sequence[float],
        quat_world_wxyz: Sequence[float],
        *,
        warn_context: str = "",
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Convert a pose given in 'world' frame (NWU) into (map-frame) NED pose.

        Returns:
        (p_cmd_ned, rpy_cmd_ned) where
            p_cmd_ned: (3,) np.ndarray
            rpy_cmd_ned: (3,) np.ndarray in euler_xyz
        Returns None if TF is not ready.
        """
        # 1) world pose (NWU) as geometry_msgs/Pose
        world_pose_nwu = self._pose_msg_from_xyz_quat_wxyz_nwu(
            xyz_world_nwu,
            quat_world_wxyz,
        )

        # 2) TF: world -> map_frame (still NWU representation, just different frame)
        map_pose_nwu = self.try_transform_pose(
            pose_in_source=world_pose_nwu,
            target_frame=self.map_frame,
            source_frame=self.world_frame,
            warn_context=warn_context or f"world_nwu_to_map_ned({self.prefix})",
        )
        if map_pose_nwu is None:
            return None

        # 3) Convert NWU pose message in map_frame to NED (p, rpy)
        p_cmd_ned, rpy_cmd_ned = PoseX.from_pose(
            xyz=np.array(
                [map_pose_nwu.position.x, map_pose_nwu.position.y, map_pose_nwu.position.z],
                dtype=float,
            ),
            rot=np.array(
                [
                    map_pose_nwu.orientation.w,
                    map_pose_nwu.orientation.x,
                    map_pose_nwu.orientation.y,
                    map_pose_nwu.orientation.z,
                ],
                dtype=float,
            ),
            rot_rep="quat_wxyz",
            frame="NWU",
        ).get_pose(frame="NED", rot_rep="euler_xyz")

        return np.asarray(p_cmd_ned, dtype=float), np.asarray(rpy_cmd_ned, dtype=float)

    def world_nwu_vect6_to_map_ned_vect6(
        self,
        vect6_world_nwu: Sequence[float],  # [vx, vy, vz, wx, wy, wz] in world NWU
        *,
        warn_context: str = "",
    ) -> Optional[np.ndarray]:
        """
        Convert a 6D twist given in 'world' frame (NWU) into a 6D twist in map frame (NED).

        Input ordering:  [vx, vy, vz, wx, wy, wz]
        Output ordering: [vx, vy, vz, wx, wy, wz] in NED components, expressed in map_frame.
        """

        # Lookup transform map <- world
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.world_frame,
                rclpy.time.Time(),
            )
        except TransformException as ex:
            msg = f"TF not ready: {self.map_frame} <- world, {ex}"
            if warn_context:
                msg = f"{warn_context}, {msg}"
            self.node.get_logger().warn(msg)
            return None

        def _xform_vec3_world_to_map(vec3_world_nwu: Sequence[float]) -> np.ndarray:
            vmsg = Vector3Stamped()
            vmsg.header.frame_id = self.world_frame
            vmsg.header.stamp = self.node.get_clock().now().to_msg()
            vmsg.vector.x = float(vec3_world_nwu[0])
            vmsg.vector.y = float(vec3_world_nwu[1])
            vmsg.vector.z = float(vec3_world_nwu[2])

            v_map = do_transform_vector3(vmsg, tf)
            return np.asarray([v_map.vector.x, v_map.vector.y, v_map.vector.z], dtype=float)

        # 1) Rotate linear and angular parts into map_frame (still NWU axes)
        v_map_nwu = _xform_vec3_world_to_map(vect6_world_nwu[0:3])
        w_map_nwu = _xform_vec3_world_to_map(vect6_world_nwu[3:6])

        # 2) Convert NWU components -> NED components (same mapping for v and w)
        v_map_ned = np.asarray([v_map_nwu[0], -v_map_nwu[1], -v_map_nwu[2]], dtype=float)
        w_map_ned = np.asarray([w_map_nwu[0], -w_map_nwu[1], -w_map_nwu[2]], dtype=float)

        return np.concatenate([v_map_ned, w_map_ned])

    def to_ned_velocity(self, J_uv, body_vel):
        ned_velocity = J_uv@body_vel
        return ned_velocity
    
    def to_body_velocity(self, J_uv, eul, ned_vel):
        body_velocity = np.linalg.inv(J_uv)@ned_vel
        body_velocity = self.vehicle_ned2body_vel(eul, ned_vel)
        return body_velocity
    
    def to_body_acceleration(self, eul, ned_vel, ned_acc, v_c):
        body_acc = self.vehicle_ned2body_acc(eul, ned_vel, ned_acc, v_c)
        return body_acc

    def publish_robot_path_callback(self):
        if not self._path_recording_enabled:
            return

        # Publish the robot trajectory path to RViz
        now_msg = self.node.get_clock().now().to_msg()
        stamp_time = now_msg.sec + now_msg.nanosec * 1e-9
        if (
            self._last_path_pub_time is not None
            and (stamp_time - self._last_path_pub_time) < self.path_publish_period
        ):
            return
        self._last_path_pub_time = stamp_time

        tra_path_msg = Path()
        tra_path_msg.header.stamp = now_msg
        tra_path_msg.header.frame_id = self.map_frame

        # Create PoseStamped from ref_pos
        traj_pose = PoseStamped()
        traj_pose.header = tra_path_msg.header
        traj_pose.pose.position.x = float(self.ned_pose[0])
        traj_pose.pose.position.y = -float(self.ned_pose[1])
        traj_pose.pose.position.z = -float(self.ned_pose[2])
        traj_pose.pose.orientation.w = 1.0  # No rotation

        # Accumulate poses
        self.traj_path_poses.append(traj_pose)
        if self.max_traj_pose_count > 0 and len(self.traj_path_poses) > self.max_traj_pose_count:
            # Keep only the most recent poses to avoid timer overruns
            self.traj_path_poses = self.traj_path_poses[-self.max_traj_pose_count:]
        tra_path_msg.poses = self.traj_path_poses

        self.trajectory_path_publisher.publish(tra_path_msg)


    def orient_towards_velocity(self, speed_threshold: float = 0.03):
        """
        Return a yaw that points along the current horizontal velocity.
        If the vehicle is moving slower than speed_threshold, do not change yaw.
        """
        vx = float(self.ned_vel[0])
        vy = float(self.ned_vel[1])

        # Only use translational velocity here
        linear_speed = np.hypot(vx, vy)

        current_yaw = float(self.ned_pose[5])

        # If we are basically not translating, keep current yaw
        if linear_speed < speed_threshold:
            return current_yaw

        # Otherwise compute the yaw that faces the velocity direction
        desired_yaw = np.arctan2(vy, vx)

        # Smooth shortest path from current to desired
        return self.normalize_angle(desired_yaw, current_yaw)

    def normalize_angle(self, desired_yaw, current_yaw):
        # Compute the smallest angular difference
        angle_diff = desired_yaw - current_yaw
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi  # Normalize to (-π, π)

        # Adjust desired_yaw to ensure the shortest rotation path
        adjusted_desired_yaw = current_yaw + angle_diff
        return adjusted_desired_yaw

    def publish_vehicle_and_arm(
        self,
        wrench_body_6: Sequence[float],
        arm_effort_5: Sequence[float],
    ) -> None:
        container = FullRobotMsg(prefix=self.prefix)
        container.set_vehicle_wrench(wrench_body_6)
        container.set_arm_effort(arm_effort_5)

        veh_msg = container.to_vehicle_dynamic_group(self.node.get_clock().now().to_msg())
        arm_msg = container.to_arm_effort_array()

        self.vehicle_effort_command_publisher.publish(veh_msg)
        self.manipulator_effort_command_publisher.publish(arm_msg)
        self._publish_controller_performance_metrics()
    
    def _now_sec(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _clear_menu_grasper_effort(self) -> None:
        if hasattr(self, "controller_lock"):
            with self.controller_lock:
                self._menu_grasper_effort = 0.0
                self._menu_grasper_until_sec = 0.0
        else:
            self._menu_grasper_effort = 0.0
            self._menu_grasper_until_sec = 0.0

    def command_grasper_from_menu(self, action: str) -> bool:
        if self.control_mode != ControlMode.PLANNER or self._active_controller_is_replay():
            self._clear_menu_grasper_effort()
            self.node.get_logger().warn(
                f"Grasper menu {action} rejected for {self.prefix}; "
                "select a feedback controller such as PID or InvDyn first."
            )
            return False

        if action == "open":
            self.arm.open_grasper()
            effort = self.grasper_menu_open_effort
        elif action == "close":
            self.arm.close_grasper()
            effort = self.grasper_menu_close_effort
        else:
            return False

        self.enable_planner_output()
        until_sec = self._now_sec() + max(0.0, self.grasper_menu_effort_duration)
        if hasattr(self, "controller_lock"):
            with self.controller_lock:
                self._menu_grasper_effort = effort
                self._menu_grasper_until_sec = until_sec
        else:
            self._menu_grasper_effort = effort
            self._menu_grasper_until_sec = until_sec
        self.node.get_logger().info(
            f"Grasper menu {action} requested for {self.prefix}; "
            f"applying axis_a effort {effort:.3f} for {self.grasper_menu_effort_duration:.2f}s."
        )
        return True

    def _apply_menu_grasper_effort(self, arm_effort_5: Sequence[float]) -> list:
        arm_effort = list(arm_effort_5)
        if len(arm_effort) < 5:
            arm_effort.extend([0.0] * (5 - len(arm_effort)))
        if self.control_mode != ControlMode.PLANNER or self._active_controller_is_replay():
            self._clear_menu_grasper_effort()
            return arm_effort

        now_sec = self._now_sec()
        if hasattr(self, "controller_lock"):
            with self.controller_lock:
                effort = self._menu_grasper_effort
                active = now_sec < self._menu_grasper_until_sec
                if not active:
                    self._menu_grasper_effort = 0.0
                    self._menu_grasper_until_sec = 0.0
        else:
            effort = self._menu_grasper_effort
            active = now_sec < self._menu_grasper_until_sec
            if not active:
                self._menu_grasper_effort = 0.0
                self._menu_grasper_until_sec = 0.0

        if active:
            arm_effort[4] = effort
        return arm_effort

    # ForwardCommandController
    def publish_commands(self, wrench_body_6: Sequence[float], arm_effort_5: Sequence[float]):
        # Vehicle, DynamicInterfaceGroupValues payload
        self.publish_vehicle_and_arm(wrench_body_6, self._apply_menu_grasper_effort(arm_effort_5))

    def publish_vehicle_pwms(self,
                             pwm_thruster_8: Sequence[float]):
        container = FullRobotMsg(prefix=self.prefix)
        container.set_vehicle_pwm(pwm_thruster_8)
        vehicle_pwm = container.to_vehicle_pwm()
        self.vehicle_pwm_command_publisher.publish(vehicle_pwm)

    def listener_callback(self, msg: DynamicJointState):
        self.update_state(msg)

    def world_robot_task_pose_callback(self):
        pose_world = self.get_frame_pose_in_frame(self.joint4_frame, self.world_frame)
        if pose_world is None:
            return
        self.task_pose_in_world = pose_world

    def _pose_to_xyz_quat_wxyz(self, pose: Pose):
        xyz = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
        quat_wxyz = np.array(
            [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z],
            dtype=float
        )
        return xyz, quat_wxyz

    def _get_vehicle_goal_from_marker(self, goal_pose:Pose):
        goal_xyz = np.array([goal_pose.position.x, goal_pose.position.y, goal_pose.position.z], dtype=float)
        goal_quat_wxyz = np.array(
            [goal_pose.orientation.w, goal_pose.orientation.x, goal_pose.orientation.y, goal_pose.orientation.z],
            dtype=float
        )
        return goal_xyz, goal_quat_wxyz

    def _log_plan_context(self, start_xyz, start_quat_wxyz, goal_xyz, goal_quat_wxyz) -> None:
        self.node.get_logger().info(
            f"Planning for {self.prefix}, "
            f"start_xyz={np.array(start_xyz, float).round(3).tolist()}, "
            f"goal_xyz={np.array(goal_xyz, float).round(3).tolist()}"
        )

    def _save_vehicle_goal_from_target(self, goal_xyz, goal_quat_wxyz):
        goal_xyz_world_nwu, goal_quat_wxyz_world = goal_xyz, goal_quat_wxyz

        # Convert world (NWU) -> map (NED)
        res = self.world_nwu_to_map_ned(
            xyz_world_nwu=goal_xyz_world_nwu,
            quat_world_wxyz=goal_quat_wxyz_world,
            warn_context=f"save goal world->map ({self.prefix})",
        )
        if res is None:
            self.final_goal_map_ned_6 = None
            return

        p_goal_ned, rpy_goal_ned = res
        self.final_goal_map_ned_6 = np.array(
            [p_goal_ned[0], p_goal_ned[1], p_goal_ned[2],
            rpy_goal_ned[0], rpy_goal_ned[1], rpy_goal_ned[2]],
            dtype=float,
        )

    def _on_planner_action_result(self, plan_result: Dict[str, Any]) -> None:
        if self.sim_reset_hold or self._replay_is_active() or not self._accept_planner_results:
            if self.planner is not None:
                self.planner.planned_result = None
            if self.vehicle_cart_traj is not None:
                self.vehicle_cart_traj.active = False
            self.node.get_logger().info(
                f"Ignoring stale planner result for {self.prefix} after reset/stop."
            )
            return

        if self.planner is not None:
            self.planner.planned_result = dict(plan_result)

        if plan_result.get("is_success", False):
            self.node.get_logger().info(
                f"Planner action produced {int(plan_result.get('count', 0))} waypoints for {self.prefix}"
            )
    
            path_xyz = np.asarray(self.planner.planned_result["xyz"], dtype=float)
            self._start_vehicle_cartesian_ruckig(self.start_xyz, self.start_quat_wxyz, path_xyz)
            self.enable_planner_output()
        else:
            self.node.get_logger().warn(
                f"Planner action failed for {self.prefix}, "
                f"message='{plan_result.get('message', '')}'"
            )

    def _start_vehicle_cartesian_ruckig(self, start_xyz, start_quat_wxyz, path_xyz: np.ndarray) -> None:
        self.vehicle_cart_traj.start_from_path(
            current_position=list(start_xyz),
            path_xyz=path_xyz,
            max_vel=self.max_traj_vel,
            max_acc=self.max_traj_acc,
            max_jerk=self.max_traj_jerk,
        )

    def plan_vehicle_trajectory_action(
        self,
        goal_pose,
        *,
        time_limit: float = 1.0,
        robot_collision_radius: float = 0.4,
    ) -> bool:
        if self._replay_is_active():
            self.abrupt_planner_stop()
            self.node.get_logger().warn(
                f"Planner request ignored for {self.prefix}; CmdReplay is active."
            )
            return False
        if self.sim_reset_hold:
            self.node.get_logger().warn(
                f"Planner request ignored for {self.prefix}; simulation is held after reset."
            )
            return False
        self.node.get_logger().info(
            f"Planning motion with {self.planner_name} for {self.prefix} to target pose..."
        )
        pose_now = self._pose_from_state_in_frame(self.world_frame)
        if pose_now is None:
            self.node.get_logger().warn("Planner action request was not sent, current pose unavailable.")
            return False
        self.abrupt_planner_stop(publish_zero=False)
        self._accept_planner_results = True
        self.hold_current_state_with_feedback()

        self.start_xyz, self.start_quat_wxyz = self._pose_to_xyz_quat_wxyz(pose_now)

        goal_xyz, goal_quat_wxyz = self._get_vehicle_goal_from_marker(goal_pose)

        self._log_plan_context(self.start_xyz, self.start_quat_wxyz, goal_xyz, goal_quat_wxyz)

        self._save_vehicle_goal_from_target(goal_xyz, goal_quat_wxyz)

        sent = self.planner_action_client.send_goal(
            start_xyz=self.start_xyz,
            start_quat_wxyz=self.start_quat_wxyz,
            goal_xyz=goal_xyz,
            goal_quat_wxyz=goal_quat_wxyz,
            planner_name=self.planner_name,
            time_limit=float(time_limit),
            robot_collision_radius=float(robot_collision_radius)
        )

        if sent:
            self.node.get_logger().info(
                f"Submitted planner action request for {self.prefix}"
            )
        else:
            self.node.get_logger().warn("Planner action request was not sent.")
        return sent

    def planner_viz_callback(self):
        k_planner = self.planner
        if k_planner is None:
            return

        stamp_now = self.node.get_clock().now().to_msg()
        if self.sim_reset_hold:
            k_planner.clear_path(stamp_now, self.world_frame)
            return
        pr = None if k_planner.planned_result is None else dict(k_planner.planned_result)
        viz_plan = bool(pr and pr.get("is_success", False) and "xyz" in pr)

        if self.control_mode == ControlMode.PLANNER and viz_plan:
            k_planner.update_path_viz(
                stamp=stamp_now,
                frame_id=self.world_frame,
                xyz_np=pr["xyz"],
                step=3,
                wp_size=0.08,
                goal_size=0.14,
            )
        else:
            k_planner.clear_path(stamp_now, self.world_frame)


    def trajectory_viz_callback(self):
        k_trajectory = self.vehicle_cart_traj
        k_planner = self.planner
        if k_planner is None or k_trajectory is None:
            return

        stamp_now = self.node.get_clock().now().to_msg()
        if self.sim_reset_hold:
            k_planner.clear_target(stamp_now, self.world_frame)
            return
        if self.control_mode == ControlMode.PLANNER and k_trajectory.active:
            target_pose_nwu = np.asarray(list(k_trajectory.out.new_position), dtype=float)
            target_vel_nwu = np.asarray(list(k_trajectory.out.new_velocity), dtype=float)
            q_arrow = self.planner.quat_wxyz_from_x_to_vec_scipy(target_vel_nwu)
            self.planner.update_target_viz(
                stamp=stamp_now,
                frame_id=self.world_frame,
                xyz=target_pose_nwu,
                quat_wxyz=q_arrow,
                as_arrow=True,
                size=0.10,
                rate_hz=30.0,
                ttl_sec=0.0,
            )
        else:
            k_planner.clear_target(stamp_now, self.world_frame)

    def com_trajectory_sampler_callback(self):
        if self.planner is None or self.vehicle_cart_traj is None:
            return
        state = self.get_state()
        if state['status'] != 'active':
            return
        if self.sim_reset_hold:
            return
        if self.control_mode != ControlMode.PLANNER:
            return
        if self.task_based_controller:
            return
        if self.final_goal_map_ned_6 is not None and self.planner.planned_result and self.planner.planned_result['is_success']:
            self.node.get_logger().debug(f"Control timer callback {self.prefix} active.")
            # Convert once to NumPy arrays
            path_xyz = np.asarray(self.planner.planned_result["xyz"], dtype=float)
            path_quat = np.asarray(self.planner.planned_result["quat_wxyz"], dtype=float)

            # Compute current manifold errors
            wp_err_trans, wp_err_rot, wp_err_joint, goal_err_trans, goal_err_rot = self.compute_errors()
            goal_xyz_error = np.linalg.norm(goal_err_trans)

            # Calculate the blend factor.
            # When pos_error >= pos_blend_threshold, blend_factor will be 0 (full velocity_yaw).
            # When pos_error == 0, blend_factor will be 1 (full target_yaw).
            self.yaw_blend_factor = np.clip((self.pos_blend_threshold - goal_xyz_error) / self.pos_blend_threshold, 0.0, 1.0)
            # self.get_logger().info(
            #     f"{robot.yaw_blend_factor} yaw_blend_factor"
            # )
            # Get the velocity-based yaw.
            adjusted_yaw = self.orient_towards_velocity()

            pos_nwu, vel_nwu, acc_nwu, res = self.vehicle_cart_traj.update(self.yaw_blend_factor)
            if pos_nwu is None:
                return
            
            target_pose_nwu = np.asarray(pos_nwu, dtype=float)
            target_vel_nwu = np.asarray(vel_nwu, dtype=float)
            target_acc_nwu = np.asarray(acc_nwu, dtype=float)

            # Pick orientation from nearest OMPL waypoint
            dists = np.linalg.norm(path_xyz - target_pose_nwu, axis=1)
            idx = int(np.argmin(dists))
            target_quat = path_quat[idx]

            stamp_now = self.node.get_clock().now().to_msg()
            self.reference_pub.publish_world_targets(
                stamp_msg=stamp_now,
                xyz_world_nwu=target_pose_nwu,
                quat_world_wxyz=target_quat,
                vel_world_nwu=target_vel_nwu,
                acc_world_nwu=target_acc_nwu,
            )


            target_pose_map_ned = self.world_nwu_to_map_ned(
                xyz_world_nwu=target_pose_nwu,
                quat_world_wxyz=target_quat,
                warn_context=f"target world->map ({self.prefix})",
            )

            tw6_world_nwu = np.zeros(6, dtype=float)
            tw6_world_nwu[0:3] = target_vel_nwu          # linear, keep angular vel zeros

            tw6_map_ned = self.world_nwu_vect6_to_map_ned_vect6(
                tw6_world_nwu,
                warn_context=f"target twist world->map ({self.prefix})",
            )

            acc6_world_nwu = np.zeros(6, dtype=float)
            acc6_world_nwu[0:3] = acc_nwu
            acc6_map_ned = self.world_nwu_vect6_to_map_ned_vect6(
                acc6_world_nwu,
                warn_context=f"target accel world->map ({self.prefix})",
            )
            
            if target_pose_map_ned is not None and tw6_map_ned is not None and acc6_map_ned is not None:
                p_cmd_ned, rpy_cmd_ned = target_pose_map_ned

                # Blend on the unwrapped shortest yaw arc. Directly averaging
                # angles near +/-pi can command a full turn through zero.
                target_yaw = self.normalize_angle(float(rpy_cmd_ned[2]), adjusted_yaw)
                rpy_cmd_ned[2] = adjusted_yaw + self.yaw_blend_factor * (target_yaw - adjusted_yaw)

                cmd_J_UV = self.vehicle_J(rpy_cmd_ned).full()
                self.node.get_logger().debug(f"v_cmd_ned {tw6_map_ned} : active.")

                body_vel_command = self.to_body_velocity(cmd_J_UV, rpy_cmd_ned, tw6_map_ned)
                self.node.get_logger().debug(f"body_vel_command {body_vel_command} : active.")

                body_acc_command = self.to_body_acceleration(rpy_cmd_ned, tw6_map_ned, acc6_map_ned, self.v_c)
                self.node.get_logger().debug(f"body_acc_command {acc6_map_ned} : active.")

                self.pose_command = [
                    float(p_cmd_ned[0]),
                    float(p_cmd_ned[1]),
                    float(p_cmd_ned[2]),
                    float(rpy_cmd_ned[0]),
                    float(rpy_cmd_ned[1]),
                    float(rpy_cmd_ned[2]),
                ]

                self.body_vel_command = [
                    float(body_vel_command[0]),
                    float(body_vel_command[1]),
                    float(body_vel_command[2]),
                    float(body_vel_command[3]),
                    float(body_vel_command[4]),
                    float(body_vel_command[5]),
                ]

                self.body_acc_command = [
                    float(body_acc_command[0]),
                    float(body_acc_command[1]),
                    float(body_acc_command[2]),
                    float(body_acc_command[3]),
                    float(body_acc_command[4]),
                    float(body_acc_command[5]),
                ]
        
    def solve_inverse_kinematics_wrt_world_frame(self, target_world_endeffector_pose: Pose):
        world_pose_now = self._pose_from_state_in_frame(self.world_frame)
        if world_pose_now is None:
            self.node.get_logger().warn(f"IK aborted, current world-frame vehicle pose is unavailable.")
            return

        state = self.get_state()
        q = np.asarray(state.get("q", []), dtype=float).reshape(-1)
        if q.size == 0:
            self.node.get_logger().warn(f"IK aborted, manipulator joint state vector is empty.")
            return

        p_now, rpy_now = PoseX.from_pose(
            xyz=np.array(
                [
                    world_pose_now.position.x,
                    world_pose_now.position.y,
                    world_pose_now.position.z,
                ],
                dtype=float,
            ),
            rot=np.array(
                [
                    world_pose_now.orientation.w,
                    world_pose_now.orientation.x,
                    world_pose_now.orientation.y,
                    world_pose_now.orientation.z,
                ],
                dtype=float,
            ),
            rot_rep="quat_wxyz",
            frame="NWU",
        ).get_pose(frame="NWU", rot_rep="euler_xyz")

        world_pose = np.concatenate([p_now, rpy_now]).astype(float)

        p_des = np.array(
            [
                target_world_endeffector_pose.position.x,
                target_world_endeffector_pose.position.y,
                target_world_endeffector_pose.position.z,
            ],
            dtype=float,
        )
        
        w_rp = 1.0
        w_reg = 0.02
        w_axis = 1.5
        w_align = float(self.ik_base_align_w)

        kp = np.array([1.0, 1.0, 1.0], dtype=float)
        k_rp = 0.2
        k_axis = 1.0
        k_align = 1.0
        
        dt = 1.0 / 500.0

        tool_axis_z_align = np.asarray(self.ik_tool_axis, dtype=float).reshape(3)
        tool_axis_x_align = np.array([1.0, 0.0, 0.0])
        target_rot = PoseX.from_pose(
            xyz=np.array(
                [
                    target_world_endeffector_pose.position.x,
                    target_world_endeffector_pose.position.y,
                    target_world_endeffector_pose.position.z,
                ],
                dtype=float,
            ),
            rot=np.array(
                [
                    target_world_endeffector_pose.orientation.w,
                    target_world_endeffector_pose.orientation.x,
                    target_world_endeffector_pose.orientation.y,
                    target_world_endeffector_pose.orientation.z,
                ],
                dtype=float,
            ),
            rot_rep="quat_wxyz",
            frame="NWU",
        ).get_rot(frame="NWU", rot_rep="matrix")
        a_des_z = target_rot @ tool_axis_z_align
        a_des_x = target_rot @ tool_axis_x_align
        x_world_next, q_next, e_p_task_star_new, e_axis_task_star_new = self.manipulator_whole_body_inverse_kinematics(
            q,
            world_pose,
            kp,
            p_des,
            w_rp,
            w_reg,
            k_rp,
            a_des_x, a_des_z,
            k_axis,
            w_axis,
            w_align, k_align,
            dt,
            np.asarray(ReachParams.base_T0_new, dtype=float),
            np.asarray(ReachParams.tipOffset, dtype=float),
        )

        def _to_1d(arr):
            if hasattr(arr, "full"):
                arr = arr.full()
            return np.asarray(arr, dtype=float).reshape(-1)

        x_world_next = _to_1d(x_world_next)
        q_next = _to_1d(q_next)
        e_p_task_star_new = _to_1d(e_p_task_star_new)
        e_axis_task_star_new = _to_1d(e_axis_task_star_new)

        if q_next.size != q.size:
            self.node.get_logger().warn(
                f"IK result invalid, joint vector size mismatch "
                f"(expected {q.size}, got {q_next.size})."
            )
            return 
        self.arm.q_command = q_next.tolist()

        if x_world_next.size != 6:
            self.node.get_logger().warn(
                f"IK result invalid, world pose vector must have size 6 "
                f"(got {x_world_next.size})."
            )
            return 
        pose_next = PoseX.from_pose(
            xyz=x_world_next[0:3],
            rot=x_world_next[3:6],
            rot_rep="euler_xyz",
            frame="NWU",
        )
        self._vehicle_desired_pose_from_ik_ = pose_next.get_pose_as_Pose_msg()
        _, quat_wxyz = pose_next.get_pose(frame="NWU", rot_rep="quat_wxyz")
        res_map_ned = self.world_nwu_to_map_ned(
            xyz_world_nwu=x_world_next[0:3],
            quat_world_wxyz=quat_wxyz,
            warn_context=f"task world->map ({self.prefix})",
        )
        if res_map_ned is None:
            self.node.get_logger().warn(
                "IK aborted, failed to convert world-frame command to map NED frame."
            )
            return 
        p_cmd_ned, rpy_cmd_ned = res_map_ned
        self.pose_command = [
            float(p_cmd_ned[0]),
            float(p_cmd_ned[1]),
            float(p_cmd_ned[2]),
            float(rpy_cmd_ned[0]),
            float(rpy_cmd_ned[1]),
            float(rpy_cmd_ned[2]),
        ]
        

    def set_control_mode(self, mode: ControlMode):
        if mode == self.control_mode:
            return
        if self.control_mode == ControlMode.REPLAY and mode != ControlMode.REPLAY:
            self._stop_replay_session_recording("mode_exit")
        if mode == ControlMode.PLANNER and self.control_mode == ControlMode.REPLAY_SETTLE:
            self.node.get_logger().warn(
                f"Planner mode ignored for {self.prefix}; CmdReplay uses replay mode."
            )
            return

        if mode == ControlMode.REPLAY and self.control_mode != ControlMode.REPLAY:
            self._mode_before_replay = self.control_mode

        self.control_mode = mode
        if mode == ControlMode.PLANNER and not self.sim_reset_hold:
            self._path_recording_enabled = True
        self._zero_teleop_commands()
        self.abrupt_planner_stop()

    def enable_planner_output(self) -> None:
        self._planner_output_enabled = True

    def disable_planner_output(self) -> None:
        self._planner_output_enabled = False

    @staticmethod
    def _safe_filename_token(value: str) -> str:
        token = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))
        return token.strip("_") or "unknown"

    def _start_replay_session_recording(self, replay_controller, state: Dict) -> None:
        if self._replay_record_handle is not None:
            return

        profile_name = self._safe_filename_token(getattr(replay_controller, "profile_name", "profile"))
        robot_name = self._safe_filename_token(self.prefix)
        pass_index = int(getattr(replay_controller, "_current_pass", 0)) + 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cmd_replay_record_dir.mkdir(parents=True, exist_ok=True)
        path = self.cmd_replay_record_dir / f"{timestamp}_{robot_name}_{profile_name}_pass{pass_index}.csv"

        fieldnames = [
            "wall_time_sec",
            "sim_time_sec",
            "replay_time_sec",
            "profile",
            "pass_index",
            "q_alpha_axis_e",
            "q_alpha_axis_d",
            "q_alpha_axis_c",
            "q_alpha_axis_b",
            "dq_alpha_axis_e",
            "dq_alpha_axis_d",
            "dq_alpha_axis_c",
            "dq_alpha_axis_b",
            "ddq_alpha_axis_e",
            "ddq_alpha_axis_d",
            "ddq_alpha_axis_c",
            "ddq_alpha_axis_b",
            "ref_alpha_axis_e",
            "ref_alpha_axis_d",
            "ref_alpha_axis_c",
            "ref_alpha_axis_b",
            "dref_alpha_axis_e",
            "dref_alpha_axis_d",
            "dref_alpha_axis_c",
            "dref_alpha_axis_b",
            "ddref_alpha_axis_e",
            "ddref_alpha_axis_d",
            "ddref_alpha_axis_c",
            "ddref_alpha_axis_b",
            "effort_alpha_axis_e",
            "effort_alpha_axis_d",
            "effort_alpha_axis_c",
            "effort_alpha_axis_b",
            "cmd_tau_axis_e",
            "cmd_tau_axis_d",
            "cmd_tau_axis_c",
            "cmd_tau_axis_b",
            "cmd_tau_axis_a",
            "vehicle_x",
            "vehicle_y",
            "vehicle_z",
            "vehicle_roll",
            "vehicle_pitch",
            "vehicle_yaw",
            "vehicle_u",
            "vehicle_v",
            "vehicle_w",
            "vehicle_p",
            "vehicle_q",
            "vehicle_r",
            "vehicle_du",
            "vehicle_dv",
            "vehicle_dw",
            "vehicle_dp",
            "vehicle_dq",
            "vehicle_dr",
            "target_vehicle_x",
            "target_vehicle_y",
            "target_vehicle_z",
            "target_vehicle_roll",
            "target_vehicle_pitch",
            "target_vehicle_yaw",
            "target_vehicle_u",
            "target_vehicle_v",
            "target_vehicle_w",
            "target_vehicle_p",
            "target_vehicle_q",
            "target_vehicle_r",
            "target_vehicle_du",
            "target_vehicle_dv",
            "target_vehicle_dw",
            "target_vehicle_dp",
            "target_vehicle_dq",
            "target_vehicle_dr",
            "wrench_vehicle_fx",
            "wrench_vehicle_fy",
            "wrench_vehicle_fz",
            "wrench_vehicle_tx",
            "wrench_vehicle_ty",
            "wrench_vehicle_tz",
            "cmd_vehicle_fx",
            "cmd_vehicle_fy",
            "cmd_vehicle_fz",
            "cmd_vehicle_tx",
            "cmd_vehicle_ty",
            "cmd_vehicle_tz",
            "payload_mass",
            "gravity",
        ]

        self._replay_record_handle = path.open("w", encoding="utf-8", newline="")
        self._replay_record_writer = csv.DictWriter(self._replay_record_handle, fieldnames=fieldnames)
        self._replay_record_writer.writeheader()
        self._replay_record_path = path
        self._replay_record_controller = replay_controller
        self._replay_last_cmd_body_wrench = [0.0] * 6
        self._replay_last_cmd_arm_tau = [0.0] * 5
        self._replay_last_recorded_sim_time = None
        self.node.get_logger().info(f"Recording CmdReplay session for {self.prefix} to {path}.")

    def _record_replay_sample(
        self,
        replay_controller,
        state: Dict,
        cmd_body_wrench: Sequence[float],
        cmd_arm_tau: Sequence[float],
    ) -> None:
        if self._replay_record_writer is None:
            return
        if replay_controller is not self._replay_record_controller:
            return
        sim_time = float(state.get("sim_time", 0.0))
        if self._replay_last_recorded_sim_time is not None and sim_time <= self._replay_last_recorded_sim_time:
            return
        self._replay_last_recorded_sim_time = sim_time

        q = list(state.get("q", []))
        dq = list(state.get("dq", []))
        ddq = list(state.get("ddq", []))
        effort = list(state.get("arm_effort", []))
        pose = list(state.get("pose", []))
        body_vel = list(state.get("body_vel", []))
        body_acc = list(state.get("body_acc", []))
        body_forces = list(state.get("body_forces", []))
        arm_cmd = list(cmd_arm_tau)
        vehicle_cmd = list(cmd_body_wrench)
        sample_index = (
            replay_controller.current_sample_index()
            if hasattr(replay_controller, "current_sample_index")
            else None
        )
        q_ref = [0.0] * 5
        dq_ref = [0.0] * 5
        ddq_ref = [0.0] * 5
        target_pose = [0.0] * 6
        target_vel = [0.0] * 6
        target_acc = [0.0] * 6
        if sample_index is not None:
            if hasattr(replay_controller, "arm_reference_at"):
                q_ref, dq_ref, ddq_ref = replay_controller.arm_reference_at(sample_index)
            if hasattr(replay_controller, "vehicle_reference_at"):
                target_pose, target_vel, target_acc = replay_controller.vehicle_reference_at(sample_index)

        def at(values, index, default=0.0):
            return float(values[index]) if index < len(values) else float(default)

        self._replay_record_writer.writerow(
            {
                "wall_time_sec": self.node.get_clock().now().nanoseconds * 1e-9,
                "sim_time_sec": sim_time,
                "replay_time_sec": float(getattr(replay_controller, "_sample_time_sec", 0.0)),
                "profile": getattr(replay_controller, "profile_name", ""),
                "pass_index": int(getattr(replay_controller, "_current_pass", 0)) + 1,
                "q_alpha_axis_e": at(q, 0),
                "q_alpha_axis_d": at(q, 1),
                "q_alpha_axis_c": at(q, 2),
                "q_alpha_axis_b": at(q, 3),
                "dq_alpha_axis_e": at(dq, 0),
                "dq_alpha_axis_d": at(dq, 1),
                "dq_alpha_axis_c": at(dq, 2),
                "dq_alpha_axis_b": at(dq, 3),
                "ddq_alpha_axis_e": at(ddq, 0),
                "ddq_alpha_axis_d": at(ddq, 1),
                "ddq_alpha_axis_c": at(ddq, 2),
                "ddq_alpha_axis_b": at(ddq, 3),
                "ref_alpha_axis_e": at(q_ref, 0),
                "ref_alpha_axis_d": at(q_ref, 1),
                "ref_alpha_axis_c": at(q_ref, 2),
                "ref_alpha_axis_b": at(q_ref, 3),
                "dref_alpha_axis_e": at(dq_ref, 0),
                "dref_alpha_axis_d": at(dq_ref, 1),
                "dref_alpha_axis_c": at(dq_ref, 2),
                "dref_alpha_axis_b": at(dq_ref, 3),
                "ddref_alpha_axis_e": at(ddq_ref, 0),
                "ddref_alpha_axis_d": at(ddq_ref, 1),
                "ddref_alpha_axis_c": at(ddq_ref, 2),
                "ddref_alpha_axis_b": at(ddq_ref, 3),
                "effort_alpha_axis_e": at(effort, 0),
                "effort_alpha_axis_d": at(effort, 1),
                "effort_alpha_axis_c": at(effort, 2),
                "effort_alpha_axis_b": at(effort, 3),
                "cmd_tau_axis_e": at(arm_cmd, 0),
                "cmd_tau_axis_d": at(arm_cmd, 1),
                "cmd_tau_axis_c": at(arm_cmd, 2),
                "cmd_tau_axis_b": at(arm_cmd, 3),
                "cmd_tau_axis_a": at(arm_cmd, 4),
                "vehicle_x": at(pose, 0),
                "vehicle_y": at(pose, 1),
                "vehicle_z": at(pose, 2),
                "vehicle_roll": at(pose, 3),
                "vehicle_pitch": at(pose, 4),
                "vehicle_yaw": at(pose, 5),
                "vehicle_u": at(body_vel, 0),
                "vehicle_v": at(body_vel, 1),
                "vehicle_w": at(body_vel, 2),
                "vehicle_p": at(body_vel, 3),
                "vehicle_q": at(body_vel, 4),
                "vehicle_r": at(body_vel, 5),
                "vehicle_du": at(body_acc, 0),
                "vehicle_dv": at(body_acc, 1),
                "vehicle_dw": at(body_acc, 2),
                "vehicle_dp": at(body_acc, 3),
                "vehicle_dq": at(body_acc, 4),
                "vehicle_dr": at(body_acc, 5),
                "target_vehicle_x": at(target_pose, 0),
                "target_vehicle_y": at(target_pose, 1),
                "target_vehicle_z": at(target_pose, 2),
                "target_vehicle_roll": at(target_pose, 3),
                "target_vehicle_pitch": at(target_pose, 4),
                "target_vehicle_yaw": at(target_pose, 5),
                "target_vehicle_u": at(target_vel, 0),
                "target_vehicle_v": at(target_vel, 1),
                "target_vehicle_w": at(target_vel, 2),
                "target_vehicle_p": at(target_vel, 3),
                "target_vehicle_q": at(target_vel, 4),
                "target_vehicle_r": at(target_vel, 5),
                "target_vehicle_du": at(target_acc, 0),
                "target_vehicle_dv": at(target_acc, 1),
                "target_vehicle_dw": at(target_acc, 2),
                "target_vehicle_dp": at(target_acc, 3),
                "target_vehicle_dq": at(target_acc, 4),
                "target_vehicle_dr": at(target_acc, 5),
                "wrench_vehicle_fx": at(body_forces, 0),
                "wrench_vehicle_fy": at(body_forces, 1),
                "wrench_vehicle_fz": at(body_forces, 2),
                "wrench_vehicle_tx": at(body_forces, 3),
                "wrench_vehicle_ty": at(body_forces, 4),
                "wrench_vehicle_tz": at(body_forces, 5),
                "cmd_vehicle_fx": at(vehicle_cmd, 0),
                "cmd_vehicle_fy": at(vehicle_cmd, 1),
                "cmd_vehicle_fz": at(vehicle_cmd, 2),
                "cmd_vehicle_tx": at(vehicle_cmd, 3),
                "cmd_vehicle_ty": at(vehicle_cmd, 4),
                "cmd_vehicle_tz": at(vehicle_cmd, 5),
                "payload_mass": float(state.get("arm_payload_mass", 0.0)),
                "gravity": float(state.get("arm_gravity", 0.0)),
            }
        )

    def _stop_replay_session_recording(self, reason: str) -> None:
        if self._replay_record_handle is None:
            return
        path = self._replay_record_path
        self._replay_record_handle.flush()
        self._replay_record_handle.close()
        self._replay_record_handle = None
        self._replay_record_writer = None
        self._replay_record_path = None
        self._replay_record_controller = None
        self._replay_last_cmd_body_wrench = [0.0] * 6
        self._replay_last_cmd_arm_tau = [0.0] * 5
        self._replay_last_recorded_sim_time = None
        self.node.get_logger().info(f"Saved CmdReplay session ({reason}) to {path}.")
    
    def abrupt_planner_stop(self, *, publish_zero: bool = True):
        self.disable_planner_output()
        self._accept_planner_results = False
        self.planner_action_client.cancel_active_goal()
        self.final_goal_map_ned_6 = None
        if self.planner is not None:
            self.planner.planned_result = None
            stamp_now = self.node.get_clock().now().to_msg()
            self.planner.clear_path(stamp_now, self.world_frame)
            self.planner.clear_target(stamp_now, self.world_frame)
        if self.vehicle_cart_traj is not None:
            self.vehicle_cart_traj.active = False
        self._zero_planner_commands()

        if publish_zero:
            self.publish_commands([0.0]*6, [0.0]*5)

    def close(self) -> None:
        self._stop_replay_session_recording("shutdown")
        self._accept_planner_results = False
        if self.vehicle_cart_traj is not None:
            self.vehicle_cart_traj.close()
            self.vehicle_cart_traj = None
        if getattr(self, "ps4_controller", None) is not None:
            try:
                self.ps4_controller.running = False
            except Exception:
                pass
        self.ps4_controller = None

    def reset_simulation(self) -> None:
        request = ResetSimUvms.Request()
        request.reset_manipulator = True
        request.reset_vehicle = True
        request.hold_commands = True
        request.use_manipulator_state = False
        request.use_vehicle_state = False
        self.reset_simulation_with_state(request)

    def reset_simulation_with_state(self, request: ResetSimUvms.Request, on_success=None, on_failure=None) -> None:
        previous_hold = self.sim_reset_hold
        self.sim_reset_hold = bool(request.hold_commands)
        self._reset_local_command_state()

        def _on_reset_success():
            if not self.sim_reset_hold:
                self._path_recording_enabled = True
            if on_success is not None:
                on_success()

        def _on_reset_failure():
            self.sim_reset_hold = previous_hold
            if on_failure is not None:
                on_failure()

        self._call_reset_state_service(
            request=request,
            service_name=f"/{self.prefix}reset_sim_uvms",
            log_label=f"[{self.prefix}] state reset",
            on_success=_on_reset_success,
            on_failure=_on_reset_failure,
        )

    def apply_sim_dynamics_from_reset_request(
        self,
        request: ResetSimUvms.Request,
        on_success=None,
        on_failure=None,
    ) -> None:
        if not (
            getattr(request, "set_manipulator_dynamics", False)
            or getattr(request, "set_vehicle_dynamics", False)
        ):
            if on_success is not None:
                on_success()
            return

        dynamics_request = SetSimDynamics.Request()
        dynamics_request.use_coupled_dynamics = bool(getattr(request, "use_coupled_dynamics", False))
        dynamics_request.set_manipulator_dynamics = bool(request.set_manipulator_dynamics)
        dynamics_request.manipulator = request.manipulator_dynamics
        dynamics_request.set_vehicle_dynamics = bool(request.set_vehicle_dynamics and "real" not in self.prefix)
        dynamics_request.vehicle = request.vehicle_dynamics

        service_name = f"/{self.prefix}set_sim_uvms_dynamics"
        if not self.set_sim_dynamics_client.wait_for_service(timeout_sec=0.2):
            self.node.get_logger().warn(
                f"[{self.prefix}] sim dynamics service {service_name} is not ready; "
                "continuing without simulator dynamics update."
            )
            if on_success is not None:
                on_success()
            return

        future = self.set_sim_dynamics_client.call_async(dynamics_request)

        def _done_callback(done_future) -> None:
            try:
                response = done_future.result()
            except Exception as exc:
                self.node.get_logger().error(f"[{self.prefix}] sim dynamics update failed: {exc}")
                if on_failure is not None:
                    on_failure()
                return

            if response is not None and response.success:
                self.node.get_logger().info(
                    f"[{self.prefix}] sim dynamics updated: {response.message}"
                )
                if on_success is not None:
                    on_success()
            else:
                message = "" if response is None else response.message
                self.node.get_logger().warn(
                    f"[{self.prefix}] sim dynamics update rejected before controller-settle: {message}"
                )
                if on_failure is not None:
                    on_failure()

        future.add_done_callback(_done_callback)

    def list_dynamics_profiles(self) -> list[str]:
        return list_robot_dynamics_profiles()

    def apply_dynamics_profile(self, profile_name: str, on_success=None, on_failure=None) -> None:
        profile = load_robot_dynamics_profile(profile_name, self.node)
        if not is_valid_robot_dynamics_profile(profile):
            if on_failure is not None:
                on_failure()
            return

        request = set_dynamics_request_from_profile(
            profile,
            include_vehicle=("real" not in self.prefix),
        )
        if not (request.set_manipulator_dynamics or request.set_vehicle_dynamics):
            self.node.get_logger().warn(
                f"[{self.prefix}] dynamics profile '{profile_name}' has no applicable parameter sections."
            )
            if on_failure is not None:
                on_failure()
            return

        service_name = f"/{self.prefix}set_sim_uvms_dynamics"
        if not self.set_sim_dynamics_client.wait_for_service(timeout_sec=0.2):
            self.node.get_logger().warn(f"[{self.prefix}] dynamics service {service_name} is not ready.")
            if on_failure is not None:
                on_failure()
            return

        future = self.set_sim_dynamics_client.call_async(request)

        def _done_callback(done_future) -> None:
            try:
                response = done_future.result()
            except Exception as exc:
                self.node.get_logger().error(f"[{self.prefix}] dynamics profile '{profile_name}' failed: {exc}")
                if on_failure is not None:
                    on_failure()
                return

            if response is not None and response.success:
                self.active_dynamics_profile = str(profile_name)
                self.node.get_logger().info(
                    f"[{self.prefix}] dynamics profile '{profile_name}' applied: {response.message}"
                )
                if on_success is not None:
                    on_success()
                return

            message = "" if response is None else response.message
            self.node.get_logger().warn(
                f"[{self.prefix}] dynamics profile '{profile_name}' rejected: {message}"
            )
            if on_failure is not None:
                on_failure()

        future.add_done_callback(_done_callback)

    def release_simulation(self) -> None:
        previous_hold = self.sim_reset_hold

        def _on_release_success():
            self.sim_reset_hold = False
            self._path_recording_enabled = True

        def _on_release_failure():
            self.sim_reset_hold = previous_hold

        self._call_uvms_service(
            client=self.release_sim_uvms_client,
            service_name=f"/{self.prefix}release_sim_uvms",
            log_label=f"[{self.prefix}] release",
            on_success=_on_release_success,
            on_failure=_on_release_failure,
        )

    def _reset_local_command_state(self) -> None:
        self.abrupt_planner_stop()
        self._path_recording_enabled = False
        self.clear_robot_path_history()
        if self.planner is not None:
            self.planner.clear_target(self.node.get_clock().now().to_msg(), self.world_frame)
        self._zero_teleop_commands()
        self.teleop_wrench_body_6 = [0.0] * 6
        self.teleop_arm_effort_5 = [0.0] * 5
        self._menu_grasper_effort = 0.0
        self._menu_grasper_until_sec = 0.0
        self.pose_command = [0.0] * 6
        self.body_vel_command = [0.0] * 6
        self.body_acc_command = [0.0] * 6
        self.arm.q_command = ReachParams.joint_home.tolist()
        self.arm.dq_command = np.zeros((4,), dtype=float).tolist()
        self.arm.ddq_command = np.zeros((4,), dtype=float).tolist()
        self.arm.close_grasper()

    def clear_robot_path_history(self) -> None:
        self.traj_path_poses = []
        self._last_path_pub_time = None
        clear_msg = Path()
        clear_msg.header.stamp = self.node.get_clock().now().to_msg()
        clear_msg.header.frame_id = self.map_frame
        self.trajectory_path_publisher.publish(clear_msg)

    def _call_uvms_service(self, client, service_name: str, log_label: str, on_success=None, on_failure=None) -> None:
        if not client.wait_for_service(timeout_sec=0.5):
            self.node.get_logger().warn(f"{log_label} skipped, service {service_name} is not ready.")
            if on_failure is not None:
                on_failure()
            return

        future = client.call_async(Trigger.Request())

        def _done_callback(done_future) -> None:
            try:
                response = done_future.result()
            except Exception as exc:
                self.node.get_logger().error(f"{log_label} failed: {exc}")
                if on_failure is not None:
                    on_failure()
                return

            if response is None:
                self.node.get_logger().error(f"{log_label} failed: empty response from {service_name}")
                if on_failure is not None:
                    on_failure()
                return

            if response.success:
                self.node.get_logger().info(f"{log_label} ok: {response.message}")
                if on_success is not None:
                    on_success()
            else:
                self.node.get_logger().warn(f"{log_label} rejected: {response.message}")
                if on_failure is not None:
                    on_failure()

        future.add_done_callback(_done_callback)

    def _call_reset_state_service(
        self,
        request: ResetSimUvms.Request,
        service_name: str,
        log_label: str,
        on_success=None,
        on_failure=None,
    ) -> None:
        if not self.reset_sim_uvms_client.wait_for_service(timeout_sec=0.5):
            self.node.get_logger().warn(f"{log_label} skipped, service {service_name} is not ready.")
            if on_failure is not None:
                on_failure()
            return

        future = self.reset_sim_uvms_client.call_async(request)

        def _done_callback(done_future) -> None:
            try:
                response = done_future.result()
            except Exception as exc:
                self.node.get_logger().error(f"{log_label} failed: {exc}")
                if on_failure is not None:
                    on_failure()
                return
            if response.success:
                self.node.get_logger().info(f"{log_label} accepted: {response.message}")
                if on_success is not None:
                    on_success()
            else:
                self.node.get_logger().warn(f"{log_label} rejected: {response.message}")
                if on_failure is not None:
                    on_failure()

        future.add_done_callback(_done_callback)


    def _zero_teleop_commands(self):
        if hasattr(self, "controller_lock"):
            with self.controller_lock:
                self.rov_surge = self.rov_sway = self.rov_z = 0.0
                self.rov_roll = self.rov_pitch = self.rov_yaw = 0.0
                self.jointe = self.jointd = self.jointc = self.jointb = self.jointa = 0.0
                self._menu_grasper_effort = 0.0
                self._menu_grasper_until_sec = 0.0

    def _zero_planner_commands(self):
        if self.control_mode == ControlMode.PLANNER:
            st = self.get_state()
            # hold current position, force roll and pitch to 0
            pose = np.asarray(st["pose"], dtype=float).copy()
            pose[3] = 0.0  # roll
            pose[4] = 0.0  # pitch

            self.pose_command = pose.tolist()
        else:
            self.pose_command = [0.0]*6
        self.body_vel_command = [0.0]*6
        self.body_acc_command = [0.0]*6

    def hold_current_state_with_feedback(self) -> None:
        """Hold the measured state with the selected feedback controller."""
        st = self.get_state()
        pose = np.asarray(st["pose"], dtype=float).copy()
        if pose.size >= 5:
            pose[3] = 0.0  # roll
            pose[4] = 0.0  # pitch
        self.pose_command = pose.tolist()
        self.body_vel_command = [0.0] * 6
        self.body_acc_command = [0.0] * 6
        self.arm.q_command = np.asarray(st["q"], dtype=float)[:4].tolist()
        self.arm.dq_command = np.zeros((4,), dtype=float).tolist()
        self.arm.ddq_command = np.zeros((4,), dtype=float).tolist()
        self.enable_planner_output()
        

    def control_loop_callback(self):
        state = self.get_state()
        if state["status"] != "active":
            return
        if self.sim_reset_hold:
            self.publish_commands([0.0] * 6, [0.0] * 5)
            return

        if self.control_mode == ControlMode.TELEOP:
            # read teleop commands and publish
            wrench = getattr(self, "teleop_wrench_body_6", [0.0]*6)
            arm = getattr(self, "teleop_arm_effort_5", [0.0]*5)
            self.publish_commands(wrench, arm)
            return

        if self.control_mode == ControlMode.REPLAY_SETTLE:
            replay_controller = self._replay_settle_controller
            arm_target = self._replay_settle_arm_target
            vehicle_target = self._replay_settle_vehicle_target
            if replay_controller is None or arm_target is None or vehicle_target is None:
                self.node.get_logger().warn(
                    f"Replay controller settle missing target/controller for {self.prefix}; returning to replay idle."
                )
                self._clear_replay_settle()
                self._return_to_replay_after_settle_failure(replay_controller)
                self.publish_commands([0.0] * 6, [0.0] * 5)
                return

            sim_time = float(state["sim_time"])
            if self._replay_settle_started_sim_time is None:
                self._replay_settle_started_sim_time = sim_time

            timeout_sec = float(self._replay_settle_config.get("timeout_sec", 20.0))
            elapsed_sec = max(0.0, sim_time - self._replay_settle_started_sim_time)
            settled, detail = self._replay_settle_is_done(state)
            if settled:
                spec = self._controllers.get("CmdReplay")
                if spec is None:
                    self.node.get_logger().error(f"CmdReplay controller missing for {self.prefix}.")
                    self._clear_replay_settle()
                    self._return_to_replay_after_settle_failure(replay_controller)
                    self.publish_commands([0.0] * 6, [0.0] * 5)
                    return

                self.vehicle_controller_fn = spec.vehicle_fn
                self.arm_controller_fn = spec.arm_fn
                self.controller_name = "CmdReplay"
                self.set_control_mode(ControlMode.REPLAY)
                replay_controller.mark_reset_succeeded()
                self._clear_replay_settle()
                self.node.get_logger().info(
                    f"Replay controller settle complete for {self.prefix}; {detail}."
                )
                self.publish_commands([0.0] * 6, [0.0] * 5)
                return

            if elapsed_sec >= timeout_sec:
                self._clear_replay_settle()
                self._return_to_replay_after_settle_failure(replay_controller)
                self.node.get_logger().warn(
                    f"Replay controller settle timed out for {self.prefix} after {elapsed_sec:.2f}s; {detail}."
                )
                self.publish_commands([0.0] * 6, [0.0] * 5)
                return

            active_controller = self.active_controller_instance()
            if active_controller is not None and hasattr(active_controller, "set_sim_time"):
                active_controller.set_sim_time(sim_time)

            self.pose_command = np.asarray(vehicle_target, dtype=float).tolist()
            self.body_vel_command = [0.0] * 6
            self.body_acc_command = [0.0] * 6
            self.arm.q_command = np.asarray(arm_target[:4], dtype=float).tolist()
            self.arm.dq_command = [0.0] * 4
            self.arm.ddq_command = [0.0] * 4

            veh_state_vec = np.array(list(state["pose"]) + list(state["body_vel"]), dtype=float)
            target_ned_pose = np.asarray(self.pose_command, dtype=float)
            target_body_vel = np.asarray(self.body_vel_command, dtype=float)
            target_body_acc = np.asarray(self.body_acc_command, dtype=float)
            q_ref = np.asarray(arm_target, dtype=float).tolist()
            dq_ref = [0.0] * 5
            ddq_ref = [0.0] * 5

            cmd_body_wrench = self.vehicle_controller_fn(
                state=veh_state_vec,
                target_pos=target_ned_pose,
                target_vel=target_body_vel,
                target_acc=target_body_acc,
                dt=state["dt"],
            )
            cmd_arm_tau = self.arm_controller_fn(
                q=list(state["q"]) + list(state["grasper_q"]),
                q_dot=list(state["dq"]) + list(state["grasper_qdot"]),
                q_ref=q_ref,
                dq_ref=dq_ref,
                ddq_ref=ddq_ref,
                dt=state["dt"],
            )

            self.publish_commands(
                np.asarray(cmd_body_wrench, float).tolist(),
                np.asarray(cmd_arm_tau, float).tolist(),
            )
            return

        if self.control_mode == ControlMode.REPLAY:
            active_controller = self.active_controller_instance()
            if active_controller is None or not hasattr(active_controller, "set_sim_time"):
                self.publish_commands([0.0] * 6, [0.0] * 5)
                return

            active_controller.set_sim_time(state["sim_time"])
            if (
                hasattr(active_controller, "has_pending_auto_start")
                and active_controller.has_pending_auto_start()
                and hasattr(active_controller, "start_playback")
            ):
                if active_controller.start_playback(sim_time_sec=state["sim_time"]):
                    if (
                        hasattr(active_controller, "recording_enabled")
                        and active_controller.recording_enabled()
                    ):
                        self._start_replay_session_recording(active_controller, state)

            if not getattr(active_controller, "enabled", False):
                self.publish_commands([0.0] * 6, [0.0] * 5)
                return

            sample_index = (
                active_controller.current_sample_index()
                if hasattr(active_controller, "current_sample_index")
                else None
            )
            feedback_controller_name = (
                active_controller.feedback_controller_name()
                if hasattr(active_controller, "feedback_controller_name")
                else "PID"
            )
            feedback_spec = self._controllers.get(feedback_controller_name)
            if feedback_spec is None and feedback_controller_name != "PID":
                if not getattr(active_controller, "_warned_invalid_feedback_controller", False):
                    self.node.get_logger().warn(
                        f"CmdReplay feedback controller '{feedback_controller_name}' is not available for "
                        f"{self.robot_name}; falling back to PID."
                    )
                    active_controller._warned_invalid_feedback_controller = True
                feedback_spec = self._controllers.get("PID")

            vehicle_mode = (
                active_controller.vehicle_subsystem_mode()
                if hasattr(active_controller, "vehicle_subsystem_mode")
                else "replay_command"
            )
            manipulator_mode = (
                active_controller.manipulator_subsystem_mode()
                if hasattr(active_controller, "manipulator_subsystem_mode")
                else "replay_command"
            )

            veh_state_vec = np.array(list(state["pose"]) + list(state["body_vel"]), dtype=float)
            if vehicle_mode == "replay_command":
                cmd_body_wrench = active_controller.vehicle_command_at(sample_index)
            elif vehicle_mode == "track_reference" and feedback_spec is not None:
                target_pos, target_vel, target_acc = active_controller.vehicle_reference_at(sample_index)
                cmd_body_wrench = feedback_spec.vehicle_fn(
                    state=veh_state_vec,
                    target_pos=np.asarray(target_pos, dtype=float),
                    target_vel=np.asarray(target_vel, dtype=float),
                    target_acc=np.asarray(target_acc, dtype=float),
                    dt=state["dt"],
                )
            elif vehicle_mode == "hold_initial" and feedback_spec is not None:
                cmd_body_wrench = feedback_spec.vehicle_fn(
                    state=veh_state_vec,
                    target_pos=np.asarray(active_controller.initial_vehicle_pose(), dtype=float),
                    target_vel=np.zeros(6, dtype=float),
                    target_acc=np.zeros(6, dtype=float),
                    dt=state["dt"],
                )
            else:
                cmd_body_wrench = np.zeros(6, dtype=float)

            if manipulator_mode == "replay_command":
                cmd_arm_tau = active_controller.arm_command_at(sample_index)
            elif manipulator_mode == "track_reference" and feedback_spec is not None:
                q_ref, dq_ref, ddq_ref = active_controller.arm_reference_at(sample_index)
                cmd_arm_tau = feedback_spec.arm_fn(
                    q=list(state["q"]) + list(state["grasper_q"]),
                    q_dot=list(state["dq"]) + list(state["grasper_qdot"]),
                    q_ref=np.asarray(q_ref, dtype=float).tolist(),
                    dq_ref=np.asarray(dq_ref, dtype=float).tolist(),
                    ddq_ref=np.asarray(ddq_ref, dtype=float).tolist(),
                    dt=state["dt"],
                )
            elif manipulator_mode == "hold_initial" and feedback_spec is not None:
                q_ref = active_controller.initial_manipulator_position()
                cmd_arm_tau = feedback_spec.arm_fn(
                    q=list(state["q"]) + list(state["grasper_q"]),
                    q_dot=list(state["dq"]) + list(state["grasper_qdot"]),
                    q_ref=q_ref,
                    dq_ref=[0.0] * 5,
                    ddq_ref=[0.0] * 5,
                    dt=state["dt"],
                )
            else:
                cmd_arm_tau = np.zeros(5, dtype=float)
            if (
                hasattr(active_controller, "needs_repeat_reset")
                and active_controller.needs_repeat_reset()
                and hasattr(active_controller, "consume_repeat_reset_request")
            ):
                self._stop_replay_session_recording("pass_complete")
                active_controller.consume_repeat_reset_request()
                request = active_controller.build_reset_request()
                def _fail_repeat_reset():
                    active_controller.mark_reset_failed()
                    self.publish_commands([0.0] * 6, [0.0] * 5)

                if hasattr(active_controller, "reset_mode") and active_controller.reset_mode() == "controller_settle":
                    self.apply_sim_dynamics_from_reset_request(
                        request,
                        on_success=lambda: self.start_replay_controller_settle(active_controller),
                        on_failure=_fail_repeat_reset,
                    )
                    self.publish_commands([0.0] * 6, [0.0] * 5)
                    return

                def _arm_next_pass():
                    self.set_control_mode(ControlMode.REPLAY)
                    active_controller.mark_reset_succeeded()

                self.reset_simulation_with_state(
                    request,
                    on_success=_arm_next_pass,
                    on_failure=_fail_repeat_reset,
                )
                self.publish_commands([0.0] * 6, [0.0] * 5)
                return

            self.publish_commands(
                np.asarray(cmd_body_wrench, float).tolist(),
                np.asarray(cmd_arm_tau, float).tolist(),
            )
            if (
                getattr(active_controller, "enabled", False)
                and hasattr(active_controller, "recording_enabled")
                and active_controller.recording_enabled()
            ):
                self._record_replay_sample(
                    active_controller,
                    state,
                    cmd_body_wrench,
                    cmd_arm_tau,
                )
                self._replay_last_cmd_body_wrench = np.asarray(cmd_body_wrench, float).tolist()
                self._replay_last_cmd_arm_tau = np.asarray(cmd_arm_tau, float).tolist()
            if hasattr(active_controller, "playback_status") and active_controller.playback_status() in (
                "complete",
                "error",
                "stopped",
                "no_csv",
            ):
                self._stop_replay_session_recording(active_controller.playback_status())
            return

        if self.control_mode == ControlMode.PLANNER:
            if not getattr(self, "_planner_output_enabled", False):
                self.publish_commands([0.0] * 6, [0.0] * 5)
                return
            active_controller = self.active_controller_instance()
            if active_controller is not None and hasattr(active_controller, "set_sim_time"):
                active_controller.set_sim_time(state["sim_time"])

            # compute model based commands and publish
            veh_state_vec = np.array(list(state["pose"]) + list(state["body_vel"]), dtype=float)

            target_ned_pose = np.asarray(self.pose_command, dtype=float)
            target_body_vel = np.asarray(self.body_vel_command, dtype=float)
            target_body_acc = np.asarray(self.body_acc_command, dtype=float)

            cmd_body_wrench = self.vehicle_controller_fn(
                state=veh_state_vec,
                target_pos=target_ned_pose,
                target_vel=target_body_vel,
                target_acc=target_body_acc,
                dt=state["dt"],
            )

            q_ref = list(self.arm.q_command) + [self.arm.grasp_command]
            dq_ref = list(self.arm.dq_command) + [0.0]
            ddq_ref = list(self.arm.ddq_command) + [0.0]

            cmd_arm_tau = self.arm_controller_fn(
                q=list(state["q"]) + list(state["grasper_q"]),
                q_dot=list(state["dq"]) + list(state["grasper_qdot"]),
                q_ref=q_ref,
                dq_ref=dq_ref,
                ddq_ref=ddq_ref,
                dt=state["dt"],
            )

            self.reference_pub.publish_map_targets_and_arm_refs(
                target_ned_pose=target_ned_pose,
                target_body_vel=target_body_vel,
                target_body_acc=target_body_acc,
                q_ref=q_ref,
                dq_ref=dq_ref,
                ddq_ref=ddq_ref,
            )

            self.publish_commands(np.asarray(cmd_body_wrench, float).tolist(),
                                np.asarray(cmd_arm_tau, float).tolist())


    def joystick_read_callback(self):
        if not self.has_joystick_interface:
            return

        with self.controller_lock:
            self.teleop_wrench_body_6 = [
                self.rov_surge, self.rov_sway, self.rov_z,
                self.rov_roll, self.rov_pitch, self.rov_yaw
            ]
            self.teleop_arm_effort_5 = [
                self.jointe, self.jointd, self.jointc, self.jointb, self.jointa
            ]
