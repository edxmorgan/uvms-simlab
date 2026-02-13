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
from pyPS4Controller.controller import Controller
import threading
import glob
from typing import Sequence, Dict, Callable, Any, Optional
from control_msgs.msg import DynamicInterfaceGroupValues
from std_msgs.msg import Float64MultiArray
from controller_msg import FullRobotMsg
from controllers import LowLevelPidController, LowLevelOptimalModelbasedController
from planner_markers import PathPlanner
from cartesian_ruckig import VehicleCartesianRuckig
from ruckig import Result
from alpha_reach import Params as alpha_params 
from frame_utils import PoseX
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
from debug_targets import DebugTargetPublisher
from planner_action_client import PlannerActionClient

class ControlSpace(str, Enum):
    JOINT_SPACE = "joint_space"
    TASK_SPACE  = "task_space"


class ControlMode(str, Enum):
    TELEOP = "teleop"
    PLANNER = "planner"

@dataclass
class ArmGains:
    Kp: list
    Ki: list
    Kd: list

@dataclass
class ControllerSpec:
    name: str
    vehicle_fn: Callable[..., Any]
    arm_fn: Callable[..., Any]
    arm_gains: Optional[ArmGains] = None

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

    floating_force_x = 'force.x'
    floating_force_y = 'force.y'
    floating_force_z = 'force.z'
    floating_torque_x = 'torque.x'
    floating_torque_y = 'torque.y'
    floating_torque_z = 'torque.z'

    sim_time = 'sim_time'
    sim_period = 'sim_period'
    
class Manipulator(Base):
    def __init__(self, node: Node, n_joint, prefix):
        self.node = node
        self.n_joint = n_joint
        self.q = [0]*n_joint
        self.dq = [0]*n_joint
        self.sim_period = [0.0]
        self.effort = [0]*n_joint
        self.alpha_axis_a = f'{prefix}_axis_a'
        self.alpha_axis_b = f'{prefix}_axis_b'
        self.alpha_axis_c = f'{prefix}_axis_c'
        self.alpha_axis_d = f'{prefix}_axis_d'
        self.alpha_axis_e = f'{prefix}_axis_e'

        self.joints = [self.alpha_axis_e, self.alpha_axis_d, self.alpha_axis_c, self.alpha_axis_b]
        self.grasper = [self.alpha_axis_a]

        self.q_command = alpha_params.joint_home.tolist()
        self.dq_command = np.zeros((4,)).tolist()
        self.ddq_command = np.zeros((4,)).tolist()

        # Initialize grasper state so get_state() is safe before first update.
        self.grasper_q = [0.0]
        self.grasper_q_dot = [0.0]
        self.close_grasper()

    def open_grasper(self):
        self.grasp_command = alpha_params.grasper_open

    def close_grasper(self):
        self.grasp_command = alpha_params.grasper_close

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
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            'arm_effort':self.effort,
            'grasper_q': self.grasper_q,
            'grasper_qdot': self.grasper_q_dot,
            'q':self.q,
            'dq':self.dq,
            'dt':self.sim_period[0]
        }

class Robot(Base):
    def __init__(self, node: Node,
                 tf_buffer: Buffer,
                  k_robot, 
                  n_joint, 
                  prefix,
                  planner=None,
                  vehicle_cart_traj=None):
        self.planner: PathPlanner = planner
        self.vehicle_cart_traj: VehicleCartesianRuckig = vehicle_cart_traj
        self.menu_handle = None
        self.final_goal_map_ned_6 = None
        self.yaw_blend_factor = 0.0
        self.tf_buffer = tf_buffer
        self.task_based_controller = False

        self.dynamics_states_sub = node.create_subscription(
                DynamicJointState,
                'dynamic_joint_states',
                self.listener_callback,
                10
            )
        
        # Latest mocap pose [x, y, z, qw, qx, qy, qz]
        self.mocap_latest = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        self.v_c = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

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
        self.dynamics_states_sub  # prevent unused variable warning
    
        package_share_directory = ament_index_python.get_package_share_directory(
                'simlab')
        fk_path = os.path.join(package_share_directory, 'manipulator/fk_eval.casadi')
        ik_path = os.path.join(package_share_directory, 'manipulator/ik_eval.casadi')

        vehicle_J_path = os.path.join(package_share_directory, 'vehicle/J_uv.casadi')
        vehicle_ned2body_acc_path = os.path.join(package_share_directory, 'vehicle/ned2body_acc.casadi')
        vehicle_ned2body_vel_path = os.path.join(package_share_directory, 'vehicle/ned2body_vel.casadi')
        ik_wb_path = os.path.join(package_share_directory, 'whole_body/ik_whole_b.casadi')

        self.fk_eval = ca.Function.load(fk_path) #  forward kinematics
        # also set a class attribute fk_eval so it can be shared
        if not hasattr(Robot, "fk_eval_cls"):
            Robot.fk_eval_cls = self.fk_eval

        self.ik_eval = ca.Function.load(ik_path) #  inverse kinematics
        # also set a class attribute ik_eval so it can be shared
        if not hasattr(Robot, "ik_eval_cls"):
            Robot.ik_eval_cls = self.ik_eval


        self.ik_wb_eval = ca.Function.load(ik_wb_path) #  inverse kinematics
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
        self.ned_vel = [0] * 6
        self.body_forces = [0] * 6
        self.prefix = prefix
        self.status = 'inactive'
        self.sim_time = 0.0
        self.start_time = 0.0
        self.joint4_frame = f"{self.prefix}joint_4"
        self.pose_command = [0.0]*6
        self.body_vel_command = [0.0]*6
        self.body_acc_command = [0.0]*6
        self.ll_statefeedback_controllers = LowLevelPidController(self.node, self.n_joint)
        self.ll_modelbased_controllers = LowLevelOptimalModelbasedController(self.node, self.n_joint)
        self.planner_action_client = PlannerActionClient(
            self.node,
            action_name="planner",
            on_result=self._on_planner_action_result,
        )
        self.debug_pub = DebugTargetPublisher(self.node, self.prefix)
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

        self.control_mode = ControlMode.TELEOP

        # inverse IK tool axis and alignment weight CONFIGURATIONS
        self.ik_tool_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        self.ik_base_align_w = 0.0

        self.traj_path_poses = []
        self.max_traj_pose_count = 2000  # cap RViz message size
        self.path_publish_period = 0.1  # seconds between stored poses
        self._last_path_pub_time = None
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
        self.control_loop_timer = self.node.create_timer(1.0/60.0, self.control_loop_callback)

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
        self.arm_gain_pack: Optional[ArmGains] = None
        self.controller_name = None

        self.world_frame = 'world'

        def wrap_arm_no_gains(fn):
            # register with arm_fn=wrap_arm_no_gains(your_fn)
            def _wrapped(**kwargs):
                kwargs.pop("Kp", None)
                kwargs.pop("Ki", None)
                kwargs.pop("Kd", None)
                return fn(**kwargs)
            return _wrapped

        # Register your existing controllers
        self.register_controller(
            name="pid",
            vehicle_fn=self.ll_statefeedback_controllers.vehicle_controller,
            arm_fn=self.ll_statefeedback_controllers.arm_controller,
            arm_gains=ArmGains(
                Kp=list(alpha_params.tau_Kp) + list(alpha_params.grasper_kp),
                Ki=list(alpha_params.tau_Ki) + list(alpha_params.grasper_ki),
                Kd=list(alpha_params.tau_Kd) + list(alpha_params.grasper_kd),
            ),
        )

        self.register_controller(
            name="model",
            vehicle_fn=self.ll_modelbased_controllers.vehicle_controller,
            arm_fn=self.ll_modelbased_controllers.arm_controller,
            arm_gains=ArmGains(
                Kp=list(alpha_params.acc_Kp) + list(alpha_params.grasper_kp),
                Ki=list(alpha_params.acc_Ki) + list(alpha_params.grasper_ki),
                Kd=list(alpha_params.acc_Kd) + list(alpha_params.grasper_kd),
            ),
        )

        # set default controller
        self.set_controller("pid")

        # Active path planner
        self.planner_name = None

        self.register_planner(
            name='RRTstar'
        )

        self.register_planner(
            name='Bitstar'
        )
        self.set_planner("Bitstar")

    def register_controller(
                self,
                name: str,
                vehicle_fn,
                arm_fn,
                arm_gains: Optional[ArmGains] = None,
            ) -> None:
        self._controllers[name] = ControllerSpec(
            name=name,
            vehicle_fn=vehicle_fn,
            arm_fn=arm_fn,
            arm_gains=arm_gains,
        )

    def set_controller(self, name: str) -> None:
        spec = self._controllers[name]  # raises KeyError if missing, good fail-fast
        self.vehicle_controller_fn = spec.vehicle_fn
        self.arm_controller_fn = spec.arm_fn
        self.arm_gain_pack = spec.arm_gains
        self.controller_name = name
        self.node.get_logger().info(f"Controller set to {name} for {self.prefix}")

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
        xq['ned_vel'] = self.ned_vel
        xq['body_forces'] = self.body_forces
        xq['status'] = self.status
        xq['sim_time'] = self.sim_time
        xq['prefix'] = self.prefix
        xq['mocap'] = self.mocap_latest
        return xq

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
    
    # ForwardCommandController
    def publish_commands(self, wrench_body_6: Sequence[float], arm_effort_5: Sequence[float]):
        # Vehicle, DynamicInterfaceGroupValues payload
        self.publish_vehicle_and_arm(wrench_body_6, arm_effort_5)

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
        if self.planner is not None:
            self.planner.planned_result = dict(plan_result)

        if plan_result.get("is_success", False):
            self.node.get_logger().info(
                f"Planner action produced {int(plan_result.get('count', 0))} waypoints for {self.prefix}"
            )
    
            path_xyz = np.asarray(self.planner.planned_result["xyz"], dtype=float)
            self._start_vehicle_cartesian_ruckig(self.start_xyz, self.start_quat_wxyz, path_xyz)
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
        self.node.get_logger().info(
            f"Planning motion with {self.planner_name} for {self.prefix} to target pose..."
        )
        self.abrupt_planner_stop()
        pose_now = self._pose_from_state_in_frame(self.world_frame)
        if pose_now is None:
            self.node.get_logger().warn("Planner action request was not sent, current pose unavailable.")
            return False

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
            self.debug_pub.publish_world_targets(
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

                cmd_J_UV = self.vehicle_J(rpy_cmd_ned).full()
                self.node.get_logger().debug(f"v_cmd_ned {tw6_map_ned} : active.")

                body_vel_command = self.to_body_velocity(cmd_J_UV, rpy_cmd_ned, tw6_map_ned)
                self.node.get_logger().debug(f"body_vel_command {body_vel_command} : active.")

                body_acc_command = self.to_body_acceleration(rpy_cmd_ned, tw6_map_ned, acc6_map_ned, self.v_c)
                self.node.get_logger().debug(f"body_acc_command {acc6_map_ned} : active.")

                # Blend yaw between velocity-based and target waypoint
                rpy_cmd_ned[2] = (1 - self.yaw_blend_factor) * adjusted_yaw + self.yaw_blend_factor * rpy_cmd_ned[2]

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
            np.asarray(alpha_params.base_T0_new, dtype=float),
            np.asarray(alpha_params.tipOffset, dtype=float),
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

        self.control_mode = mode
        self._zero_teleop_commands()
        self.abrupt_planner_stop()
    
    def abrupt_planner_stop(self):
        self.final_goal_map_ned_6 = None
        if self.planner is not None:
            self.planner.planned_result = None
        if self.vehicle_cart_traj is not None:
            self.vehicle_cart_traj.active = False
        self._zero_planner_commands()

        # publish zeros once immediately
        self.publish_commands([0.0]*6, [0.0]*5)


    def _zero_teleop_commands(self):
        if hasattr(self, "controller_lock"):
            with self.controller_lock:
                self.rov_surge = self.rov_sway = self.rov_z = 0.0
                self.rov_roll = self.rov_pitch = self.rov_yaw = 0.0
                self.jointe = self.jointd = self.jointc = self.jointb = self.jointa = 0.0

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
        

    def control_loop_callback(self):
        state = self.get_state()
        if state["status"] != "active":
            return

        if self.control_mode == ControlMode.TELEOP:
            # read teleop commands and publish
            wrench = getattr(self, "teleop_wrench_body_6", [0.0]*6)
            arm = getattr(self, "teleop_arm_effort_5", [0.0]*5)
            self.publish_commands(wrench, arm)
            return

        if self.control_mode == ControlMode.PLANNER:
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

            g = self.arm_gain_pack
            if g is None:
                raise RuntimeError(f"Arm gains not set for controller {self.controller_name}")

            q_ref = list(self.arm.q_command) + [self.arm.grasp_command]
            dq_ref = list(self.arm.dq_command) + [0.0]
            ddq_ref = list(self.arm.ddq_command) + [0.0]

            cmd_arm_tau = self.arm_controller_fn(
                q=list(state["q"]) + list(state["grasper_q"]),
                q_dot=list(state["dq"]) + list(state["grasper_qdot"]),
                q_ref=q_ref,
                dq_ref=dq_ref,
                ddq_ref=ddq_ref,
                Kp=g.Kp,
                Ki=g.Ki,
                Kd=g.Kd,
                dt=state["dt"],
                u_max=list(alpha_params.u_max) + list(alpha_params.grasper_u_max),
                u_min=list(alpha_params.u_min) + list(alpha_params.grasper_u_min),
                model_param=alpha_params.sim_p,
            )

            self.debug_pub.publish_map_targets_and_arm_refs(
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
