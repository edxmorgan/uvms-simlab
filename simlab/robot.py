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
import casadi as ca
from nav_msgs.msg import Path
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, QoSHistoryPolicy
import csv
from datetime import datetime
import copy
from std_msgs.msg import Float32
from pyPS4Controller.controller import Controller
import threading
import glob
from typing import Sequence, Dict
from control_msgs.msg import DynamicInterfaceGroupValues
from std_msgs.msg import Float64MultiArray, String
from controller_msg import FullRobotMsg
from controllers import LowLevelController
from planner_markers import PathPlanner
from cartesian_ruckig import VehicleCartesianRuckig, EndeffectorCartesianRuckig
from alpha_reach import Params as alpha_params
class PS4Controller(Controller):
    def __init__(self, ros_node, prefix, **kwargs):
        super().__init__(**kwargs)
        self.ros_node = ros_node
        
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

        # # Create a lock specifically for updating gain values.
        # self.gain_lock = threading.Lock()
        # # Start a thread to update the gain every few seconds.
        # # gain randomization for good data collection
        # self.gain_thread = threading.Thread(target=self._update_gain, daemon=True)
        # self.gain_thread.start()

    # def _update_gain(self):
    #     """Randomize the gain value every few seconds and update the torque parameters."""
    #     while True:
    #         # For example, choose a new gain between 4 and 6.
    #         new_gain = random.uniform(3, 8)
    #         with self.gain_lock:
    #             self.gain = new_gain

    #             self.max_torque = self.gain * 2.0
    #             self.heave_max_torque = self.gain * 3.0
    #             self.orient_max_torque = self.gain * 0.7
    #             self.yaw_max_torque = self.gain * 0.2
    #         # Keep this gain for 8 seconds.
    #         time.sleep(8)

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

    imu_roll = "imu_roll"
    imu_pitch = "imu_pitch"
    imu_yaw = "imu_yaw"

    imu_roll_unwrap = "imu_roll_unwrap"
    imu_pitch_unwrap = "imu_pitch_unwrap"
    imu_yaw_unwrap = "imu_yaw_unwrap"

    imu_q_w = "imu_orientation_w"
    imu_q_x = "imu_orientation_x"
    imu_q_y = "imu_orientation_y"
    imu_q_z = "imu_orientation_z"

    imu_wx = "imu_angular_vel_x"
    imu_wy = "imu_angular_vel_y"
    imu_wz = "imu_angular_vel_z"

    imu_ax = "imu_linear_acceleration_x"
    imu_ay = "imu_linear_acceleration_y"
    imu_az = "imu_linear_acceleration_z"

    depth_pressure2 = "depth_from_pressure2"

    dvl_roll = "dvl_gyro_roll"
    dvl_pitch = "dvl_gyro_pitch"
    dvl_yaw = "dvl_gyro_yaw"

    dvl_speed_x = "dvl_speed_x"
    dvl_speed_y = "dvl_speed_y"
    dvl_speed_z = "dvl_speed_z"
    
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

        self.q_command = alpha_params.joint_home.tolist()
        self.dq_command = np.zeros((4,)).tolist()
        self.ddq_command = np.zeros((4,)).tolist()

    def update_state(self, msg: DynamicJointState):
        self.q = self.get_interface_value(
            msg,
            self.joints,
            [Axis_Interface_names.manipulator_position] * 4
        )
        self.filtered_q = self.get_interface_value(
            msg,
            self.joints,
            [Axis_Interface_names.manipulator_filtered_position] * 4
        )
        self.dq = self.get_interface_value(
            msg,
            self.joints,
            [Axis_Interface_names.manipulator_velocity] * 4
        )
        self.filtered_dq = self.get_interface_value(
            msg,
            self.joints,
            [Axis_Interface_names.manipulator_filtered_velocity] * 4
        )
        self.estimated_ddq = self.get_interface_value(
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
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            'arm_effort':self.effort,
            'q':self.q,
            'dq':self.dq,
            'dt':self.sim_period[0]
        }

class Robot(Base):
    def __init__(self, node: Node,
                  k_robot, 
                  n_joint, 
                  prefix,
                  record=False,  
                  controller='pid'):
        self.planner: PathPlanner = None
        self.vehicle_cart_traj: VehicleCartesianRuckig = None
        self.endeffector_cart_traj: EndeffectorCartesianRuckig = None
        self.menu_handle = None
        self.final_goal = None
        self.yaw_blend_factor = 0.0
        self.dynamics_states_sub = node.create_subscription(
                DynamicJointState,
                'dynamic_joint_states',
                self.listener_callback,
                10
            )
        
        # Latest mocap pose [x, y, z, qw, qx, qy, qz]
        self.mocap_latest = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        # Subscribe to the ENU, origin offset pose from MocapPathBuilder
        # Topic name must match MocapPathBuilder.mocap_pose_topic, default 'mocap_pose'
        self.mocap_pose_sub = node.create_subscription(
            PoseStamped,
            'mocap_pose',
            self._mocap_pose_cb,
            10
        )
        


        self.k_robot = k_robot
        self.robot_name = f'uvms {prefix}: {k_robot}'
        self.dynamics_states_sub  # prevent unused variable warning
    
        package_share_directory = ament_index_python.get_package_share_directory(
                'simlab')
        fk_path = os.path.join(package_share_directory, 'manipulator/fk_eval.casadi')
        ik_path = os.path.join(package_share_directory, 'manipulator/ik_eval.casadi')

        vehicle_J_path = os.path.join(package_share_directory, 'vehicle/J_uv.casadi')

        whole_body_ik_path = os.path.join(package_share_directory, 'whole_body/whole_body_ik.casadi')

        self.fk_eval = ca.Function.load(fk_path) #  forward kinematics
        # also set a class attribute fk_eval so it can be shared
        if not hasattr(Robot, "fk_eval_cls"):
            Robot.fk_eval_cls = self.fk_eval

        self.ik_eval = ca.Function.load(ik_path) #  inverse kinematics
        # also set a class attribute ik_eval so it can be shared
        if not hasattr(Robot, "ik_eval_cls"):
            Robot.ik_eval_cls = self.ik_eval

        self.vehicle_J = ca.Function.load(vehicle_J_path)

        self.whole_body_ik_eval = ca.Function.load(whole_body_ik_path)
        # also set a class attribute whole_body_ik_eval so it can be shared
        if not hasattr(Robot, "wb_ik_eval_cls"):
            Robot.wb_ik_eval_cls = self.whole_body_ik_eval

        self.node = node
        self.sensors = [
            Axis_Interface_names.imu_roll,
            Axis_Interface_names.imu_pitch,
            Axis_Interface_names.imu_yaw,
            Axis_Interface_names.imu_roll_unwrap,
            Axis_Interface_names.imu_pitch_unwrap,
            Axis_Interface_names.imu_yaw_unwrap,
            Axis_Interface_names.imu_q_w,
            Axis_Interface_names.imu_q_x,
            Axis_Interface_names.imu_q_y,
            Axis_Interface_names.imu_q_z,
            Axis_Interface_names.imu_wx,
            Axis_Interface_names.imu_wy,
            Axis_Interface_names.imu_wz,
            Axis_Interface_names.imu_ax,
            Axis_Interface_names.imu_ay,
            Axis_Interface_names.imu_az,
            Axis_Interface_names.depth_pressure2,
            Axis_Interface_names.dvl_roll,
            Axis_Interface_names.dvl_pitch,
            Axis_Interface_names.dvl_yaw,
            Axis_Interface_names.dvl_speed_x,
            Axis_Interface_names.dvl_speed_y,
            Axis_Interface_names.dvl_speed_z
            ]
        self.prediction_interfaces = [
            "position.x", "position.y", "position.z", "roll", "pitch", "yaw",
            "orientation.w", "orientation.x", "orientation.y", "orientation.z", 
            "velocity.x", "velocity.y", "velocity.z", 
            "angular_velocity.x", "angular_velocity.y", "angular_velocity.z",
        ]

        self.state_estimate_interfaces = [
            "position_estimate.x", "position_estimate.y", "position_estimate.z",
            "roll_estimate", "pitch_estimate", "yaw_estimate",
            "orientation_estimate.w", "orientation_estimate.x", "orientation_estimate.y", "orientation_estimate.z",
            "velocity_estimate.x", "velocity_estimate.y", "velocity_estimate.z",
            "angular_velocity_estimate.x", "angular_velocity_estimate.y", "angular_velocity_estimate.z",
            "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z",
            "angular_acceleration.x", "angular_acceleration.y", "angular_acceleration.z",
            "P_x_x", "P_y_y", "P_z_z", "P_roll_roll", "P_pitch_pitch", "P_yaw_yaw",
            "P_u_u", "P_v_v", "P_w_w", "P_p_p", "P_q_q", "P_r_r",
        ]

        self.payload_state_interfaces = ["payload.mass", "payload.Ixx", "payload.Iyy", "payload.Izz"]

        self.n_joint = n_joint
        self.floating_base_IOs = f'{prefix}IOs'
        self.arm_IOs = f'{prefix}_arm_IOs'
        self.arm = Manipulator(node, n_joint, prefix)
        self.ned_pose = [0] * 6
        self.body_vel = [0] * 6
        self.ned_vel = [0] * 6
        self.sensor_reading = [0] * len(self.sensors)
        self.prediction_readings = [0] * len(self.prediction_interfaces)
        self.state_estimate_readings = [0] * len(self.state_estimate_interfaces)
        self.payload_state_readings = [0] * len(self.payload_state_interfaces)
        self.body_forces = [0] * 6
        self.gt_measurements = [0] * 6
        self.prefix = prefix
        self.status = 'inactive'
        self.sim_time = 0.0
        self.start_time = 0.0

        # self.use_controller = controller
        self.pose_command = [0.0]*6
        self.body_vel_command = [0.0]*6
        self.body_acc_command = [0.0]*6
        self.ll_controllers = LowLevelController(self.n_joint)

 
        self.uvms_ll = [-1000, -1000, 0.0, -np.pi/6, -np.pi/6, -1000, 1, 0.01, 0.01, 0.01]
        self.uvms_ul = [ 1000, 1000, 1000, np.pi/6, np.pi/6, 1000, 5.50, 3.40, 3.40, 5.70]
        self.k0 = [1,1,1 , 1,1,1, 1,1,1,1]
        self.base_pose = [0.190, 0.000, -0.120, np.pi, 0.000, 0.000] #floating base mount
        self.world_pose = [0.0, 0.0, 0, 0, 0, 0]
        self.vec_g = [0, 0, 9.81]

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

        self.overlay_text_publisher = self.node.create_publisher(
            String,
            "chatter",
            qos_profile
        )

        
        self.overlay_text_timer = self.node.create_timer(1.0 / 30, self.publish_overlay_text_callback)

        self.traj_path_poses = []
        self.max_traj_pose_count = 2000  # cap RViz message size
        self.path_publish_period = 0.1  # seconds between stored poses
        self._last_path_pub_time = None

        self.record = record

        self.initiaize_data_writer()

        self.node_name = node.get_name()
        # if self.node_name in ['joystick_controller','']:
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
    
    @classmethod
    def uvms_Forward_kinematics(cls, joint_qx, base_T0, world_pose, tipOffset):
        return cls.fk_eval_cls(joint_qx, base_T0, world_pose, tipOffset)

    @classmethod
    def manipulator_inverse_kinematics(cls, target_position):
        return cls.ik_eval_cls(target_position).full().flatten().tolist()

    @classmethod
    def uvms_body_inverse_kinematics(cls, target_pose):
        return cls.wb_ik_eval_cls(target_pose, alpha_params.base_T0_new, alpha_params.tipOffset,
                                   alpha_params.joint_min, alpha_params.joint_max).full().flatten().tolist()
    
    def _mocap_pose_cb(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        # Order matches your CSV header: x, y, z, qw, qx, qy, qz
        self.mocap_latest = [float(p.x), float(p.y), float(p.z),
                            float(q.w), float(q.x), float(q.y), float(q.z)]

    def compute_errors(self):
        st = self.get_state()
        goal_xyz, goal_quat_xyz = self.final_goal
        X_goal_des = list(goal_xyz) + list(goal_quat_xyz)
        # vehicle part
        X_curr = np.asarray(st["pose"], dtype=float)
        X_wp_des  = np.asarray(self.pose_command, dtype=float)

        err_wp = (X_wp_des - X_curr)
        err_wp_trans = np.abs(err_wp[:3])
        err_wp_rotation = np.abs(err_wp[3:])

        err_goal = (X_goal_des - X_curr)
        err_goal_trans = np.abs(err_goal[:3])
        err_goal_rotation = np.abs(err_goal[3:])

        # manipulator part, subtract per joint
        q_curr = np.asarray(st["q"], dtype=float).tolist()
        q_des  = np.asarray(self.arm.q_command, dtype=float).tolist()

        err_joints = [np.abs((Xd - Xc)) for Xd, Xc in zip(q_des, q_curr)]
        return err_wp_trans, err_wp_rotation, err_joints , err_goal_trans, err_goal_rotation

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

        self.ned_vel = self.to_ned_velocity(self.body_vel, self.ned_pose)
        
        self.sensor_reading = self.get_interface_value(
            msg,
            [self.floating_base_IOs] * len(self.sensors),
            self.sensors
        )

        self.prediction_readings = self.get_interface_value(
            msg,
            [self.floating_base_IOs] * len(self.prediction_interfaces),
            self.prediction_interfaces
        )
        self.state_estimate_readings = self.get_interface_value(
            msg,
            [self.floating_base_IOs] * len(self.state_estimate_interfaces),
            self.state_estimate_interfaces
        )

        self.payload_state_readings = self.get_interface_value(
            msg,
            [self.arm_IOs] * len(self.payload_state_interfaces),
            self.payload_state_interfaces
        )

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
        xq['raw_sensor_readings'] = self.sensor_reading
        xq['mocap'] = self.mocap_latest
        # self.node.get_logger().info(f"body forces {xq['raw_sensor_readings']}")
        return xq

    def to_ned_velocity(self, body_vel, pose):
        velocity_ned = copy.copy(body_vel)
        J_UV_REF = self.vehicle_J(pose[3:6])
        velocity_ned[:6] = J_UV_REF.full()@body_vel[:6]
        return velocity_ned

    def publish_robot_path(self):
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
        tra_path_msg.header.frame_id = f"{self.prefix}map" 

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

    def initiaize_data_writer(self):
        if self.record:
            # Create a timestamp string
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create a folder with the timestamp as its name (in the current working directory)
            folder_path = os.path.join(os.getcwd(), timestamp_str)
            os.makedirs(folder_path, exist_ok=True)
            
            # Create a timestamped filename for the CSV
            filename = f"{timestamp_str}_{self.prefix}.csv"
            file_path = os.path.join(folder_path, filename)
            
            # Open the CSV file and prepare to write data
            self.csv_file = open(file_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)

            # Write a header row for clarity
            columns = [
                'timestamp',
                'ros_time',
                'base_x_force', 'base_y_force', 'base_z_force', 'base_x_torque', 'base_y_torque', 'base_z_torque',
                'base_x', 'base_y', 'base_z', 'base_roll', 'base_pitch', 'base_yaw',
                'base_dx', 'base_dy', 'base_dz', 'base_vel_roll', 'base_vel_pitch', 'base_vel_yaw',
                
                'effort_alpha_axis_e', 'effort_alpha_axis_d', 'effort_alpha_axis_c', 'effort_alpha_axis_b',
                'q_alpha_axis_e', 'q_alpha_axis_d', 'q_alpha_axis_c', 'q_alpha_axis_b',
                'dq_alpha_axis_e', 'dq_alpha_axis_d', 'dq_alpha_axis_c', 'dq_alpha_axis_b',

                'imu_roll', 'imu_pitch', 'imu_yaw',
                'imu_roll_unwrap', 'imu_pitch_unwrap', 'imu_yaw_unwrap',
                'imu_q_w', 'imu_q_x', 'imu_q_y', 'imu_q_z',
                'imu_ang_vel_x', 'imu_ang_vel_y','imu_ang_vel_z',
                'imu_linear_acc_x', 'imu_linear_acc_y','imu_linear_acc_z',
                'depth_from_pressure2',
                'dvl_roll', 'dvl_pitch', 'dvl_yaw',
                'dvl_speed_x', 'dvl_speed_y', 'dvl_speed_z',

                'base_x_ref', 'base_y_ref', 'base_z_ref', 'base_roll_ref', 'base_pitch_ref', 'base_yaw_ref',
                'q_alpha_axis_e_ref', 'q_alpha_axis_d_ref', 'q_alpha_axis_c_ref', 'q_alpha_axis_b_ref', 'q_alpha_axis_a_ref',

                "position.x", "position.y", "position.z", "roll", "pitch", "yaw",
                "orientation.w", "orientation.x", "orientation.y", "orientation.z", 
                "velocity.x", "velocity.y", "velocity.z", 
                "angular_velocity.x", "angular_velocity.y", "angular_velocity.z",

                "position_estimate.x", "position_estimate.y", "position_estimate.z",
                "roll_estimate", "pitch_estimate", "yaw_estimate",
                "orientation_estimate.w", "orientation_estimate.x", "orientation_estimate.y", "orientation_estimate.z",
                "velocity_estimate.x", "velocity_estimate.y", "velocity_estimate.z",
                "angular_velocity_estimate.x", "angular_velocity_estimate.y", "angular_velocity_estimate.z",
                "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z",
                "angular_acceleration.x", "angular_acceleration.y", "angular_acceleration.z",
                "P_x_x", "P_y_y", "P_z_z", "P_roll_roll", "P_pitch_pitch", "P_yaw_yaw",
                "P_u_u", "P_v_v", "P_w_w", "P_p_p", "P_q_q", "P_r_r",

                "payload.mass", "payload.Ixx", "payload.Iyy", "payload.Izz",

                "mocap_x", "mocap_y", "mocap_z", "mocap_q_w", "mocap_q_x", "mocap_q_y", "mocap_q_z",
            ]

            self.csv_writer.writerow(columns)
    
    def write_data_to_file(self, ref=[0,0,0, 0,0,0, 0,0,0,0, 0]):
        if self.record:
            t_ros = float(self.node.get_clock().now().nanoseconds) * 1e-9
            row_data = []
            info = self.get_state()
            
            row_data.extend([info['sim_time']])
            row_data.extend([t_ros])
            
            row_data.extend(info['body_forces'])
            row_data.extend(info['pose'])
            row_data.extend(info['body_vel'])

            row_data.extend(info['arm_effort'])
            row_data.extend(info['q'])
            row_data.extend(info['dq'])
            
            row_data.extend(info['raw_sensor_readings'])
            row_data.extend(ref)
            row_data.extend(self.prediction_readings)
            row_data.extend(self.state_estimate_readings)
            row_data.extend(self.payload_state_readings)
            row_data.extend(info['mocap'])

            if all(value == 0 for value in row_data):
                return

            """Write a single row of data to the CSV file."""
            self.csv_writer.writerow(row_data)
            self.csv_file.flush()

    def publish_overlay_text_callback(self) -> None:
        str_msg = String()
        str_msg.data = f"© {datetime.now().year} Louisiana State University. Research use."
        self.overlay_text_publisher.publish(str_msg)

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

    def close_csv(self):
        # Close the CSV file when the node is destroyed
        self.csv_file.close()

    def listener_callback(self, msg: DynamicJointState):
        self.update_state(msg)
