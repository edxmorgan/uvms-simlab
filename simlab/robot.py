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
from geometry_msgs.msg import PoseStamped, Pose
from rclpy.qos import QoSProfile, QoSHistoryPolicy
import copy
from std_msgs.msg import Float32
from pyPS4Controller.controller import Controller
import threading
import glob
from typing import Sequence, Dict
from control_msgs.msg import DynamicInterfaceGroupValues
from std_msgs.msg import Float64MultiArray
from controller_msg import FullRobotMsg
from controllers import LowLevelController
from planner_markers import PathPlanner
from cartesian_ruckig import VehicleCartesianRuckig, EndeffectorCartesianRuckig
from alpha_reach import Params as alpha_params 
from frame_utils import PoseX
from tf2_ros import TransformException, Buffer
from tf2_geometry_msgs import do_transform_pose
from typing import Optional
from geometry_msgs.msg import Pose
from typing import Optional, Tuple, Sequence
import numpy as np
from geometry_msgs.msg import Pose

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

    # def on_L3_down(self, value):
    #     scaled = self.max_torque * (value / 32767.0)
    #     with self.ros_node.controller_lock:
    #         self.ros_node.rov_surge = -scaled

    def on_L3_down(self, value):
        scaled = self.max_torque * (value / 32767.0)
        with self.ros_node.controller_lock:
            self.ros_node.rov_surge = scaled

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
        self.joint_desired = [0.0]*n_joint

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
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            'arm_effort':self.effort,
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
                  controller='pid'):
        self.planner: PathPlanner = None
        self.vehicle_cart_traj: VehicleCartesianRuckig = None
        self.endeffector_cart_traj: EndeffectorCartesianRuckig = None
        self.menu_handle = None
        self.final_goal_in_world = None
        self.final_goal_map_ned_6 = None
        self.yaw_blend_factor = 0.0
        self.tf_buffer = tf_buffer
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

        self.fk_eval = ca.Function.load(fk_path) #  forward kinematics
        # also set a class attribute fk_eval so it can be shared
        if not hasattr(Robot, "fk_eval_cls"):
            Robot.fk_eval_cls = self.fk_eval

        self.ik_eval = ca.Function.load(ik_path) #  inverse kinematics
        # also set a class attribute ik_eval so it can be shared
        if not hasattr(Robot, "ik_eval_cls"):
            Robot.ik_eval_cls = self.ik_eval

        self.vehicle_J = ca.Function.load(vehicle_J_path)

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
        # self.use_controller = controller
        self.pose_command = [0.0]*6
        self.body_vel_command = [0.0]*6
        self.body_acc_command = [0.0]*6
        self.ll_controllers = LowLevelController(self.n_joint)

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

        self.traj_path_poses = []
        self.max_traj_pose_count = 2000  # cap RViz message size
        self.path_publish_period = 0.1  # seconds between stored poses
        self._last_path_pub_time = None
        self.joint_4_in_world = None

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
        self.planner_viz_timer = self.node.create_timer(1.0 / 10.0, self.planner_viz_callback)
        self.control_frequency = 60.0  # Hz
        self.control_timer = self.node.create_timer(1.0 / self.control_frequency, self.control_timer_callback)
        # Define a threshold error at which we start yaw blending.
        self.pos_blend_threshold = 1.1
        self.world_task_pose_timer = self.node.create_timer(1.0 / self.control_frequency, self.world_robot_task_pose_callback)

    @classmethod
    def uvms_Forward_kinematics(cls, joint_qx, base_T0, world_pose, tipOffset):
        return cls.fk_eval_cls(joint_qx, base_T0, world_pose, tipOffset)

    @classmethod
    def manipulator_inverse_kinematics(cls, target_position):
        return cls.ik_eval_cls(target_position).full().flatten().tolist()

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

        self.ned_vel = self.to_ned_velocity(self.body_vel, self.ned_pose)

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
            source_frame="world",
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

    def to_ned_velocity(self, body_vel, pose):
        velocity_ned = copy.copy(body_vel)
        J_UV_REF = self.vehicle_J(pose[3:6])
        velocity_ned[:6] = J_UV_REF.full()@body_vel[:6]
        return velocity_ned

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

    def planner_viz_callback(self):
        stamp_now = self.node.get_clock().now().to_msg()
        k_planner = self.planner
        if k_planner.planned_result and k_planner.planned_result['is_success']:
            k_planner.update_path_viz(
                stamp=stamp_now,
                frame_id="world",
                xyz_np=k_planner.planned_result["xyz"],
                step=3,
                wp_size=0.08,
                goal_size=0.14,
            )

    def world_robot_task_pose_callback(self):
        pose_world = self.get_frame_pose_in_frame(self.joint4_frame, "world")
        if pose_world is None:
            return
        self.joint_4_in_world = pose_world

    def control_timer_callback(self):
        state = self.get_state()
        if state['status'] == 'active':
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

                if pos_nwu is not None:
                    target_nwu = np.asarray(pos_nwu, dtype=float)

                    # Pick orientation from nearest OMPL waypoint
                    dists = np.linalg.norm(path_xyz - target_nwu, axis=1)
                    idx = int(np.argmin(dists))
                    target_quat = path_quat[idx]

                    q_arrow = self.planner.quat_wxyz_from_x_to_vec_scipy(vel_nwu)

                    self.planner.update_target_viz(
                        stamp=self.node.get_clock().now().to_msg(),
                        frame_id="world",
                        xyz=target_nwu,
                        quat_wxyz=q_arrow,
                        as_arrow=True,
                        size=0.10,
                        rate_hz=30.0,
                        ttl_sec=0.0,
                    )

                    res_map_ned = self.world_nwu_to_map_ned(
                        xyz_world_nwu=target_nwu,
                        quat_world_wxyz=target_quat,
                        warn_context=f"target world->map ({self.prefix})",
                    )
                    if res_map_ned is not None:
                        p_cmd_ned, rpy_cmd_ned = res_map_ned

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
            
            self.arm.q_command = self.arm.joint_desired  

        veh_state_vec = np.array(
            list(state['pose']) + list(state['body_vel']),
            dtype=float
        )
        # log to terminal
        # self.get_logger().info(f"robot command = {robot.pose_command}")

        cmd_body_wrench = self.ll_controllers.vehicle_controller(
            state=veh_state_vec,
            target=np.array(self.pose_command, dtype=float),
            dt=state["dt"]
        )

        # cmd_body_wrench = np.zeros(6)
        # cmd_body_wrench = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
        # Arm PID
        cmd_arm_tau = self.ll_controllers.arm_controller(
            q=state["q"],
            q_dot=state["dq"],
            q_ref=self.arm.q_command,
            Kp=alpha_params.Kp,
            Ki=alpha_params.Ki,
            Kd=alpha_params.Kd,
            dt=state["dt"],
            u_max=alpha_params.u_max,
            u_min=alpha_params.u_min,
            model_param=alpha_params.sim_p,
        )

        arm_tau_list = list(np.asarray(cmd_arm_tau, dtype=float).reshape(-1))
        # always produce 5 values, slice if longer, pad if shorter
        arm_tau_list = arm_tau_list[:5] + [0.0]

        self.publish_commands(cmd_body_wrench, arm_tau_list)