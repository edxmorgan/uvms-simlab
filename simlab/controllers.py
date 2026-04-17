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
import ament_index_python
import os
import casadi as ca
from rclpy.node import Node
from namor import OGES, build_weight_vector
from namor import (
    load_alpha_reach_params,
    load_blue_rov_params,
    load_manipulator_model_function,
    load_uv_model_function,
)

class LowLevelPidController:
    def __init__(self, node: Node, arm_dof: int = 4):
        package_share_directory = ament_index_python.get_package_share_directory('simlab')

        # Vehicle PID
        uv_pid_controller_path = os.path.join(package_share_directory, 'vehicle/uv_reg_pid_controller.casadi')
        self.uv_pid_controller = ca.Function.load(uv_pid_controller_path)
        self.vehicle_pid_i_buffer = np.zeros(6, dtype=float)  # 6 dof vehicle integral buffer

        # Integral buffer hardening, clamp and leak
        self.vehicle_i_limit = np.array([3, 3, 3, 3, 3, 3], dtype=float)  # per axis clamp

        # Arm PID
        arm_pid_controller_path = os.path.join(package_share_directory, 'manipulator/arm_pid.casadi')
        self.arm_pid_controller = ca.Function.load(arm_pid_controller_path)

        self.arm_dof = int(arm_dof)
        self.arm_pid_i_buffer = np.zeros(self.arm_dof+1, dtype=float)  # arm integral buffer

        self.vehicle_model_params = [3.72028553e+01, 2.21828075e+01, 6.61734807e+01, 3.38909801e+00,
                                  6.41362046e-01, 6.41362034e-01, 3.38909800e+00, 1.39646394e+00,
                                  4.98032205e-01, 2.53118738e+00, 1.05000000e+02, 9.78296453e+01,
                                  8.27479545e-01, 1.36822559e-01, 4.25841171e+00, -7.36416666e+01,
                                  -3.36082112e+01, -8.94055107e+01, -2.98736214e+00, -1.57921531e+00,
                                  -3.39766499e+00, -1.47912104e-04, -5.16373030e-04, -9.85522538e+01,
                                  -3.05907788e-02, -1.27877517e-01, -1.63514832e+00]

        self.kp = np.array([40.0, 40.0, 40.0, 10, 10, 10.0])
        self.ki = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.kd = np.array([15.0, 15.0, 15.0, 2, 2, 5.0])

    def vehicle_controller(self, state: np.ndarray, target_pos: np.ndarray, 
                           target_vel: np.ndarray, 
                           target_acc: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute vehicle body wrench with a CasADi PID function.

        state  shape (12,) = [x, y, z, roll, pitch, yaw, u, v, w, p, q, r]
        target_pos shape (6,)  = [x, y, z, roll, pitch, yaw]
        """
        state  = np.asarray(state, dtype=float).reshape(-1)
        target_pos = np.asarray(target_pos, dtype=float).reshape(-1)
        target_vel = np.asarray(target_vel, dtype=float).reshape(-1)

        if state.size != 12:
            raise ValueError(f"state must have 12 elements, got {state.size}")
        if target_pos.size != 6:
            raise ValueError(f"target_pos must have 6 elements, got {target_pos.size}")
        if target_vel.size != 6:
            raise ValueError(f"target_vel must have 6 elements, got {target_vel.size}")

        buf = np.zeros(6, dtype=float)  # Disable integral action for now

        pid_control, i_buf_next = self.uv_pid_controller(
            self.vehicle_model_params,
            self.kp,
            self.ki,
            self.kd,
            ca.DM(buf),
            ca.DM(state),
            ca.DM(target_pos),
            ca.DM(target_vel),
            float(dt),
        )

        # Update and clamp the returned integral buffer
        self.vehicle_pid_i_buffer = np.clip(
            np.asarray(i_buf_next).reshape(-1)[:6],
            -self.vehicle_i_limit,
            self.vehicle_i_limit,
        )

        return np.asarray(pid_control).reshape(-1)

    def arm_controller(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        q_ref: np.ndarray,
        dq_ref: np.ndarray,
        ddq_ref: np.ndarray,
        Kp: np.ndarray,
        Ki: np.ndarray,
        Kd: np.ndarray,
        dt: float,
        u_max: np.ndarray,
        u_min: np.ndarray,
        model_param: np.ndarray,
    ) -> np.ndarray:
        """
        Compute arm joint torques with the simple PID CasADi function you built.

        All vectors are length arm_dof.
        Returns saturated torque command, and updates the internal integral buffer.
        """
        # Coerce shapes
        def v(x):
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.size != self.arm_dof+1:
                raise ValueError(f"expected length {self.arm_dof+1}, got {x.size}")
            return x

        q     = v(q)
        q_dot = v(q_dot)
        q_ref = v(q_ref)
        Kp    = v(Kp)
        Ki    = v(Ki)
        Kd    = v(Kd)
        u_max = v(u_max)
        u_min = v(u_min)

        buf = np.asarray(self.arm_pid_i_buffer, dtype=float).reshape(-1)
        if buf.size != self.arm_dof+1:
            buf = np.zeros(self.arm_dof+1, dtype=float)

        u_sat, err, buf_next = self.arm_pid_controller(
            ca.DM(q),
            ca.DM(q_dot),
            ca.DM(q_ref),
            ca.DM(Kp),
            ca.DM(Ki),
            ca.DM(Kd),
            ca.DM(buf),
            float(dt),
            ca.DM(u_max),
            ca.DM(u_min),
            ca.DM(model_param)
        )

        self.arm_pid_i_buffer = np.asarray(buf_next).reshape(-1)[: self.arm_dof+1]
        return np.asarray(u_sat).reshape(-1)



class LowLevelOptimalModelbasedController:
    def __init__(self, node: Node, arm_dof: int = 4):
        self.node = node
        package_share_directory = ament_index_python.get_package_share_directory('simlab')
        tracking_uv_path = os.path.join(package_share_directory, 'vehicle/uv_trackingController.casadi')

        self.tracking_uv_controller = ca.Function.load(tracking_uv_path)

        # Arm tracking PID
        tracking_pid_path = os.path.join(package_share_directory, 'manipulator/tracking_pid.casadi')
        self.tracking_pid_controller = ca.Function.load(tracking_pid_path)

        self.arm_dof = int(arm_dof)
        self.arm_pid_i_buffer = np.zeros(self.arm_dof+1, dtype=float)  # arm integral buffer

        self.vehicle_model_params = [3.72028553e+01, 2.21828075e+01, 6.61734807e+01, 3.38909801e+00,
                                  6.41362046e-01, 6.41362034e-01, 3.38909800e+00, 1.39646394e+00,
                                  4.98032205e-01, 2.53118738e+00, 1.05000000e+02, 9.78296453e+01,
                                  8.27479545e-01, 1.36822559e-01, 4.25841171e+00, -7.36416666e+01,
                                  -3.36082112e+01, -8.94055107e+01, -2.98736214e+00, -1.57921531e+00,
                                  -3.39766499e+00, -1.47912104e-04, -5.16373030e-04, -9.85522538e+01,
                                  -3.05907788e-02, -1.27877517e-01, -1.63514832e+00]
        
        self.uv_u_min = np.array([-20, -20, -20, -5, -5, -5])
        self.uv_u_max = np.array([20, 20, 20, 5, 5, 5])

        # Integral buffer hardening, clamp and leak
        self.vehicle_i_limit = np.array([3, 3, 3, 3, 3, 3], dtype=float)  # per axis clamp

        self.kp = np.array([3.0, 3.0, 3.0, 0.5, 5.0, 0.4])
        self.ki = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.kd = np.array([5.0, 5.0, 5.0, 1.5, 10.0, 1.5])
        self.v_c =  np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def vehicle_controller(self, state: np.ndarray, target_pos: np.ndarray, target_vel: np.ndarray, target_acc: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute vehicle body wrench with a CasADi PID function.

        state  shape (12,) = [x, y, z, roll, pitch, yaw, u, v, w, p, q, r]
        target_pos shape (6,)  = [x, y, z, roll, pitch, yaw]
        """
        state  = np.asarray(state, dtype=float).reshape(-1)
        target_pos = np.asarray(target_pos, dtype=float).reshape(-1)
        target_vel = np.asarray(target_vel, dtype=float).reshape(-1)
        target_acc = np.asarray(target_acc, dtype=float).reshape(-1)

        if state.size != 12:
            raise ValueError(f"state must have 12 elements, got {state.size}")
        if target_pos.size != 6:
            raise ValueError(f"target_pos must have 6 elements, got {target_pos.size}")
        if target_vel.size != 6:
            raise ValueError(f"target_vel must have 6 elements, got {target_vel.size}")
        if target_acc.size != 6:
            raise ValueError(f"target_acc must have 6 elements, got {target_acc.size}")
        
        buf = np.zeros(6, dtype=float)  # Disable integral action for now

        self.node.get_logger().debug(f"internal target_pos {target_pos} : controller active.")
        self.node.get_logger().debug(f"internal target_vel {target_vel} : controller active.")
        self.node.get_logger().debug(f"internal target_acc {target_acc} : controller active.")

        pid_control, i_buf_next = self.tracking_uv_controller(
            self.vehicle_model_params,
            self.kp,
            self.ki,
            self.kd,
            ca.DM(buf),
            ca.DM(state),
            ca.DM(target_pos),
            ca.DM(target_vel),
            ca.DM(target_acc),
            float(dt),
            ca.DM(self.v_c)
        )

        self.node.get_logger().debug(f"internal cmd_body_wrench {pid_control} : controller active.")

        # beta_sym, Kp, Ki, Kd, sum_e_buffer, sim_x, nd, xb_d, des_acc_b, dt, v_c
        # Update and clamp the returned integral buffer
        self.vehicle_pid_i_buffer = np.clip(
            np.asarray(i_buf_next).reshape(-1)[:6],
            -self.vehicle_i_limit,
            self.vehicle_i_limit,
        )

        return np.asarray(pid_control).reshape(-1)

    def arm_controller(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        q_ref: np.ndarray,
        dq_ref: np.ndarray,
        ddq_ref: np.ndarray,
        Kp: np.ndarray,
        Ki: np.ndarray,
        Kd: np.ndarray,
        dt: float,
        u_max: np.ndarray,
        u_min: np.ndarray,
        model_param: np.ndarray,
    ) -> np.ndarray:
        """
        Compute arm joint torques with the simple PID CasADi function you built.

        All vectors are length arm_dof.
        Returns saturated torque command, and updates the internal integral buffer.
        """
        # Coerce shapes
        def v(x):
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.size != self.arm_dof+1:
                raise ValueError(f"expected length {self.arm_dof+1}, got {x.size}")
            return x

        q     = v(q)
        q_dot = v(q_dot)
        q_ref = v(q_ref)
        Kp    = v(Kp)
        Ki    = v(Ki)
        Kd    = v(Kd)
        u_max = v(u_max)
        u_min = v(u_min)

        buf = np.asarray(self.arm_pid_i_buffer, dtype=float).reshape(-1)
        if buf.size != self.arm_dof+1:
            buf = np.zeros(self.arm_dof+1, dtype=float)

        u_sat, err, buf_next = self.tracking_pid_controller(
            ca.DM(q),
            ca.DM(q_dot),
            ca.DM(q_ref),
            ca.DM(dq_ref),
            ca.DM(ddq_ref),
            ca.DM(Kp),
            ca.DM(Ki),
            ca.DM(Kd),
            ca.DM(buf),
            float(dt),
            ca.DM(u_max),
            ca.DM(u_min),
            ca.DM(model_param)
        )

        self.arm_pid_i_buffer = np.asarray(buf_next).reshape(-1)[: self.arm_dof+1]
        return np.asarray(u_sat).reshape(-1)


class OgesModelbasedController:
    def __init__(self, node: Node, arm_dof: int = 4):
        self.node = node
        self.arm_dof = int(arm_dof)
        self.use_vehicle_control_filter = True
        self.use_arm_control_filter = False

        uv_oges = OGES(n_dof=6, use_jit=True, cyclic_dims=(3, 4, 5))
        uv_A, uv_b, uv_V = uv_oges.define_lyapunov_joint_constraints()
        self.vehicle_policy = uv_oges.controller(
            uv_A,
            uv_b,
            uv_V,
            include_constraint_violation=True,
            filter_control=self.use_vehicle_control_filter,
        )

        arm_oges = OGES(n_dof=self.arm_dof, use_jit=True)
        arm_A, arm_b, arm_V = arm_oges.define_lyapunov_joint_constraints()
        self.arm_policy = arm_oges.controller(
            arm_A,
            arm_b,
            arm_V,
            include_constraint_violation=True,
            filter_control=self.use_arm_control_filter,
        )

        self.vehicle_weights = build_weight_vector(
            a1=[15, 15, 30, 15, 15, 15],
            a2=[1, 1, 5, 0.2, 0.2, 0.2],
            cross_ratio=0.5,
            decay_rate=0.001,
        )

        self.arm_weights = build_weight_vector(
            a1=[100, 100, 100, 100],
            a2=[4, 3, 2, 0.04],
            cross_ratio=0.95,
            decay_rate=0.001,
        )

        self.blue = load_blue_rov_params()
        self.alpha_params = load_alpha_reach_params()

        self.M_uv_matrix = load_uv_model_function("M_id.casadi")
        self.C_uv_mat = load_uv_model_function("C_id.casadi")
        self.g_uv_vec = load_uv_model_function("g_id.casadi")
        self.Dp_uv_vec = load_uv_model_function("body_damping_matrix_id.casadi")
        self.J_uv = load_uv_model_function("J_uv.casadi")

        self.M_arm_matrix = load_manipulator_model_function("alpha_id_D.casadi")
        self.Cqot_arm_vector = load_manipulator_model_function("alpha_id_Cqot.casadi")
        self.g_arm_vec = load_manipulator_model_function("alpha_id_g.casadi")
        self.B_arm_vec = load_manipulator_model_function("alpha_id_B.casadi")

        self.vehicle_u_prev = np.zeros(6, dtype=float)
        self.arm_u_prev = np.zeros(self.arm_dof, dtype=float)
        self.vehicle_w_scale = 0.1
        self.arm_w_scale = 1.0
        self.vehicle_lowpass_tau = np.full(6, 0.0)
        self.arm_lowpass_tau = np.array([0.1, 0.1, 0.1, 0.25])

        self.node.get_logger().info(f"\033[96mOGES vehicle controller {self.vehicle_policy} : controller active.\033[0m")
        self.node.get_logger().info(f"\033[93mOGES controller parameters{self.blue.sim_p} : controller active.\033[0m")

    def vehicle_controller(self, state: np.ndarray, target_pos: np.ndarray, target_vel: np.ndarray, target_acc: np.ndarray, dt: float) -> np.ndarray:
        dt = float(dt)

        vr_i = state[6:12].reshape(-1, 1)
        eul = state[3:6].reshape(-1, 1)

        H_i = self.M_uv_matrix(self.blue.sim_p)
        C_i = self.C_uv_mat(vr_i, self.blue.sim_p)
        g_i = self.g_uv_vec(eul, self.blue.sim_p)
        Dp_i = self.Dp_uv_vec(vr_i, self.blue.sim_p)

        F_i = g_i + C_i @ vr_i + Dp_i @ vr_i
        N_i = np.linalg.inv(H_i)
        Jk = self.J_uv(eul)
        Jk_ref = self.J_uv(target_pos[3:6])

        target_body_vel_ref = target_vel.copy()
        target_body_acc_ref = target_acc.copy()
        tau_nullspace = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

        vehicle_policy_args = [state,
            self.vehicle_weights,
            N_i,
            H_i,
            F_i,
            target_pos,
            target_body_vel_ref,
            target_body_acc_ref,
            Jk,
            Jk_ref,
            self.vehicle_w_scale,
            self.blue.u_min,
            self.blue.u_max,
            tau_nullspace]
        
        if self.use_vehicle_control_filter:
            vehicle_policy_args.extend([
                self.vehicle_u_prev,
                self.vehicle_lowpass_tau,
                dt,
            ])
        
        u, V, null_err, idem_err, metric_err, clf_violation = self.vehicle_policy(*vehicle_policy_args)
        u = np.asarray(u.full(), dtype=float).reshape(-1)
        self.vehicle_u_prev = u.copy()
        return u

    def arm_controller(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        q_ref: np.ndarray,
        dq_ref: np.ndarray,
        ddq_ref: np.ndarray,
        Kp: np.ndarray,
        Ki: np.ndarray,
        Kd: np.ndarray,
        dt: float,
        u_max: np.ndarray,
        u_min: np.ndarray,
        model_param: np.ndarray,
    ) -> np.ndarray:

        dt = float(dt)

        q_arm = q[: self.arm_dof]
        qdot_arm = q_dot[: self.arm_dof]
        qref_arm = q_ref[: self.arm_dof]
        dqref_arm = dq_ref[: self.arm_dof]
        ddqref_arm = ddq_ref[: self.arm_dof]

        H_i = self.M_arm_matrix(q_arm, self.alpha_params.step_model_params)
        Cqot_i = self.Cqot_arm_vector(q_arm, qdot_arm, self.alpha_params.step_model_params)
        g_i = self.g_arm_vec(q_arm, self.alpha_params.step_model_params)
        B_i = self.B_arm_vec(qdot_arm, self.alpha_params.step_model_params)

        F_i = Cqot_i + g_i + B_i
        N_i = np.linalg.inv(H_i)
        Jk = np.eye(self.arm_dof, dtype=float)
        Jk_ref = np.eye(self.arm_dof, dtype=float)
        tau_nullspace = np.zeros(self.arm_dof, dtype=float)

        arm_policy_args = [
            np.concatenate((q_arm, qdot_arm)),
            self.arm_weights,
            N_i,
            H_i,
            F_i,
            qref_arm,
            dqref_arm,
            ddqref_arm,
            Jk,
            Jk_ref,
            self.arm_w_scale,
            u_min[: self.arm_dof],
            u_max[: self.arm_dof],
            tau_nullspace,
        ]
            
        if self.use_arm_control_filter:
            arm_policy_args.extend([
                self.arm_u_prev,
                self.arm_lowpass_tau,
                dt,
            ])

        arm_tau, V, null_err, idem_err, metric_err, clf_violation = self.arm_policy(*arm_policy_args)

        arm_tau = np.asarray(arm_tau.full(), dtype=float).reshape(-1)
        self.arm_u_prev = arm_tau.copy()

        grasp_err = q_ref[-1] - q[-1]
        grasp_d_err = dq_ref[-1] - q_dot[-1]
        grasper_tau = Kp[-1] * grasp_err + Kd[-1] * grasp_d_err
        return np.concatenate((arm_tau, np.array([grasper_tau], dtype=float)))
