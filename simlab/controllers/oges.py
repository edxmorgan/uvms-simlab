import numpy as np
from namor import OGES, build_weight_vector
from namor import (
    load_alpha_reach_params,
    load_blue_rov_params,
    load_manipulator_model_function,
    load_uv_model_function,
)
from rclpy.node import Node

from simlab.controllers.base import ControllerTemplate


class OgesModelbasedController(ControllerTemplate):
    name = "OGES"
    registry_name = "Ours"
    arm_gain_profile = "tau"

    def __init__(self, node: Node, arm_dof: int = 4):
        super().__init__(node, arm_dof)
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

        self.node.get_logger().info(
            f"\033[96mOGES vehicle controller {self.vehicle_policy} : controller active.\033[0m"
        )
        self.node.get_logger().info(
            f"\033[93mOGES controller parameters{self.blue.sim_p} : controller active.\033[0m"
        )

    def vehicle_controller(
        self,
        state: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_acc: np.ndarray,
        dt: float,
    ) -> np.ndarray:
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

        vehicle_policy_args = [
            state,
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
            tau_nullspace,
        ]

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
