import numpy as np
from rclpy.node import Node

from simlab.controllers.oges import OgesModelbasedController

try:
    from namor import DRCLFOGES, ResidualBuffer, estimate_force_residual_from_velocity_error
    NAMOR_DR_IMPORT_ERROR = None
except ImportError as exc:
    DRCLFOGES = None
    ResidualBuffer = None
    estimate_force_residual_from_velocity_error = None
    NAMOR_DR_IMPORT_ERROR = exc


def _array(values) -> np.ndarray:
    if hasattr(values, "full"):
        values = values.full()
    return np.asarray(values, dtype=float)


class DistributionallyRobustOgesController(OgesModelbasedController):
    name = "DR-OGES"
    registry_name = "DR-Ours"

    def __init__(self, node: Node, arm_dof: int = 4, robot_prefix: str = ""):
        super().__init__(node, arm_dof, robot_prefix)
        if NAMOR_DR_IMPORT_ERROR is not None:
            raise ImportError(
                "DistributionallyRobustOgesController requires a namor version "
                "with DRCLFOGES support."
            ) from NAMOR_DR_IMPORT_ERROR
        self.vehicle_residual_buffer = ResidualBuffer(n_dof=6, maxlen=200)
        self.arm_residual_buffer = ResidualBuffer(n_dof=self.arm_dof, maxlen=200)
        self.vehicle_dr_projector = DRCLFOGES(
            n_dof=6,
            residual_buffer=self.vehicle_residual_buffer,
            wasserstein_radius=0.01,
            min_samples=50,
            effect_deadband=0.05,
            confidence_scale=2.0,
            projection_deadband=1e-6,
            slack_weight=1e8,
        )
        self.arm_dr_projector = DRCLFOGES(
            n_dof=self.arm_dof,
            residual_buffer=self.arm_residual_buffer,
            wasserstein_radius=0.005,
            min_samples=50,
            effect_deadband=0.01,
            confidence_scale=2.0,
            projection_deadband=1e-6,
            slack_weight=1e8,
        )
        self._prev_vehicle_sample = None
        self._prev_arm_sample = None
        self.last_vehicle_dr_result = None
        self.last_arm_dr_result = None
        self.node.get_logger().info(
            "\033[96mDistributionally robust OGES controller active as DR-Ours.\033[0m"
        )

    def reset_controller_state(self) -> None:
        self.vehicle_u_prev = np.zeros(6, dtype=float)
        self.arm_u_prev = np.zeros(self.arm_dof, dtype=float)
        self.vehicle_residual_buffer.clear()
        self.arm_residual_buffer.clear()
        self._prev_vehicle_sample = None
        self._prev_arm_sample = None
        self.last_vehicle_dr_result = None
        self.last_arm_dr_result = None

    def _update_vehicle_residual(self, qdot_now: np.ndarray) -> None:
        sample = self._prev_vehicle_sample
        if sample is None:
            return
        try:
            residual = estimate_force_residual_from_velocity_error(
                sample["H"],
                sample["qdot"],
                qdot_now,
                sample["u"],
                sample["F"],
                sample["dt"],
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.node.get_logger().debug(f"DR-Ours skipped vehicle residual update: {exc}")
            return
        if np.all(np.isfinite(residual)):
            self.vehicle_residual_buffer.append(residual)

    def _update_arm_residual(self, qdot_now: np.ndarray) -> None:
        sample = self._prev_arm_sample
        if sample is None:
            return
        try:
            residual = estimate_force_residual_from_velocity_error(
                sample["H"],
                sample["qdot"],
                qdot_now,
                sample["u"],
                sample["F"],
                sample["dt"],
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.node.get_logger().debug(f"DR-Ours skipped arm residual update: {exc}")
            return
        if np.all(np.isfinite(residual)):
            self.arm_residual_buffer.append(residual)

    def vehicle_controller(
        self,
        state: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_acc: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        dt = float(dt)

        vr_i = np.asarray(state[6:12], dtype=float).reshape(-1)
        eul = np.asarray(state[3:6], dtype=float).reshape(-1)
        self._update_vehicle_residual(vr_i)

        H_i = self.M_uv_matrix(self.blue.sim_p)
        C_i = self.C_uv_mat(vr_i.reshape(-1, 1), self.blue.sim_p)
        g_i = self.g_uv_vec(eul.reshape(-1, 1), self.blue.sim_p)
        Dp_i = self.Dp_uv_vec(vr_i.reshape(-1, 1), self.blue.sim_p)

        F_i = g_i + C_i @ vr_i.reshape(-1, 1) + Dp_i @ vr_i.reshape(-1, 1)
        H_np = _array(H_i)
        F_np = _array(F_i).reshape(-1)
        N_i = np.linalg.inv(H_np)
        Jk = self.J_uv(eul)
        Jk_ref = self.J_uv(target_pos[3:6])

        target_body_vel_ref = target_vel.copy()
        target_body_acc_ref = target_acc.copy()
        tau_nullspace = np.zeros(6, dtype=float)

        vehicle_policy_args = [
            state,
            self.vehicle_weights,
            N_i,
            H_np,
            F_np,
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

        u_nominal, V, null_err, idem_err, metric_err, clf_violation = self.vehicle_policy(*vehicle_policy_args)
        u_nominal = _array(u_nominal).reshape(-1)

        A, b, _ = self.uv_oges.A_b_V_func(
            state[:6],
            vr_i,
            target_pos,
            target_body_vel_ref,
            Jk_ref,
            target_body_acc_ref,
            self.vehicle_weights,
            Jk,
            self.vehicle_w_scale,
        )

        result = self.vehicle_dr_projector.project(
            u_nominal=u_nominal,
            A=_array(A).reshape(-1),
            b=float(_array(b).reshape(-1)[0]),
            H=H_np,
            F=F_np,
            u_min=self.blue.u_min,
            u_max=self.blue.u_max,
            u_prev=self.vehicle_u_prev,
            dt=dt,
        )
        self.last_vehicle_dr_result = result
        u = result.control
        self.vehicle_u_prev = u.copy()
        self._prev_vehicle_sample = {
            "H": H_np.copy(),
            "F": F_np.copy(),
            "qdot": vr_i.copy(),
            "u": u.copy(),
            "dt": dt,
        }
        return u

    def arm_controller(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        q_ref: np.ndarray,
        dq_ref: np.ndarray,
        ddq_ref: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        dt = float(dt)

        q_arm = np.asarray(q[: self.arm_dof], dtype=float)
        qdot_arm = np.asarray(q_dot[: self.arm_dof], dtype=float)
        qref_arm = np.asarray(q_ref[: self.arm_dof], dtype=float)
        dqref_arm = np.asarray(dq_ref[: self.arm_dof], dtype=float)
        ddqref_arm = np.asarray(ddq_ref[: self.arm_dof], dtype=float)
        self._update_arm_residual(qdot_arm)

        H_i = self.M_arm_matrix(q_arm, self.alpha_params.step_model_params)
        Cqot_i = self.Cqot_arm_vector(q_arm, qdot_arm, self.alpha_params.step_model_params)
        g_i = self.g_arm_vec(q_arm, self.alpha_params.step_model_params)
        B_i = self.B_arm_vec(qdot_arm, self.alpha_params.step_model_params)

        F_i = Cqot_i + g_i + B_i
        H_np = _array(H_i)
        F_np = _array(F_i).reshape(-1)
        N_i = np.linalg.inv(H_np)
        Jk = np.eye(self.arm_dof, dtype=float)
        Jk_ref = np.eye(self.arm_dof, dtype=float)
        tau_nullspace = np.zeros(self.arm_dof, dtype=float)

        arm_policy_args = [
            np.concatenate((q_arm, qdot_arm)),
            self.arm_weights,
            N_i,
            H_np,
            F_np,
            qref_arm,
            dqref_arm,
            ddqref_arm,
            Jk,
            Jk_ref,
            self.arm_w_scale,
            self.arm_u_min[: self.arm_dof],
            self.arm_u_max[: self.arm_dof],
            tau_nullspace,
        ]

        if self.use_arm_control_filter:
            arm_policy_args.extend([
                self.arm_u_prev,
                self.arm_lowpass_tau,
                dt,
            ])

        arm_nominal, V, null_err, idem_err, metric_err, clf_violation = self.arm_policy(*arm_policy_args)
        arm_nominal = _array(arm_nominal).reshape(-1)

        A, b, _ = self.arm_oges.A_b_V_func(
            q_arm,
            qdot_arm,
            qref_arm,
            dqref_arm,
            Jk_ref,
            ddqref_arm,
            self.arm_weights,
            Jk,
            self.arm_w_scale,
        )
        result = self.arm_dr_projector.project(
            u_nominal=arm_nominal,
            A=_array(A).reshape(-1),
            b=float(_array(b).reshape(-1)[0]),
            H=H_np,
            F=F_np,
            u_min=self.arm_u_min[: self.arm_dof],
            u_max=self.arm_u_max[: self.arm_dof],
            u_prev=self.arm_u_prev,
            dt=dt,
        )
        self.last_arm_dr_result = result
        arm_tau = result.control
        self.arm_u_prev = arm_tau.copy()
        self._prev_arm_sample = {
            "H": H_np.copy(),
            "F": F_np.copy(),
            "qdot": qdot_arm.copy(),
            "u": arm_tau.copy(),
            "dt": dt,
        }

        grasp_err = q_ref[-1] - q[-1]
        grasp_d_err = dq_ref[-1] - q_dot[-1]
        grasper_tau = self.grasper_kp * grasp_err + self.grasper_kd * grasp_d_err
        return np.concatenate((arm_tau, np.array([grasper_tau], dtype=float)))


__all__ = [
    "DistributionallyRobustOgesController",
    "NAMOR_DR_IMPORT_ERROR",
]
