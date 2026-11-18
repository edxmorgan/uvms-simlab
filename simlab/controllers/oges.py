import numpy as np
from rclpy.node import Node

from simlab.controllers.base import ControllerTemplate
from simlab.dynamics_profiles import NominalModel, load_robot_dynamics_profile
from scipy.spatial.transform import Rotation
from simlab.uvms_parameters import select_robot_params

try:
    from namor import OGES, build_weight_vector
    from namor import (
        load_alpha_reach_params,
        load_blue_rov_params,
        load_manipulator_model_function,
        load_uv_model_function,
    )
    NAMOR_IMPORT_ERROR = None
except ImportError as exc:
    OGES = None
    build_weight_vector = None
    load_alpha_reach_params = None
    load_blue_rov_params = None
    load_manipulator_model_function = None
    load_uv_model_function = None
    NAMOR_IMPORT_ERROR = exc

class OgesModelbasedController(ControllerTemplate):
    name = "OGES"
    registry_name = "Ours"
    attitude_error = 'so3_log_body'
    configuration_fields = ('attitude_error', 'vehicle_weights', 'arm_weights',
                            'vehicle_w_scale', 'arm_w_scale', 'arm_u_min', 'arm_u_max',
                            'grasper_kp', 'grasper_kd')

    def set_model_profile(self, model) -> None:
        # Loaders may return parameter classes. Instantiate before assigning to
        # avoid modifying defaults shared by other robots/controllers.
        import copy
        blue = self.blue() if isinstance(self.blue, type) else copy.copy(self.blue)
        alpha = self.alpha_params() if isinstance(self.alpha_params, type) else copy.copy(self.alpha_params)
        blue.sim_p = model.vehicle_oges
        alpha.step_model_params = model.arm_oges
        self.blue, self.alpha_params = blue, alpha
        self.model_profile_sha256 = model.sha256
        self.reset_controller_state()

    def configuration_snapshot(self) -> dict:
        result = super().configuration_snapshot()
        result['parameters'].update(
            vehicle_model=np.asarray(self.blue.sim_p, dtype=float).reshape(-1).tolist(),
            arm_model=np.asarray(self.alpha_params.step_model_params, dtype=float).reshape(-1).tolist(),
            vehicle_u_min=np.asarray(self.blue.u_min, dtype=float).tolist(),
            vehicle_u_max=np.asarray(self.blue.u_max, dtype=float).tolist())
        return result

    def __init__(self, node: Node, arm_dof: int = 4, robot_prefix: str = ""):
        super().__init__(node, arm_dof, robot_prefix)
        if NAMOR_IMPORT_ERROR is not None:
            raise ImportError(
                "OgesModelbasedController requires the optional 'namor' dependency."
            ) from NAMOR_IMPORT_ERROR
        self.uv_oges = OGES(n_dof=6, use_jit=True, cyclic_dims=(3, 4, 5))
        uv_A, uv_b, uv_V = self.uv_oges.define_lyapunov_joint_constraints()
        self.vehicle_policy = self.uv_oges.controller(
            uv_A,
            uv_b,
            uv_V,
            filter_control=False,
            nominal_feedback=True,
        )

        self.arm_oges = OGES(n_dof=self.arm_dof, use_jit=True)
        arm_A, arm_b, arm_V = self.arm_oges.define_lyapunov_joint_constraints()
        self.arm_policy = self.arm_oges.controller(
            arm_A,
            arm_b,
            arm_V,
            filter_control=False,
        )

        self.vehicle_weights = build_weight_vector(
            a1=[15, 15, 30, 15, 15, 15],
            a2=[1, 1, 5, 0.2, 0.2, 0.2],
            cross_ratio=0.5,
            decay_rate=0.001,
        )

        self.arm_weights = build_weight_vector(
            a1=[100, 100, 100, 100],
            a2=[4., 3., 2., 100./225.],
            cross_ratio=0.95,
            decay_rate=0.001,
        )

        self.blue = load_blue_rov_params()
        self.alpha_params = load_alpha_reach_params()
        self.arm_params, self.vehicle_params = select_robot_params(self.robot_prefix)
        # Model layout differs from the adapters, but the physical defaults do
        # not: include the same mounting pose, current and actuator limits.
        import copy
        self.blue = self.blue() if isinstance(self.blue, type) else copy.copy(self.blue)
        self.alpha_params = (self.alpha_params() if isinstance(self.alpha_params, type)
                             else copy.copy(self.alpha_params))
        self.blue.sim_p = np.r_[np.asarray(self.vehicle_params.model_params).ravel(),
                               np.asarray(self.vehicle_params.v_c).ravel()]
        self.blue.u_min = np.asarray(self.vehicle_params.u_min).copy()
        self.blue.u_max = np.asarray(self.vehicle_params.u_max).copy()
        self.alpha_params.step_model_params = np.r_[
            np.asarray(self.arm_params.sim_p).ravel(), self.arm_params.tipOffset]

        self.M_uv_matrix = load_uv_model_function("M_id.casadi")
        self.C_uv_mat = load_uv_model_function("C_id.casadi")
        self.g_uv_vec = load_uv_model_function("g_depth_id.casadi")
        self.Dp_uv_vec = load_uv_model_function("body_damping_matrix_id.casadi")

        self.M_arm_matrix = load_manipulator_model_function("alpha_id_D.casadi")
        self.Cqot_arm_vector = load_manipulator_model_function("alpha_id_Cqot.casadi")
        self.g_arm_vec = load_manipulator_model_function("alpha_id_g.casadi")
        self.B_arm_vec = load_manipulator_model_function("alpha_id_B.casadi")

        self.arm_u_min = self.arm_vector(
            list(self.arm_params.u_min) + list(self.arm_params.grasper_u_min),
            "arm_u_min",
        )
        self.arm_u_max = self.arm_vector(
            list(self.arm_params.u_max) + list(self.arm_params.grasper_u_max),
            "arm_u_max",
        )
        self.grasper_kp = float(self.arm_params.grasper_kp[0])
        self.grasper_kd = float(self.arm_params.grasper_kd[0])
        self.vehicle_w_scale = 0.1
        self.arm_w_scale = 1.0
        if 'real' not in robot_prefix:
            # Shared engineer-selected nominal model, independent of plant changes.
            model = NominalModel.from_profile(load_robot_dynamics_profile('dory_alpha', node))
            self.blue.sim_p = model.vehicle_oges
            self.alpha_params.step_model_params = model.arm_oges
            self.model_profile_sha256 = model.sha256

    def bounded_arm_effort(self, arm_tau, grasper_tau):
        effort = self.arm_vector(np.r_[arm_tau, grasper_tau], 'arm effort')
        if not np.isfinite(effort).all():
            raise ValueError('OGES returned nonfinite arm effort')
        return np.clip(effort, self.arm_u_min, self.arm_u_max)

    def _evaluate_policy(self, subsystem, args, dt):
        """Call the controller API; DR overrides only this final evaluation."""
        policy = getattr(self, subsystem + '_policy')
        return np.asarray(policy(*args)[0], dtype=float).reshape(-1)

    def vehicle_controller(
        self,
        state: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_acc: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('OGES vehicle dt must be finite and positive')

        vr_i = state[6:12].reshape(-1, 1)

        H_i = self.M_uv_matrix(self.blue.sim_p)
        C_i = self.C_uv_mat(vr_i, self.blue.sim_p)
        g_i = self.g_uv_vec(state[:6], self.blue.sim_p)
        Dp_i = self.Dp_uv_vec(vr_i, self.blue.sim_p)

        F_i = g_i + C_i @ vr_i + Dp_i @ vr_i
        N_i = np.linalg.inv(np.asarray(H_i, dtype=float))
        # Physical twists in world axes, not Euler-coordinate rates.
        R = Rotation.from_euler('xyz', state[3:6]).as_matrix()
        Rd = Rotation.from_euler('xyz', target_pos[3:6]).as_matrix()
        Jk = np.zeros((6, 6))
        Jk_ref = np.zeros((6, 6))
        Jk[:3, :3] = Jk[3:, 3:] = R
        Jk_ref[:3, :3] = Jk_ref[3:, 3:] = Rd

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

        return self._evaluate_policy('vehicle', vehicle_policy_args, dt)

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
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('OGES arm dt must be finite and positive')

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
        N_i = np.linalg.inv(np.asarray(H_i, dtype=float))
        Jk = np.eye(self.arm_dof, dtype=float)
        Jk_ref = np.eye(self.arm_dof, dtype=float)
        tau_nullspace = np.zeros(self.arm_dof, dtype=float)
        arm_policy_args = [
            np.concatenate((q_arm, qdot_arm)), self.arm_weights, N_i, H_i, F_i,
            qref_arm, dqref_arm, ddqref_arm, Jk, Jk_ref, self.arm_w_scale,
            self.arm_u_min[:self.arm_dof], self.arm_u_max[:self.arm_dof], tau_nullspace,
        ]
        arm_tau = self._evaluate_policy('arm', arm_policy_args, dt)

        grasp_err = q_ref[-1] - q[-1]
        grasp_d_err = dq_ref[-1] - q_dot[-1]
        grasper_tau = self.grasper_kp * grasp_err + self.grasper_kd * grasp_d_err
        return self.bounded_arm_effort(arm_tau, grasper_tau)
