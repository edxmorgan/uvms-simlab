import os

import ament_index_python
import casadi as ca
import numpy as np
from rclpy.node import Node

from simlab.alpha_reach import Params as alpha_params
from simlab.controllers.base import ControllerTemplate


class LowLevelOptimalModelbasedController(ControllerTemplate):
    name = "ModelBased"
    registry_name = "InvDyn"

    def __init__(self, node: Node, arm_dof: int = 4):
        super().__init__(node, arm_dof)
        package_share_directory = ament_index_python.get_package_share_directory("simlab")

        tracking_uv_path = os.path.join(
            package_share_directory,
            "vehicle/uv_trackingController.casadi",
        )
        self.tracking_uv_controller = ca.Function.load(tracking_uv_path)

        tracking_pid_path = os.path.join(
            package_share_directory,
            "manipulator/tracking_pid.casadi",
        )
        self.tracking_pid_controller = ca.Function.load(tracking_pid_path)

        self.arm_pid_i_buffer = np.zeros(self.arm_dof + 1, dtype=float)
        self.arm_kp = self.arm_vector(
            list(alpha_params.acc_Kp) + list(alpha_params.grasper_kp),
            "arm_kp",
        )
        self.arm_ki = self.arm_vector(
            list(alpha_params.acc_Ki) + list(alpha_params.grasper_ki),
            "arm_ki",
        )
        self.arm_kd = self.arm_vector(
            list(alpha_params.acc_Kd) + list(alpha_params.grasper_kd),
            "arm_kd",
        )
        self.arm_u_max = self.arm_vector(
            list(alpha_params.u_max) + list(alpha_params.grasper_u_max),
            "arm_u_max",
        )
        self.arm_u_min = self.arm_vector(
            list(alpha_params.u_min) + list(alpha_params.grasper_u_min),
            "arm_u_min",
        )
        self.arm_model_params = alpha_params.sim_p
        self.vehicle_model_params = [
            3.72028553e+01,
            2.21828075e+01,
            6.61734807e+01,
            3.38909801e+00,
            6.41362046e-01,
            6.41362034e-01,
            3.38909800e+00,
            1.39646394e+00,
            4.98032205e-01,
            2.53118738e+00,
            1.05000000e+02,
            9.78296453e+01,
            8.27479545e-01,
            1.36822559e-01,
            4.25841171e+00,
            -7.36416666e+01,
            -3.36082112e+01,
            -8.94055107e+01,
            -2.98736214e+00,
            -1.57921531e+00,
            -3.39766499e+00,
            -1.47912104e-04,
            -5.16373030e-04,
            -9.85522538e+01,
            -3.05907788e-02,
            -1.27877517e-01,
            -1.63514832e+00,
        ]
        self.uv_u_min = np.array([-20, -20, -20, -5, -5, -5])
        self.uv_u_max = np.array([20, 20, 20, 5, 5, 5])
        self.vehicle_i_limit = np.array([3, 3, 3, 3, 3, 3], dtype=float)

        self.kp = np.array([3.0, 3.0, 3.0, 0.5, 5.0, 0.4])
        self.ki = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.kd = np.array([5.0, 5.0, 5.0, 1.5, 10.0, 1.5])
        self.v_c = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def vehicle_controller(
        self,
        state: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_acc: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        state = self.vector(state, 12, "state")
        target_pos = self.vector(target_pos, 6, "target_pos")
        target_vel = self.vector(target_vel, 6, "target_vel")
        target_acc = self.vector(target_acc, 6, "target_acc")
        buf = np.zeros(6, dtype=float)

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
            ca.DM(self.v_c),
        )

        self.node.get_logger().debug(f"internal cmd_body_wrench {pid_control} : controller active.")
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
        dt: float,
    ) -> np.ndarray:
        q = self.arm_vector(q, "q")
        q_dot = self.arm_vector(q_dot, "q_dot")
        q_ref = self.arm_vector(q_ref, "q_ref")

        buf = np.asarray(self.arm_pid_i_buffer, dtype=float).reshape(-1)
        if buf.size != self.arm_dof + 1:
            buf = np.zeros(self.arm_dof + 1, dtype=float)

        u_sat, err, buf_next = self.tracking_pid_controller(
            ca.DM(q),
            ca.DM(q_dot),
            ca.DM(q_ref),
            ca.DM(dq_ref),
            ca.DM(ddq_ref),
            ca.DM(self.arm_kp),
            ca.DM(self.arm_ki),
            ca.DM(self.arm_kd),
            ca.DM(buf),
            float(dt),
            ca.DM(self.arm_u_max),
            ca.DM(self.arm_u_min),
            ca.DM(self.arm_model_params),
        )

        self.arm_pid_i_buffer = np.asarray(buf_next).reshape(-1)[: self.arm_dof + 1]
        return np.asarray(u_sat).reshape(-1)
