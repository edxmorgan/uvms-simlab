import os

import ament_index_python
import casadi as ca
import numpy as np
from rclpy.node import Node

from simlab.controllers.base import ControllerTemplate, VEHICLE_MODEL_PARAMS


class LowLevelPidController(ControllerTemplate):
    name = "PID"
    registry_name = "PID"
    arm_gain_profile = "tau"

    def __init__(self, node: Node, arm_dof: int = 4):
        super().__init__(node, arm_dof)
        package_share_directory = ament_index_python.get_package_share_directory("simlab")

        uv_pid_controller_path = os.path.join(
            package_share_directory,
            "vehicle/uv_reg_pid_controller.casadi",
        )
        self.uv_pid_controller = ca.Function.load(uv_pid_controller_path)
        self.vehicle_pid_i_buffer = np.zeros(6, dtype=float)
        self.vehicle_i_limit = np.array([3, 3, 3, 3, 3, 3], dtype=float)

        arm_pid_controller_path = os.path.join(
            package_share_directory,
            "manipulator/arm_pid.casadi",
        )
        self.arm_pid_controller = ca.Function.load(arm_pid_controller_path)
        self.arm_pid_i_buffer = np.zeros(self.arm_dof + 1, dtype=float)

        self.vehicle_model_params = VEHICLE_MODEL_PARAMS
        self.kp = np.array([40.0, 40.0, 40.0, 10, 10, 10.0])
        self.ki = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.kd = np.array([15.0, 15.0, 15.0, 2, 2, 5.0])

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

        buf = np.zeros(6, dtype=float)
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
        q = self.arm_vector(q, "q")
        q_dot = self.arm_vector(q_dot, "q_dot")
        q_ref = self.arm_vector(q_ref, "q_ref")
        Kp = self.arm_vector(Kp, "Kp")
        Ki = self.arm_vector(Ki, "Ki")
        Kd = self.arm_vector(Kd, "Kd")
        u_max = self.arm_vector(u_max, "u_max")
        u_min = self.arm_vector(u_min, "u_min")

        buf = np.asarray(self.arm_pid_i_buffer, dtype=float).reshape(-1)
        if buf.size != self.arm_dof + 1:
            buf = np.zeros(self.arm_dof + 1, dtype=float)

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
            ca.DM(model_param),
        )

        self.arm_pid_i_buffer = np.asarray(buf_next).reshape(-1)[: self.arm_dof + 1]
        return np.asarray(u_sat).reshape(-1)
