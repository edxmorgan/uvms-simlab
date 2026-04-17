import os

import ament_index_python
import casadi as ca
import numpy as np
from rclpy.node import Node

from simlab.controllers.base import ControllerTemplate, VEHICLE_MODEL_PARAMS


class LowLevelOptimalModelbasedController(ControllerTemplate):
    name = "ModelBased"
    registry_name = "InvDyn"
    arm_gain_profile = "acc"

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
        self.vehicle_model_params = VEHICLE_MODEL_PARAMS
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
            ca.DM(model_param),
        )

        self.arm_pid_i_buffer = np.asarray(buf_next).reshape(-1)[: self.arm_dof + 1]
        return np.asarray(u_sat).reshape(-1)
