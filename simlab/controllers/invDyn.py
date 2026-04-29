import os

import ament_index_python
import casadi as ca
import numpy as np
from rclpy.node import Node

from simlab.controllers.base import ControllerTemplate
from simlab.uvms_parameters import select_robot_params


class LowLevelInvDynController(ControllerTemplate):
    name = "InvDynController"
    registry_name = "InvDyn"

    def __init__(self, node: Node, arm_dof: int = 4, robot_prefix: str = ""):
        super().__init__(node, arm_dof, robot_prefix)
        package_share_directory = ament_index_python.get_package_share_directory("simlab")

        tracking_uv_path = os.path.join(
            package_share_directory,
            "model_functions/vehicle/uv_trackingController.casadi",
        )
        self.tracking_uv_controller = ca.Function.load(tracking_uv_path)

        tracking_pid_path = os.path.join(
            package_share_directory,
            "model_functions/arm/tracking_pid.casadi",
        )
        self.tracking_pid_controller = ca.Function.load(tracking_pid_path)

        self.arm_params, self.vehicle_params = select_robot_params(self.robot_prefix)
        self.arm_pid_i_buffer = np.zeros(self.arm_dof + 1, dtype=float)
        self.arm_kp = self.arm_vector(
            list(self.arm_params.invdyn_kp) + list(self.arm_params.grasper_kp),
            "arm_kp",
        )
        self.arm_ki = self.arm_vector(
            list(self.arm_params.invdyn_ki) + list(self.arm_params.grasper_ki),
            "arm_ki",
        )
        self.arm_kd = self.arm_vector(
            list(self.arm_params.invdyn_kd) + list(self.arm_params.grasper_kd),
            "arm_kd",
        )
        self.arm_u_max = self.arm_vector(
            list(self.arm_params.u_max) + list(self.arm_params.grasper_u_max),
            "arm_u_max",
        )
        self.arm_u_min = self.arm_vector(
            list(self.arm_params.u_min) + list(self.arm_params.grasper_u_min),
            "arm_u_min",
        )
        self.arm_model_params = self.arm_params.sim_p
        self.vehicle_model_params = self.vehicle_params.model_params
        self.uv_u_min = self.vehicle_params.u_min
        self.uv_u_max = self.vehicle_params.u_max
        self.vehicle_i_limit = self.vehicle_params.i_limit

        self.kp = self.vehicle_params.invdyn_kp
        self.ki = self.vehicle_params.invdyn_ki
        self.kd = self.vehicle_params.invdyn_kd
        self.v_c = self.vehicle_params.v_c
        self.node.get_logger().info(
            f"InvDyn params for {self.robot_prefix or 'default'}: "
            f"arm_profile={self.arm_params.profile_name}, "
            f"vehicle_profile={self.vehicle_params.profile_name}"
        )

    def reset_controller_state(self) -> None:
        self.arm_pid_i_buffer = np.zeros(self.arm_dof + 1, dtype=float)

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
