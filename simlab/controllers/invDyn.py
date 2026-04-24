import os

import ament_index_python
import casadi as ca
import numpy as np
from rclpy.node import Node

from simlab.controllers.base import ControllerTemplate
from simlab.uvms_parameters import ReachParams, VehicleControllerParams


class LowLevelInvDynController(ControllerTemplate):
    name = "InvDynController"
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
            list(ReachParams.invdyn_kp) + list(ReachParams.grasper_kp),
            "arm_kp",
        )
        self.arm_ki = self.arm_vector(
            list(ReachParams.invdyn_ki) + list(ReachParams.grasper_ki),
            "arm_ki",
        )
        self.arm_kd = self.arm_vector(
            list(ReachParams.invdyn_kd) + list(ReachParams.grasper_kd),
            "arm_kd",
        )
        self.arm_u_max = self.arm_vector(
            list(ReachParams.u_max) + list(ReachParams.grasper_u_max),
            "arm_u_max",
        )
        self.arm_u_min = self.arm_vector(
            list(ReachParams.u_min) + list(ReachParams.grasper_u_min),
            "arm_u_min",
        )
        self.arm_model_params = ReachParams.sim_p
        self.vehicle_model_params = VehicleControllerParams.model_params.copy()
        self.uv_u_min = VehicleControllerParams.u_min.copy()
        self.uv_u_max = VehicleControllerParams.u_max.copy()
        self.vehicle_i_limit = VehicleControllerParams.i_limit.copy()

        self.kp = VehicleControllerParams.invdyn_kp.copy()
        self.ki = VehicleControllerParams.invdyn_ki.copy()
        self.kd = VehicleControllerParams.invdyn_kd.copy()
        self.v_c = VehicleControllerParams.v_c.copy()

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
