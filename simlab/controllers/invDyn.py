import os

import ament_index_python
import casadi as ca
import numpy as np
from rclpy.node import Node

from simlab.controllers.base import ControllerTemplate, bounded_integral_control
from simlab.uvms_parameters import select_robot_params


class LowLevelInvDynController(ControllerTemplate):
    name = "InvDynController"
    registry_name = "InvDyn"
    attitude_error = 'so3_log_body'
    configuration_fields = ('attitude_error', 'kp', 'ki', 'kd', 'vehicle_i_limit', 'arm_kp', 'arm_ki',
                            'arm_kd', 'arm_u_min', 'arm_u_max', 'uv_u_min', 'uv_u_max',
                            'vehicle_model_params', 'arm_model_params', 'gripper_parameters', 'v_c')

    def set_model_profile(self, model) -> None:
        gripper = self.valid_gripper(model.gripper_parameters)
        self.vehicle_model_params = model.vehicle_baseline
        self.arm_model_params = model.arm_baseline
        self.gripper_parameters = gripper
        self.v_c = model.current_velocity
        self.model_profile_sha256 = model.sha256

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
        self.vehicle_pid_i_buffer = np.zeros(6, dtype=float)
        self.arm_pid_i_buffer = np.zeros(self.arm_dof + 1, dtype=float)
        self.arm_kp = self.arm_vector(
            list(self.arm_params.invdyn_kp) + list(self.arm_params.invdyn_grasper_kp),
            "arm_kp",
        )
        self.arm_ki = self.arm_vector(
            list(self.arm_params.invdyn_ki) + list(self.arm_params.invdyn_grasper_ki),
            "arm_ki",
        )
        self.arm_kd = self.arm_vector(
            list(self.arm_params.invdyn_kd) + list(self.arm_params.invdyn_grasper_kd),
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
        self.gripper_parameters = self.valid_gripper(self.arm_params.gripper_model)
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

    @classmethod
    def valid_gripper(cls, values):
        value = cls.vector(values, 3, 'gripper model').copy()
        if not np.isfinite(value).all() or value[0] <= 0 or np.any(value[1:] < 0):
            raise ValueError('Invalid inverse-dynamics gripper model')
        return value

    def reset_controller_state(self) -> None:
        self.vehicle_pid_i_buffer = np.zeros(6, dtype=float)
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
        buf = self.vector(self.vehicle_pid_i_buffer, 6, 'vehicle integral state')

        def evaluate(buffer, step):
            return self.tracking_uv_controller(
                self.vehicle_model_params, self.kp, self.ki, self.kd,
                ca.DM(buffer), ca.DM(state), ca.DM(target_pos), ca.DM(target_vel),
                ca.DM(target_acc), step, ca.DM(self.v_c))

        command, integral = bounded_integral_control(
            evaluate, buf, self.ki, self.vehicle_i_limit,
            self.uv_u_min, self.uv_u_max, float(dt))
        self.vehicle_pid_i_buffer = integral
        return command

    def arm_controller(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        q_ref: np.ndarray,
        dq_ref: np.ndarray,
        ddq_ref: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('inverse dynamics dt must be finite and positive')
        q = self.arm_vector(q, "q")
        q_dot = self.arm_vector(q_dot, "q_dot")
        q_ref = self.arm_vector(q_ref, "q_ref")
        dq_ref = self.arm_vector(dq_ref, 'dq_ref')
        ddq_ref = self.arm_vector(ddq_ref, 'ddq_ref')
        if not np.isfinite(np.r_[q, q_dot, q_ref, dq_ref, ddq_ref]).all():
            raise ValueError('inverse dynamics inputs must be finite')
        buf = self.arm_vector(self.arm_pid_i_buffer, 'arm integral state')

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
            ca.DM(self.gripper_parameters),
        )

        command = self.arm_vector(u_sat, 'arm effort')
        integral = self.arm_vector(buf_next, 'arm integral state')
        if not np.isfinite(command).all() or not np.isfinite(integral).all():
            raise ValueError('inverse dynamics returned nonfinite arm output')
        self.arm_pid_i_buffer = integral.copy()
        return np.clip(command, self.arm_u_min, self.arm_u_max)
