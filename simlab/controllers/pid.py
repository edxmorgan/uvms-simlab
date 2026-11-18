import os

import ament_index_python
import casadi as ca
import numpy as np
from rclpy.node import Node

from simlab.controllers.base import ControllerTemplate, bounded_integral_control
from simlab.uvms_parameters import select_robot_params


class LowLevelPidController(ControllerTemplate):
    name = "PID"
    registry_name = "PID"
    attitude_error = 'so3_log_body'
    configuration_fields = ('attitude_error', 'kp', 'ki', 'kd', 'vehicle_i_limit', 'arm_kp', 'arm_ki',
                            'arm_kd', 'arm_u_min', 'arm_u_max',
                            'vehicle_model_params', 'arm_model_params', 'uv_u_min', 'uv_u_max')

    def set_model_profile(self, model) -> None:
        self.vehicle_model_params = model.vehicle_baseline
        self.arm_model_params = model.arm_baseline
        self.model_profile_sha256 = model.sha256

    def __init__(self, node: Node, arm_dof: int = 4, robot_prefix: str = ""):
        super().__init__(node, arm_dof, robot_prefix)
        package_share_directory = ament_index_python.get_package_share_directory("simlab")

        uv_pid_controller_path = os.path.join(
            package_share_directory,
            "model_functions/vehicle/uv_reg_pid_controller.casadi",
        )
        self.uv_pid_controller = ca.Function.load(uv_pid_controller_path)
        self.arm_params, self.vehicle_params = select_robot_params(self.robot_prefix)
        self.vehicle_pid_i_buffer = np.zeros(6, dtype=float)
        self.vehicle_i_limit = self.vehicle_params.i_limit
        self.vehicle_model_params = self.vehicle_params.model_params
        self.uv_u_min = np.asarray(self.vehicle_params.u_min, dtype=float)
        self.uv_u_max = np.asarray(self.vehicle_params.u_max, dtype=float)
        self.kp = self.vehicle_params.pid_kp
        self.ki = self.vehicle_params.pid_ki
        self.kd = self.vehicle_params.pid_kd

        arm_pid_controller_path = os.path.join(
            package_share_directory,
            "model_functions/arm/arm_pid.casadi",
        )
        self.arm_pid_controller = ca.Function.load(arm_pid_controller_path)
        self.arm_pid_i_buffer = np.zeros(self.arm_dof + 1, dtype=float)
        self.arm_kp = self.arm_vector(
            list(self.arm_params.pid_kp) + list(self.arm_params.grasper_kp),
            "arm_kp",
        )
        self.arm_ki = self.arm_vector(
            list(self.arm_params.pid_ki) + list(self.arm_params.grasper_ki),
            "arm_ki",
        )
        self.arm_kd = self.arm_vector(
            list(self.arm_params.pid_kd) + list(self.arm_params.grasper_kd),
            "arm_kd",
        )
        # Startup-only tuning overrides leave the gripper and effort limits intact.
        for field in ('arm_kp', 'arm_ki', 'arm_kd'):
            name = f'{self.robot_prefix}pid_{field}'
            defaults = getattr(self, field)
            if not node.has_parameter(name):
                node.declare_parameter(name, defaults[:self.arm_dof].tolist())
            values = self.vector(node.get_parameter(name).value, self.arm_dof, name)
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f'{name} requires finite nonnegative gains')
            setattr(self, field, np.r_[values, defaults[self.arm_dof:]])
        self.arm_u_max = self.arm_vector(
            list(self.arm_params.u_max) + list(self.arm_params.grasper_u_max),
            "arm_u_max",
        )
        self.arm_u_min = self.arm_vector(
            list(self.arm_params.u_min) + list(self.arm_params.grasper_u_min),
            "arm_u_min",
        )
        self.arm_model_params = self.arm_params.sim_p
        self.node.get_logger().info(
            f"PID params for {self.robot_prefix or 'default'}: "
            f"arm_profile={self.arm_params.profile_name}, "
            f"vehicle_profile={self.vehicle_params.profile_name}, "
            f"arm_kp={self.arm_kp[:self.arm_dof].tolist()}, "
            f"arm_ki={self.arm_ki[:self.arm_dof].tolist()}, "
            f"arm_kd={self.arm_kd[:self.arm_dof].tolist()}"
        )

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

        buf = self.vector(self.vehicle_pid_i_buffer, 6, 'vehicle integral state')
        def evaluate(buffer, step):
            return self.uv_pid_controller(
                self.vehicle_model_params, self.kp, self.ki, self.kd,
                ca.DM(buffer), ca.DM(state), ca.DM(target_pos), ca.DM(target_vel), step)

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
            raise ValueError('PID arm dt must be finite and positive')
        q = self.arm_vector(q, "q")
        q_dot = self.arm_vector(q_dot, "q_dot")
        q_ref = self.arm_vector(q_ref, "q_ref")

        buf = self.arm_vector(self.arm_pid_i_buffer, 'arm integral state')

        u_sat, err, buf_next = self.arm_pid_controller(
            ca.DM(q),
            ca.DM(q_dot),
            ca.DM(q_ref),
            ca.DM(self.arm_kp),
            ca.DM(self.arm_ki),
            ca.DM(self.arm_kd),
            ca.DM(buf),
            float(dt),
            ca.DM(self.arm_u_max),
            ca.DM(self.arm_u_min),
            ca.DM(self.arm_model_params),
        )

        command = self.arm_vector(u_sat, 'arm effort')
        error = self.arm_vector(err, 'arm error')
        next_integral = self.arm_vector(buf_next, 'arm integral state')
        if not np.all(np.isfinite(np.r_[command, error, next_integral])):
            raise ValueError('PID returned nonfinite arm output')
        # Conditional integration: saturation may not accumulate error that
        # pushes farther into the effort limit. Opposite error can unwind it.
        integral_direction = self.arm_ki * error
        blocked = ((command >= self.arm_u_max) & (integral_direction > 0)) | (
            (command <= self.arm_u_min) & (integral_direction < 0))
        self.arm_pid_i_buffer = np.where(blocked | (self.arm_ki == 0), buf, next_integral)
        return np.clip(command, self.arm_u_min, self.arm_u_max)
