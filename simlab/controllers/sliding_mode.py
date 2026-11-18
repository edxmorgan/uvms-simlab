"""Vehicle and arm SMC using the exported CasADi controllers."""
from pathlib import Path

import ament_index_python
import casadi as ca
import numpy as np

from simlab.controllers.base import ControllerTemplate
from simlab.uvms_parameters import select_robot_params


class SlidingModeController(ControllerTemplate):
    name = registry_name = 'SMC'
    lazy_initialization = True
    configuration_fields = (
        'vehicle_surface', 'vehicle_damping', 'vehicle_switching', 'vehicle_boundary',
        'arm_surface', 'arm_damping', 'arm_switching', 'arm_boundary',
        'vehicle_model_params', 'arm_model_params', 'gripper_parameters',
        'uv_u_min', 'uv_u_max', 'arm_u_min', 'arm_u_max',
    )

    def __init__(self, node, arm_dof=4, robot_prefix=''):
        super().__init__(node, arm_dof, robot_prefix)
        if arm_dof != 4:
            raise ValueError('SMC exports require four arm joints plus gripper')
        package_share = Path(ament_index_python.get_package_share_directory('simlab'))
        model_directory = package_share / 'model_functions'
        self.vehicle_function = ca.Function.load(
            str(model_directory / 'vehicle/uv_sliding_mode_controller.casadi')
        )
        self.arm_function = ca.Function.load(
            str(model_directory / 'arm/arm_sliding_mode_controller.casadi')
        )

        arm, vehicle = select_robot_params(robot_prefix)
        self.vehicle_surface = self.finite(vehicle.smc_surface, 6, 'vehicle surface')
        self.vehicle_damping = self.finite(vehicle.smc_damping, 6, 'vehicle damping')
        self.vehicle_switching = self.finite(vehicle.smc_switching, 6, 'vehicle switching')
        self.vehicle_boundary = self.finite(vehicle.smc_boundary, 6, 'vehicle boundary')
        self.arm_surface = self.finite(arm.smc_surface, 5, 'arm surface')
        self.arm_damping = self.finite(arm.smc_damping, 5, 'arm damping')
        self.arm_switching = self.finite(arm.smc_switching, 5, 'arm switching')
        self.arm_boundary = self.finite(arm.smc_boundary, 5, 'arm boundary')

        for gains in (self.vehicle_surface, self.vehicle_boundary,
                      self.arm_surface, self.arm_boundary):
            if np.any(gains <= 0):
                raise ValueError('SMC surface and boundary gains must be positive')
        for gains in (self.vehicle_damping, self.vehicle_switching,
                      self.arm_damping, self.arm_switching):
            if np.any(gains < 0):
                raise ValueError('SMC damping and switching gains must be nonnegative')

        self.vehicle_model_params = self.finite(vehicle.model_params, 27, 'vehicle model')
        self.arm_model_params = self.finite(arm.sim_p, 75, 'arm model')
        self.gripper_parameters = self.finite(arm.gripper_model, 3, 'gripper model')
        self.uv_u_min = self.finite(vehicle.u_min, 6, 'vehicle effort minimum')
        self.uv_u_max = self.finite(vehicle.u_max, 6, 'vehicle effort maximum')
        self.arm_u_min = self.finite(
            list(arm.u_min) + list(arm.grasper_u_min), 5, 'arm effort minimum'
        )
        self.arm_u_max = self.finite(
            list(arm.u_max) + list(arm.grasper_u_max), 5, 'arm effort maximum'
        )
        if np.any(self.arm_u_min > self.arm_u_max) or np.any(self.uv_u_min > self.uv_u_max):
            raise ValueError('SMC effort limits are reversed')
        if np.any(np.asarray(vehicle.v_c)):
            raise ValueError('SMC export requires a zero-current nominal model')
        if self.gripper_parameters[0] <= 0 or np.any(self.gripper_parameters[1:] < 0):
            raise ValueError('Invalid SMC gripper model')

    @classmethod
    def finite(cls, values, size, label):
        value = cls.vector(values, size, label)
        if not np.all(np.isfinite(value)):
            raise ValueError(f'SMC {label} must be finite')
        return value.copy()

    def set_model_profile(self, model):
        if np.any(model.current_velocity):
            raise ValueError('SMC export requires a zero-current nominal model')
        # Validate before assigning, so a rejected model leaves the old model intact.
        vehicle = self.finite(model.vehicle_baseline, 27, 'vehicle model')
        arm = self.finite(model.arm_baseline, 75, 'arm model')
        gripper = self.finite(model.gripper_parameters, 3, 'gripper model')
        if gripper[0] <= 0 or np.any(gripper[1:] < 0):
            raise ValueError('Invalid SMC gripper model')
        self.vehicle_model_params = vehicle
        self.arm_model_params = arm
        self.gripper_parameters = gripper
        self.model_profile_sha256 = model.sha256

    def vehicle_controller(self, state, target_pos, target_vel, target_acc, dt):
        result = self.vehicle_function(
            self.vehicle_model_params,
            self.finite(state, 12, 'state'),
            self.finite(target_pos, 6, 'vehicle target'),
            self.finite(target_vel, 6, 'desired body velocity'),
            self.finite(target_acc, 6, 'desired body acceleration'),
            self.vehicle_surface,
            self.vehicle_damping,
            self.vehicle_switching,
            self.vehicle_boundary,
        )
        wrench = self.finite(result[0], 6, 'vehicle wrench')
        return np.clip(wrench, self.uv_u_min, self.uv_u_max)

    def arm_controller(self, q, q_dot, q_ref, dq_ref, ddq_ref, dt):
        result = self.arm_function(
            self.finite(q, 5, 'q'),
            self.finite(q_dot, 5, 'velocity'),
            self.finite(q_ref, 5, 'joint target'),
            self.finite(dq_ref, 5, 'joint target velocity'),
            self.finite(ddq_ref, 5, 'joint target acceleration'),
            self.arm_surface,
            self.arm_damping,
            self.arm_switching,
            self.arm_boundary,
            self.arm_u_min,
            self.arm_u_max,
            self.arm_model_params,
            self.gripper_parameters,
        )
        effort = self.finite(result[0], 5, 'arm effort')
        return np.clip(effort, self.arm_u_min, self.arm_u_max)
