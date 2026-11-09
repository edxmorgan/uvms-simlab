"""Decode public C++ vehicle and arm disturbance EKF estimates."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DisturbanceEstimate:
    generation: int
    step: int
    stamp: float
    period: float
    force: np.ndarray
    variance: np.ndarray
    model: np.ndarray


def decode_observation(msg, prefix, subsystem):
    groups = {name: dict(zip(v.interface_names, v.values))
              for name, v in zip(msg.joint_names, msg.interface_values)}
    if subsystem == 'vehicle':
        meta = groups[prefix + 'IOs']
        axes = ('force.x', 'force.y', 'force.z', 'torque.x', 'torque.y', 'torque.z')
        parameter_count = 33
    elif subsystem == 'arm':
        meta = groups[prefix + '_arm_IOs']
        # A partially informed contact posterior is not an independently
        # identified four-dimensional load observation for DR statistics.
        # Fail closed for older telemetry without explicit lock status too.
        if any(meta['estimator_joint_locked.' + axis] != 0.0 for axis in 'edcb'):
            raise ValueError('arm disturbance evidence unavailable while a joint is locked')
        if any(meta['estimator_disturbance_observable.' + axis] != 1.0 for axis in 'edcb'):
            raise ValueError('arm disturbance evidence unavailable during joint-limit contact')
        axes = tuple('torque.' + axis for axis in 'edcb')
        parameter_count = 81
    else:
        raise ValueError('unknown disturbance subsystem')
    if meta['estimator_valid'] != 1.0:
        raise ValueError(f'{subsystem} EKF is not valid')
    force = np.array([meta['estimated_disturbance.' + a] for a in axes])
    variance = np.array([meta['estimated_disturbance.variance.' + a] for a in axes])
    model = np.array([meta[f'estimator_nominal_parameter.{i}']
                      for i in range(parameter_count)])
    step = meta['estimator_step']
    generation = meta['sim_generation']
    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    if (not np.all(np.isfinite(np.r_[force, variance, model, step, generation, stamp]))
            or np.any(variance < 0) or step < 1 or step != int(step)
            or generation < 0 or generation != int(generation)):
        raise ValueError(f'invalid {subsystem} EKF telemetry')
    # Additive generalized force in native coordinates: body FRD for the
    # vehicle, joint order e/d/c/b for the arm. No differentiation in Python.
    return DisturbanceEstimate(int(generation), int(step), stamp, 0.0,
                              force, variance, model)


def reset_dynamics_observer(robot):
    previous = getattr(robot, '_dynamics_observer', None)
    if previous is not None:
        previous.reset_observations()
    robot._dynamics_observer = None


def forward_dynamics_observations(robot, msg):
    """Route only to the executing controller, including CmdReplay feedback."""
    controller = robot.active_controller_instance()
    mode = robot.control_mode.name
    if mode == 'REPLAY':
        if controller is not None and getattr(controller, 'enabled', False):
            controller = robot.controller_instance(controller.feedback_controller_name())
        else:
            controller = None
    if mode == 'TELEOP' or robot.sim_reset_hold:
        controller = None
    if not getattr(controller, 'uses_dynamics_observations', False):
        controller = None
    previous = getattr(robot, '_dynamics_observer', None)
    if previous is not controller:
        if previous is not None:
            previous.reset_observations()
        if controller is not None:
            controller.reset_observations()
        robot._dynamics_observer = controller
    if controller is None:
        return
    for subsystem in ('vehicle', 'arm'):
        try:
            observation = decode_observation(msg, robot.prefix, subsystem)
        except (KeyError, ValueError, TypeError):
            observation = None
        if observation is not None:
            age = robot.node.get_clock().now().nanoseconds * 1e-9 - observation.stamp
            if age < 0 or age > 0.5:
                observation = None
        controller.observe_dynamics(subsystem, observation)
