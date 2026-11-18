"""Thin DR-OGES API adapter using native observer feedback."""
from dataclasses import dataclass

import numpy as np
from rclpy.node import Node

from simlab.controllers.oges import OgesModelbasedController


try:
    from namor.controls.dr_clf_oges import DRCLFOGES, ResidualBuffer
    NAMOR_DR_IMPORT_ERROR = None
except ImportError as exc:
    DRCLFOGES = None
    ResidualBuffer = None
    NAMOR_DR_IMPORT_ERROR = exc


class DistributionallyRobustOgesController(OgesModelbasedController):
    name = "DR-OGES"
    registry_name = "DR-Ours"
    uses_dynamics_observations = True
    lazy_initialization = True
    def configuration_snapshot(self) -> dict:
        result = super().configuration_snapshot()
        result['parameters']['vehicle_residual_estimator'] = 'cpp_disturbance_ekf'
        result['parameters']['arm_residual_estimator'] = 'cpp_disturbance_ekf'
        for subsystem in ('vehicle', 'arm'):
            projector = getattr(self, subsystem + '_dr_projector')
            buffer = getattr(self, subsystem + '_residual_buffer')
            if projector.residual_buffer is not buffer:
                raise RuntimeError(f'{subsystem} residual collector is disconnected')
            result['parameters'][subsystem + '_dr'] = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in vars(projector).items()
                if isinstance(value, (int, float, bool, str, np.ndarray, type(None)))}
            result['parameters'][subsystem + '_dr']['history_seconds'] = buffer.window_seconds
        return result

    def __init__(self, node: Node, arm_dof: int = 4, robot_prefix: str = ""):
        super().__init__(node, arm_dof, robot_prefix)
        if NAMOR_DR_IMPORT_ERROR is not None:
            raise ImportError(
                "DistributionallyRobustOgesController requires a namor version "
                "with DRCLFOGES support."
            ) from NAMOR_DR_IMPORT_ERROR
        self.vehicle_residual_buffer = ResidualBuffer(n_dof=6, maxlen=2000, window_seconds=2.)
        self.arm_residual_buffer = ResidualBuffer(n_dof=self.arm_dof, maxlen=2000, window_seconds=2.)
        self.vehicle_dr_projector = DRCLFOGES(
            n_dof=6,
            residual_buffer=self.vehicle_residual_buffer,
            wasserstein_radius=0.01,
            min_samples=10,
            min_history_seconds=.5,
            effect_deadband=0.05,
            confidence_scale=2.0,
            projection_deadband=1e-6,
            slack_weight=1e8,
        )
        self.arm_dr_projector = DRCLFOGES(
            n_dof=self.arm_dof,
            residual_buffer=self.arm_residual_buffer,
            wasserstein_radius=0.005,
            min_samples=10,
            min_history_seconds=.5,
            effect_deadband=0.01,
            confidence_scale=2.0,
            projection_deadband=1e-6,
            slack_weight=1e8,
        )
        self._observations = {'vehicle': None, 'arm': None}
        self.last_vehicle_dr_result = None
        self.last_arm_dr_result = None
        self.node.get_logger().info(
            "\033[96mDistributionally robust OGES controller active as DR-Ours.\033[0m"
        )

    def reset_controller_state(self) -> None:
        super().reset_controller_state()
        self.reset_observations()
        self.last_vehicle_dr_result = None
        self.last_arm_dr_result = None

    def reset_observations(self) -> None:
        self._observations = {'vehicle': None, 'arm': None}
        self.vehicle_residual_buffer.clear()
        self.arm_residual_buffer.clear()

    def observe_dynamics(self, subsystem, observation) -> None:
        """Consume native EKF posterior means with the same contract for both subsystems.

        Posterior means are correlated estimates; empirical DR statistics are
        not certified independent-sample confidence bounds.
        """
        if subsystem not in ('vehicle', 'arm'):
            raise ValueError('unknown disturbance subsystem')
        previous = self._observations[subsystem]
        buffer = getattr(self, subsystem + '_residual_buffer')
        nominal = (self.blue.sim_p if subsystem == 'vehicle'
                   else self.alpha_params.step_model_params)
        reason = None
        if observation is None:
            reason = 'missing, stale or invalid C++ disturbance estimate'
        elif not np.allclose(observation.model, np.asarray(nominal).reshape(-1),
                             rtol=1e-9, atol=1e-10):
            reason = 'C++ estimator nominal model differs from DR nominal model'
        if reason is not None:
            buffer.clear()
            self._observations[subsystem] = None
            if getattr(self, '_' + subsystem + '_rejection', None) != reason:
                self.node.get_logger().warning(f'DR {subsystem} evidence unavailable: ' + reason)
            setattr(self, '_' + subsystem + '_rejection', reason)
            return
        setattr(self, '_' + subsystem + '_rejection', None)
        if previous is not None:
            if observation.stamp <= previous.stamp or observation.generation < previous.generation:
                return
            if observation.generation == previous.generation and observation.step <= previous.step:
                return
            if (observation.generation != previous.generation
                    or observation.stamp - previous.stamp > 0.5):
                buffer.clear()
        self._observations[subsystem] = observation
        buffer.append(observation.force, observation.stamp)

    def _expire_disturbance(self, subsystem):
        observation = self._observations[subsystem]
        if observation is not None:
            age = self.node.get_clock().now().nanoseconds * 1e-9 - observation.stamp
            if age < 0 or age > 0.5:
                self.observe_dynamics(subsystem, None)

    def _evaluate_policy(self, subsystem, args, dt):
        control = super()._evaluate_policy(subsystem, args, dt)
        self._expire_disturbance(subsystem)
        state, weights, _, H, F, reference, velocity, acceleration, J, J_ref, scale, lower, upper, _ = args
        model = self.uv_oges if subsystem == 'vehicle' else self.arm_oges
        n = 6 if subsystem == 'vehicle' else self.arm_dof
        A, b, _ = model.A_b_V_func(
            state[:n], state[n:], reference, velocity, J_ref, acceleration,
            weights, J, scale)
        result = getattr(self, subsystem + '_dr_projector').project(
            u_nominal=control, A=np.asarray(A).ravel(), b=float(b),
            H=np.asarray(H), F=np.asarray(F).ravel(), u_min=lower, u_max=upper,
            dt=dt)
        setattr(self, 'last_' + subsystem + '_dr_result', result)
        return result.control


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
