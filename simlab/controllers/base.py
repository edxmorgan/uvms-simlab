from abc import ABC, abstractmethod

import numpy as np
from rclpy.node import Node


class ControllerTemplate(ABC):
    """Base interface for UVMS vehicle and arm controllers."""

    name: str = ""
    registry_name: str = ""
    configuration_fields: tuple = ()
    uses_dynamics_observations = False

    def __init__(self, node: Node, arm_dof: int = 4, robot_prefix: str = ""):
        self.node = node
        self.arm_dof = int(arm_dof)
        self.robot_prefix = str(robot_prefix)

    @staticmethod
    def vector(values, expected_size: int, label: str) -> np.ndarray:
        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.size != expected_size:
            raise ValueError(f"{label} must have {expected_size} elements, got {vector.size}")
        return vector

    def arm_vector(self, values, label: str) -> np.ndarray:
        return self.vector(values, self.arm_dof + 1, label)

    def reset_controller_state(self) -> None:
        """Clear controller memory before a new independent control segment."""
        return

    def reset_observations(self) -> None:
        """Discard estimator evidence at an execution boundary (optional)."""
        return

    def observe_dynamics(self, subsystem, observation) -> None:
        """Observe a completed hardware step; never called from command evaluation."""
        return

    def set_model_profile(self, model) -> None:
        """Opt-in contract; never infer model layout for an unknown plugin."""
        raise NotImplementedError(f'{self.registry_name} does not support model profiles')

    def configuration_snapshot(self) -> dict:
        """Read effective arrays, not just the name of the requested profile."""
        def value(v):
            if hasattr(v, 'full'):
                v = v.full()
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, tuple):
                return list(v)
            if isinstance(v, np.generic):
                return v.item()
            return v
        return dict(schema_version=1, controller=self.registry_name,
                    implementation=f'{type(self).__module__}.{type(self).__qualname__}',
                    model_profile_sha256=getattr(self, 'model_profile_sha256', ''),
                    parameters={name: value(getattr(self, name)) for name in self.configuration_fields})

    @abstractmethod
    def vehicle_controller(
        self,
        state: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_acc: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def arm_controller(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        q_ref: np.ndarray,
        dq_ref: np.ndarray,
        ddq_ref: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        raise NotImplementedError


def bounded_integral_control(evaluate, integral, gains, limits, lower, upper, dt):
    """Evaluate at accepted integral state; reject increments worsening saturation.

    Compare changes in actual body wrench, not world-coordinate error: rotations
    and inertia couple integral channels to actuator limits. The export remains
    the sole source of controller equations. ``evaluate(buffer, dt)`` returns
    raw wrench and next integral state.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('controller dt must be finite and positive')

    def checked(buffer, step):
        wrench, next_buffer = evaluate(buffer, step)
        wrench = np.asarray(wrench, dtype=float).reshape(-1)
        next_buffer = np.asarray(next_buffer, dtype=float).reshape(-1)
        if wrench.shape != (6,) or next_buffer.shape != (6,) or not np.isfinite(
                np.r_[wrench, next_buffer]).all():
            raise ValueError('controller returned invalid wrench or integral state')
        return wrench, next_buffer

    if not np.any(gains):
        # No latent integral history while integral feedback is disabled.
        buffer = np.zeros(6)
        wrench, _ = checked(buffer, 0.)
        return np.clip(wrench, lower, upper), buffer
    buffer = np.clip(np.asarray(integral, dtype=float), -limits, limits)
    old_wrench, _ = checked(buffer, 0.)
    _, candidate = checked(buffer, dt)
    candidate = np.clip(candidate, -limits, limits)
    new_wrench, _ = checked(candidate, 0.)
    excess = new_wrench - np.clip(new_wrench, lower, upper)
    if np.any(excess * (new_wrench - old_wrench) > 0):
        return np.clip(old_wrench, lower, upper), buffer.copy()
    return np.clip(new_wrench, lower, upper), candidate.copy()
