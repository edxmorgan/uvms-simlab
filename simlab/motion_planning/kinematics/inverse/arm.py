"""Arm-base position IK using the same exported FK as the controller."""
from pathlib import Path
import threading
import casadi as ca
import numpy as np
from scipy.optimize import least_squares
import time

from .base import ManipulatorIKTemplate, ManipulatorIKRequest, IKResult, IKStatus, finite_vector


class ManipulatorPositionIK(ManipulatorIKTemplate):
    """Bounded, measured-pose-seeded position IK; legacy analytic evaluation remains explicit."""
    registry_name = 'manipulator_position'

    def __init__(self, resource_directory):
        self.solver = ca.Function.load(str(Path(resource_directory) / 'model_functions/arm/ik_eval.casadi'))
        self.last_report = {}
        self.lock = threading.Lock()
        self.fk = ca.Function.load(str(Path(resource_directory) / 'model_functions/arm/fk_eval.casadi'))

    def solve_request(self, request):
        if not isinstance(request, ManipulatorIKRequest):
            return IKResult(IKStatus.INVALID_INPUT, message='Expected arm-base position request')
        # Old offline callers can still evaluate the analytic model. Robot command
        # requests always supply measured joints, actuator bounds and tool offset.
        if all(getattr(request, key) is None for key in ('q', 'lower', 'upper', 'tip')):
            return super().solve_request(request)
        try:
            target = finite_vector(request.target_position, 3, 'target')
            q, lower, upper = (finite_vector(getattr(request, key), 4, key)
                               for key in ('q', 'lower', 'upper'))
            tip = finite_vector(request.tip, 6, 'tip')
            if np.any(lower >= upper):
                raise ValueError('Invalid actuator bounds')
            if not np.isfinite(request.position_tolerance) or request.position_tolerance <= 0:
                raise ValueError('Position tolerance must be finite and positive')
        except (ValueError, TypeError) as error:
            return IKResult(IKStatus.INVALID_INPUT, message=str(error))
        start = time.monotonic()
        try:
            def position_error(joints):
                pose = np.asarray(self.fk(joints, np.zeros(6), np.zeros(6), tip)[-1]).reshape(-1)
                return finite_vector(pose, 7, 'FK pose')[:3] - target
            def residual(joints):
                # Resolve position-only redundancy toward the measured posture.
                return np.r_[position_error(joints), 1e-5 * (joints-q)]
            with self.lock:
                # Bounds constrain commands, not observations. Keep q unchanged
                # in the posture objective; project only the optimizer seed.
                result = least_squares(residual, np.clip(q, lower, upper), bounds=(lower, upper),
                                       max_nfev=100, ftol=1e-10, xtol=1e-10, gtol=1e-10)
                error = float(np.linalg.norm(position_error(result.x)))
            if not result.success or error > request.position_tolerance:
                raise ValueError(f'Arm IK could not reach target within bounds: position error {error:.4f} m')
            return IKResult(IKStatus.SUCCESS, joints=tuple(result.x),
                            guarantees=frozenset({'finite_solution', 'joint_limits', 'position_endpoint'}),
                            diagnostics={'position_error_m': error, 'elapsed_seconds': time.monotonic()-start})
        except (ValueError, RuntimeError) as error:
            return IKResult(IKStatus.FAILED, message=str(error))

    def solve(self, target_position):
        # Keep the existing evaluation and output ordering exactly as before.
        with self.lock:
            result = self.solver(target_position).full().flatten().tolist()
        self.last_report = {'task': self.task}
        return result
