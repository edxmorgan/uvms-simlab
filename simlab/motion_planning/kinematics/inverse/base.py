"""ROS-independent IK contracts; an endpoint never certifies a safe trajectory.

Frame convention: arm requests use arm-base XYZ; whole-body requests use world
NWU XYZ/rotation matrices and vehicle XYZ/extrinsic XYZ Euler angles (radians).
Joint coordinates and mounting/tip transforms retain the exported model order.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import time
from numpy.typing import ArrayLike

import numpy as np
from scipy.spatial.transform import Rotation


class IKCapability(Enum):
    MANIPULATOR_POSITION = 'position'
    WHOLE_BODY_POSE = 'pose'


class IKStatus(Enum):
    SUCCESS = 'success'
    INVALID_INPUT = 'invalid_input'
    FAILED = 'failed'


@dataclass(frozen=True)
class ManipulatorIKRequest:
    target_position: ArrayLike


@dataclass(frozen=True)
class WholeBodyIKRequest:
    q: ArrayLike
    world_pose: ArrayLike
    target_position: ArrayLike
    target_rotation: ArrayLike
    base: ArrayLike
    tip: ArrayLike
    lower: ArrayLike
    upper: ArrayLike
    position_tolerance: float = .01
    orientation_tolerance: float = .05
    base_align_weight: float = 0.


@dataclass(frozen=True)
class IKResult:
    status: IKStatus
    joints: tuple[float, ...] | None = None
    vehicle_pose: tuple[float, ...] | None = None
    message: str = ''
    # Success means a solution was returned. Guarantees state what was checked.
    guarantees: frozenset[str] = frozenset()
    diagnostics: dict = field(default_factory=dict)

    @property
    def success(self):
        return self.status is IKStatus.SUCCESS


def finite_vector(value, size, name):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f'{name} must be a finite vector of length {size}')
    return vector.copy()


def validate_pose_request(request):
    for name, size in (('q', 4), ('world_pose', 6), ('target_position', 3),
                       ('base', 6), ('tip', 6), ('lower', 4), ('upper', 4)):
        finite_vector(getattr(request, name), size, name)
    rotation = np.asarray(request.target_rotation, dtype=float)
    if (rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)) or
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0) or
            not np.isclose(np.linalg.det(rotation), 1., atol=1e-8, rtol=0)):
        raise ValueError('Target rotation must belong to SO(3)')
    lower, upper, q = map(np.asarray, (request.lower, request.upper, request.q))
    if np.any(lower >= upper) or np.any(q < lower-1e-9) or np.any(q > upper+1e-9):
        raise ValueError('Invalid joint bounds or measured joints outside bounds')
    for tolerance in (request.position_tolerance, request.orientation_tolerance):
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('IK tolerances must be finite and positive')
    if not np.isfinite(request.base_align_weight) or request.base_align_weight < 0:
        raise ValueError('Alignment weight must be finite and nonnegative')


def validate_pose_endpoint(request, vehicle_pose, joints, fk):
    """Independent model check, shared by adapters and command acceptance."""
    pose = finite_vector(vehicle_pose, 6, 'vehicle solution')
    q = finite_vector(joints, 4, 'joint solution')
    if np.any(q < np.asarray(request.lower)-1e-9) or np.any(q > np.asarray(request.upper)+1e-9):
        raise ValueError('Joint solution outside limits')
    actual = finite_vector(np.asarray(fk(q, request.base, pose, request.tip)[-1]).reshape(-1), 7, 'FK pose')
    pe = float(np.linalg.norm(actual[:3] - request.target_position))
    oe = float((Rotation.from_matrix(request.target_rotation).inv() *
                Rotation.from_quat(actual[[4, 5, 6, 3]])).magnitude())
    if pe > request.position_tolerance or oe > request.orientation_tolerance:
        raise ValueError(f'Endpoint error {pe:.4f} m, {np.degrees(oe):.2f} deg')
    return dict(position_error_m=pe, orientation_error_rad=oe)


class IKTemplate(ABC):
    registry_name = ''
    capability: IKCapability

    @abstractmethod
    def solve_request(self, request) -> IKResult:
        """Return a per-call result; failure must contain no command coordinates."""


class ManipulatorIKTemplate(IKTemplate):
    capability = IKCapability.MANIPULATOR_POSITION
    task = capability.value  # Compatibility for existing Python callers.

    def solve_request(self, request):
        if not isinstance(request, ManipulatorIKRequest):
            return IKResult(IKStatus.INVALID_INPUT, message='Expected arm-base position request')
        try:
            target = finite_vector(request.target_position, 3, 'target position')
        except (ValueError, TypeError) as error:
            return IKResult(IKStatus.INVALID_INPUT, message=str(error))
        start = time.monotonic()
        try:
            joints = finite_vector(self.solve(target), 4, 'joint solution')
            return IKResult(IKStatus.SUCCESS, joints=tuple(joints),
                            guarantees=frozenset({'finite_solution'}),
                            diagnostics={'task': self.task, 'elapsed_seconds': time.monotonic()-start})
        except (ValueError, RuntimeError) as error:
            return IKResult(IKStatus.FAILED, message=str(error),
                            diagnostics={'elapsed_seconds': time.monotonic()-start})

    @abstractmethod
    def solve(self, target_position):
        """Existing analytic evaluator; no orientation or endpoint guarantee."""


class WholeBodyIKTemplate(IKTemplate):
    capability = IKCapability.WHOLE_BODY_POSE
    task = capability.value

    def solve_request(self, request):
        if not isinstance(request, WholeBodyIKRequest):
            return IKResult(IKStatus.INVALID_INPUT, message='Expected world-frame pose request')
        try:
            validate_pose_request(request)
        except (ValueError, TypeError) as error:
            return IKResult(IKStatus.INVALID_INPUT, message=str(error))
        start = time.monotonic()
        try:
            pose, joints = self.solve(
                request.q, request.world_pose, request.target_position, request.target_rotation,
                request.base, request.tip, request.lower, request.upper,
                request.position_tolerance, request.orientation_tolerance,
                base_align_weight=request.base_align_weight)
            errors = validate_pose_endpoint(request, pose, joints, self.fk)
            errors['elapsed_seconds'] = time.monotonic()-start
            return IKResult(IKStatus.SUCCESS, joints=tuple(joints), vehicle_pose=tuple(pose),
                            guarantees=frozenset({'finite_solution', 'joint_limits', 'endpoint_pose'}),
                            diagnostics=errors)
        except (ValueError, RuntimeError) as error:
            return IKResult(IKStatus.FAILED, message=str(error),
                            diagnostics={'elapsed_seconds': time.monotonic()-start})

    @abstractmethod
    def solve(self, q, world_pose, target_position, target_rotation, base, tip,
              lower, upper, position_tolerance=.01, orientation_tolerance=.05,
              base_align_weight=0.):
        """Numerical evaluator; provide fk for independent endpoint validation."""
