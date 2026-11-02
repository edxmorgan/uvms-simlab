"""SE(3) endpoint IK. Returned configurations are not collision-checked paths."""
from pathlib import Path
import threading
import time
import casadi as ca
import numpy as np
from scipy.spatial.transform import Rotation


def aligned_target_rotation(marker_rotation, axis):
    """Point tool +Z along the selected marker axis, with deterministic roll.

    Keep marker +X as tool +X where perpendicular; for marker X alignment,
    use marker +Y as tool +X. All three choices are proper rotations.
    """
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,) or not any(np.array_equal(axis, a) for a in np.eye(3)):
        raise ValueError('IK alignment axis must be +X, +Y or +Z')
    x = np.array([0., 1., 0.]) if axis[0] == 1 else np.array([1., 0., 0.])
    return np.asarray(marker_rotation) @ np.column_stack((x, np.cross(axis, x), axis))


class EndpointIK:
    """Bounded endpoint IK; returned configurations do not certify a safe path."""
    def __init__(self, resource_directory):
        root = Path(resource_directory)
        self.solver = ca.Function.load(str(root/'whole_body/endpoint_ik_ipopt.casadi'))
        self.fk = ca.Function.load(str(root/'model_functions/arm/fk_eval.casadi'))
        self.lock = threading.Lock()
        self.last_report = {}

    def solve(self, q, world_pose, target_position, target_rotation, base, tip,
              lower, upper, position_tolerance=.01, orientation_tolerance=.05, base_align_weight=0.):
        q, world, lower, upper = [np.asarray(v, dtype=float).reshape(-1)
                                  for v in (q, world_pose, lower, upper)]
        if q.shape != (4,) or world.shape != (6,) or lower.shape != (4,) or upper.shape != (4,):
            raise ValueError('Invalid IK configuration dimensions')
        if (not np.all(np.isfinite(np.r_[q, world, lower, upper])) or
                np.any(lower >= upper) or np.any(q < lower-1e-9) or np.any(q > upper+1e-9)):
            raise ValueError('Measured configuration or joint bounds are invalid')
        rot = Rotation.from_matrix(target_rotation)
        measured = np.r_[world[:3], Rotation.from_euler('xyz',world[3:]).as_quat()[[3,0,1,2]], q]
        goal = np.r_[target_position, rot.as_quat()[[3,0,1,2]]]
        if not np.isfinite(base_align_weight) or base_align_weight < 0:
            raise ValueError('Arm/base alignment weight must be finite and nonnegative')
        parameters = np.r_[measured, goal, base, tip, base_align_weight]
        if not np.all(np.isfinite(parameters)):
            raise ValueError('Nonfinite IK input')
        seeds = [measured.copy()]
        # Wrist branches: optimizer seeds only; never issued as intermediate commands.
        for fraction in (.01, .5, .99):
            seed = measured.copy()
            seed[10] = lower[3] + fraction*(upper[3]-lower[3])
            seeds.append(seed)
        candidates, attempts = [], []
        start = time.monotonic()
        with self.lock:
            for seed in seeds:
                if attempts and time.monotonic()-start >= 1.0:
                    break
                try:
                    result = self.solver(x0=seed, p=parameters,
                        lbx=np.r_[[-np.inf]*3,[-1.]*4,lower],
                        ubx=np.r_[[np.inf]*3,[1.]*4,upper], lbg=0., ubg=0.)
                    stats = self.solver.stats()
                    attempts.append(str(stats.get('return_status','unknown')))
                    if not stats.get('success',False):
                        continue
                    z = np.asarray(result['x'],dtype=float).reshape(-1)
                    if not np.all(np.isfinite(z)) or abs(np.linalg.norm(z[3:7])-1)>1e-6:
                        continue
                    if np.any(z[7:] < lower-1e-9) or np.any(z[7:] > upper+1e-9):
                        continue
                    pose = np.r_[z[:3],Rotation.from_quat(z[[4,5,6,3]]).as_euler('xyz')]
                    actual = np.asarray(self.fk(z[7:],base,pose,tip)[-1],dtype=float).reshape(-1)
                    pe = float(np.linalg.norm(actual[:3]-goal[:3]))
                    oe = float((rot.inv()*Rotation.from_quat(actual[[4,5,6,3]])).magnitude())
                    if not np.isfinite(pe+oe) or pe>position_tolerance or oe>orientation_tolerance:
                        continue
                    candidates.append((float(result['f']),pose,z[7:].copy(),pe,oe))
                except RuntimeError:
                    attempts.append('solver_exception')
            self.last_report = dict(attempts=attempts,valid_candidates=len(candidates),
                                    elapsed_seconds=time.monotonic()-start)
        if not candidates:
            raise RuntimeError('No converged, in-bounds IK endpoint meets pose tolerances')
        _, pose, joints, pe, oe = min(candidates,key=lambda candidate:candidate[0])
        self.last_report.update(position_error_m=pe,orientation_error_rad=oe)
        return pose,joints
