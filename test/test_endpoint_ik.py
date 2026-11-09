from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
from simlab.motion_planning.kinematics.inverse.whole_body import EndpointIK
from simlab.uvms_parameters_sim import ReachParams as P

ROOT=Path(__file__).resolve().parents[1]/'resource'


def test_ik_registry_enforces_capability_and_preserves_endpoint_implementation():
    from simlab.motion_planning.kinematics.inverse import ik_class, ManipulatorPositionIK
    from simlab.motion_planning.kinematics.inverse.base import WholeBodyIKTemplate, ManipulatorIKTemplate
    assert ik_class('whole_body_endpoint', task='pose') is EndpointIK
    assert issubclass(EndpointIK, WholeBodyIKTemplate)
    assert ik_class('manipulator_position', task='position') is ManipulatorPositionIK
    assert issubclass(ManipulatorPositionIK, ManipulatorIKTemplate)
    with pytest.raises(ValueError):
        ik_class('manipulator_position', task='pose')
    with pytest.raises(KeyError):
        ik_class('missing', task='pose')


@pytest.mark.parametrize('target', [[.2, 0., -.4], [.1, .1, -.3], [.3, -.1, -.2]])
def test_modular_arm_ik_matches_original_casadi_evaluation(target):
    import casadi as ca
    from simlab.motion_planning.kinematics.inverse import ik_class
    from simlab.robot import Robot
    from types import SimpleNamespace
    original = ca.Function.load(str(ROOT/'model_functions/arm/ik_eval.casadi'))
    robot = SimpleNamespace(manipulator_ik=ik_class('manipulator_position', task='position')(ROOT))
    actual = Robot.manipulator_inverse_kinematics(robot, target)
    np.testing.assert_array_equal(actual, original(target).full().flatten().tolist())


def test_manipulator_solver_is_selected_per_robot():
    from simlab.robot import Robot
    from types import SimpleNamespace
    from unittest.mock import Mock
    first = SimpleNamespace(manipulator_ik=Mock())
    second = SimpleNamespace(manipulator_ik=Mock())
    from simlab.motion_planning.kinematics.inverse.base import IKResult, IKStatus
    first.manipulator_ik.solve_request.return_value = IKResult(IKStatus.SUCCESS, joints=(1., 2., 3., 4.))
    assert Robot.manipulator_inverse_kinematics(first, [.1, .2, .3]) == [1., 2., 3., 4.]
    assert first.manipulator_ik.solve_request.call_args.args[0].target_position == [.1, .2, .3]
    second.manipulator_ik.solve_request.assert_not_called()

@pytest.mark.parametrize('angle',[0.,90.,179.9,180.])
def test_full_pose_with_large_orientation_changes(angle):
    solver=EndpointIK(ROOT)
    q=np.array([3.1,.7,.4,2.1]);world=np.array([0.,0.,-2.,0.,0.,0.])
    initial=np.array(solver.fk(q,P.base_T0_new,world,P.tipOffset)[-1]).ravel()
    target=R.from_quat(initial[[4,5,6,3]])*R.from_rotvec([np.radians(angle),0,0])
    x,j=solver.solve(q,world,initial[:3],target.as_matrix(),P.base_T0_new,P.tipOffset,[.01]*4,[6.27,3.4,3.4,5.7])
    final=np.array(solver.fk(j,P.base_T0_new,x,P.tipOffset)[-1]).ravel()
    assert np.linalg.norm(final[:3]-initial[:3])<.01
    assert (target.inv()*R.from_quat(final[[4,5,6,3]])).magnitude()<.05
    assert np.all(j>=.01-1e-9) and np.all(j<=np.array([6.27,3.4,3.4,5.7])+1e-9)
    assert 1<=len(solver.last_report['attempts'])<=4


def test_invalid_start_is_not_wrapped_or_silently_clamped():
    solver=EndpointIK(ROOT)
    with pytest.raises(ValueError):
        solver.solve([3.1,.7,.4,-1.],np.zeros(6),np.zeros(3),np.eye(3),P.base_T0_new,P.tipOffset,[.01]*4,[6.27,3.4,3.4,5.7])


def test_alignment_toggle_preserves_tool_pose_and_changes_posture():
    solver=EndpointIK(ROOT)
    q=[1.5,.7,.4,2.1];world=[0,0,-2,0,0,0]
    initial=np.array(solver.fk(q,P.base_T0_new,world,P.tipOffset)[-1]).ravel()
    target=R.from_quat(initial[[4,5,6,3]])
    errors=[]
    for weight in (0.,1.):
        x,j=solver.solve(q,world,initial[:3],target.as_matrix(),P.base_T0_new,
            P.tipOffset,[.01]*4,[6.27,3.4,3.4,5.7],base_align_weight=weight)
        first=np.array(solver.fk(j,P.base_T0_new,np.zeros(6),P.tipOffset)[0]).ravel()
        errors.append(np.arccos(np.clip(-R.from_quat(first[[4,5,6,3]]).as_matrix()[0,0],-1,1)))
        assert solver.last_report['position_error_m']<1e-6
        assert solver.last_report['orientation_error_rad']<1e-6
    assert errors[0]>np.radians(80)
    assert errors[1]<np.radians(10)


def test_alignment_objective_matches_first_link_fk_for_mount_rotations():
    solver=EndpointIK(ROOT)
    objective=solver.solver.get_function('nlp_f')
    for angle in (.01,1.5,np.pi,6.27):
        for mount in (P.base_T0_new,[.19,0,-.12,.3,-.4,.7]):
            z=np.r_[[0,0,-2,1,0,0,0],[angle,.7,.4,2.1]]
            p=np.r_[z,[0,0,-2,1,0,0,0],mount,P.tipOffset,0.]
            off=float(objective(z,p));p[-1]=1
            on=float(objective(z,p))
            first=np.array(solver.fk(z[7:],mount,np.zeros(6),P.tipOffset)[0]).ravel()
            expected=1+R.from_quat(first[[4,5,6,3]]).as_matrix()[0,0]
            assert on-off==pytest.approx(expected,abs=1e-10)


@pytest.mark.parametrize('axis', np.eye(3))
def test_axis_target_is_proper_rotation_without_parallel_axis_singularity(axis):
    from simlab.motion_planning.kinematics.inverse.whole_body import aligned_target_rotation
    marker=R.from_euler('xyz',[.2,-.4,.8]).as_matrix()
    target=aligned_target_rotation(marker,axis)
    np.testing.assert_allclose(target.T@target,np.eye(3),atol=1e-12)
    assert np.linalg.det(target)==pytest.approx(1.)
    np.testing.assert_allclose(target[:,2],marker@axis,atol=1e-12)
    if axis[2]==1:
        np.testing.assert_allclose(target,marker,atol=1e-12)


def test_typed_ik_rejects_wrong_request_without_invoking_solver():
    from simlab.motion_planning.kinematics.inverse import ik_class
    from simlab.motion_planning.kinematics.inverse.base import ManipulatorIKRequest, IKStatus
    solver = ik_class('whole_body_endpoint', task='pose')(ROOT)
    result = solver.solve_request(ManipulatorIKRequest([0., 0., 0.]))
    assert result.status is IKStatus.INVALID_INPUT
    assert result.joints is None and result.vehicle_pose is None


@pytest.mark.parametrize('target', [[float('nan'), 0., 0.], [0., 0.], [float('inf'), 0., 0.]])
def test_typed_arm_ik_rejects_invalid_targets(target):
    from simlab.motion_planning.kinematics.inverse import ManipulatorPositionIK
    from simlab.motion_planning.kinematics.inverse.base import ManipulatorIKRequest, IKStatus
    result = ManipulatorPositionIK(ROOT).solve_request(ManipulatorIKRequest(target))
    assert result.status is IKStatus.INVALID_INPUT
    assert result.joints is None


def test_arm_result_does_not_claim_pose_validation():
    from simlab.motion_planning.kinematics.inverse import ManipulatorPositionIK
    from simlab.motion_planning.kinematics.inverse.base import ManipulatorIKRequest
    solver = ManipulatorPositionIK(ROOT)
    result = solver.solve_request(ManipulatorIKRequest([.2, 0., -.4]))
    assert result.success
    assert result.vehicle_pose is None
    assert result.guarantees == frozenset({'finite_solution'})
    # A later failure cannot leak the previous solution or validation status.
    solver.solver = lambda target: (_ for _ in ()).throw(RuntimeError('failed'))
    failed = solver.solve_request(ManipulatorIKRequest([.2, 0., -.4]))
    assert not failed.success and failed.joints is None and not failed.guarantees
    assert result.success


def test_typed_whole_body_validates_rotation_and_nonfinite_fk():
    from dataclasses import replace
    from simlab.motion_planning.kinematics.inverse.base import WholeBodyIKRequest, IKStatus, validate_pose_endpoint
    solver = EndpointIK(ROOT)
    q = np.array([3.1, .7, .4, 2.1]); world = np.array([0., 0., -2., 0., 0., 0.])
    actual = np.array(solver.fk(q, P.base_T0_new, world, P.tipOffset)[-1]).ravel()
    request = WholeBodyIKRequest(q, world, actual[:3], R.from_quat(actual[[4,5,6,3]]).as_matrix(),
                                P.base_T0_new, P.tipOffset, [.01]*4, [6.27,3.4,3.4,5.7])
    good = solver.solve_request(request)
    assert good.success and 'endpoint_pose' in good.guarantees
    assert good.diagnostics['position_error_m'] < .01
    for bad in (replace(request, target_rotation=np.zeros((3,3))),
                replace(request, target_rotation=np.diag([1.,1.,-1.])),
                replace(request, position_tolerance=float('nan'))):
        rejected = solver.solve_request(bad)
        assert rejected.status is IKStatus.INVALID_INPUT and rejected.joints is None
    with pytest.raises(ValueError, match='FK pose'):
        validate_pose_endpoint(request, world, q, lambda *args: [np.full(7, np.nan)])


def test_duplicate_registry_names_are_rejected(monkeypatch):
    import simlab.motion_planning.kinematics.inverse as registry
    monkeypatch.setattr(registry, 'DEFAULT_IK_CLASSES', [EndpointIK, EndpointIK])
    with pytest.raises(ValueError, match='Duplicate'):
        registry.ik_class('whole_body_endpoint', task='pose')


class Solver:
    def __init__(self, candidates):
        self.candidates = iter(candidates)

    def __call__(self, **kwargs):
        self.current = next(self.candidates)
        return {'x': self.current[0], 'f': self.current[1]}

    def stats(self):
        return {'success': True, 'return_status': 'Solve_Succeeded'}


def candidate(joints=.5, position=0., cost=1.):
    return (np.r_[[position,0.,0.], [1.,0.,0.,0.], np.full(4,joints)], cost)


def solve(candidates):
    def fk(q, base, world, tip):
        return [np.r_[world[:3], [1.,0.,0.,0.]]]
    policy = EndpointIK(ROOT)
    policy.solver = Solver(candidates)
    policy.fk = fk
    result = policy.solve(np.full(4,.5), np.zeros(6), np.zeros(3), np.eye(3),
                          np.zeros(6), np.zeros(6), np.zeros(4), np.ones(4))
    return policy, result


def test_rank_only_converged_in_bounds_pose_valid_candidates():
    policy, (_, joints) = solve([candidate(cost=3.), candidate(joints=2.,cost=-10.),
                                candidate(position=1.,cost=-20.), candidate(joints=.8,cost=1.)])
    np.testing.assert_array_equal(joints, np.full(4,.8))
    assert policy.last_report['valid_candidates'] == 2


def test_solver_success_cannot_override_fk_endpoint_rejection():
    with pytest.raises(RuntimeError, match='No converged'):
        solve([candidate(position=1.)]*4)


def test_solver_success_cannot_override_invalid_quaternion():
    z, cost = candidate()
    z[3:7] = 0.
    with pytest.raises(RuntimeError, match='No converged'):
        solve([(z,cost)]*4)
