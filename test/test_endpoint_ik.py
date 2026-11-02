from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
from simlab.motion_planning.endpoint_ik import EndpointIK
from simlab.uvms_parameters_sim import ReachParams as P

ROOT=Path(__file__).resolve().parents[1]/'resource'

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
    from simlab.motion_planning.endpoint_ik import aligned_target_rotation
    marker=R.from_euler('xyz',[.2,-.4,.8]).as_matrix()
    target=aligned_target_rotation(marker,axis)
    np.testing.assert_allclose(target.T@target,np.eye(3),atol=1e-12)
    assert np.linalg.det(target)==pytest.approx(1.)
    np.testing.assert_allclose(target[:,2],marker@axis,atol=1e-12)
    if axis[2]==1:
        np.testing.assert_allclose(target,marker,atol=1e-12)
