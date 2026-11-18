from pathlib import Path
from unittest.mock import Mock
import casadi as ca
import numpy as np
import pytest
from simlab.controllers.pid import LowLevelPidController
from simlab.controllers.invDyn import LowLevelInvDynController
from simlab.uvms_parameters import select_robot_params


@pytest.fixture(params=['PID', 'InvDyn'])
def vehicle(request):
    _, p = select_robot_params('robot_real_')
    cls = LowLevelPidController if request.param == 'PID' else LowLevelInvDynController
    c = cls.__new__(cls);c.node=Mock()
    root=Path(__file__).parents[1]/'resource/model_functions/vehicle'
    c.uv_pid_controller=ca.Function.load(str(root/'uv_reg_pid_controller.casadi'))
    c.tracking_uv_controller=ca.Function.load(str(root/'uv_trackingController.casadi'))
    c.vehicle_model_params=p.model_params;c.v_c=p.v_c
    c.vehicle_pid_i_buffer=np.zeros(6);c.vehicle_i_limit=np.ones(6)*100
    c.uv_u_min=np.asarray(p.u_min);c.uv_u_max=np.asarray(p.u_max)
    c.kp=np.ones(6);c.ki=np.ones(6);c.kd=np.ones(6)
    return c


def test_vehicle_saturation_and_unwinding(vehicle):
    c=vehicle;x=np.zeros(12);target=np.zeros(6);target[0]=100;z=np.zeros(6)
    for _ in range(30):
        u=c.vehicle_controller(x,target,z,z,.01)
        assert u[0]==c.uv_u_max[0]
    np.testing.assert_array_equal(c.vehicle_pid_i_buffer,z)
    c.kp[:]=0;c.vehicle_pid_i_buffer[0]=100;target[0]=-1
    c.vehicle_controller(x,target,z,z,.01)
    assert c.vehicle_pid_i_buffer[0]==pytest.approx(99.99)
    c.ki[:]=0
    c.vehicle_controller(x,target,z,z,.01)
    np.testing.assert_array_equal(c.vehicle_pid_i_buffer,z)


def test_vehicle_integrates_unsaturated_and_rejects_bad_output_atomically(vehicle):
    c=vehicle;x=np.zeros(12);t=np.zeros(6);t[0]=.001;z=np.zeros(6)
    c.vehicle_controller(x,t,z,z,.01)
    assert c.vehicle_pid_i_buffer[0]==pytest.approx(.00001)
    before=c.vehicle_pid_i_buffer.copy()
    with pytest.raises(ValueError):c.vehicle_controller(x,t,z,z,float('nan'))
    np.testing.assert_array_equal(c.vehicle_pid_i_buffer,before)


@pytest.fixture(scope='module',params=['Ours','DR-Ours'])
def oges(request):
    pytest.importorskip('namor')
    from simlab.controllers.oges import OgesModelbasedController
    from simlab.controllers.dr_oges import DistributionallyRobustOgesController
    cls={'Ours':OgesModelbasedController,'DR-Ours':DistributionallyRobustOgesController}[request.param]
    node = Mock()
    node.get_clock().now().nanoseconds = 0
    return cls(node,4,'robot_real_')


def test_oges_shared_model_and_gripper_caps(oges):
    c=oges;p,v=select_robot_params('robot_real_')
    np.testing.assert_allclose(np.asarray(c.alpha_params.step_model_params)[:75],np.asarray(p.sim_p).ravel())
    np.testing.assert_allclose(np.asarray(c.blue.sim_p)[:27],np.asarray(v.model_params).ravel())
    q=np.array([3.1,.7,.4,2.1,.01]);z=np.zeros(5)
    for sign in [-1,1]:
        t=q.copy();t[-1]+=sign*.03
        effort=c.arm_controller(q,z,t,z,z,.005)
        assert effort[-1]==pytest.approx(sign*10)


def test_arm_clf_weights_have_positive_storage_and_schur_nominal_error_dynamics(oges):
    # This checks the bandwidth design's corresponding nominal error system;
    # nonlinear, saturated closed-loop performance requires the separate trials.
    w=np.asarray(oges.arm_weights).ravel()
    p1,p2,p12=w[:4],w[4:8],w[8:12]
    assert np.all(p1*p2-p12**2>0)
    for h in (.005,.01,.02,.05):
        for kp,kd in zip(p1/p2,2*p12/p2):
            transition=np.array([[1-.5*h*h*kp,h-.5*h*h*kd],[-h*kp,1-h*kd]])
            assert max(abs(np.linalg.eigvals(transition)))<1.


def test_oges_arm_exact_tracking_preserves_inverse_dynamics(oges):
    c=oges;q=np.array([3.1,.7,.4,2.1,.01]);z=np.zeros(5)
    for moving in [False,True]:
        v=np.array([.01,-.01,.01,-.01,0]) if moving else z
        a=np.array([.001]*4+[0]) if moving else z
        p=c.alpha_params.step_model_params
        expected=np.asarray(c.M_arm_matrix(q[:4],p)@a[:4]+c.Cqot_arm_vector(q[:4],v[:4],p)
                            +c.g_arm_vec(q[:4],p)+c.B_arm_vec(v[:4],p)).ravel()
        c.reset_controller_state()
        got=c.arm_controller(q,v,q,v,a,.005)
        np.testing.assert_allclose(got[:4],np.clip(expected,c.arm_u_min[:4],c.arm_u_max[:4]),atol=1e-9)


def test_oges_vehicle_equilibrium_retains_bias(oges):
    c=oges;x=np.zeros(12);x[2]=2.;z=np.zeros(6)
    c.reset_controller_state()
    expected=np.asarray(c.g_uv_vec(x[:6],c.blue.sim_p)).ravel()
    got=c.vehicle_controller(x,x[:6],z,z,.005)
    np.testing.assert_allclose(got,np.clip(expected,c.blue.u_min,c.blue.u_max),atol=1e-9)


def test_vehicle_windup_gate_uses_rotated_body_effort(vehicle):
    c=vehicle;x=np.zeros(12);x[5]=np.pi/2
    target=x[:6].copy();target[0]=100;z=np.zeros(6)
    for _ in range(10):
        u=c.vehicle_controller(x,target,z,z,.01)
    assert u[1]==pytest.approx(c.uv_u_min[1])
    np.testing.assert_array_equal(c.vehicle_pid_i_buffer,z)


def test_oges_vehicle_small_error_keeps_linear_restoring_gain(oges):
    c = oges
    x = np.zeros(12)
    x[2] = 2.
    z = np.zeros(6)
    efforts = []
    for error in (1e-4, 1e-6):
        target = x[:6].copy()
        target[0] += error
        c.reset_controller_state()
        efforts.append(c.vehicle_controller(x, target, z, z, .01)[0])
    assert efforts[1] > 0
    # An absolute projection regularizer must not erase the restoring gain
    # as the error approaches zero (the old equality law was cubic here).
    assert efforts[1] / efforts[0] == pytest.approx(.01, rel=.01)



@pytest.mark.parametrize('dt', [0., -.01, float('nan'), float('inf')])
def test_pid_arm_rejects_invalid_interval_before_mutating_integral(dt):
    c = LowLevelPidController.__new__(LowLevelPidController)
    c.arm_dof = 4
    c.arm_pid_i_buffer = np.arange(5, dtype=float)
    c.arm_pid_controller = Mock()
    before = c.arm_pid_i_buffer.copy()
    q = np.zeros(5)
    with pytest.raises(ValueError, match='dt must be finite and positive'):
        c.arm_controller(q, q, q, q, q, dt)
    np.testing.assert_array_equal(c.arm_pid_i_buffer, before)
    c.arm_pid_controller.assert_not_called()


def test_oges_returns_api_command_without_additional_control_state(oges, monkeypatch):
    policy = Mock(wraps=oges.vehicle_policy)
    monkeypatch.setattr(oges, 'vehicle_policy', policy)
    state = np.array([.1, -.2, 2., .3, -.4, .5, .01, -.02, .03, 0., 0., 0.])
    target = state[:6] + .01
    zero = np.zeros(6)
    actual = oges.vehicle_controller(state, target, zero, zero, .02)
    policy.assert_called_once()
    expected = np.asarray(policy._mock_wraps(*policy.call_args.args)[0]).ravel()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(oges.vehicle_controller(state, target, zero, zero, .02), actual)
