"""SO(3) controller regression: representation, cut locus, and feedforward."""
from pathlib import Path
import casadi as ca
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from simlab.uvms_parameters import select_robot_params

ROOT = Path(__file__).parents[1]/'resource/model_functions/vehicle'


def variants(rpy):
    a = np.asarray(rpy)
    return [a, a+np.array([4*np.pi, -2*np.pi, 6*np.pi]),
            np.array([a[0]+np.pi, np.pi-a[1], a[2]+np.pi])]


def controller(name, state, target, vel=None, acc=None, gains=None):
    _, p = select_robot_params('robot_1_')
    f = ca.Function.load(str(ROOT/name))
    gains = np.ones(6) if gains is None else gains
    args = [p.model_params, gains, np.zeros(6), np.ones(6), np.zeros(6),
            state, target, np.zeros(6) if vel is None else vel]
    if 'tracking' in name:
        args += [np.zeros(6) if acc is None else acc, .01, np.zeros(6)]
    else:
        args += [.01]
    return [np.asarray(x).ravel() for x in f(*args)]


@pytest.mark.parametrize('name', ['uv_reg_pid_controller.casadi', 'uv_trackingController.casadi'])
def test_controller_equivalent_euler_representations(name):
    state = np.r_[[1., 2., 3.], [.4, 1.2, -.7], [.1]*6]
    target = np.r_[[2., 1., 4.], [-.3, -.8, 2.5]]
    vel = np.arange(6.)*.03
    expected = controller(name, state, target, vel)
    for measured in variants(state[3:6]):
        for desired in variants(target[3:]):
            s, t = state.copy(), target.copy()
            s[3:6], t[3:] = measured, desired
            actual = controller(name, s, t, vel)
            np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_rotation_log_independent_and_finite_jacobians():
    log = ca.Function.load(str(ROOT/'rotation_log.casadi'))
    r = ca.SX.sym('r', 3, 3)
    derivative = ca.Function('derivative', [r], [ca.jacobian(log(r), ca.vec(r))])
    rng = np.random.default_rng(641)
    for angle in [0., 1e-10, .1, 2., np.pi-1e-8, np.pi]:
        for _ in range(10):
            axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
            R = Rotation.from_rotvec(angle*axis).as_matrix()
            result = np.asarray(log(R)).ravel()
            assert np.isfinite(np.asarray(derivative(R))).all()
            assert np.linalg.norm(result) <= np.pi+1e-12
            np.testing.assert_allclose(Rotation.from_rotvec(result).as_matrix(), R, atol=1e-10)
            if angle < np.pi-1e-9:
                np.testing.assert_allclose(result, angle*axis, atol=1e-10)


@pytest.mark.parametrize('name', ['uv_reg_pid_controller.casadi', 'uv_trackingController.casadi'])
@pytest.mark.parametrize('pitch', [np.pi/2, -np.pi/2, np.pi/2-1e-10])
def test_controller_no_euler_rate_singularity(name, pitch):
    state = np.r_[np.zeros(3), [.4, pitch, .8], np.zeros(6)]
    target = state[:6].copy()
    target[5] += .3
    assert all(np.isfinite(x).all() for x in controller(name, state, target))


def test_invdyn_reference_transport_derivative():
    # Zero feedback gains isolate feedforward. Reproduce the old function at
    # current=target with independently transported velocity/acceleration.
    state = np.r_[[0.,0.,2.], [.4,-.5,.7], [.1,-.2,.3,.2,-.3,.4]]
    target = np.r_[[0.,0.,2.], [-.7,.4,-.9]]
    vel, acc = np.arange(6.)*.06, np.arange(6.)*.03
    R = Rotation.from_euler('xyz',state[3:6]).as_matrix()
    Rd = Rotation.from_euler('xyz',target[3:]).as_matrix()
    h = 1e-6
    def rotated(t):
        Rt = R@Rotation.from_rotvec(t*state[9:12]).as_matrix()
        Rdt = Rd@Rotation.from_rotvec(t*vel[3:]).as_matrix()
        A = Rt.T@Rdt
        return np.r_[A@(vel[:3]+t*acc[:3]), A@(vel[3:]+t*acc[3:])]
    converted = rotated(0.)
    derivative = (rotated(h)-rotated(-h))/(2*h)
    actual = controller('uv_trackingController.casadi',state,target,vel,acc,np.zeros(6))[0]
    # At equal orientation transport still has angular-rate derivative terms;
    # compensate them to compare the same physical reference acceleration.
    omega_delta = converted[3:]-state[9:12]
    corrected = derivative-np.r_[np.cross(omega_delta,converted[:3]),
                                np.cross(omega_delta,converted[3:])]
    expected = controller('uv_trackingController.casadi',state,state[:6],converted,corrected,np.zeros(6))[0]
    np.testing.assert_allclose(actual,expected,atol=1e-7)


def test_geometric_clf_directional_derivative_and_invariance():
    OGES = pytest.importorskip("namor").OGES
    c = OGES(6, use_jit=False, cyclic_dims=(3, 4, 5))
    f = c.A_b_V_func
    s = c.ss
    matrices = ca.Function('weights', [s.opt_weight], [s.Q1,s.Q2,s.Q3])
    weights = np.r_[[.2,.2,.2,.1,.8,.3], [.3,.3,.3,.2,.5,.4], np.ones(6)*.04, np.ones(6)*.2]
    Q1,Q2,Q3 = [np.asarray(v) for v in matrices(weights)]
    rng = np.random.default_rng(774)
    log = ca.Function.load(str(ROOT/'rotation_log.casadi'))
    def G(rpy):
        R = Rotation.from_euler('xyz',rpy).as_matrix()
        return np.block([[R,np.zeros((3,3))],[np.zeros((3,3)),R]])
    for _ in range(20):
        q,qr,v,vr,a,ar = rng.normal(size=(6,6))*.4
        q[2]=qr[2]=2.
        def evaluate(q,v,qr,vr):
            return f(q,v,qr,vr,G(qr[3:]),ar,weights,G(q[3:]),1.)
        A,b,V = evaluate(q,v,qr,vr)
        for variant in variants(q[3:]):
            changed=q.copy();changed[3:]=variant
            results=evaluate(changed,v,qr,vr)
            for actual,expected in zip(results,(A,b,V)):
                np.testing.assert_allclose(actual,expected,atol=1e-10)
        R,Rd=G(q[3:])[:3,:3],G(qr[3:])[:3,:3]
        error=np.r_[q[:3]-qr[:3],-np.asarray(log(R.T@Rd)).ravel()]
        dv=np.r_[R@v[:3]-Rd@vr[:3],v[3:]-R.T@Rd@vr[3:]]
        decay=.5*error@Q1@error+.5*dv@Q2@dv+dv@Q3@error
        def step(q,v,a,t):
            rotation=Rotation.from_euler('xyz',q[3:])
            return np.r_[q[:3]+t*rotation.apply(v[:3]),
                         (rotation*Rotation.from_rotvec(t*v[3:])).as_euler('xyz')], v+t*a
        h=1e-6
        values=[]
        for t in (-h,h):
            qt,vt=step(q,v,a,t);qrt,vrt=step(qr,vr,ar,t)
            values.append(float(evaluate(qt,vt,qrt,vrt)[2]))
        np.testing.assert_allclose((values[1]-values[0])/(2*h)+decay,
                                   float(A@a-b),atol=1e-7)
    # The rest pose must not generate a NaN from differentiating the log at zero.
    q=np.zeros(6)
    for value in f(q,q,q,q,np.eye(6),q,weights,np.eye(6),1.):
        assert np.isfinite(np.asarray(value)).all()
