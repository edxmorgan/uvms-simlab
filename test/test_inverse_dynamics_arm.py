from pathlib import Path
from unittest.mock import Mock
import casadi as ca
import numpy as np
import pytest
from simlab.controllers.invDyn import LowLevelInvDynController
from simlab.controllers.sliding_mode import SlidingModeController


@pytest.fixture
def controller(monkeypatch):
    monkeypatch.setattr('ament_index_python.get_package_share_directory',
                        lambda _: str(Path(__file__).parents[1]/'resource'))
    c = LowLevelInvDynController(Mock(), 4, 'robot_real_')
    c.arm_u_min = np.full(5, -1e6)
    c.arm_u_max = np.full(5, 1e6)
    return c


def test_complete_bias_matches_existing_smc(controller):
    c = controller
    smc = SlidingModeController(Mock(), 4, 'robot_real_')
    smc.arm_u_min, smc.arm_u_max = c.arm_u_min, c.arm_u_max
    smc.arm_switching = np.zeros(5)
    q = np.array([3.1, .7, .4, 2.1, .01])
    v = np.array([.1, -.2, .3, -.4, .002])
    acc = np.array([.2, .1, -.1, .3, .001])
    np.testing.assert_allclose(c.arm_controller(q,v,q,v,acc,.005),
                               smc.arm_controller(q,v,q,v,acc,.005), atol=1e-10)


def test_viscous_friction_and_scalar_gripper_physics(controller):
    c = controller
    q = np.array([3.1,.7,.4,2.1,.01]);v=np.array([.1,-.2,.3,-.4,.002]);a=np.zeros(5)
    first=c.arm_controller(q,v,q,v,a,.005)
    p=np.asarray(c.arm_model_params).ravel().copy();p[40:44]+=2;c.arm_model_params=p
    change=c.arm_controller(q,v,q,v,a,.005)-first
    np.testing.assert_allclose(change[:4],2*v[:4],atol=1e-10)
    c.gripper_parameters=np.array([123.,17.,11.]);a[-1]=.03
    assert c.arm_controller(q,v,q,v,a,.005)[-1] == pytest.approx(123*.03+17*.002+11*.01)


def test_saturation_blocks_windup_but_allows_unwinding(controller):
    c=controller;c.arm_kp=np.zeros(5);c.arm_kd=np.zeros(5);c.arm_ki=np.array([0,0,0,0,1.])
    c.gripper_parameters=np.array([1.,0.,0.]);c.arm_u_min[-1]=-.1;c.arm_u_max[-1]=.1
    q=np.zeros(5);target=q.copy();target[-1]=1
    for _ in range(100): c.arm_controller(q,q,target,q,q,.01)
    assert c.arm_pid_i_buffer[-1] <= .1+1e-12
    c.arm_pid_i_buffer[-1]=2.;target[-1]=-1
    c.arm_controller(q,q,target,q,q,.01)
    assert c.arm_pid_i_buffer[-1] == pytest.approx(1.99)
    c.arm_ki[:]=0
    c.arm_controller(q,q,target,q,q,.01)
    np.testing.assert_array_equal(c.arm_pid_i_buffer,np.zeros(5))


def test_gripper_nominal_tracking_converges(controller):
    c=controller;q=np.array([3.1,.7,.4,2.1,.01]);target=q.copy();target[-1]=.02
    v=np.zeros(5);z=np.zeros(5);dt=.005;m,b,k=c.gripper_parameters
    for _ in range(1400):
        u=c.arm_controller(q,v,target,z,z,dt)[-1]
        v[-1]+=dt*(u-b*v[-1]-k*q[-1])/m
        q[-1]+=dt*v[-1]
    assert abs(q[-1]-target[-1]) < 1e-5
    assert abs(v[-1]) < 1e-5


def test_bad_input_does_not_poison_integrator(controller):
    c=controller;q=np.zeros(5);bad=q.copy();bad[2]=np.nan
    with pytest.raises(ValueError,match='finite'):c.arm_controller(q,q,q,bad,q,.005)
    with pytest.raises(ValueError,match='dt'):c.arm_controller(q,q,q,q,q,0)
    np.testing.assert_array_equal(c.arm_pid_i_buffer,np.zeros(5))


def test_integral_gate_respects_cross_joint_inertia(controller):
    c=controller;c.arm_kp=np.zeros(5);c.arm_kd=np.zeros(5);c.arm_ki=np.zeros(5)
    q=np.array([3.1,.7,.4,2.1,.01]);z=np.zeros(5)
    bias=c.arm_controller(q,z,q,z,z,.01)
    columns=[]
    for j in range(4):
        acc=z.copy();acc[j]=1
        columns.append(c.arm_controller(q,z,q,z,acc,.01)-bias)
    M=np.array(columns).T
    coupling=M[:4,:4].copy();np.fill_diagonal(coupling,0)
    i,j=np.unravel_index(np.argmax(np.abs(coupling)),coupling.shape)
    assert abs(M[i,j])>1e-5
    c.arm_ki[j]=1
    target=q.copy();target[j]+=np.sign(M[i,j])
    c.arm_u_max[i]=bias[i]+abs(M[i,j])*.001
    c.arm_controller(q,z,target,z,z,.01)
    assert c.arm_pid_i_buffer[j]==0  # another actuator's limit blocks this increment


def test_profile_updates_gripper_atomically(controller):
    from dataclasses import replace
    from simlab.dynamics_profiles import NominalModel
    from simlab.dynamics_profiles import load_robot_dynamics_profile
    model=NominalModel.from_profile(load_robot_dynamics_profile('dory_alpha'))
    controller.set_model_profile(replace(model,gripper_parameters=(123.,17.,11.)))
    np.testing.assert_array_equal(controller.gripper_parameters,[123.,17.,11.])
    before=controller.configuration_snapshot()
    with pytest.raises(ValueError,match='gripper'):
        controller.set_model_profile(replace(model,gripper_parameters=(-1.,17.,11.)))
    assert controller.configuration_snapshot()==before
