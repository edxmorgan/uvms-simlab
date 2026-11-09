from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from simlab.controllers import DEFAULT_CONTROLLER_CLASSES
from simlab.controllers.sliding_mode import SlidingModeController
from simlab.controllers.model_profile import NominalModel
from simlab.dynamics_profiles import load_robot_dynamics_profile


@pytest.fixture
def controller(monkeypatch):
    monkeypatch.setattr('ament_index_python.get_package_share_directory',
                        lambda _: str(Path(__file__).parents[1]/'resource'))
    return SlidingModeController(Mock(),4,'robot_1_')


def test_public_registry_and_output_contract(controller):
    assert any(c.registry_name=='SMC' for c in DEFAULT_CONTROLLER_CLASSES)
    assert DEFAULT_CONTROLLER_CLASSES[0].registry_name=='PID'
    x=np.zeros(12);x[2]=2
    u=controller.vehicle_controller(x,x[:6],np.zeros(6),np.zeros(6),.01)
    assert u.shape==(6,) and np.isfinite(u).all()
    q=np.array([3.1,.7,.4,2.1,.01])
    tau=controller.arm_controller(q,np.zeros(5),q,np.zeros(5),np.zeros(5),.01)
    assert tau.shape==(5,) and np.isfinite(tau).all()
    assert np.all(tau<=controller.arm_u_max) and np.all(tau>=controller.arm_u_min)
    # Gripper spring=0: holding still must not invent a gripper force.
    assert tau[-1]==pytest.approx(0)
    x[0]=float('nan')
    with pytest.raises(ValueError,match='finite'):
        controller.vehicle_controller(x,np.zeros(6),np.zeros(6),np.zeros(6),.01)


def test_model_profile_and_export_inputs(controller,monkeypatch):
    profile=load_robot_dynamics_profile('dory_alpha')
    profile['manipulator']['endeffector_mass']=123.
    controller.set_model_profile(NominalModel.from_profile(profile))
    assert controller.gripper_parameters[0]==123.
    assert len(controller.vehicle_model_params)==27
    snapshot=controller.configuration_snapshot()
    assert snapshot['parameters']['gripper_parameters'][0]==123.
    profile['vehicle']['current_velocity'][0]=.1
    before=controller.configuration_snapshot()
    with pytest.raises(ValueError,match='zero-current'):
        controller.set_model_profile(NominalModel.from_profile(profile))
    assert controller.configuration_snapshot()==before


@pytest.mark.parametrize("source,destination", [("PID", "SMC"), ("SMC", "PID")])
def test_smc_switch_preserves_active_plan(controller, monkeypatch, source, destination):
    from simlab.robot import Robot, ControlMode
    monkeypatch.setattr('simlab.robot.reset_dynamics_observer', lambda _: None)
    class Peer:
        def __init__(self): self.reset_controller_state = Mock()
        def arm_controller(self, *args): pass
        def vehicle_controller(self, *args): pass
    controller.reset_controller_state = Mock()
    controllers = {'SMC': controller, 'PID': Peer()}
    specs = {}
    for name, c in controllers.items():
        spec = SimpleNamespace(vehicle_fn=c.vehicle_controller, arm_fn=c.arm_controller)
        spec.resolve = lambda spec=spec: spec
        specs[name] = spec
    r = SimpleNamespace(controller_name=source,
        reset_vehicle_command_yaw_memory=lambda: None, control_mode=ControlMode.PLANNER,
        _controllers=specs, _controller_is_replay=lambda c: False,
        _active_controller_is_replay=lambda: False,
        node=Mock(), prefix='robot_1_', abrupt_planner_stop=Mock(),
        hold_current_state_with_feedback=Mock(), arm=SimpleNamespace(grasp_command=.02),
        pose_command=[1, 2, 3, .1, .2, .3], planner=object())
    r.active_controller_instance = lambda: controllers[r.controller_name]
    # Exercise the real mode setter: remaining in planner mode must not stop it.
    r.set_control_mode = lambda mode: Robot.set_control_mode(r, mode)
    planner, reference = r.planner, r.pose_command
    assert Robot.set_controller(r, destination)
    controllers[destination].reset_controller_state.assert_called_once()
    controllers[source].reset_controller_state.assert_not_called()
    r.abrupt_planner_stop.assert_not_called()
    r.hold_current_state_with_feedback.assert_not_called()
    assert r.planner is planner and r.pose_command is reference
    assert r.arm.grasp_command == .02
    assert r.vehicle_controller_fn == controllers[destination].vehicle_controller
    assert r.arm_controller_fn == controllers[destination].arm_controller


@pytest.mark.parametrize("source,destination", [("PID", "SMC"), ("SMC", "PID")])
def test_smc_selection_preserves_waypoint_mission(source, destination):
    from simlab.uvms_backend import UVMSBackendCore
    mission = Mock()
    backend = SimpleNamespace(vehicle_waypoint_missions={0: mission})
    robot = SimpleNamespace(controller_name=source, prefix='robot_1_', k_robot=0,
        list_controllers=lambda: ['SMC', 'PID'], set_controller=Mock(return_value=True))
    assert UVMSBackendCore.set_robot_controller(backend, robot, destination)[0]
    robot.set_controller.assert_called_once_with(destination)
    mission.stop.assert_not_called()


def test_gain_configuration_is_atomic_and_held_only(controller):
    from simlab.controller_configuration import controller_configuration
    import json
    replay=SimpleNamespace(playback_status=lambda:'stopped')
    robot=SimpleNamespace(controller_instance=lambda name:replay if name=='CmdReplay' else controller,
        _controller_is_replay=lambda c:False,prefix='robot_1_',status='active',sim_reset_hold=True)
    original=controller.arm_surface.copy()
    request=SimpleNamespace(controller_name='SMC',operation='set_gains',gains_json=json.dumps(
        {'arm_surface':(original*1.1).tolist(),'arm_boundary':[0]*5}))
    with pytest.raises(ValueError):controller_configuration(robot,request)
    np.testing.assert_array_equal(controller.arm_surface,original)
    request.gains_json=json.dumps({'arm_surface':(original*1.1).tolist()})
    robot.sim_reset_hold=False
    with pytest.raises(ValueError,match='inactive or held'):controller_configuration(robot,request)
    robot.sim_reset_hold=True
    result=json.loads(controller_configuration(robot,request))
    np.testing.assert_allclose(result['parameters']['arm_surface'],original*1.1)
    request.gains_json=json.dumps({'arm_u_max':[100]*5})
    with pytest.raises(ValueError,match='gain arrays'):controller_configuration(robot,request)


def test_pid_uses_same_vehicle_limits(controller):
    from simlab.controllers.pid import LowLevelPidController
    pid=LowLevelPidController(Mock(),4,'robot_1_')
    target=np.array([1e4,-1e4,1e4,0,0,0])
    effort=pid.vehicle_controller(np.zeros(12),target,np.zeros(6),np.zeros(6),.01)
    assert np.all(effort<=controller.uv_u_max) and np.all(effort>=controller.uv_u_min)
    assert effort[0]==controller.uv_u_max[0]



def test_switch_between_existing_controllers_resets_destination_memory(monkeypatch):
    from simlab.robot import Robot,ControlMode
    monkeypatch.setattr('simlab.robot.reset_dynamics_observer',lambda _:None)
    class Controller:
        def __init__(self):self.reset_controller_state=Mock()
        def arm_controller(self,*args):pass
        def vehicle_controller(self,*args):pass
    controllers={name:Controller() for name in ['PID','InvDyn']}
    specs={}
    for name,c in controllers.items():
        spec=SimpleNamespace(arm_fn=c.arm_controller,vehicle_fn=c.vehicle_controller)
        spec.resolve=lambda spec=spec:spec
        specs[name]=spec
    r=SimpleNamespace(controller_name='PID',control_mode=ControlMode.PLANNER,
        reset_vehicle_command_yaw_memory=lambda:None,_controllers=specs,
        _controller_is_replay=lambda c:False,_active_controller_is_replay=lambda:False,
        set_control_mode=lambda m:None,node=Mock(),prefix='robot_1_')
    r.active_controller_instance=lambda:controllers[r.controller_name]
    Robot.set_controller(r,'InvDyn');controllers['InvDyn'].reset_controller_state.assert_called_once()
    Robot.set_controller(r,'InvDyn');controllers['InvDyn'].reset_controller_state.assert_called_once()
    Robot.set_controller(r,'PID');controllers['PID'].reset_controller_state.assert_called_once()
