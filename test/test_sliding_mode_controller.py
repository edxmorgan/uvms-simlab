import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from simlab.controllers import DEFAULT_CONTROLLER_CLASSES
from simlab.controllers.sliding_mode import SlidingModeController
from simlab.dynamics_profiles import NominalModel
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
    r._record_controller_context = lambda: Robot._record_controller_context(r)
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
    from types import SimpleNamespace
    node = Mock()
    parameters = {}
    node.has_parameter.side_effect = lambda name: name in parameters
    node.declare_parameter.side_effect = lambda name, value: parameters.setdefault(name, value)
    node.get_parameter.side_effect = lambda name: SimpleNamespace(value=parameters[name])
    pid=LowLevelPidController(node,4,'robot_1_')
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
    r._record_controller_context=lambda:Robot._record_controller_context(r)
    r.active_controller_instance=lambda:controllers[r.controller_name]
    Robot.set_controller(r,'InvDyn');controllers['InvDyn'].reset_controller_state.assert_called_once()
    Robot.set_controller(r,'InvDyn');controllers['InvDyn'].reset_controller_state.assert_called_once()
    Robot.set_controller(r,'PID');controllers['PID'].reset_controller_state.assert_called_once()


@pytest.mark.parametrize('error', [ValueError('SMC currently has simulation gains only'),
                                  RuntimeError('invalid export'), OSError('missing export')])
def test_failed_lazy_controller_selection_preserves_active_motion(error):
    from simlab.robot import Robot, ControllerSpec, ControlMode
    from simlab.uvms_backend import UVMSBackendCore
    from types import SimpleNamespace
    from unittest.mock import Mock
    factory = Mock(side_effect=error)
    spec = ControllerSpec('SMC', None, None, factory=factory)
    arm, vehicle = Mock(), Mock()
    r = SimpleNamespace(_controllers={'SMC': spec}, controller_name='PID',
        arm_controller_fn=arm, vehicle_controller_fn=vehicle, node=Mock(),
        control_mode=ControlMode.REPLAY_SETTLE, _hold_after_arm_replay=True,
        reset_vehicle_command_yaw_memory=Mock(), cancel_replay_settle=Mock(),
        active_controller_instance=Mock(), prefix='robot_real_',
        list_controllers=lambda: ['SMC'])
    r.set_controller = lambda name: Robot.set_controller(r, name)
    ok, message = UVMSBackendCore.set_robot_controller(None, r, 'SMC')
    assert not ok and str(error) in message
    assert r.controller_name == 'PID'
    assert r.arm_controller_fn is arm and r.vehicle_controller_fn is vehicle
    assert r.control_mode is ControlMode.REPLAY_SETTLE
    assert r._hold_after_arm_replay
    r.reset_vehicle_command_yaw_memory.assert_not_called()
    r.cancel_replay_settle.assert_not_called()
    r.active_controller_instance.assert_not_called()
    assert spec.factory is factory and spec.arm_fn is None


def test_controller_runtime_exception_stops_instead_of_escaping_executor(monkeypatch):
    from simlab.robot import Robot
    from types import SimpleNamespace
    from unittest.mock import Mock
    r=SimpleNamespace(stop_execution=Mock(),node=Mock())
    monkeypatch.setattr(Robot,'_control_loop_impl',Mock(side_effect=RuntimeError('solver failed')))
    r._control_tick = time.monotonic() - .01
    Robot.control_loop_callback(r)
    r.stop_execution.assert_called_once()
    assert 'solver failed' in r.node.get_logger().error.call_args.args[0]


def test_command_boundary_rejects_nonfinite_and_preserves_finite_commands():
    from simlab.robot import Robot
    from types import SimpleNamespace
    from unittest.mock import Mock
    r=SimpleNamespace(_apply_menu_grasper_effort=lambda v:v,publish_vehicle_and_arm=Mock())
    with pytest.raises(ValueError):Robot.publish_commands(r,[float('nan')]*6,[0.]*5)
    r.publish_vehicle_and_arm.assert_not_called()
    Robot.publish_commands(r,[1.]*6,[.1]*5)
    r.publish_vehicle_and_arm.assert_called_once_with([1.]*6,[.1]*5)


def test_hardware_smc_uses_real_profile_and_preserves_simulation_defaults(monkeypatch):
    from simlab.uvms_parameters_real import ReachParams as Real
    from simlab.uvms_parameters_sim import ReachParams as Sim
    monkeypatch.setattr('ament_index_python.get_package_share_directory',
                        lambda _: str(Path(__file__).parents[1]/'resource'))
    before=Sim.smc_surface.copy()
    real_before=Real.smc_surface.copy()
    hardware=SlidingModeController(Mock(),4,'robot_real_')
    np.testing.assert_array_equal(hardware.arm_surface,Real.smc_surface)
    np.testing.assert_array_equal(hardware.arm_u_max[:4],Real.u_max)
    np.testing.assert_array_equal(hardware.arm_model_params,np.asarray(Real.sim_p).reshape(-1))
    assert not np.array_equal(hardware.arm_surface[:4],before[:4])
    hardware.arm_surface[0]=real_before[0]+1.
    np.testing.assert_array_equal(Sim.smc_surface,before)
    np.testing.assert_array_equal(Real.smc_surface,real_before)


def test_hardware_smc_finite_bounded_outputs_across_joint_ranges(monkeypatch):
    monkeypatch.setattr('ament_index_python.get_package_share_directory',
                        lambda _: str(Path(__file__).parents[1]/'resource'))
    controller=SlidingModeController(Mock(),4,'robot_real_')
    rng=np.random.default_rng(20261112)
    for _ in range(300):
        q=np.r_[rng.uniform([1.,.01,.01,.01],[5.5,3.4,3.4,5.7]),.005]
        velocity=np.r_[rng.uniform(-.4,.4,4),0.]
        target=q+np.r_[rng.uniform(-.2,.2,4),0.]
        tau=controller.arm_controller(q,velocity,target,np.zeros(5),np.zeros(5),.005)
        assert np.isfinite(tau).all()
        assert np.all(tau >= controller.arm_u_min) and np.all(tau <= controller.arm_u_max)
    with pytest.raises(ValueError,match='finite'):
        controller.arm_controller(q,np.full(5,np.nan),target,np.zeros(5),np.zeros(5),.005)


def test_arm_adapter_enforces_limits_even_if_export_returns_unbounded_output(controller):
    controller.arm_function=Mock(return_value=[np.full(5,1e6)])
    result=controller.arm_controller(np.zeros(5),np.zeros(5),np.zeros(5),np.zeros(5),np.zeros(5),.005)
    np.testing.assert_array_equal(result,controller.arm_u_max)



def test_hardware_gripper_smc_reduces_noise_without_changing_arm(monkeypatch):
    root = Path(__file__).parents[1] / 'resource'
    monkeypatch.setattr('ament_index_python.get_package_share_directory', lambda _: str(root))
    tuned = SlidingModeController(Mock(), 4, 'robot_real_')
    old = SlidingModeController(Mock(), 4, 'robot_real_')
    old.arm_damping[4] = 4.
    old.arm_switching[4] = .005
    old.arm_boundary[4] = .002
    for position in (.0004, .0137):
        old_efforts, tuned_efforts = [], []
        for velocity in np.linspace(-.0004, .0004, 41):
            q = np.array([3.1, .7, .4, 2.1, position])
            v = np.array([.01, -.01, .01, -.01, velocity])
            ref = q + np.array([.02, -.02, .02, -.02, 0.])
            a = old.arm_controller(q, v, ref, np.zeros(5), np.zeros(5), .005)
            b = tuned.arm_controller(q, v, ref, np.zeros(5), np.zeros(5), .005)
            np.testing.assert_array_equal(a[:4], b[:4])
            # The fifth output must still come directly from the SMC export.
            exported = tuned.arm_function(q, v, ref, np.zeros(5), np.zeros(5),
                tuned.arm_surface, tuned.arm_damping, tuned.arm_switching,
                tuned.arm_boundary, tuned.arm_u_min, tuned.arm_u_max,
                tuned.arm_model_params, tuned.gripper_parameters)[0]
            assert b[4] == pytest.approx(float(exported[4]))
            old_efforts.append(a[4]); tuned_efforts.append(b[4])
        assert np.linalg.norm(tuned_efforts) < .15 * np.linalg.norm(old_efforts)
        # A real position error still produces correction in either direction.
        for error in (-.001, .001):
            ref = q.copy(); ref[4] += error
            effort = tuned.arm_controller(q, np.zeros(5), ref, np.zeros(5), np.zeros(5), .005)
            assert effort[4] * error > 0
