"""Execution permission is distinct from readiness and simulator release."""
import time
import pytest
from types import SimpleNamespace
from unittest.mock import Mock
from simlab.execution_gate import ExecutionGate


def gate(paused=True):
    g = ExecutionGate.__new__(ExecutionGate)
    g.startup_paused, g.releasing, g.authorized = paused, False, False
    g.controllers_ready, g.checked_at = True, time.monotonic()
    g.robot = SimpleNamespace(_last_state_received=time.monotonic(), sim_reset_hold=False,
        get_state=lambda: {'status': 'active'}, release_simulation=Mock(),
        node=SimpleNamespace(get_logger=lambda: Mock()))
    return g


def test_ready_does_not_enable_output():
    g = gate(False)
    assert g.ready() and not g.allows_output()


def test_start_requires_release_ack():
    g = gate()
    assert g.authorize() and not g.allows_output()
    g.robot.release_simulation.call_args.kwargs['on_success']()
    assert g.allows_output()


def test_stop_during_release_cannot_reauthorize():
    g = gate()
    g.authorize(); g.stop()
    g.robot.release_simulation.call_args.kwargs['on_success']()
    assert not g.allows_output() and g.startup_paused


def test_failed_release_remains_paused():
    g = gate()
    g.authorize()
    g.robot.release_simulation.call_args.kwargs['on_failure']()
    assert g.startup_paused and not g.authorized


def test_stale_feedback_rejects_execution():
    g = gate(False)
    g.robot._last_state_received -= 1
    assert not g.authorize() and not g.allows_output()
    g.robot.release_simulation.assert_not_called()


def test_hardware_path_never_calls_sim_release():
    g = gate(False)
    assert g.authorize() and g.allows_output()
    g.stop()
    assert not g.allows_output()
    g.robot.release_simulation.assert_not_called()


def test_controller_liveness_is_required_after_authorization():
    g = gate(False)
    assert g.authorize()
    g.checked_at -= 3
    assert not g.allows_output()


def test_failed_planner_result_never_releases_startup():
    from simlab.robot import Robot
    g = gate()
    robot = SimpleNamespace(scene_switch_hold=False, sim_reset_hold=False,
        _replay_is_active=lambda: False, _accept_planner_results=True,
        _preserve_active_plan_on_failure=False, planner=SimpleNamespace(planned_result=None),
        node=SimpleNamespace(get_logger=lambda: Mock()), prefix='robot_1_',
        execution_gate=g, authorize_motion=Mock(side_effect=g.authorize))
    Robot._on_planner_action_result(robot, {'is_success': False, 'message': 'No path'})
    robot.authorize_motion.assert_not_called()
    assert g.startup_paused and not g.authorized


def test_moved_start_rejects_path_but_keeps_authorized_hold():
    from simlab.robot import Robot
    from geometry_msgs.msg import Pose
    p = Pose(); p.position.x = .2
    g = gate(False); g.authorized = True
    robot = SimpleNamespace(scene_switch_hold=False, sim_reset_hold=False,
        _replay_is_active=lambda: False, _accept_planner_results=True,
        _preserve_active_plan_on_failure=False, start_xyz=[0., 0., 0.],
        planner=SimpleNamespace(planned_result=None), pose_command=[1., 2., 3., 0., 0., 0.],
        _pose_from_state_in_frame=lambda _: p, world_frame='world', execution_gate=g,
        abrupt_planner_stop=Mock(side_effect=lambda **_: g.stop()),
        hold_current_state_with_feedback=Mock(), authorize_motion=Mock(side_effect=g.authorize),
        node=SimpleNamespace(get_logger=lambda: Mock()))
    Robot._on_planner_action_result(robot, {'is_success': True})
    robot.abrupt_planner_stop.assert_not_called()
    robot.hold_current_state_with_feedback.assert_not_called()
    robot.authorize_motion.assert_not_called()
    assert robot.pose_command == [1., 2., 3., 0., 0., 0.]
    assert not robot._accept_planner_results
    assert g.authorized and g.allows_output()


def test_interactive_reset_returns_to_startup_only_after_acknowledged_feedback():
    from simlab.robot import Robot
    g = gate(False); g.authorized = True
    robot = SimpleNamespace(execution_gate=g, sim_reset_hold=True,
                            reset_simulation_with_state=Mock())
    Robot.reset_simulation(robot)
    call = robot.reset_simulation_with_state.call_args
    assert call.args[0].hold_commands and call.args[0].pause_dynamics
    assert robot.sim_reset_hold and not g.startup_paused
    call.kwargs['on_success']()
    assert not robot.sim_reset_hold and g.startup_paused and not g.authorized
    g.robot.release_simulation.assert_not_called()


def test_controller_liveness_timeout_pauses_without_claiming_deactivation():
    g = gate(False); g.authorized = True; g.checked_at -= 1.
    g.query = None; g.last_status = None; g.publisher = Mock()
    g.client = Mock(); g.client.service_is_ready.return_value = False
    g.robot.prefix = 'robot_1_'
    g.robot.stop_on_control_fault = Mock(side_effect=lambda _: g.stop())
    g.poll()
    assert g.authorized and g.allows_output()
    g.robot.stop_on_control_fault.assert_not_called()
    g.checked_at -= 2.
    g.poll()
    assert not g.authorized and not g.allows_output()
    g.robot.stop_on_control_fault.assert_called_once_with('Controller liveness confirmation timed out')


def test_confirmed_controller_deactivation_cancels_execution():
    from concurrent.futures import Future
    g = gate(False); g.authorized = True
    g.query = None; g.last_status = None; g.publisher = Mock()
    future = Future()
    g.client = Mock(); g.client.service_is_ready.return_value = True
    g.client.call_async.return_value = future
    g.robot.prefix = 'robot_1_'
    g.robot.stop_on_control_fault = Mock(side_effect=lambda _: g.stop())
    g.poll()
    future.set_result(SimpleNamespace(controller=[]))
    g.robot.stop_on_control_fault.assert_called_once()
    assert not g.authorized and not g.allows_output()


def settling_robot(monkeypatch):
    from simlab.robot import Robot, ControlMode
    from geometry_msgs.msg import Pose
    clock = [0.0]
    monkeypatch.setattr('simlab.robot.time.monotonic', lambda: clock[0])
    robot = Robot.__new__(Robot)
    robot.node = Mock()
    robot.execution_gate = SimpleNamespace(authorized=True, ready=lambda: True)
    robot._navigation_epoch = 1
    robot.sim_reset_hold = False
    robot.task_based_controller = False
    robot.control_mode = ControlMode.PLANNER
    robot.planner_name = 'Bitstar'
    robot._replay_is_active = lambda: False
    state = {'body_vel': [0.1, 0, 0, 0, 0, 0]}
    robot.get_state = lambda: state
    robot._send_vehicle_plan_from_current_state = Mock(return_value=True)
    assert robot._wait_for_planning_settle(Pose(), 1., .574, 0.)
    callback = robot.node.create_timer.call_args.args[1]
    return robot, state, clock, callback


def test_planning_waits_for_continuous_fresh_low_speed(monkeypatch):
    robot, state, clock, tick = settling_robot(monkeypatch)
    tick()
    robot._send_vehicle_plan_from_current_state.assert_not_called()
    state['body_vel'] = [0.] * 6
    clock[0] = 1.; tick()
    clock[0] = 1.3; tick()
    robot._send_vehicle_plan_from_current_state.assert_not_called()
    robot.execution_gate.ready = lambda: False
    clock[0] = 1.4; tick()
    robot.execution_gate.ready = lambda: True
    clock[0] = 2.; tick()
    clock[0] = 2.5; tick()
    robot._send_vehicle_plan_from_current_state.assert_called_once()
    assert robot._planning_settle_timer is None


def test_planning_settle_timeout_retains_hold(monkeypatch):
    robot, state, clock, tick = settling_robot(monkeypatch)
    clock[0] = 10.; tick()
    robot._send_vehicle_plan_from_current_state.assert_not_called()
    assert robot.execution_gate.authorized
    assert robot._planning_settle_timer is None


def test_stop_or_superseding_request_cannot_dispatch_old_plan(monkeypatch):
    robot, state, clock, tick = settling_robot(monkeypatch)
    state['body_vel'] = [0.] * 6
    tick()
    robot._cancel_planning_settle()
    robot._navigation_epoch += 1
    clock[0] = 1.; tick()
    robot._send_vehicle_plan_from_current_state.assert_not_called()


def test_task_space_switch_cancels_pending_plan(monkeypatch):
    robot, state, clock, tick = settling_robot(monkeypatch)
    robot.task_based_controller = True
    tick()
    assert robot._planning_settle_timer is None
    robot._send_vehicle_plan_from_current_state.assert_not_called()


def test_dispatch_samples_pose_at_send_time():
    from simlab.robot import Robot
    from geometry_msgs.msg import Pose
    robot = Robot.__new__(Robot)
    robot.node = Mock(); robot.prefix = 'robot_1_'; robot.world_frame = 'world'
    robot.planner_name = 'Bitstar'; robot.planner_action_client = Mock()
    robot.planner_action_client.send_goal.return_value = True
    current = Pose(); current.position.x = 2.; current.orientation.w = 1.
    robot._pose_from_state_in_frame = lambda _: current
    robot._pose_to_xyz_quat_wxyz = lambda p: ([p.position.x, 0., 0.], [1., 0., 0., 0.])
    robot._get_vehicle_goal_from_marker = lambda p: ([3., 0., 0.], [1., 0., 0., 0.])
    robot._log_plan_context = Mock(); robot._save_vehicle_goal_from_target = Mock()
    assert robot._send_vehicle_plan_from_current_state(Pose(), 1., .574, 0.)
    assert robot.planner_action_client.send_goal.call_args.kwargs['start_xyz'] == [2., 0., 0.]


def test_startup_and_nonpreempting_plans_skip_settling():
    from simlab.robot import Robot
    from geometry_msgs.msg import Pose
    for paused, preempt in [(True, True), (False, False)]:
        robot = Robot.__new__(Robot)
        robot.node = Mock(); robot.prefix = 'robot_1_'; robot.world_frame = 'world'
        robot.planner_name = 'Bitstar'; robot.sim_reset_hold = False
        robot._replay_is_active = lambda: False
        robot._pose_from_state_in_frame = lambda _: Pose()
        robot.execution_gate = SimpleNamespace(startup_paused=paused, authorized=False, ready=lambda: True)
        robot.abrupt_planner_stop = Mock(); robot.hold_current_state_with_feedback = Mock()
        robot.authorize_motion = Mock()
        robot._wait_for_planning_settle = Mock()
        robot._send_vehicle_plan_from_current_state = Mock(return_value=True)
        assert robot.plan_vehicle_trajectory_action(Pose(), preempt_current=preempt)
        robot._wait_for_planning_settle.assert_not_called()
        robot._send_vehicle_plan_from_current_state.assert_called_once()


def motion_calculation_robot():
    import numpy as np
    from simlab.robot import Robot, ControlMode
    robot = Robot.__new__(Robot)
    robot.node = Mock()
    robot.control_mode = ControlMode.PLANNER
    robot.task_based_controller = False
    robot.ik_tool_axis = np.array([0., 0., 1.])
    robot.sim_reset_hold = robot.scene_switch_hold = False
    robot._replay_is_active = lambda: False
    robot._navigation_epoch = 0
    process, receiver = Mock(), Mock()
    process.is_alive.return_value = True
    receiver.poll.return_value = True
    receiver.recv.return_value = (True, 'calculated result')
    robot._motion_worker = (process, receiver)
    return robot


def test_motion_result_is_applied_only_by_executor_poll():
    robot = motion_calculation_robot()
    apply = Mock()
    robot._submit_motion_calculation(abs, (-1,), apply)
    apply.assert_not_called()
    poll = robot.node.create_timer.call_args.args[1]
    poll()
    apply.assert_called_once_with('calculated result')
    assert robot._motion_calculation is None
    # Successful jobs reuse the worker; shutdown still releases it.
    robot._motion_worker[0].terminate.assert_not_called()
    robot._cancel_motion_calculation()


def test_changed_navigation_epoch_discards_completed_motion_result():
    robot = motion_calculation_robot()
    apply = Mock()
    robot._submit_motion_calculation(abs, (-1,), apply)
    robot._navigation_epoch += 1
    robot.node.create_timer.call_args.args[1]()
    apply.assert_not_called()
    assert robot._motion_calculation is None and robot._motion_worker is None


def test_scene_hold_discards_completed_motion_result():
    robot = motion_calculation_robot()
    apply = Mock()
    robot._submit_motion_calculation(abs, (-1,), apply)
    robot.scene_switch_hold = True
    robot.node.create_timer.call_args.args[1]()
    apply.assert_not_called()
    assert robot._motion_worker is None


def test_queued_old_poll_cannot_apply_or_cancel_a_new_request():
    robot = motion_calculation_robot()
    first, second = Mock(), Mock()
    robot.node.create_timer.side_effect = [Mock(), Mock()]
    robot._submit_motion_calculation(abs, (-1,), first)
    old_poll = robot.node.create_timer.call_args.args[1]
    # Simulate completing the first request before dispatching its queued callback.
    robot._cancel_motion_calculation(keep_worker=True)
    robot._submit_motion_calculation(abs, (-2,), second)
    old_poll()
    first.assert_not_called()
    second.assert_not_called()
    assert robot._motion_calculation is not None
    robot.node.create_timer.call_args.args[1]()
    second.assert_called_once()
    robot._cancel_motion_calculation()


def test_arm_action_does_not_release_or_command_paused_vehicle():
    g = gate()
    g.simulated_components = frozenset({'vehicle', 'manipulator'})
    assert g.authorize({'manipulator'})
    call = g.robot.release_simulation.call_args.kwargs
    assert call['components'] == {'manipulator'}
    call['on_success']()
    assert g.startup_paused
    assert g.allows_output('manipulator') and not g.allows_output('vehicle')
    assert g.authorize({'vehicle'})
    call = g.robot.release_simulation.call_args.kwargs
    assert call['components'] == {'vehicle'}
    call['on_success']()
    assert g.allows_output('vehicle') and g.allows_output('manipulator')


def test_physical_arm_authorization_leaves_sim_vehicle_paused():
    g = gate()
    g.simulated_components = frozenset({'vehicle'})
    g.startup_paused = True
    assert g.authorize({'manipulator'})
    g.robot.release_simulation.assert_not_called()
    assert g.allows_output('manipulator') and not g.allows_output('vehicle')
    assert g.paused_components == {'vehicle'}


def test_inflight_release_cannot_silently_authorize_other_component():
    g = gate()
    assert g.authorize({'manipulator'})
    assert not g.authorize({'vehicle'})
    g.robot.release_simulation.call_args.kwargs['on_success']()
    assert not g.allows_output('vehicle')
    assert g.authorize({'vehicle'})
    g.stop()
    g.robot.release_simulation.call_args.kwargs['on_success']()
    assert not g.allows_output()


def test_release_routes_only_simulated_components():
    from simlab.robot import Robot
    g = gate()
    g.simulated_components = frozenset({'vehicle'})
    g.startup_paused = True
    r = Robot.__new__(Robot)
    r.sim_reset_hold = False; r.execution_gate = g; r.prefix = 'robot_real_'
    r.release_sim_component_clients = {'vehicle': object()}
    r._call_uvms_service = Mock()
    completed = Mock()
    r.release_simulation(components={'manipulator'}, on_success=completed)
    r._call_uvms_service.assert_not_called()
    completed.assert_called_once()
    assert g.startup_paused
    r.release_simulation(components={'vehicle'})
    assert r._call_uvms_service.call_args.kwargs['service_name'] == '/robot_real_release_sim_vehicle'
    r._call_uvms_service.call_args.kwargs['on_success']()
    assert not g.startup_paused


def test_arm_only_settle_does_not_wait_for_vehicle_target():
    import numpy as np
    from simlab.robot import Robot
    r = Robot.__new__(Robot)
    r._replay_settle_config = dict(position_tolerance=.01, velocity_tolerance=.01,
        vehicle_position_tolerance=.01, vehicle_velocity_tolerance=.01)
    r._replay_settle_arm_target = np.zeros(5)
    r._replay_settle_vehicle_target = np.ones(6) * 100
    r._replay_settle_controller = SimpleNamespace(
        vehicle_subsystem_mode=lambda: 'zero_command',
        manipulator_subsystem_mode=lambda: 'track_reference')
    state = dict(q=[0.]*4, dq=[0.]*4, grasper_q=[0.], grasper_qdot=[0.],
                 pose=[0.]*6, body_vel=[0.]*6)
    assert r._replay_settle_is_done(state)[0]
    state['q'][0] = .1
    assert not r._replay_settle_is_done(state)[0]


def test_component_permission_masks_actual_published_commands():
    from builtin_interfaces.msg import Time
    from simlab.robot import Robot
    g = gate()
    g.simulated_components = frozenset({'vehicle'})
    g.startup_paused = True
    g.authorize({'manipulator'})
    r = Robot.__new__(Robot)
    r.execution_gate = g; r.prefix = 'robot_real_'; r.node = Mock()
    r.node.get_clock.return_value.now.return_value.to_msg.return_value = Time()
    r.vehicle_effort_command_publisher = Mock()
    r.manipulator_effort_command_publisher = Mock()
    r._publish_controller_performance_metrics = Mock()
    r.publish_vehicle_and_arm([10.]*6, [.1]*5)
    vehicle = r.vehicle_effort_command_publisher.publish.call_args.args[0]
    arm = r.manipulator_effort_command_publisher.publish.call_args.args[0]
    assert list(vehicle.interface_values[0].values) == [0.]*6
    assert list(arm.data) == [.1]*5
    g.stop()
    r.publish_vehicle_and_arm([10.]*6, [.1]*5)
    assert list(r.manipulator_effort_command_publisher.publish.call_args.args[0].data) == [0.]*5


def test_arm_teleop_does_not_authorize_vehicle_and_can_later_request_it():
    from simlab.robot import Robot, ControlMode
    g = gate()
    g.simulated_components = frozenset({'vehicle'})
    g.startup_paused = True
    r = Robot.__new__(Robot)
    r.execution_gate = g; r.sim_reset_hold = False
    r.get_state = lambda: {'status':'active'}
    r.control_mode = ControlMode.TELEOP
    r.teleop_wrench_body_6 = [0.]*6
    r.teleop_arm_effort_5 = [.1]*5
    r.authorize_motion = lambda **kw:g.authorize(kw['components'])
    r.publish_commands = Mock()
    r._control_tick = time.monotonic() - .01
    r.control_loop_callback()
    assert g.active_components == {'manipulator'}
    g.robot.release_simulation.assert_not_called()
    r.teleop_wrench_body_6[0] = 1.
    r._control_tick = time.monotonic() - .01
    r.control_loop_callback()
    assert g.robot.release_simulation.call_args.kwargs['components'] == {'vehicle'}


def test_controller_fault_pauses_only_simulated_components():
    from simlab.robot import Robot
    g = gate(False)
    g.simulated_components = frozenset({'vehicle'})
    r = SimpleNamespace(execution_gate=g, prefix='robot_real_',
                        pause_sim_component_clients={'vehicle': Mock()},
                        _call_uvms_service=Mock())
    Robot.pause_simulation_on_fault(r)
    assert not g.authorized and r.sim_reset_hold
    assert g.paused_components == {'vehicle'}
    assert r._call_uvms_service.call_count == 1
    assert r._call_uvms_service.call_args.kwargs['service_name'] == '/robot_real_pause_sim_vehicle'
    r._call_uvms_service.call_args.kwargs['on_success']()
    assert not r.sim_reset_hold
    assert g.startup_paused and not g.authorized


def test_controller_clock_is_independent_of_zero_reset_period(monkeypatch):
    from simlab.robot import Robot, ControlMode
    r = Robot.__new__(Robot)
    r._control_tick = 10.
    r.node = Mock()
    r.sim_reset_hold = False
    r.control_mode = ControlMode.ARM_HOLD
    state = dict(status='active', dt=0.)
    r.get_state = lambda: state
    r.publish_arm_hold = Mock()
    monkeypatch.setattr(time, 'monotonic', lambda: 10.01)
    r.control_loop_callback()
    import pytest
    assert r.publish_arm_hold.call_args.args[0]['dt'] == pytest.approx(.01)
    assert state['dt'] == 0.  # Do not falsify plant telemetry.


def transition_robot():
    from simlab.robot import Robot
    r = Robot.__new__(Robot)
    r.node = Mock(); r.prefix = 'robot_1_'; r.sim_reset_hold = False
    r._last_state_received = time.monotonic()
    r.get_state = lambda: {'status': 'active'}
    r.execution_gate = gate()
    r.execution_gate.simulated_components = frozenset({'vehicle', 'manipulator'})
    r.execution_gate.robot = r
    r.release_sim_component_clients = {part: Mock() for part in ('vehicle', 'manipulator')}
    r.pause_sim_component_clients = {part: Mock() for part in ('vehicle', 'manipulator')}
    r._call_uvms_service = Mock()
    return r


def test_partial_release_failure_rolls_back_both_components():
    r = transition_robot(); g = r.execution_gate
    assert g.authorize()
    first = r._call_uvms_service.call_args.kwargs
    assert first['service_name'].endswith('release_sim_manipulator')
    first['on_success']()
    second = r._call_uvms_service.call_args.kwargs
    assert second['service_name'].endswith('release_sim_vehicle')
    second['on_failure']()
    pause = r._call_uvms_service.call_args_list[-2:]
    assert {call.kwargs['service_name'] for call in pause} == {
        '/robot_1_pause_sim_manipulator', '/robot_1_pause_sim_vehicle'}
    assert r.sim_reset_hold and not g.authorized
    pause[0].kwargs['on_success']()
    assert r.sim_reset_hold
    pause[1].kwargs['on_success']()
    assert not r.sim_reset_hold and g.startup_paused and not g.allows_output()


def test_stop_during_release_compensates_late_ack_without_releasing_next():
    r = transition_robot(); g = r.execution_gate
    assert g.authorize()
    release = r._call_uvms_service.call_args.kwargs
    g.stop()
    old_pause = r._call_uvms_service.call_args_list[-2:]
    release['on_success']()  # Service still executed despite local cancellation.
    current_pause = r._call_uvms_service.call_args_list[-2:]
    for call in old_pause:
        call.kwargs['on_success']()
    assert r.sim_reset_hold  # Old acknowledgements cannot clear the new barrier.
    assert not any(call.kwargs['service_name'].endswith('release_sim_vehicle')
                   for call in r._call_uvms_service.call_args_list)
    for call in current_pause:
        call.kwargs['on_success']()
    assert not g.authorized and g.startup_paused and not r.sim_reset_hold


def test_repeated_fault_pause_is_idempotent_and_preserves_reset_hold():
    r = transition_robot()
    r.pause_simulation_on_fault()
    r.pause_simulation_on_fault()
    assert r._call_uvms_service.call_count == 2
    r._reset_feedback_pending = object()
    for call in r._call_uvms_service.call_args_list:
        call.kwargs['on_success']()
    assert r.sim_reset_hold
    assert not r._simulation_pause_pending


def test_release_response_timeout_retains_late_success_compensation(monkeypatch):
    from concurrent.futures import Future
    from simlab.robot import Robot
    now = [0.]
    monkeypatch.setattr(time, 'monotonic', lambda: now[0])
    r = SimpleNamespace(node=Mock())
    client = Mock(); client.service_is_ready.return_value = True
    future = Future(); client.call_async.return_value = future
    success, failure, late = Mock(), Mock(), Mock()
    Robot._call_uvms_service(r, client, '/release', 'release', success, failure, late)
    expiry = r.node.create_timer.call_args.args[1]
    now[0] = 2.01; expiry(); expiry()
    failure.assert_called_once(); success.assert_not_called()
    assert not future.cancelled()
    future.set_result(SimpleNamespace(success=True, message='released'))
    late.assert_called_once(); success.assert_not_called()


def test_release_unavailable_does_not_block_executor():
    from simlab.robot import Robot
    r = SimpleNamespace(node=Mock())
    client = Mock(); client.service_is_ready.return_value = False
    failure = Mock()
    Robot._call_uvms_service(r, client, '/release', 'release', on_failure=failure)
    failure.assert_called_once()
    client.wait_for_service.assert_not_called()
    client.call_async.assert_not_called()


def test_ungated_fault_pause_requires_explicit_release():
    from simlab.robot import Robot
    r = SimpleNamespace(simulated_components={'vehicle'}, prefix='robot_1_',
                        pause_sim_component_clients={'vehicle': Mock()},
                        _call_uvms_service=Mock())
    Robot.pause_simulation_on_fault(r)
    r._call_uvms_service.call_args.kwargs['on_success']()
    assert r.sim_reset_hold and not r._simulation_pause_pending


def test_late_service_response_cannot_beat_queued_timeout(monkeypatch):
    from concurrent.futures import Future
    from simlab.robot import Robot
    now = [0.]
    monkeypatch.setattr(time, 'monotonic', lambda: now[0])
    r = SimpleNamespace(node=Mock())
    client = Mock(); client.service_is_ready.return_value = True
    future = Future(); client.call_async.return_value = future
    success, failure, late = Mock(), Mock(), Mock()
    Robot._call_uvms_service(r, client, '/release', 'release', success, failure, late)
    now[0] = 3.
    future.set_result(SimpleNamespace(success=True, message='released'))
    failure.assert_called_once(); late.assert_called_once(); success.assert_not_called()


def test_gate_initialization_and_status_deduplication():
    robot = SimpleNamespace(node=Mock(), prefix='robot_1_',
        sim_reset_hold=False, _last_state_received=time.monotonic(),
        get_state=lambda: {'status': 'active'})
    g = ExecutionGate(robot, startup_paused=True, simulated_components={'vehicle'})
    g.client.service_is_ready.return_value = False
    assert g.paused_components == {'vehicle'} and not g.authorized
    g.poll(); g.poll()
    assert g.publisher.publish.call_count == 1
    assert g.last_status == 'Waiting for feedback/controllers'
    g.controllers_ready = True; g.checked_at = time.monotonic()
    g.poll()
    assert g.last_status == 'Ready — simulation paused'
    g.paused_components.clear(); g.poll()
    assert g.last_status == 'Ready — idle'
    g.authorized = True; g.poll()
    assert g.last_status == 'Control active'
    g.releasing = True; g.poll()
    assert g.last_status == 'Starting execution'


@pytest.mark.parametrize('components', [set(), {'invalid'}, {'vehicle', 'invalid'}])
def test_invalid_component_requests_are_passive(components):
    g = gate()
    assert not g.authorize(components)
    assert not g.authorized
    g.robot.release_simulation.assert_not_called()


@pytest.mark.parametrize('bad_field,bad_value', [
    ('_last_state_received', None), ('_last_state_received', 0.),
    ('_feedback_fault', True), ('_hardware_estimator_valid', False),
    ('sim_reset_hold', True),
    ('_simulation_pause_pending', True)])
def test_readiness_failures_never_release(bad_field, bad_value):
    g = gate(); setattr(g.robot, bad_field, bad_value)
    assert not g.ready() and not g.authorize()
    g.robot.release_simulation.assert_not_called()


def test_stale_release_failure_cannot_cancel_new_authorization():
    g = gate(); assert g.authorize()
    old_failure = g.robot.release_simulation.call_args.kwargs['on_failure']
    g.stop(); assert g.authorize()
    old_failure()
    assert g.authorized and g.releasing


@pytest.mark.parametrize('outcome', ['active', 'exception', 'cancel_timeout', 'submit_error'])
@pytest.mark.parametrize('authorized', [False, True])
def test_controller_readiness_service_outcomes(outcome, authorized, monkeypatch):
    from concurrent.futures import Future
    now = [time.monotonic()]
    monkeypatch.setattr(time, 'monotonic', lambda: now[0])
    g = gate(False); g.authorized = authorized
    g.query = None; g.last_status = None; g.publisher = Mock()
    g.client = Mock(); g.client.service_is_ready.return_value = True
    future = Future(); next_future = Future()
    g.client.call_async.side_effect = [future, next_future]
    g.robot.prefix = 'robot_1_'
    g.robot.stop_on_control_fault = Mock(side_effect=lambda _: g.stop())
    if outcome == 'submit_error':
        g.client.call_async.side_effect = RuntimeError('disconnected')
    g.poll()
    if outcome == 'active':
        future.set_result(SimpleNamespace(controller=[
            SimpleNamespace(name='vehicle_effort_controller_robot_1_', state='active'),
            SimpleNamespace(name='manipulation_effort_controller_robot_1_', state='active')]))
        assert g.controllers_ready
        g.robot.stop_on_control_fault.assert_not_called()
    elif outcome == 'exception':
        future.set_exception(RuntimeError('disconnected'))
        assert not g.controllers_ready
        assert g.robot.stop_on_control_fault.call_count == int(authorized)
    elif outcome == 'cancel_timeout':
        now[0] += 2.1; g.poll()
        assert future.cancelled() and g.query is next_future
        assert g.robot.stop_on_control_fault.call_count == int(authorized)
    else:
        assert g.query is None and not g.controllers_ready
        assert g.robot.stop_on_control_fault.call_count == int(authorized)


@pytest.mark.parametrize('blocker', ['_release_transaction', '_simulation_pause_pending', '_reset_feedback_pending'])
@pytest.mark.parametrize('callback', [False, True])
def test_overlapping_transition_cannot_submit_release(blocker, callback):
    r = transition_robot(); setattr(r, blocker, object())
    fail = Mock() if callback else None
    r.release_simulation(on_failure=fail)
    r._call_uvms_service.assert_not_called()
    if fail is not None:
        fail.assert_called_once()


def test_cancelled_release_failure_does_not_repeat_pause():
    r = transition_robot(); r.release_simulation()
    failure = r._call_uvms_service.call_args.kwargs['on_failure']
    r.cancel_simulation_release()
    count = r._call_uvms_service.call_count
    failure()
    assert r._call_uvms_service.call_count == count
    assert r._simulation_pause_pending


def test_release_failure_without_user_callback_still_pauses():
    r = transition_robot(); r.release_simulation()
    r._call_uvms_service.call_args.kwargs['on_failure']()
    assert r._simulation_pause_pending and not r.execution_gate.authorized


def test_ungated_release_completes_and_all_real_pause_is_passive():
    from simlab.robot import Robot
    r = Robot.__new__(Robot); r.node = Mock(); r.prefix = 'robot_1_'
    r.simulated_components = {'vehicle'}; r.sim_reset_hold = True
    r.release_sim_component_clients = {'vehicle': Mock()}; r._call_uvms_service = Mock()
    success = Mock(); r.release_simulation(on_success=success)
    r._call_uvms_service.call_args.kwargs['on_success']()
    assert not r.sim_reset_hold
    success.assert_called_once()
    r.simulated_components = set(); r._call_uvms_service.reset_mock()
    r.pause_simulation_on_fault()
    r._call_uvms_service.assert_not_called()


@pytest.mark.parametrize('outcome', ['success', 'rejected', 'empty', 'future_error', 'submit_error', 'unavailable', 'timeout'])
@pytest.mark.parametrize('callbacks', [False, True])
def test_service_outcome_matrix(outcome, callbacks, monkeypatch):
    from concurrent.futures import Future
    from simlab.robot import Robot
    now = [0.]; monkeypatch.setattr(time, 'monotonic', lambda: now[0])
    r = SimpleNamespace(node=Mock()); client = Mock()
    client.service_is_ready.return_value = outcome != 'unavailable'
    future = Future(); client.call_async.return_value = future
    if outcome == 'submit_error':
        client.call_async.side_effect = RuntimeError('disconnected')
    success = Mock() if callbacks else None
    failure = Mock() if callbacks else None
    Robot._call_uvms_service(r, client, '/service', 'test service', success, failure)
    if outcome in ('unavailable', 'submit_error'):
        assert not r.node.create_timer.called
    else:
        timer = r.node.create_timer.return_value
        expiry = r.node.create_timer.call_args.args[1]
        expiry()  # An early watchdog tick cannot reject a pending request.
        if failure is not None:
            failure.assert_not_called()
        if outcome == 'timeout':
            now[0] = 2.; expiry()
            future.set_result(SimpleNamespace(success=False, message='late rejected'))
        elif outcome == 'future_error':
            future.set_exception(RuntimeError('disconnected'))
        else:
            future.set_result(None if outcome == 'empty' else SimpleNamespace(
                success=outcome == 'success', message=outcome))
        timer.cancel.assert_called_once()
        r.node.destroy_timer.assert_called_once_with(timer)
    if callbacks:
        assert success.call_count == int(outcome == 'success')
        assert failure.call_count == int(outcome != 'success')


@pytest.mark.parametrize('authorized', [False, True])
def test_late_controller_status_response_cannot_refresh_liveness(authorized, monkeypatch):
    from concurrent.futures import Future
    now = [0.]; monkeypatch.setattr(time, 'monotonic', lambda: now[0])
    g = gate(False); g.authorized = authorized
    g.query = None; g.last_status = None; g.publisher = Mock()
    g.client = Mock(); g.client.service_is_ready.return_value = True
    future = Future(); g.client.call_async.return_value = future
    g.robot.prefix = 'robot_1_'
    g.robot.stop_on_control_fault = Mock(side_effect=lambda _: g.stop())
    g.poll(); now[0] = 2.1
    future.set_result(SimpleNamespace(controller=[
        SimpleNamespace(name='vehicle_effort_controller_robot_1_', state='active'),
        SimpleNamespace(name='manipulation_effort_controller_robot_1_', state='active')]))
    assert not g.controllers_ready and g.checked_at == 0.
    assert g.robot.stop_on_control_fault.call_count == int(authorized)


def test_fault_cleanup_error_revokes_gate_and_still_pauses():
    from simlab.robot import Robot
    r = transition_robot(); r.execution_gate.authorized = True
    r.stop_execution = Mock(side_effect=RuntimeError('broken plugin cleanup'))
    r.publish_vehicle_and_arm = Mock()
    Robot.stop_on_control_fault(r, 'controller failed')
    assert not r.execution_gate.authorized
    r.publish_vehicle_and_arm.assert_called_once_with([0.] * 6, [0.] * 5)
    assert r._simulation_pause_pending
    assert r._call_uvms_service.call_count == 2


def test_native_pause_dispatch_error_is_reported_without_escaping(monkeypatch):
    from simlab.robot import Robot
    r = SimpleNamespace(node=Mock(), stop_execution=Mock())
    monkeypatch.setattr(Robot, 'pause_simulation_on_fault', Mock(side_effect=RuntimeError('transport failed')))
    Robot.stop_on_control_fault(r, 'controller failed')
    r.stop_execution.assert_called_once()
    errors = [call.args[0] for call in r.node.get_logger().error.call_args_list]
    assert any('Simulation fault pause failed: transport failed' in message for message in errors)
    assert any('controller failed' in message for message in errors)


@pytest.mark.parametrize('field', ['wrench', 'integral'])
@pytest.mark.parametrize('corruption', ['shape', 'nonfinite'])
def test_invalid_controller_export_cannot_commit_integral(field, corruption):
    import numpy as np
    from simlab.controllers.base import bounded_integral_control
    integral = np.ones(6) * .2; before = integral.copy()
    wrench, next_integral = np.zeros(6), np.ones(6)
    if corruption == 'shape':
        if field == 'wrench':
            wrench = np.zeros(5)
        else:
            next_integral = np.zeros(5)
    elif field == 'wrench':
        wrench[0] = np.nan
    else:
        next_integral[0] = np.inf
    with pytest.raises(ValueError, match='invalid wrench or integral state'):
        bounded_integral_control(lambda *_: (wrench, next_integral), integral,
                                 np.ones(6), np.ones(6), -np.ones(6), np.ones(6), .01)
    np.testing.assert_array_equal(integral, before)


def execution_robot(components=('vehicle', 'manipulator')):
    """Real gate and publication methods; only ROS endpoints are mocked."""
    import numpy as np
    from simlab.robot import Robot, ControlMode
    robot = Robot.__new__(Robot)
    robot.node = Mock()
    from builtin_interfaces.msg import Time
    robot.node.get_clock.return_value.now.return_value.to_msg.return_value = Time()
    robot.prefix = 'robot_1_'
    robot.world_frame = 'world'
    robot.control_mode = ControlMode.PLANNER
    robot.task_based_controller = False
    robot.ik_tool_axis = np.array([0., 0., 1.])
    robot._replay_is_active = lambda: False
    robot.sim_reset_hold = False
    robot._last_state_received = time.monotonic()
    state = {'status': 'active', 'pose': np.array([1., 2., -3., .2, -.3, .4]),
             'q': np.array([.4, .5, .6, .7])}
    robot.get_state = lambda: state
    robot.arm = SimpleNamespace()
    robot.planner = None
    robot.planner_action_client = Mock()
    robot.vehicle_cart_traj = SimpleNamespace(active=True)
    robot.active_controller_instance = Mock(return_value=None)
    robot.cancel_simulation_release = Mock()
    robot.release_simulation = Mock()
    robot.vehicle_effort_command_publisher = Mock()
    robot.manipulator_effort_command_publisher = Mock()
    robot._publish_controller_performance_metrics = Mock()
    g = gate(False)
    g.robot = robot
    robot.execution_gate = g
    if components:
        assert g.authorize(components)
    return robot, g, state


@pytest.mark.parametrize('velocity', [[float('nan'), 0., 0.], [0., 0.]])
def test_vehicle_velocity_lookup_rejects_invalid_measurements(velocity):
    robot, _, _ = execution_robot()
    robot.ned_vel = velocity
    with pytest.raises(ValueError, match='velocity is invalid'):
        robot._current_vehicle_velocity_world_nwu()


def test_vehicle_velocity_lookup_rejects_missing_transform():
    from tf2_ros import TransformException
    robot, _, _ = execution_robot()
    robot.ned_vel = [.2, .1, 0.]
    robot.map_frame = 'robot_1_map'
    robot.tf_buffer = Mock()
    robot.tf_buffer.lookup_transform.side_effect = TransformException('missing')
    with pytest.raises(ValueError, match='velocity transform'):
        robot._current_vehicle_velocity_world_nwu()


def test_generated_vehicle_trajectory_activates_without_collision_admission():
    import numpy as np
    from geometry_msgs.msg import Pose
    from simlab.motion_planning.trajectory_generators.ruckig import RuckigVehicleTrajectoryGenerator
    robot, g, _ = execution_robot(())
    robot.max_traj_vel = [.3]*3
    robot.max_traj_acc = [.1]*3
    robot.max_traj_jerk = [.5]*3
    robot.reset_vehicle_command_yaw_memory = Mock()
    robot._current_vehicle_velocity_world_nwu = lambda: np.zeros(3)
    robot._pose_from_state_in_frame = lambda _: Pose()
    previous = RuckigVehicleTrajectoryGenerator(robot.node, 3, .02, 100)
    robot.vehicle_cart_traj = previous
    robot._submit_motion_calculation = lambda fn, args, accept: accept(fn(*args))
    robot._start_vehicle_trajectory(np.zeros(3), [1., 0., 0., 0.],
                                   np.array([[0., 0., 0.], [1., 0., 0.]]))
    assert robot.vehicle_cart_traj is not previous
    assert g.allows_output('vehicle')
    robot.vehicle_cart_traj.close()




def handoff_robot(monkeypatch):
    import threading
    import simlab.robot as module
    robot, gate, state = execution_robot()
    monkeypatch.setattr(module, 'reset_dynamics_observer', Mock())
    robot.controller_lock = threading.Lock()
    robot._pending_mode_toggle = False
    robot._joystick_disconnected = False
    robot.has_joystick_interface = True
    robot._control_interval = 1. / 150.
    robot.reset_vehicle_command_yaw_memory = Mock()
    robot.publish_commands = Mock()
    robot.arm.q_command = [9.] * 4
    robot.arm.dq_command = [8.] * 4
    robot.arm.ddq_command = [7.] * 4
    robot.arm.grasp_command = 6.
    state['grasper_q'] = [.2]
    robot._zero_teleop_commands()
    robot.on_control_mode_changed = Mock()
    return robot, gate, state


def test_mode_handoff_clears_cached_efforts_before_next_control_tick(monkeypatch):
    from simlab.robot import ControlMode
    robot, gate, _ = handoff_robot(monkeypatch)
    for _ in range(3):
        robot.rov_surge = 7.
        robot.jointe = 3.
        robot.joystick_read_callback()
        robot.set_control_mode(ControlMode.TELEOP)
        robot._control_loop_impl()
        assert robot.teleop_wrench_body_6 == [0.] * 6
        assert robot.teleop_arm_effort_5 == [0.] * 5
        robot.publish_commands.assert_called_with([0.] * 6, [0.] * 5)
        assert not gate.authorized
        robot.set_control_mode(ControlMode.PLANNER)
    assert robot._control_mode_epoch == 6


def test_return_to_interactive_captures_measured_references_without_starting(monkeypatch):
    import numpy as np
    from simlab.robot import ControlMode
    robot, gate, state = handoff_robot(monkeypatch)
    robot.set_control_mode(ControlMode.TELEOP)
    robot.set_control_mode(ControlMode.PLANNER)
    np.testing.assert_allclose(robot.pose_command, state['pose'])
    np.testing.assert_allclose(robot.arm.q_command, state['q'])
    assert robot.arm.dq_command == [0.] * 4
    assert robot.arm.ddq_command == [0.] * 4
    assert robot.arm.grasp_command == .2
    assert robot._arm_feedback_hold
    assert not robot._planner_output_enabled and not gate.authorized
    robot.on_control_mode_changed.assert_called()


def test_share_is_processed_by_executor_not_joystick_thread(monkeypatch):
    from simlab.robot import PS4Controller, ControlMode
    robot, _, _ = handoff_robot(monkeypatch)
    controller = SimpleNamespace(ros_node=robot)
    PS4Controller.on_share_press(controller)
    assert robot.control_mode == ControlMode.PLANNER
    robot.joystick_read_callback()
    assert robot.control_mode == ControlMode.TELEOP
    PS4Controller.on_share_press(controller)
    robot.joystick_read_callback()
    assert robot.control_mode == ControlMode.PLANNER


@pytest.mark.parametrize('mode', ['planner', 'teleop'])
def test_disconnected_joystick_only_stops_its_active_control_mode(monkeypatch, mode):
    from simlab.robot import ControlMode
    robot, gate, _ = handoff_robot(monkeypatch)
    robot.control_mode = ControlMode(mode)
    robot._joystick_disconnected = True
    robot.stop_execution = Mock()
    robot.joystick_read_callback()
    assert robot.stop_execution.called == (mode == 'teleop')
    assert not robot.has_joystick_interface
    assert gate.authorized # no background-thread gate mutation


def test_listener_exit_defers_mode_specific_cleanup_to_executor(monkeypatch):
    import simlab.robot as module
    robot, gate, _ = handoff_robot(monkeypatch)
    robot.k_robot = 0
    captured = {}
    def thread(*, target, daemon):
        captured['listen'] = target
        return Mock()
    monkeypatch.setattr(module.threading, 'Thread', thread)
    monkeypatch.setattr(module, 'PS4Controller', Mock())
    robot.start_joystick('/unused-test-device')
    captured['listen']()
    assert robot._joystick_disconnected
    assert gate.authorized
    robot.joystick_read_callback()
    assert gate.authorized # disconnecting an idle joystick does not revoke planner


def test_new_mode_change_discards_queued_share_request(monkeypatch):
    from simlab.robot import PS4Controller, ControlMode
    robot, _, _ = handoff_robot(monkeypatch)
    PS4Controller.on_share_press(SimpleNamespace(ros_node=robot))
    robot.set_control_mode(ControlMode.TELEOP) # an explicit newer command
    robot.joystick_read_callback()
    assert robot.control_mode == ControlMode.TELEOP


def test_scene_transition_preserves_existing_output_but_blocks_new_authorization():
    g = gate(False)
    assert g.authorize()
    g.robot.scene_switch_hold = True
    assert g.ready()
    assert g.allows_output()
    assert not g.authorize()
