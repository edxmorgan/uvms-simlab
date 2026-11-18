import time
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from simlab.robot import Robot, ControlMode


def robot_fixture():
    robot = Robot.__new__(Robot)
    state = dict(status='active', q=[3., .7, .4, 2.], dq=[0.]*4,
                 grasper_q=[.001], grasper_qdot=[0.], dt=.01, sim_time=1.)
    robot.get_state = lambda: state
    robot.execution_gate = SimpleNamespace(ready=lambda: True, allows_output=lambda part: True)
    controller = SimpleNamespace(arm_controller=Mock(return_value=[.1]*5),
        vehicle_controller=Mock(), reset_controller_state=Mock())
    robot.controller_instance = lambda name: controller
    robot.arm_controller_fn = controller.arm_controller
    robot.active_controller_instance = lambda: controller
    robot.ik_joint_lower = np.array([1., .01, .01, .01])
    robot.ik_joint_upper = np.array([5.5, 3.4, 3.4, 5.7])
    robot.ik_joint_velocity = np.full(4, .5)
    for method in ('abrupt_planner_stop', 'cancel_replay_settle', '_zero_teleop_commands',
                   '_clear_menu_grasper_effort', 'publish_commands'):
        setattr(robot, method, Mock())
    robot.authorize_motion = Mock(return_value=True)
    return robot, controller, state


def replay_fixture(robot):
    q = robot._arm_hold_target.copy()
    positions = np.array([q, q + [.01, 0., 0., 0., 0.], q])
    return SimpleNamespace(
        manipulator_subsystem_mode=lambda: 'track_reference', vehicle_subsystem_mode=lambda: 'zero_command',
        repeats=1, has_valid_playback=lambda: True, feedback_controller_name=lambda: 'PID',
        times_sec=np.array([0., 1., 2.]), arm_reference_position=positions,
        arm_reference_velocity=np.zeros((3,5)), arm_reference_acceleration=np.zeros((3,5)),
        arm_reference_at=lambda i: (positions[i].copy(), np.zeros(5), np.zeros(5)),
        begin_sequence=Mock(return_value=True), mark_reset_succeeded=Mock(),
        arm_controller=Mock(), vehicle_controller=Mock())


def test_hold_retains_integral_and_only_authorizes_arm():
    robot, controller, state = robot_fixture()
    robot.start_arm_hold()
    assert robot.control_mode == ControlMode.ARM_HOLD
    robot.abrupt_planner_stop.assert_called_once_with(publish_zero=False)
    controller.reset_controller_state.assert_not_called()
    robot.authorize_motion.assert_called_once_with(reset_controller=False, components=('manipulator',))
    robot.publish_arm_hold(state)
    robot.publish_commands.assert_called_once_with([0.]*6, [.1]*5)


def test_unauthorized_hold_cannot_publish_torque():
    robot, controller, state = robot_fixture()
    robot.start_arm_hold()
    robot.execution_gate.allows_output = lambda part: False
    robot.publish_arm_hold(state)
    controller.arm_controller.assert_not_called()
    robot.publish_commands.assert_called_once_with([0.]*6, [0.]*5)


def test_stale_hold_rejected_without_mutation():
    robot, controller, state = robot_fixture()
    robot.execution_gate.ready = lambda: False
    with pytest.raises(ValueError, match='fresh feedback'):
        robot.start_arm_hold()
    robot.abrupt_planner_stop.assert_not_called()
    robot.authorize_motion.assert_not_called()


def test_continuous_reference_handoff_does_not_zero_or_reset_controller():
    robot, controller, state = robot_fixture()
    robot.start_arm_hold()
    replay = replay_fixture(robot)
    robot.start_arm_replay_from_hold(replay)
    assert robot.control_mode == ControlMode.REPLAY and robot._hold_after_arm_replay
    replay.mark_reset_succeeded.assert_called_once()
    robot.publish_commands.assert_not_called()
    controller.reset_controller_state.assert_not_called()


@pytest.mark.parametrize('fault', ['jump', 'limit', 'speed', 'nan', 'repeats', 'vehicle', 'gripper', 'sample_jump'])
def test_bad_reference_keeps_existing_hold(fault):
    robot, controller, state = robot_fixture()
    robot.start_arm_hold()
    replay = replay_fixture(robot)
    if fault == 'jump': replay.arm_reference_position[0, 1] += .1
    if fault == 'limit': replay.arm_reference_position[1, 1] = -.1
    if fault == 'speed': replay.arm_reference_velocity[1, 1] = .6
    if fault == 'nan': replay.arm_reference_acceleration[1, 1] = np.nan
    if fault == 'repeats': replay.repeats = 2
    if fault == 'sample_jump': replay.arm_reference_position[1, 1] += 1.
    if fault == 'gripper': replay.arm_reference_position[1, 4] += .001
    if fault == 'vehicle': replay.vehicle_subsystem_mode = lambda: 'replay_command'
    with pytest.raises(ValueError): robot.start_arm_replay_from_hold(replay)
    assert robot.control_mode == ControlMode.ARM_HOLD
    replay.begin_sequence.assert_not_called()
    robot.publish_commands.assert_not_called()


def test_explicit_stop_cancels_hold_and_sends_zero():
    robot, controller, state = robot_fixture()
    robot.start_arm_hold()
    robot._hold_after_arm_replay = True
    robot.publish_vehicle_and_arm = Mock()
    Robot.stop_execution(robot)
    assert not robot._hold_after_arm_replay
    robot.abrupt_planner_stop.assert_called_with()
    robot.publish_vehicle_and_arm.assert_called_once_with([0.]*6, [0.]*5)


def test_completion_holds_final_reference_in_same_control_tick():
    robot, controller, state = robot_fixture()
    robot.start_arm_hold()
    replay = replay_fixture(robot)
    robot.start_arm_replay_from_hold(replay)
    robot.sim_reset_hold = False
    replay.enabled = True
    replay.set_sim_time = Mock()
    replay.start_playback = Mock()
    replay.playback_status = lambda: 'complete'
    def complete():
        replay.enabled = False
        return None
    replay.current_sample_index = complete
    robot.active_controller_instance = lambda: replay
    robot._stop_replay_session_recording = Mock()
    robot._control_tick = time.monotonic() - .01
    robot.control_loop_callback()
    assert robot.control_mode == ControlMode.ARM_HOLD
    robot.publish_commands.assert_called_once_with([0.]*6, [.1]*5)
    controller.reset_controller_state.assert_not_called()


@pytest.mark.parametrize('failure', ['exception', 'nonfinite'])
def test_hold_controller_failure_stops_output(failure):
    robot, controller, state = robot_fixture()
    robot.start_arm_hold()
    robot.stop_execution = Mock()
    if failure == 'exception': controller.arm_controller.side_effect = RuntimeError('failed')
    else: controller.arm_controller.return_value = [float('nan')]*5
    robot.node = Mock()
    robot.sim_reset_hold = False
    robot.execution_gate.simulated_components = frozenset()
    robot._control_tick = time.monotonic() - .01
    robot.control_loop_callback()
    robot.stop_execution.assert_called_once()
    robot.publish_commands.assert_not_called()
