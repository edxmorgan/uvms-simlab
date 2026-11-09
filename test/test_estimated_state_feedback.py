"""Truth and posterior deliberately disagree: feedback must use the posterior."""
from types import SimpleNamespace
import numpy as np
import pytest
from control_msgs.msg import DynamicJointState, InterfaceValue
from simlab.robot import Robot, Manipulator, Axis_Interface_names


def make_feedback():
    arm = Manipulator.__new__(Manipulator)
    arm.joints = ['e', 'd', 'c', 'b']
    arm.grasper = ['a']
    arm.alpha_axis_e = 'e'
    robot = Robot.__new__(Robot)
    robot.arm = arm
    robot.floating_base_IOs = 'vehicle'
    robot.arm_IOs = 'arm'
    robot.status = 'inactive'
    robot.prefix = 'robot_1_'
    robot.sim_time = 0.0
    robot.mocap_latest = None
    robot.vehicle_J = lambda _: SimpleNamespace(full=lambda: np.eye(6))
    msg = DynamicJointState()
    vehicle = {value: 0.0 for key, value in vars(Axis_Interface_names).items()
               if key.startswith('floating_')}
    vehicle.update({'sim_time': 1.0, 'sensor_noise_enabled': 1.0})
    for key in list(vehicle):
        if 'estimate' in key: vehicle[key] = .25
    # Truth exists and is very different; never select it or fall back to it.
    vehicle.update({k: 99.0 for k in ['position.x','position.y','position.z',
        'roll','pitch','yaw','velocity.x','velocity.y','velocity.z',
        'angular_velocity.x','angular_velocity.y','angular_velocity.z',
        'linear_acceleration.x','linear_acceleration.y','linear_acceleration.z',
        'angular_acceleration.x','angular_acceleration.y','angular_acceleration.z']})
    fields = {'vehicle': vehicle, 'arm': {'payload.mass': 0., 'gravity': 9.81,
        'control_power_abs': 0., 'control_energy_abs': 0.}}
    for joint in arm.joints + arm.grasper:
        fields[joint] = {'position':99., 'velocity':99., 'filtered_position':.4,
            'filtered_velocity':.2, 'estimated_acceleration':.1, 'effort':0., 'sim_period':.005}
    for name, values in fields.items():
        msg.joint_names.append(name)
        msg.interface_values.append(InterfaceValue(interface_names=list(values), values=list(values.values())))
    return robot, msg


def test_shared_feedback_uses_estimates_with_truth_present():
    robot, msg = make_feedback()
    robot.update_state(msg)
    np.testing.assert_array_equal(robot.ned_pose, [.25]*6)
    np.testing.assert_array_equal(robot.body_vel, [.25]*6)
    np.testing.assert_array_equal(robot.body_acc, [.25]*6)
    np.testing.assert_array_equal(robot.arm.q, [.4]*4)
    np.testing.assert_array_equal(robot.arm.dq, [.2]*4)
    assert robot.arm.grasper_q == [.4]
    assert robot.arm.grasper_q_dot == [.2]
    # This public snapshot is consumed by controllers, IK, hold and replay logic.
    state = robot.get_state()
    assert state['pose'] == [.25]*6
    assert state['body_vel'] == [.25]*6
    assert state['body_acc'] == [.25]*6
    assert state['q'] == [.4]*4
    assert state['dq'] == [.2]*4


def test_missing_estimate_never_falls_back_to_truth():
    robot, msg = make_feedback()
    values = msg.interface_values[0]
    i = values.interface_names.index('position_estimate.x')
    del values.interface_names[i]
    del values.values[i]
    with pytest.raises(ValueError):
        robot.update_state(msg)


@pytest.mark.parametrize('joint,field,bad', [
    ('vehicle', 'position_estimate.x', float('nan')),
    ('vehicle', 'angular_acceleration_estimate.z', float('inf')),
    ('a', 'estimated_acceleration', float('nan')),
    ('arm', 'gravity', float('nan')),
])
def test_invalid_sample_does_not_partially_commit(joint, field, bad):
    robot, msg = make_feedback()
    robot.update_state(msg)
    before = robot.get_state().copy()
    values = msg.interface_values[msg.joint_names.index('e')]
    values.values[values.interface_names.index('filtered_position')] = 12.0
    values = msg.interface_values[msg.joint_names.index(joint)]
    values.values[values.interface_names.index(field)] = bad
    with pytest.raises(ValueError):
        robot.update_state(msg)
    assert robot.get_state() == before


def test_missing_late_field_does_not_partially_commit():
    robot, msg = make_feedback()
    robot.update_state(msg)
    before = robot.get_state().copy()
    values = msg.interface_values[msg.joint_names.index('a')]
    i = values.interface_names.index('effort')
    del values.interface_names[i]
    del values.values[i]
    with pytest.raises(ValueError):
        robot.update_state(msg)
    assert robot.get_state() == before


def test_listener_rejects_sample_without_refreshing_age(monkeypatch):
    from unittest.mock import Mock
    monkeypatch.setattr('simlab.robot.forward_dynamics_observations', lambda *_: None)
    robot, msg = make_feedback()
    robot.node = Mock()
    robot.listener_callback(msg)
    previous = robot._last_state_received
    values = msg.interface_values[0]
    values.values[values.interface_names.index('position_estimate.x')] = float('nan')
    robot.listener_callback(msg)
    assert robot._feedback_fault
    assert robot._last_state_received == previous
    values.values[values.interface_names.index('position_estimate.x')] = .25
    robot.listener_callback(msg)
    assert not robot._feedback_fault
    assert robot._last_state_received >= previous


@pytest.mark.parametrize('fault,last_received', [(True, 10.), (False, 9.)])
def test_feedback_failure_cancels_once_and_publishes_zero(monkeypatch, fault, last_received):
    from unittest.mock import Mock
    monkeypatch.setattr('simlab.robot.time.monotonic', lambda: 10.)
    robot = Robot.__new__(Robot)
    robot._feedback_fault = fault
    robot._last_state_received = last_received
    robot._feedback_stale_since = 9.0
    robot.prefix = 'robot_1_'
    robot.node = Mock()
    robot.abrupt_planner_stop = Mock()
    robot.cancel_replay_settle = Mock()
    robot._zero_teleop_commands = Mock()
    controller = Mock()
    robot.active_controller_instance = Mock(return_value=controller)
    robot.publish_commands = Mock()
    robot.get_state = Mock()
    robot.control_loop_callback()
    robot.control_loop_callback()
    robot.abrupt_planner_stop.assert_called_once()
    controller.mark_reset_failed.assert_called_once()
    robot.get_state.assert_not_called()
    assert robot.publish_commands.call_count == 2
    robot.publish_commands.assert_called_with([0.] * 6, [0.] * 5)


def test_executor_delay_suppresses_command_then_allows_feedback_to_drain(monkeypatch):
    from unittest.mock import Mock
    monkeypatch.setattr('simlab.robot.time.monotonic', lambda: 10.)
    robot = Robot.__new__(Robot)
    robot._last_state_received = 9.
    robot.publish_vehicle_and_arm = Mock()
    robot.abrupt_planner_stop = Mock()
    robot.control_loop_callback()
    robot.publish_vehicle_and_arm.assert_called_once_with([0.] * 6, [0.] * 5)
    robot.abrupt_planner_stop.assert_not_called()
    # The queued valid update arrives before the next timer callback. No plan
    # cancellation or reference rewrite is allowed for this scheduling delay.
    robot._last_state_received = 10.
    robot.get_state = Mock(return_value={'status': 'inactive'})
    robot.control_loop_callback()
    robot.abrupt_planner_stop.assert_not_called()
    assert robot._feedback_stale_since is None


@pytest.mark.parametrize('sampling', [False, True])
@pytest.mark.parametrize('valid', [0., 1.])
def test_hardware_guard_independent_of_optional_identification(monkeypatch, sampling, valid):
    from unittest.mock import Mock
    monkeypatch.setattr('simlab.robot.forward_dynamics_observations', lambda *_: None)
    robot = Robot.__new__(Robot)
    robot.prefix = 'robot_real_'
    robot.node = Mock()
    robot.node.get_clock.return_value.now.return_value.nanoseconds = 123
    robot.update_state = Mock()
    names = ['estimator_valid'] + (['identification.valid'] if sampling else [])
    values = [valid] + ([1.] if sampling else [])
    msg = DynamicJointState(joint_names=['robot_real_IOs'], interface_values=[
        InterfaceValue(interface_names=names, values=values)])
    robot.listener_callback(msg)
    assert robot._hardware_estimator_valid is (valid == 1.)
    assert robot._hardware_estimator_received == 123


def test_simulated_feedback_does_not_enable_hardware_guard(monkeypatch):
    from unittest.mock import Mock
    monkeypatch.setattr('simlab.robot.forward_dynamics_observations', lambda *_: None)
    robot = Robot.__new__(Robot)
    robot.prefix = 'robot_1_'
    robot.update_state = Mock()
    msg = DynamicJointState(joint_names=['robot_1_IOs'], interface_values=[
        InterfaceValue(interface_names=['estimator_valid', 'sim_step'], values=[0., 1.])])
    robot.listener_callback(msg)
    assert not hasattr(robot, '_hardware_estimator_valid')
