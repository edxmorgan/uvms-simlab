"""Read-only integration of state/reference metrics with the ROS message."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import time
import numpy as np

from simlab.robot import Robot, ControlMode
from simlab.performance_metrics import ControllerPerformanceMetrics
from simlab.motion_planning.kinematics.forward import load_forward_kinematics


def robot_fixture():
    robot = Robot.__new__(Robot)
    robot.prefix = 'robot_1_'
    robot.map_frame = 'robot_1_map'
    robot.node = Mock()
    from builtin_interfaces.msg import Time
    robot.node.get_clock().now().to_msg.return_value = Time()
    robot.performance_publisher = Mock()
    robot.performance_metrics = ControllerPerformanceMetrics()
    robot._last_state_received = time.monotonic()
    robot.sim_reset_hold = False
    robot.control_mode = ControlMode.PLANNER
    robot.pose_command = [0., 0., 2., 0., 0., 0.]
    robot.body_vel_command = robot.body_acc_command = [0.] * 6
    q = [3.1, .7, .4, 2.1]
    robot.arm = SimpleNamespace(q_command=q, dq_command=[0.]*4, ddq_command=[0.]*4)
    state = dict(status='active', pose=robot.pose_command.copy(), body_vel=[0.]*6,
                 body_acc=[0.]*6, q=q, dq=[0.]*4, ddq=[0.]*4, sim_time=0.)
    robot.get_state = lambda: state.copy()
    robot._reference_trajectory_direction_ned = lambda: np.zeros(3)
    robot.fk_eval = load_forward_kinematics(Path(__file__).parents[1] / 'resource')
    return robot, state


def test_publication_exposes_validity_and_coherent_tool_tracking():
    robot, state = robot_fixture()
    robot._publish_controller_performance_metrics()
    msg = robot.performance_publisher.publish.call_args.args[0]
    assert msg.metric_version == 2 and msg.valid and msg.tool_valid
    assert msg.tool_position_error_m < 1e-12
    assert msg.tool_attitude_error_rad < 1e-12
    assert np.isnan(msg.tracking_score_rms)  # no elapsed support yet
    assert not msg.effort_valid and np.isnan(msg.normalized_control_effort)
    state['sim_time'] = .1
    robot._publish_controller_performance_metrics()
    msg = robot.performance_publisher.publish.call_args.args[0]
    assert msg.valid_duration_sec == .1
    assert msg.tracking_score_rms == 0.
    np.testing.assert_array_equal(msg.arm_position_error_rad, np.zeros(4))


def test_stale_feedback_publishes_unavailable_not_zero():
    robot, state = robot_fixture()
    robot._last_state_received = time.monotonic() - 1.
    robot._publish_controller_performance_metrics()
    msg = robot.performance_publisher.publish.call_args.args[0]
    assert msg.active and not msg.valid
    assert np.isnan(msg.tracking_score)
    assert np.isnan(msg.vehicle_n_attitude)
    assert not msg.tool_valid


def test_inactive_tracking_publishes_unavailable():
    robot, state = robot_fixture()
    robot.control_mode = ControlMode.TELEOP
    robot._publish_controller_performance_metrics()
    msg = robot.performance_publisher.publish.call_args.args[0]
    assert not msg.active and not msg.valid
    assert msg.status == 'inactive'
    assert np.isnan(msg.tracking_score)


def test_overlay_snapshot_expires_without_new_command_publications():
    robot, state = robot_fixture()
    robot._update_controller_performance_metrics()
    assert robot.get_controller_performance_metrics()['valid']
    robot._last_state_received = time.monotonic() - 1.
    snapshot = robot.get_controller_performance_metrics()
    assert not snapshot['valid'] and not snapshot['tool_valid']
    assert snapshot['feedback_age_sec'] >= 1.
    robot.control_mode = ControlMode.TELEOP
    assert not robot.get_controller_performance_metrics()['active']


def test_missing_feedback_is_explicit_false_for_inactive_display():
    robot, state = robot_fixture()
    robot._last_state_received = None
    robot.control_mode = ControlMode.TELEOP
    snapshot = robot.get_controller_performance_metrics()
    assert snapshot['feedback_valid'] is False
    assert snapshot['active'] is False
