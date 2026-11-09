from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from geometry_msgs.msg import Pose
from builtin_interfaces.msg import Time

from simlab.robot import Robot, ControlMode, ControllerSpec
from simlab.reference_targets import ReferenceTargetPublisher






def test_replay_message_is_fresh_and_transforms_body_derivatives():
    node = Mock()
    publisher = ReferenceTargetPublisher(node, 'robot_1_')
    publisher._last_msg = object()  # A stale planner cache must not be accessed.
    pose = Pose()
    pose.position.x = 10.
    pose.orientation.w = 1.
    publisher.publish_replay_targets(
        stamp_msg=Time(sec=42), world_pose=pose,
        target_ned_pose=[1., 2., 3., 0., 0., 0.],
        target_body_vel=[2., 0., 0., 0., 0., 1.],
        target_body_acc=[1., 0., 0., 0., 0., 2.],
        q_ref=[1.] * 5, dq_ref=[2.] * 5, ddq_ref=[3.] * 5,
    )
    msg = publisher.pub_targets.publish.call_args.args[0]
    assert msg.header.frame_id == 'world'
    assert msg.header.stamp.sec == 42
    assert msg.world_pose.position.x == 10.
    assert list(msg.arm_position) == [1.] * 5
    assert list(msg.arm_velocity) == [2.] * 5
    assert list(msg.arm_acceleration) == [3.] * 5
    assert msg.world_twist.linear.x == 2.
    assert msg.world_twist.angular.z == -1.
    assert msg.world_accel.linear.x == 1.
    assert msg.world_accel.linear.y == -2.  # omega cross v, then FRD -> FLU
    assert msg.world_accel.angular.z == -2.


@pytest.mark.parametrize('vehicle_mode', ['track_reference', 'hold_initial', 'replay_command', 'zero_command'])
@pytest.mark.parametrize('arm_mode', ['track_reference', 'hold_initial', 'replay_command', 'zero_command'])
@pytest.mark.parametrize('tf_ready', [True, False])
@pytest.mark.parametrize('deferred', [True, False])
def test_replay_publishes_controller_inputs_without_changing_commands(vehicle_mode, arm_mode, tf_ready, deferred):
    robot = Robot.__new__(Robot)
    robot.node = Mock()
    robot.prefix, robot.map_frame, robot.world_frame = 'robot_1_', 'robot_1_map', 'world'
    robot.sim_reset_hold = False
    robot.control_mode = ControlMode.REPLAY
    state = dict(status='active', sim_time=1., pose=[0.] * 6,
                 body_vel=[0.] * 6, q=[0.] * 4, dq=[0.] * 4,
                 grasper_q=[0.], grasper_qdot=[0.], dt=.01)
    robot.get_state = lambda: state
    spec = ControllerSpec('PID', vehicle_fn=Mock(return_value=[11.] * 6),
                          arm_fn=Mock(return_value=[12.] * 5))
    if deferred:
        instance = SimpleNamespace(vehicle_controller=spec.vehicle_fn, arm_controller=spec.arm_fn)
        factory = Mock(return_value=instance)
        spec.factory = factory
        spec.vehicle_fn = spec.arm_fn = None
    robot._controllers = {'PID': spec}
    replay = SimpleNamespace(
        enabled=True, set_sim_time=Mock(), current_sample_index=lambda: 7,
        feedback_controller_name=lambda: 'PID',
        vehicle_subsystem_mode=lambda: vehicle_mode,
        manipulator_subsystem_mode=lambda: arm_mode,
        vehicle_reference_at=Mock(return_value=([1.] * 6, [2.] * 6, [3.] * 6)),
        arm_reference_at=Mock(return_value=([4.] * 5, [5.] * 5, [6.] * 5)),
        initial_vehicle_pose=lambda: [7.] * 6,
        initial_manipulator_position=lambda: [8.] * 5,
        vehicle_command_at=lambda _: [9.] * 6, arm_command_at=lambda _: [10.] * 5,
    )
    robot.active_controller_instance = lambda: replay
    robot.continuous_vehicle_reference_pose = lambda pose, **_: np.asarray(pose)
    robot.try_transform_pose = Mock(return_value=Pose() if tf_ready else None)
    robot.reference_pub = Mock()
    robot.publish_commands = Mock()
    robot.control_loop_callback()
    supported = ('track_reference', 'hold_initial')
    if deferred:
        if vehicle_mode in supported or arm_mode in supported:
            factory.assert_called_once_with()
        else:
            factory.assert_not_called()
    if vehicle_mode in supported and arm_mode in supported:
        reference = robot._replay_tracking_reference
        vehicle = spec.vehicle_fn.call_args.kwargs
        arm = spec.arm_fn.call_args.kwargs
        for metric, control in [('pose', 'target_pos'), ('body_vel', 'target_vel'), ('body_acc', 'target_acc')]:
            np.testing.assert_array_equal(reference[metric], vehicle[control])
        for metric, control in [('q', 'q_ref'), ('dq', 'dq_ref'), ('ddq', 'ddq_ref')]:
            np.testing.assert_array_equal(reference[metric], arm[control])
    else:
        assert robot._replay_tracking_reference is None
    if tf_ready and vehicle_mode in supported and arm_mode in supported:
        args = robot.reference_pub.publish_replay_targets.call_args.kwargs
        vehicle = spec.vehicle_fn.call_args.kwargs
        arm = spec.arm_fn.call_args.kwargs
        for published, used in [('target_ned_pose', 'target_pos'), ('target_body_vel', 'target_vel'), ('target_body_acc', 'target_acc')]:
            np.testing.assert_array_equal(args[published], vehicle[used])
        for key in ('q_ref', 'dq_ref', 'ddq_ref'):
            np.testing.assert_array_equal(args[key], arm[key])
        assert robot.try_transform_pose.call_args.kwargs['source_frame'] == 'robot_1_map'
    else:
        robot.reference_pub.publish_replay_targets.assert_not_called()
    expected_vehicle = 11. if vehicle_mode in supported else (9. if vehicle_mode == 'replay_command' else 0.)
    expected_arm = 12. if arm_mode in supported else (10. if arm_mode == 'replay_command' else 0.)
    robot.publish_commands.assert_called_once_with([expected_vehicle] * 6, [expected_arm] * 5)


def test_oges_reset_clears_only_command_memory():
    from simlab.controllers.oges import OgesModelbasedController
    controller = OgesModelbasedController.__new__(OgesModelbasedController)
    controller.arm_dof = 4
    controller.vehicle_u_prev = np.ones(6)
    controller.arm_u_prev = np.ones(4)
    controller.vehicle_weights = np.arange(6.)
    controller.reset_controller_state()
    np.testing.assert_array_equal(controller.vehicle_u_prev, np.zeros(6))
    np.testing.assert_array_equal(controller.arm_u_prev, np.zeros(4))
    np.testing.assert_array_equal(controller.vehicle_weights, np.arange(6.))


@pytest.mark.parametrize('reference', [None, dict(pose=[1.] * 6)])
def test_metrics_never_use_planner_cache_for_replay(reference):
    robot = Robot.__new__(Robot)
    robot.get_state = lambda: dict(status='active')
    robot.sim_reset_hold = False
    robot.control_mode = ControlMode.REPLAY
    robot.pose_command = robot.body_vel_command = robot.body_acc_command = [999.] * 6
    robot.arm = SimpleNamespace(q_command=[999.] * 4, dq_command=[999.] * 4, ddq_command=[999.] * 4)
    robot._reference_trajectory_direction_ned = lambda: np.zeros(3)
    robot._replay_tracking_reference = reference
    robot.performance_metrics = Mock()
    robot._update_controller_performance_metrics()
    args = robot.performance_metrics.update.call_args.kwargs
    assert args['active'] is (reference is not None)
    assert args['reference'] == (reference or {})


@pytest.mark.parametrize('available', [True, False])
def test_replay_pass_resets_selected_feedback_or_rejects_missing_controller(available):
    robot = Robot.__new__(Robot)
    robot.node = Mock()
    robot.control_mode = ControlMode.REPLAY
    robot.sim_reset_hold = False
    robot.get_state = lambda: dict(status='active', sim_time=1.)
    controller = SimpleNamespace(reset_controller_state=Mock()) if available else None
    robot.controller_instance = lambda name: controller
    robot.performance_metrics = Mock()
    robot.publish_commands = Mock()
    replay = SimpleNamespace(
        set_sim_time=Mock(), has_pending_auto_start=lambda: True,
        feedback_controller_name=lambda: 'Ours',
        vehicle_subsystem_mode=lambda: 'track_reference',
        manipulator_subsystem_mode=lambda: 'track_reference',
        start_playback=Mock(return_value=False), mark_reset_failed=Mock(), enabled=False)
    robot.active_controller_instance = lambda: replay
    robot.control_loop_callback()
    if available:
        controller.reset_controller_state.assert_called_once()
        robot.performance_metrics.reset.assert_called_once()
        replay.start_playback.assert_called_once_with(sim_time_sec=1.)
    else:
        replay.start_playback.assert_not_called()
        replay.mark_reset_failed.assert_called_once()
    robot.publish_commands.assert_called_once_with([0.] * 6, [0.] * 5)


def test_reset_waits_for_state_published_after_service_completion(monkeypatch):
    from simlab.robot import Robot
    from ros2_control_blue_reach_5.srv import ResetSimUvms
    from control_msgs.msg import DynamicJointState
    monkeypatch.setattr('simlab.robot.forward_dynamics_observations', lambda *_: None)
    callbacks={};success=Mock();failure=Mock()
    robot=SimpleNamespace(sim_reset_hold=True,_reset_local_command_state=Mock(),prefix='robot_1_',
        node=Mock(),update_state=Mock(),_call_reset_state_service=lambda **kw:callbacks.update(kw))
    robot.node.get_clock.return_value.now.return_value.nanoseconds=100
    Robot.reset_simulation_with_state(robot,ResetSimUvms.Request(),success,failure)
    msg=DynamicJointState();msg.header.stamp.nanosec=99
    Robot.listener_callback(robot,msg)
    success.assert_not_called()
    callbacks['on_success']()
    Robot.listener_callback(robot,msg)
    success.assert_not_called()
    msg.header.stamp.nanosec=100
    Robot.listener_callback(robot,msg)
    success.assert_not_called()
    msg.header.stamp.nanosec=101
    Robot.listener_callback(robot,msg)
    success.assert_called_once();failure.assert_not_called()
    assert robot._reset_feedback_pending is None
    Robot.listener_callback(robot,msg)
    success.assert_called_once()


def test_reset_feedback_timeout_holds_commands():
    from simlab.robot import Robot
    failure=Mock()
    robot=SimpleNamespace(_reset_feedback_pending={'deadline':0.,'failure':failure},
        publish_commands=Mock(),node=Mock(),sim_reset_hold=False,get_state=Mock())
    Robot.control_loop_callback(robot)
    robot.publish_commands.assert_called_once_with([0.]*6,[0.]*5)
    robot.get_state.assert_not_called();failure.assert_called_once()
    assert robot.sim_reset_hold and robot._reset_feedback_pending is None
