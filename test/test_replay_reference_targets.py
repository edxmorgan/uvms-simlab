from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from geometry_msgs.msg import Pose
from builtin_interfaces.msg import Time

from simlab.robot import Robot, ControlMode
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
def test_replay_publishes_controller_inputs_without_changing_commands(vehicle_mode, arm_mode, tf_ready):
    robot = Robot.__new__(Robot)
    robot.node = Mock()
    robot.prefix, robot.map_frame, robot.world_frame = 'robot_1_', 'robot_1_map', 'world'
    robot.sim_reset_hold = False
    robot.control_mode = ControlMode.REPLAY
    state = dict(status='active', sim_time=1., pose=[0.] * 6,
                 body_vel=[0.] * 6, q=[0.] * 4, dq=[0.] * 4,
                 grasper_q=[0.], grasper_qdot=[0.], dt=.01)
    robot.get_state = lambda: state
    spec = SimpleNamespace(vehicle_fn=Mock(return_value=[11.] * 6),
                           arm_fn=Mock(return_value=[12.] * 5))
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
