"""Geometric wrench mapping: physical motion, frames, offsets and action contract."""
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from simlab.motion_planning.kinematics.forward import load_forward_kinematics
from simlab.motion_planning.kinematics.forward.tool_wrench import ToolWrenchMapper
from simlab.uvms_parameters_sim import ReachParams
from simlab.disturbance_waveforms import parse_args, display_channels


@pytest.fixture
def mapper():
    root = Path(__file__).parents[1]/'resource'
    return ToolWrenchMapper(root, ReachParams.base_T0_new, ReachParams.tipOffset)


def test_export_matches_ik_and_physical_velocity(mapper):
    fk = load_forward_kinematics(Path(__file__).parents[1]/'resource')
    rng = np.random.default_rng(485)
    for _ in range(20):
        q = rng.uniform(.2, 2.5, 4)
        nu = rng.normal(size=10)
        J, T = mapper.geometry(q)
        pose = np.asarray(fk(q, mapper.mounting, np.zeros(6), mapper.tip)[-1]).ravel()
        np.testing.assert_allclose(T[:3, 3], pose[:3], atol=1e-10)
        np.testing.assert_allclose(T[:3, :3], Rotation.from_quat(pose[[4,5,6,3]]).as_matrix(), atol=1e-9)
        def moved(t):
            _, local = mapper.geometry(q+t*nu[6:])
            base = np.eye(4)
            base[:3,:3] = Rotation.from_rotvec(t*nu[3:6]*[1,-1,-1]).as_matrix()
            base[:3,3] = t*nu[:3]*[1,-1,-1]
            return base @ local
        h = 1e-6
        plus, minus = moved(h), moved(-h)
        angular = ((plus[:3,:3]-minus[:3,:3])/(2*h)) @ T[:3,:3].T
        twist = np.r_[(plus[:3,3]-minus[:3,3])/(2*h), angular[2,1],angular[0,2],angular[1,0]]
        np.testing.assert_allclose(J @ nu, twist, atol=2e-9)
        wrench = rng.normal(size=6)
        assert mapper.map(q, wrench, 'body') @ nu == pytest.approx(wrench @ twist, abs=1e-8)


def test_force_lever_arm_and_frame_equivalence(mapper):
    q = [.8,1.6,.9,1.1]
    _, T = mapper.geometry(q)
    wrench = np.array([1.,2.,3.,.1,-.2,.3])
    body = np.r_[T[:3,:3] @ wrench[:3],T[:3,:3] @ wrench[3:]]
    mapped, recorded = mapper.resolve(q, wrench, 'tool')
    np.testing.assert_allclose(recorded, wrench, atol=1e-12)
    np.testing.assert_allclose(mapped[:3], body[:3]*[1,-1,-1], atol=1e-12)
    np.testing.assert_allclose(mapped[3:6], (body[3:]+np.cross(T[:3,3],body[:3]))*[1,-1,-1], atol=1e-12)
    rotation = Rotation.from_euler('xyz', [.4,np.pi/2,.8]).as_matrix()
    world = np.r_[rotation @ body[:3],rotation @ body[3:]]
    np.testing.assert_allclose(mapper.map(q, world, 'world', rotation), mapped, atol=1e-12)
    np.testing.assert_allclose(mapper.map(q, body, 'body'), mapped, atol=1e-12)
    with pytest.raises(ValueError):
        mapper.map(q, world, 'world')


def test_task_cli_action_and_display():
    from simlab.action import RunDisturbance
    from simlab.disturbance_injection_node import goal_options
    args = parse_args(['--space','task','--dof','fx','tz','--amplitude','1','.1',
                       '--wrench-frame','world'])
    goal = RunDisturbance.Goal(space='task', wrench_frame='world', dofs=args.dof,
        amplitudes=[1.,.1], waveforms=['sine'], phases_deg=[0.], delays_s=[0.],
        frequency_hz=.1, cycles=3)
    options = goal_options(goal)
    assert options.space == 'task' and options.wrench_frame == 'world'
    assert [c.dof for c in display_channels(options)] == [
        'fx','fy','fz','tx','ty','tz']
    assert parse_args(['--space','task']).dof == ['fx']
    assert parse_args([]).dof == ['surge']
    for bad in [['--space','task','--dof','arm_d'], ['--dof','fx']]:
        with pytest.raises(SystemExit):
            parse_args(bad)


@pytest.mark.parametrize('frame', ['tool', 'body', 'world'])
def test_native_task_action_maps_and_releases(monkeypatch, mapper, frame):
    """Exercise the real action node with isolated receiver telemetry, no plant."""
    import time
    import json
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from rclpy.executors import SingleThreadedExecutor
    from control_msgs.msg import DynamicJointState, InterfaceValue
    from geometry_msgs.msg import TransformStamped
    from tf2_ros import StaticTransformBroadcaster
    from ros2_control_blue_reach_5.msg import DisturbanceCommand, DisturbanceState
    from simlab.action import RunDisturbance
    from simlab.disturbance_injection_node import DisturbanceInjectionNode
    monkeypatch.setenv('ROS_DOMAIN_ID', '176')
    rclpy.init()
    node = DisturbanceInjectionNode()
    fixture = Node('task_receiver_fixture')
    executor = SingleThreadedExecutor()
    executor.add_node(node); executor.add_node(fixture)
    states = {c: DisturbanceState(component=c) for c in ('vehicle','arm')}
    pubs = {c: fixture.create_publisher(DisturbanceState,
        '/robot_1_/disturbance/'+c+'/state', 10) for c in states}
    joints = fixture.create_publisher(DynamicJointState, '/dynamic_joint_states', 10)
    received = []
    q = [.8,1.6,.9,1.1]
    publish_joints = [True]
    broadcast = StaticTransformBroadcaster(fixture)
    world_rotation = Rotation.from_euler('xyz', [.4,np.pi/2,.8])
    request_wrench = [1.,-.5,.3,.1,-.2,.4]
    def receive(msg):
        received.append(msg)
        for state in states.values():
            state.lease_active = not msg.release
            state.lease_owner = '' if msg.release else msg.session_id
    subscription = fixture.create_subscription(DisturbanceCommand,
        '/robot_1_/disturbance/command', receive, 10)
    def tick():
        msg = DynamicJointState(joint_names=['robot_1__axis_'+a for a in 'edcb'],
            interface_values=[InterfaceValue(interface_names=['position'], values=[v]) for v in q])
        msg.joint_names.append('robot_1_IOs')
        msg.interface_values.append(InterfaceValue(
            interface_names=['roll', 'pitch', 'yaw'], values=[0., 0., 0.]))
        msg.header.stamp = fixture.get_clock().now().to_msg()
        transform = TransformStamped()
        transform.header.stamp = msg.header.stamp
        transform.header.frame_id = 'world'
        transform.child_frame_id = 'robot_1_map'
        quat = world_rotation.as_quat()
        transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w = map(float, quat)
        broadcast.sendTransform(transform)
        if publish_joints[0]:
            joints.publish(msg)
        for c,state in states.items():
            if not state.paused:
                state.dynamics_time += .1
            pubs[c].publish(state)
    timer = fixture.create_timer(.02, tick)
    client = ActionClient(fixture, RunDisturbance, '/robot_1_/disturbance/run')
    try:
        deadline = time.monotonic()+5
        while (not node.receivers_ready() or not node.joint_states) and time.monotonic()<deadline:
            executor.spin_once(timeout_sec=.02)
        assert client.wait_for_server(timeout_sec=2)
        goal = RunDisturbance.Goal(space='task', wrench_frame=frame, dofs=['fx','fy','fz','tx','ty','tz'],
            amplitudes=request_wrench, waveforms=['step'], phases_deg=[0.], delays_s=[0.],
            frequency_hz=1., cycles=1)
        future = client.send_goal_async(goal)
        executor.spin_until_future_complete(future, timeout_sec=3)
        assert future.done() and future.result().accepted
        result = future.result().get_result_async()
        deadline = time.monotonic()+3
        while (node.run is None or node.run.elapsed < 2.1) and time.monotonic()<deadline:
            executor.spin_once(timeout_sec=.01)
        assert node.run is not None
        for state in states.values():
            state.paused = True
        deadline = time.monotonic()+.1
        while time.monotonic()<deadline:
            executor.spin_once(timeout_sec=.01)
        paused_time = node.status.elapsed_s
        deadline = time.monotonic()+.2
        while time.monotonic()<deadline:
            executor.spin_once(timeout_sec=.01)
        assert node.status.elapsed_s == paused_time and not result.done()
        for state in states.values():
            state.paused = False
        executor.spin_until_future_complete(result, timeout_sec=6)
        assert result.done() and result.result().result.success
        expected = mapper.map(q, request_wrench, frame, world_rotation.as_matrix())
        nonzero = []
        for msg in received:
            force, torque = msg.body_wrench.force, msg.body_wrench.torque
            values = [force.x,force.y,force.z,torque.x,torque.y,torque.z,*msg.joint_torques]
            if any(values):
                nonzero.append(values)
                np.testing.assert_allclose(values, expected, atol=1e-10)
        assert nonzero
        assert received[-1].release
        assert json.loads(node.status.configuration_json)['application_frame'] == 'robot_1_joint_4'
        # A subsequent run must release its lease when mapping telemetry goes stale.
        goal.cycles = 20
        again = client.send_goal_async(goal)
        executor.spin_until_future_complete(again, timeout_sec=3)
        assert again.done() and again.result().accepted
        publish_joints[0] = False
        failed = again.result().get_result_async()
        executor.spin_until_future_complete(failed, timeout_sec=3)
        assert failed.done() and not failed.result().result.success
        assert 'telemetry' in failed.result().result.message or 'stale' in failed.result().result.message
        assert received[-1].release
        # Recover after failure, then check explicit action cancellation.
        publish_joints[0] = True
        deadline = time.monotonic()+.2
        while time.monotonic()<deadline:
            executor.spin_once(timeout_sec=.01)
        again = client.send_goal_async(goal)
        executor.spin_until_future_complete(again, timeout_sec=3)
        assert again.done() and again.result().accepted
        cancel = again.result().cancel_goal_async()
        executor.spin_until_future_complete(cancel, timeout_sec=3)
        canceled = again.result().get_result_async()
        executor.spin_until_future_complete(canceled, timeout_sec=3)
        assert canceled.done() and canceled.result().result.message == 'Canceled'
        assert received[-1].release
        assert subscription is not None and timer is not None
    finally:
        client.destroy(); executor.shutdown(); fixture.destroy_node(); node.destroy_node(); rclpy.shutdown()


@pytest.mark.parametrize('fault', ['none','missing_joint','nan_joint','future','stale_stamp','stale_receipt','missing_tf','missing_attitude'])
def test_task_mapping_requires_fresh_consistent_state(mapper, fault):
    import time
    from types import SimpleNamespace
    from collections import deque
    from rclpy.clock import Clock, ClockType
    from rclpy.duration import Duration
    from tf2_ros import Buffer
    from geometry_msgs.msg import TransformStamped
    from control_msgs.msg import DynamicJointState, InterfaceValue
    from simlab.disturbance_injection_node import DisturbanceInjectionNode
    clock = Clock(clock_type=ClockType.ROS_TIME)
    now = clock.now()
    state = DynamicJointState(joint_names=['robot_1__axis_'+a for a in 'edcb'],
        interface_values=[InterfaceValue(interface_names=['position'], values=[v]) for v in [.8,1.6,.9,1.1]])
    state.joint_names.append('robot_1_IOs')
    state.interface_values.append(InterfaceValue(
        interface_names=['roll','pitch','yaw','roll_estimate','pitch_estimate','yaw_estimate'],
        values=[.2,-.3,.7,1.,1.,1.]))
    if fault == 'missing_attitude':
        state.interface_values[-1].values[0] = float('nan')
    state.header.stamp = (now-Duration(seconds=.1)).to_msg()
    receipt = time.monotonic()
    if fault == 'missing_joint':
        state.joint_names[0] = 'another_robot_axis_e'
    elif fault == 'nan_joint':
        state.interface_values[0].values[0] = float('nan')
    elif fault == 'future':
        state.header.stamp = (now+Duration(seconds=1.)).to_msg()
    elif fault == 'stale_stamp':
        state.header.stamp = (now-Duration(seconds=1.)).to_msg()
    elif fault == 'stale_receipt':
        receipt -= 1.
    buffer = Buffer()
    if fault != 'missing_tf':
        for delay in (.2,.05):
            tf = TransformStamped()
            tf.header.frame_id, tf.child_frame_id = 'world','robot_1_map'
            tf.header.stamp = (now-Duration(seconds=delay)).to_msg()
            tf.transform.rotation.w = 1.
            buffer.set_transform_static(tf, 'test')
    probe = SimpleNamespace(joint_states=deque([(state,receipt)]), prefix='robot_1_',
        get_clock=lambda:clock, tf_buffer=buffer, task_mapper=mapper)
    if fault == 'none':
        q,rotation,stamp = DisturbanceInjectionNode.task_state(probe,'world')
        assert q == [.8,1.6,.9,1.1]
        np.testing.assert_allclose(rotation, Rotation.from_euler('xyz',[.2,.3,-.7]).as_matrix(), atol=1e-14)
        assert stamp == state.header.stamp
    else:
        with pytest.raises(ValueError):
            DisturbanceInjectionNode.task_state(probe,'world')
