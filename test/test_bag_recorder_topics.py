from simlab.bag_recorder import BagRecorder
from rclpy.qos import QoSDurabilityPolicy, QoSReliabilityPolicy


def test_bag_recorder_records_dynamic_obstacle_topics_with_transient_local_qos():
    specs = BagRecorder._dynamic_obstacle_topic_specs()
    by_name = {spec.name: spec for spec in specs}

    assert "/dynamic_obstacles" in by_name
    assert "/dynamic_obstacle_markers" in by_name
    assert by_name["/dynamic_obstacles"].type_str == "ros2_control_blue_reach_5/msg/DynamicObstacleArray"
    assert by_name["/dynamic_obstacle_markers"].type_str == "visualization_msgs/msg/MarkerArray"

    for topic_name in ("/dynamic_obstacles", "/dynamic_obstacle_markers"):
        qos = by_name[topic_name].qos_depth
        assert qos.durability == QoSDurabilityPolicy.TRANSIENT_LOCAL
        assert qos.reliability == QoSReliabilityPolicy.RELIABLE


def test_bag_recorder_disturbance_topics_are_per_robot():
    specs = BagRecorder._disturbance_topic_specs(['robot_1_', 'robot_2_'])
    by_name = {spec.name: spec for spec in specs}
    assert len(by_name) == 12
    for prefix in ('robot_1_', 'robot_2_'):
        base = f'/{prefix}/disturbance/'
        assert by_name[base+'command'].type_str.endswith('/DisturbanceCommand')
        assert by_name[base+'vehicle/state'].type_str.endswith('/DisturbanceState')
        assert by_name[base+'arm/state'].type_str.endswith('/DisturbanceState')
        assert by_name[base+'status'].type_str == 'simlab/msg/DisturbanceRunStatus'
        assert by_name[base+'task_wrench'].type_str == 'geometry_msgs/msg/WrenchStamped'


def test_standard_mcap_roundtrip_preserves_injection_and_estimates(tmp_path, monkeypatch):
    import time
    import rclpy
    import rosbag2_py
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.serialization import deserialize_message
    from control_msgs.msg import DynamicJointState, InterfaceValue
    from geometry_msgs.msg import WrenchStamped
    from ros2_control_blue_reach_5.msg import DisturbanceCommand, DisturbanceState
    from simlab.msg import DisturbanceRunStatus, TaskWrenchEstimate
    from simlab.bag_recorder import TopicSpec

    # Isolated graph: this test records synthetic messages, never drives a simulator.
    monkeypatch.setenv('ROS_DOMAIN_ID', '121')
    rclpy.init()
    recorder = BagRecorder([TopicSpec('dynamic_joint_states', DynamicJointState,
        'control_msgs/msg/DynamicJointState')], bag_base_dir=str(tmp_path))
    publisher = Node('test_standard_disturbance_recording')
    executor = SingleThreadedExecutor()
    executor.add_node(recorder)
    executor.add_node(publisher)
    base = '/robot_1_/disturbance/'
    command = DisturbanceCommand(sequence=7, session_id='recording-test')
    command.body_wrench.force.x = .3
    vehicle = DisturbanceState(component='vehicle', dynamics_time=2., step=400)
    vehicle.applied_body_wrench.force.x = .3
    arm = DisturbanceState(component='arm', dynamics_time=2., step=400,
        applied_joint_torques=[.01, -.02, .03, -.04])
    status = DisturbanceRunStatus(session_id='recording-test', state='running',
        configuration_json='{"frequency_hz":0.1,"cycles":3}')
    # Deliberately distinct truth/estimate channels must survive MCAP unchanged.
    joints = DynamicJointState(joint_names=['robot_1_IOs', 'robot_1__axis_e'], interface_values=[
        InterfaceValue(interface_names=['position.x','position_estimate.x',
            'velocity.x','velocity_estimate.x','roll','roll_estimate',
            'estimated_disturbance.force.x'], values=[1.,1.1,2.,2.1,.3,.4,.25]),
        InterfaceValue(interface_names=['position','filtered_position','velocity','filtered_velocity'],
                       values=[.5,.6,.7,.8])])
    task_wrench = WrenchStamped()
    task_wrench.header.frame_id = 'robot_1_joint_4'
    task_wrench.wrench.force.x = 1.
    samples = {base+'task_estimate':TaskWrenchEstimate(valid=True, rank=6),
               base+'task_wrench':task_wrench, base+'command':command, base+'vehicle/state':vehicle,
               base+'arm/state':arm, base+'status':status, 'dynamic_joint_states':joints}
    from tf2_msgs.msg import TFMessage
    from geometry_msgs.msg import TransformStamped
    from simlab.bag_recorder import transient_local_qos
    for index, topic in enumerate(['/tf','/tf_static','/simulation_truth/tf','/simulation_truth/tf_static']):
        transform = TransformStamped(child_frame_id='frame_'+str(index))
        transform.header.frame_id = 'world'
        transform.transform.rotation.w = 1.
        transform.transform.translation.x = float(index)
        samples[topic] = TFMessage(transforms=[transform])
    pubs = {topic:publisher.create_publisher(type(msg), topic,
                transient_local_qos(100) if topic.endswith('tf_static') else 10)
            for topic,msg in samples.items()}
    try:
        assert recorder.start_recording()[0]
        deadline = time.monotonic()+5.
        while time.monotonic()<deadline and not all(p.get_subscription_count() for p in pubs.values()):
            executor.spin_once(timeout_sec=.02)
        assert all(p.get_subscription_count() for p in pubs.values())
        for topic,msg in samples.items():
            pubs[topic].publish(msg)
        deadline = time.monotonic()+.5
        while time.monotonic()<deadline:
            executor.spin_once(timeout_sec=.02)
        assert recorder.stop_recording()[0]
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=recorder.bag_dir, storage_id='mcap'),
                    rosbag2_py.ConverterOptions('', ''))
        restored = {}
        while reader.has_next():
            topic,data,_ = reader.read_next()
            if topic in samples:
                restored[topic] = deserialize_message(data, type(samples[topic]))
        assert restored == samples
    finally:
        recorder.stop_recording()
        executor.shutdown()
        recorder.destroy_node()
        publisher.destroy_node()
        rclpy.shutdown()


def test_evaluation_trace_is_opt_in_and_per_robot(monkeypatch):
    monkeypatch.delenv('UVMS_EKF_EVALUATION_TRACE', raising=False)
    assert BagRecorder._evaluation_trace_topic_specs(['robot_1_']) == []
    monkeypatch.setenv('UVMS_EKF_EVALUATION_TRACE', '1')
    specs = BagRecorder._evaluation_trace_topic_specs(['robot_1_', 'robot_2_'])
    assert [s.name for s in specs] == ['/robot_1_ekf_evaluation_trace', '/robot_2_ekf_evaluation_trace']
    assert all(s.type_str == 'std_msgs/msg/Float64MultiArray' and s.qos_depth == 2048 for s in specs)


def test_standard_recording_keeps_both_transform_trees():
    specs = {s.name: s for s in BagRecorder._transform_topic_specs()}
    assert set(specs) == {'/tf', '/tf_static', '/simulation_truth/tf', '/simulation_truth/tf_static'}
    for topic in ['/tf_static', '/simulation_truth/tf_static']:
        assert specs[topic].qos_depth.durability == QoSDurabilityPolicy.TRANSIENT_LOCAL
