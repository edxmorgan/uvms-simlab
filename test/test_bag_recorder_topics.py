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
