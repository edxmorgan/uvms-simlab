from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, List

import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message

import rosbag2_py


@dataclass(frozen=True)
class TopicSpec:
    name: str
    msg_cls: Any
    type_str: str
    qos_depth: int = 10


class BagRecorder(Node):
    def __init__(
        self,
        topics: List[TopicSpec],
        bag_base_dir: str = "uvms_bag",
        storage_id: str = "mcap",
        serialization_format: str = "cdr",
    ):
        super().__init__("bag_recorder")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.bag_dir = f"{bag_base_dir}_{ts}"
        self.serialization_format = serialization_format

        self.writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_dir, storage_id=storage_id)
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format=self.serialization_format,
            output_serialization_format=self.serialization_format,
        )
        self.writer.open(storage_options, converter_options)

        self._subs = []
        self._topic_names: List[str] = []

        # Register topics and create subscriptions
        for topic_id, spec in enumerate(topics):
            # 1) create topic metadata in bag
            self.writer.create_topic(
                rosbag2_py.TopicMetadata(
                    topic_id,
                    spec.name,
                    spec.type_str,
                    self.serialization_format
                )
            )

            # 2) create subscription that writes back to the same topic
            cb = self._make_write_cb(spec.name)
            sub = self.create_subscription(spec.msg_cls, spec.name, cb, spec.qos_depth)

            self._subs.append(sub)
            self._topic_names.append(spec.name)

        self.get_logger().info(f"Recording {len(self._topic_names)} topics to {self.bag_dir}")
        for n in self._topic_names[:10]:
            self.get_logger().info(f"  - {n}")
        if len(self._topic_names) > 10:
            self.get_logger().info(f"  ... (+{len(self._topic_names) - 10} more)")

    def _make_write_cb(self, topic_name: str) -> Callable[[Any], None]:
        def _cb(msg: Any) -> None:
            self.writer.write(
                topic_name,
                serialize_message(msg),
                self.get_clock().now().nanoseconds,
            )
        return _cb


def main(args=None):
    rclpy.init(args=args)

    from control_msgs.msg import DynamicJointState
    from geometry_msgs.msg import PoseStamped

    topics = [
        TopicSpec(
            name="dynamic_joint_states",
            msg_cls=DynamicJointState,
            type_str="control_msgs/msg/DynamicJointState",
            qos_depth=10,
        ),
        TopicSpec(
            name="mocap_pose",
            msg_cls=PoseStamped,
            type_str="geometry_msgs/msg/PoseStamped",
            qos_depth=10,
        ),
    ]

    node = BagRecorder(topics=topics)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
