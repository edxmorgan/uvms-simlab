from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, List

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.serialization import serialize_message
from std_srvs.srv import Trigger

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

        self.declare_parameter("bag_base_dir", bag_base_dir)
        self.declare_parameter("storage_id", storage_id)
        self.declare_parameter("serialization_format", serialization_format)
        self.declare_parameter("autostart_recording", False)
        self.declare_parameter("robots_prefix", ["robot_1_"])

        self.bag_base_dir = str(self.get_parameter("bag_base_dir").value)
        self.storage_id = str(self.get_parameter("storage_id").value)
        self.serialization_format = str(self.get_parameter("serialization_format").value)
        robot_prefixes = [str(prefix) for prefix in self.get_parameter("robots_prefix").value]

        self.bag_dir = ""
        self.writer = None
        self._recording = False

        self._subs = []
        self._topic_names: List[str] = []
        self._topics = list(topics)
        self._topics.extend(self._desired_target_topic_specs(robot_prefixes))

        for spec in self._topics:
            cb = self._make_write_cb(spec.name)
            sub = self.create_subscription(spec.msg_cls, spec.name, cb, spec.qos_depth)
            self._subs.append(sub)
            self._topic_names.append(spec.name)

        self.create_service(Trigger, "/bag_recorder_node/start_recording", self.start_recording_callback)
        self.create_service(Trigger, "/bag_recorder_node/stop_recording", self.stop_recording_callback)

        self.get_logger().info(
            f"Bag recorder ready for {len(self._topic_names)} topics. "
            "Use /bag_recorder_node/start_recording and /bag_recorder_node/stop_recording."
        )
        for n in self._topic_names[:10]:
            self.get_logger().info(f"  - {n}")
        if len(self._topic_names) > 10:
            self.get_logger().info(f"  ... (+{len(self._topic_names) - 10} more)")

        if bool(self.get_parameter("autostart_recording").value):
            self.start_recording()

    @staticmethod
    def _desired_target_topic_specs(robot_prefixes: List[str]) -> List[TopicSpec]:
        from simlab_msgs.msg import ControllerPerformance, ReferenceTargets

        topics: List[TopicSpec] = []
        for prefix in robot_prefixes:
            topics.append(
                TopicSpec(
                    name=f"/{prefix}/reference/targets",
                    msg_cls=ReferenceTargets,
                    type_str="simlab_msgs/msg/ReferenceTargets",
                    qos_depth=10,
                )
            )
            topics.append(
                TopicSpec(
                    name=f"/{prefix}/performance/controller",
                    msg_cls=ControllerPerformance,
                    type_str="simlab_msgs/msg/ControllerPerformance",
                    qos_depth=10,
                )
            )
        return topics

    def start_recording(self) -> tuple[bool, str]:
        if self._recording:
            return True, f"already recording to {self.bag_dir}"

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.bag_dir = f"{self.bag_base_dir}_{ts}"

        writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_dir, storage_id=self.storage_id)
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format=self.serialization_format,
            output_serialization_format=self.serialization_format,
        )
        writer.open(storage_options, converter_options)

        for topic_id, spec in enumerate(self._topics):
            writer.create_topic(
                rosbag2_py.TopicMetadata(
                    topic_id,
                    spec.name,
                    spec.type_str,
                    self.serialization_format,
                )
            )

        self.writer = writer
        self._recording = True
        message = f"recording {len(self._topic_names)} topics to {self.bag_dir}"
        self.get_logger().info(message)
        return True, message

    def stop_recording(self) -> tuple[bool, str]:
        if not self._recording:
            return True, "recording is not active"

        path = self.bag_dir
        self.writer = None
        self._recording = False
        message = f"stopped recording {path}"
        self.get_logger().info(message)
        return True, message

    def start_recording_callback(self, request, response):
        del request
        response.success, response.message = self.start_recording()
        return response

    def stop_recording_callback(self, request, response):
        del request
        response.success, response.message = self.stop_recording()
        return response

    def _make_write_cb(self, topic_name: str) -> Callable[[Any], None]:
        def _cb(msg: Any) -> None:
            if not self._recording or self.writer is None:
                return
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
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
