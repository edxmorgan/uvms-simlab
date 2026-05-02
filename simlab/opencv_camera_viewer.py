# Copyright (C) 2026 Edward Morgan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
from typing import Dict, List

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


@dataclass
class TopicState:
    window_name: str
    frame_count: int = 0
    last_log_time: float = 0.0
    last_hash: int | None = None
    static_count: int = 0


class OpenCVCameraViewer(Node):
    def __init__(self, topics: List[str], window_name: str, reliable: bool):
        super().__init__("camera_viewer")
        self.declare_parameter("image_topics", [""])
        self.declare_parameter("robot_prefixes", [""])
        param_topics = [
            str(topic)
            for topic in self.get_parameter("image_topics").value
            if str(topic)
        ]
        param_robot_topics = [
            f"/{str(prefix).rstrip('_')}/camera/image_raw"
            for prefix in self.get_parameter("robot_prefixes").value
            if str(prefix)
        ]
        self.topics = topics or param_topics or param_robot_topics or ["/robot_1/camera/image_raw"]
        self.topic_states: Dict[str, TopicState] = {}
        self.image_subscriptions = []

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
        )
        for topic in self.topics:
            topic_window_name = window_name if len(self.topics) == 1 else f"{window_name}: {topic}"
            self.topic_states[topic] = TopicState(
                window_name=topic_window_name,
                last_log_time=time.monotonic(),
            )
            self.image_subscriptions.append(
                self.create_subscription(
                    Image,
                    topic,
                    lambda msg, image_topic=topic: self.image_callback(image_topic, msg),
                    qos,
                )
            )
            cv2.namedWindow(topic_window_name, cv2.WINDOW_NORMAL)
        self.get_logger().info(
            f"Viewing {self.topics}. Press q or Esc in any image window to exit."
        )

    def image_callback(self, topic: str, msg: Image) -> None:
        state = self.topic_states[topic]
        image = self._message_to_bgr(topic, msg)
        if image is None:
            return

        state.frame_count += 1
        frame_hash = hash(image.tobytes())
        if frame_hash == state.last_hash:
            state.static_count += 1
        else:
            state.static_count = 0
        state.last_hash = frame_hash

        overlay = (
            f"{topic} | {msg.header.frame_id} | "
            f"{msg.width}x{msg.height} | frame {state.frame_count} | "
            f"static {state.static_count}"
        )
        cv2.putText(
            image,
            overlay,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(state.window_name, image)

        now = time.monotonic()
        if now - state.last_log_time >= 2.0:
            state.last_log_time = now
            self.get_logger().info(
                f"topic={topic} frame={state.frame_count} "
                f"stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d} "
                f"min={int(image.min())} max={int(image.max())} mean={float(image.mean()):.2f} "
                f"static_count={state.static_count}"
            )

    def _message_to_bgr(self, topic: str, msg: Image):
        if msg.encoding not in {"bgr8", "rgb8", "mono8"}:
            self.get_logger().warn(
                f"{topic}: unsupported image encoding: {msg.encoding}",
                throttle_duration_sec=2.0,
            )
            return None

        channels = 1 if msg.encoding == "mono8" else 3
        expected_size = int(msg.height) * int(msg.width) * channels
        data = np.frombuffer(msg.data, dtype=np.uint8)
        if data.size < expected_size:
            self.get_logger().warn(
                f"{topic}: image data too small: got {data.size}, expected {expected_size}",
                throttle_duration_sec=2.0,
            )
            return None

        image = data[:expected_size].reshape((msg.height, msg.width, channels))
        if msg.encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif msg.encoding == "mono8":
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image = image.copy()
        return image


def main(args=None):
    parser = argparse.ArgumentParser(description="Display one or more ROS 2 Image topics with OpenCV.")
    parser.add_argument(
        "topics",
        nargs="*",
        help="Image topics to display. If omitted, uses the image_topics ROS parameter.",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        default=[],
        help="Robot prefixes to display, for example robot_1_ robot_2_ robot_real_.",
    )
    parser.add_argument("--window-name", default="UVMS Camera", help="OpenCV window title.")
    parser.add_argument(
        "--best-effort",
        action="store_true",
        help="Use best-effort QoS instead of reliable QoS.",
    )
    parsed, ros_args = parser.parse_known_args(args)

    rclpy.init(args=ros_args)
    robot_topics = [f"/{prefix.rstrip('_')}/camera/image_raw" for prefix in parsed.robots]
    node = OpenCVCameraViewer([*parsed.topics, *robot_topics], parsed.window_name, reliable=not parsed.best_effort)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
