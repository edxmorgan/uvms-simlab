# Copyright (C) 2026 Edward Morgan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class OpenCVCameraViewer(Node):
    def __init__(self, topic: str, window_name: str, reliable: bool):
        super().__init__("opencv_camera_viewer")
        self.topic = topic
        self.window_name = window_name
        self.frame_count = 0
        self.last_log_time = time.monotonic()
        self.last_hash = None
        self.static_count = 0

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(Image, topic, self.image_callback, qos)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.get_logger().info(f"Viewing {topic}. Press q or Esc in the image window to exit.")

    def image_callback(self, msg: Image) -> None:
        image = self._message_to_bgr(msg)
        if image is None:
            return

        self.frame_count += 1
        frame_hash = hash(image.tobytes())
        if frame_hash == self.last_hash:
            self.static_count += 1
        else:
            self.static_count = 0
        self.last_hash = frame_hash

        overlay = (
            f"{self.topic} | {msg.header.frame_id} | "
            f"{msg.width}x{msg.height} | frame {self.frame_count} | "
            f"static {self.static_count}"
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
        cv2.imshow(self.window_name, image)

        now = time.monotonic()
        if now - self.last_log_time >= 2.0:
            self.last_log_time = now
            self.get_logger().info(
                f"frame={self.frame_count} stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d} "
                f"min={int(image.min())} max={int(image.max())} mean={float(image.mean()):.2f} "
                f"static_count={self.static_count}"
            )

    def _message_to_bgr(self, msg: Image):
        if msg.encoding not in {"bgr8", "rgb8", "mono8"}:
            self.get_logger().warn(f"Unsupported image encoding: {msg.encoding}", throttle_duration_sec=2.0)
            return None

        channels = 1 if msg.encoding == "mono8" else 3
        expected_size = int(msg.height) * int(msg.width) * channels
        data = np.frombuffer(msg.data, dtype=np.uint8)
        if data.size < expected_size:
            self.get_logger().warn(
                f"Image data too small: got {data.size}, expected {expected_size}",
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
    parser = argparse.ArgumentParser(description="Display a ROS 2 Image topic with OpenCV.")
    parser.add_argument(
        "topic",
        nargs="?",
        default="/robot_1/camera/image_raw",
        help="Image topic to display.",
    )
    parser.add_argument("--window-name", default="UVMS Camera", help="OpenCV window title.")
    parser.add_argument(
        "--best-effort",
        action="store_true",
        help="Use best-effort QoS instead of reliable QoS.",
    )
    parsed, ros_args = parser.parse_known_args(args)

    rclpy.init(args=ros_args)
    node = OpenCVCameraViewer(parsed.topic, parsed.window_name, reliable=not parsed.best_effort)
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
