from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from rclpy.node import Node
from simlab.msg import ReferenceTargets


@dataclass
class ReferenceTopics:
    targets: str


def default_reference_topics(prefix: str) -> ReferenceTopics:
    return ReferenceTopics(targets=f"/{prefix}/reference/targets")


class ReferenceTargetPublisher:
    """
    Publishes the reference sample used by planning/control and MCAP replay capture.
    """

    def __init__(
        self,
        node: Node,
        prefix: str,
        topics: Optional[ReferenceTopics] = None,
        queue_size: int = 10,
        world_frame: str = "world",
    ):
        self.node = node
        self.prefix = prefix
        self.world_frame = world_frame
        self.topics = topics or default_reference_topics(prefix)
        self.pub_targets = node.create_publisher(ReferenceTargets, self.topics.targets, queue_size)

    def publish_targets(
        self, *, stamp_msg, world_pose, target_ned_pose,
        target_body_vel, target_body_acc, q_ref, dq_ref, ddq_ref,
    ) -> None:
        """Publish one coherent control or replay reference sample.

        Body quantities are FRD; world_pose describes the ROS FLU body in
        the world frame. Body acceleration is the derivative of body velocity.
        """
        msg = ReferenceTargets()
        msg.header.stamp = stamp_msg
        msg.header.frame_id = self.world_frame
        msg.world_pose = world_pose
        msg.ned_pose = np.asarray(target_ned_pose, dtype=float).tolist()
        msg.body_velocity = np.asarray(target_body_vel, dtype=float).tolist()
        msg.body_acceleration = np.asarray(target_body_acc, dtype=float).tolist()
        msg.arm_position = np.asarray(q_ref, dtype=float).tolist()
        msg.arm_velocity = np.asarray(dq_ref, dtype=float).tolist()
        msg.arm_acceleration = np.asarray(ddq_ref, dtype=float).tolist()
        q = world_pose.orientation
        body_to_world = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix() @ np.diag([1., -1., -1.])
        velocity = np.asarray(target_body_vel, dtype=float)
        acceleration = np.asarray(target_body_acc, dtype=float)
        for field, value in (
            (msg.world_twist.linear, body_to_world @ velocity[:3]),
            (msg.world_twist.angular, body_to_world @ velocity[3:]),
            (msg.world_accel.linear, body_to_world @ (
                acceleration[:3] + np.cross(velocity[3:], velocity[:3]))),
            (msg.world_accel.angular, body_to_world @ acceleration[3:]),
        ):
            field.x, field.y, field.z = map(float, value)
        self.pub_targets.publish(msg)
