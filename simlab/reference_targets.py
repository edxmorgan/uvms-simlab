from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from rclpy.node import Node
from simlab_msgs.msg import ReferenceTargets


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
        self._last_msg: Optional[ReferenceTargets] = None

    def publish_world_targets(
        self,
        *,
        stamp_msg,
        xyz_world_nwu: Sequence[float],
        quat_world_wxyz: Sequence[float],
        vel_world_nwu: Optional[Sequence[float]] = None,   # [vx, vy, vz]
        acc_world_nwu: Optional[Sequence[float]] = None,   # [ax, ay, az]
    ) -> None:
        msg = self._last_msg or ReferenceTargets()
        msg.header.stamp = stamp_msg
        msg.header.frame_id = self.world_frame
        msg.world_pose.position.x = float(xyz_world_nwu[0])
        msg.world_pose.position.y = float(xyz_world_nwu[1])
        msg.world_pose.position.z = float(xyz_world_nwu[2])
        msg.world_pose.orientation.w = float(quat_world_wxyz[0])
        msg.world_pose.orientation.x = float(quat_world_wxyz[1])
        msg.world_pose.orientation.y = float(quat_world_wxyz[2])
        msg.world_pose.orientation.z = float(quat_world_wxyz[3])

        if vel_world_nwu is not None:
            msg.world_twist.linear.x = float(vel_world_nwu[0])
            msg.world_twist.linear.y = float(vel_world_nwu[1])
            msg.world_twist.linear.z = float(vel_world_nwu[2])

        if acc_world_nwu is not None:
            msg.world_accel.linear.x = float(acc_world_nwu[0])
            msg.world_accel.linear.y = float(acc_world_nwu[1])
            msg.world_accel.linear.z = float(acc_world_nwu[2])

        self._last_msg = msg

    def publish_map_targets_and_arm_refs(
        self,
        *,
        target_ned_pose: Sequence[float],
        target_body_vel: Sequence[float],
        target_body_acc: Sequence[float],
        q_ref: Sequence[float],
        dq_ref: Sequence[float],
        ddq_ref: Sequence[float],
    ) -> None:
        msg = self._last_msg or ReferenceTargets()
        msg.ned_pose = np.asarray(target_ned_pose, dtype=float).tolist()
        msg.body_velocity = np.asarray(target_body_vel, dtype=float).tolist()
        msg.body_acceleration = np.asarray(target_body_acc, dtype=float).tolist()
        msg.arm_position = np.asarray(q_ref, dtype=float).tolist()
        msg.arm_velocity = np.asarray(dq_ref, dtype=float).tolist()
        msg.arm_acceleration = np.asarray(ddq_ref, dtype=float).tolist()
        self._last_msg = msg
        self.pub_targets.publish(msg)
