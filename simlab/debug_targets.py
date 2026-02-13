# debug_targets.py
# Keep this file in your package (same folder as your Robot file, or in a utils/ folder).

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped, TwistStamped, AccelStamped


@dataclass
class DebugTopics:
    world_pose: str
    world_twist: str
    world_accel: str
    map_ned_pose: str
    body_vel: str
    body_acc: str
    arm_ref: str
    arm_dref: str
    arm_ddref: str


def default_debug_topics(prefix: str) -> DebugTopics:
    base = f"/{prefix}/debug"
    return DebugTopics(
        world_pose=f"{base}/target_world_pose",
        world_twist=f"{base}/target_world_twist",
        world_accel=f"{base}/target_world_accel",
        map_ned_pose=f"{base}/target_ned_pose",
        body_vel=f"{base}/target_body_vel",
        body_acc=f"{base}/target_body_acc",
        arm_ref=f"{base}/arm_ref",
        arm_dref=f"{base}/arm_dref",
        arm_ddref=f"{base}/arm_ddref",
    )


class DebugTargetPublisher:
    """
    Publishes planner/control targets for inspection, logging, rqt_plot, and RViz.

    WORLD (NWU):
      - PoseStamped: target pose
      - TwistStamped: target linear velocity (angular left 0)
      - AccelStamped: target linear acceleration (angular left 0)

    MAP (NED) + ARM refs:
      - Float64MultiArray: target_ned_pose [x y z roll pitch yaw]
      - Float64MultiArray: target_body_vel (6)
      - Float64MultiArray: target_body_acc (6)
      - Float64MultiArray: arm_ref, arm_dref, arm_ddref (arm joints + grasper)
    """

    def __init__(self, node: Node, prefix: str, topics: Optional[DebugTopics] = None, queue_size: int = 10):
        self.node = node
        self.prefix = prefix
        self.topics = topics or default_debug_topics(prefix)

        self.pub_world_pose = node.create_publisher(PoseStamped, self.topics.world_pose, queue_size)
        self.pub_world_twist = node.create_publisher(TwistStamped, self.topics.world_twist, queue_size)
        self.pub_world_accel = node.create_publisher(AccelStamped, self.topics.world_accel, queue_size)

        self.pub_map_pose = node.create_publisher(Float64MultiArray, self.topics.map_ned_pose, queue_size)
        self.pub_body_vel = node.create_publisher(Float64MultiArray, self.topics.body_vel, queue_size)
        self.pub_body_acc = node.create_publisher(Float64MultiArray, self.topics.body_acc, queue_size)

        self.pub_arm_ref = node.create_publisher(Float64MultiArray, self.topics.arm_ref, queue_size)
        self.pub_arm_dref = node.create_publisher(Float64MultiArray, self.topics.arm_dref, queue_size)
        self.pub_arm_ddref = node.create_publisher(Float64MultiArray, self.topics.arm_ddref, queue_size)

    def publish_world_targets(
        self,
        *,
        stamp_msg,
        xyz_world_nwu: Sequence[float],
        quat_world_wxyz: Sequence[float],
        vel_world_nwu: Optional[Sequence[float]] = None,   # [vx, vy, vz]
        acc_world_nwu: Optional[Sequence[float]] = None,   # [ax, ay, az]
    ) -> None:
        ps = PoseStamped()
        ps.header.stamp = stamp_msg
        ps.header.frame_id = "world"
        ps.pose.position.x = float(xyz_world_nwu[0])
        ps.pose.position.y = float(xyz_world_nwu[1])
        ps.pose.position.z = float(xyz_world_nwu[2])
        ps.pose.orientation.w = float(quat_world_wxyz[0])
        ps.pose.orientation.x = float(quat_world_wxyz[1])
        ps.pose.orientation.y = float(quat_world_wxyz[2])
        ps.pose.orientation.z = float(quat_world_wxyz[3])
        self.pub_world_pose.publish(ps)

        if vel_world_nwu is not None:
            ts = TwistStamped()
            ts.header.stamp = stamp_msg
            ts.header.frame_id = "world"
            ts.twist.linear.x = float(vel_world_nwu[0])
            ts.twist.linear.y = float(vel_world_nwu[1])
            ts.twist.linear.z = float(vel_world_nwu[2])
            self.pub_world_twist.publish(ts)

        if acc_world_nwu is not None:
            a = AccelStamped()
            a.header.stamp = stamp_msg
            a.header.frame_id = "world"
            a.accel.linear.x = float(acc_world_nwu[0])
            a.accel.linear.y = float(acc_world_nwu[1])
            a.accel.linear.z = float(acc_world_nwu[2])
            self.pub_world_accel.publish(a)

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
        msg = Float64MultiArray()

        msg.data = np.asarray(target_ned_pose, dtype=float).tolist()
        self.pub_map_pose.publish(msg)

        msg.data = np.asarray(target_body_vel, dtype=float).tolist()
        self.pub_body_vel.publish(msg)

        msg.data = np.asarray(target_body_acc, dtype=float).tolist()
        self.pub_body_acc.publish(msg)

        msg.data = np.asarray(q_ref, dtype=float).tolist()
        self.pub_arm_ref.publish(msg)

        msg.data = np.asarray(dq_ref, dtype=float).tolist()
        self.pub_arm_dref.publish(msg)

        msg.data = np.asarray(ddq_ref, dtype=float).tolist()
        self.pub_arm_ddref.publish(msg)
