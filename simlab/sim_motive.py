# rigid_bodies_pub.py
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point, Quaternion
from mocap4r2_msgs.msg import RigidBodies, RigidBody, Marker
from scipy.spatial.transform import Rotation as R
from robot import Robot
from typing import List, Tuple
import numpy as np
import numpy as np
import tf2_ros
from tf2_ros import TransformException
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion

def apply_transform_pose(p_in: Pose, ts: TransformStamped) -> Pose:
    """
    Apply a geometry_msgs/TransformStamped to a geometry_msgs/Pose.
    ts is target_frame <- source_frame from tf2_ros.Buffer.lookup_transform(target, source, ...)
    p_in is expressed in source_frame
    returns p_out expressed in target_frame
    """
    # translation and rotation from the transform
    t = ts.transform.translation
    q = ts.transform.rotation

    # rotate and translate position
    p = np.array([p_in.position.x, p_in.position.y, p_in.position.z])
    R_t = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    p_out = R_t @ p + np.array([t.x, t.y, t.z])

    # compose orientations, active rotation, left multiply
    q_out = quat_mul(q, p_in.orientation)

    # build Pose
    pout = Pose()
    pout.position.x, pout.position.y, pout.position.z = p_out.tolist()
    pout.orientation = q_out
    return pout


def quat_mul(q1: Quaternion, q2: Quaternion) -> Quaternion:
    r1 = R.from_quat([q1.x, q1.y, q1.z, q1.w])
    r2 = R.from_quat([q2.x, q2.y, q2.z, q2.w])
    r_out = r1 * r2
    x, y, z, w = r_out.as_quat()
    return Quaternion(x=x, y=y, z=z, w=w)


def euler_xyz_to_quat(rpy_xyz: Tuple[float, float, float]) -> Quaternion:
    q = R.from_euler('xyz', rpy_xyz).as_quat()  # [x, y, z, w]
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


def ned_list_to_point_rpy(pose_list) -> Tuple[Point, Tuple[float, float, float]]:
    """pose_list is [x, y, z, roll, pitch, yaw] in NED, radians"""
    if not isinstance(pose_list, (list, tuple)) or len(pose_list) < 6:
        raise TypeError("Expected list or tuple [x, y, z, roll, pitch, yaw]")
    x, y, z, roll, pitch, yaw = map(float, pose_list[:6])
    return Point(x=x, y=y, z=z), (roll, pitch, yaw)


def ned_to_enu_pose(p_ned: Point, rpy_ned_xyz: Tuple[float, float, float]) -> Pose:
    """
    NED to RViz ENU, match your C++ path
    position, x = x, y = −y, z = −z
    orientation, roll stays, pitch and yaw negate
    rpy_ned_xyz must be radians
    """
    # position
    p_enu = Point(
        x=p_ned.x,
        y=-p_ned.y,
        z=-p_ned.z,
    )

    # orientation
    roll, pitch, yaw = rpy_ned_xyz
    q_enu = euler_xyz_to_quat((roll, -pitch, -yaw))

    return Pose(position=p_enu, orientation=q_enu)


def transform_enu_to_optitrack(pose_enu: Pose) -> Pose:
    # Position, ENU [x, y, z] to OptiTrack world [X=Y_enu, Y=Z_enu, Z=X_enu]
    px, py, pz = pose_enu.position.x, pose_enu.position.y, pose_enu.position.z
    pos_opt = Point(x=py, y=pz, z=px)
    # Orientation, apply fixed permutation quaternion on the left
    q_perm_inv = Quaternion(x=-0.5, y=-0.5, z=-0.5, w=0.5)
    q_in = pose_enu.orientation
    q_opt = quat_mul(q_perm_inv, q_in)
    return Pose(position=pos_opt, orientation=q_opt)

class RigidBodiesPublisher(Node):
    def __init__(self):
        super().__init__(
            'rigid_bodies_publisher',
            automatically_declare_parameters_from_overrides=True
        )

        # Parameters
        self.no_robot = self.get_parameter('no_robot').value
        self.no_efforts = self.get_parameter('no_efforts').value
        self.robots_prefix = self.get_parameter('robots_prefix').value
        self.record = self.get_parameter('record_data').value
        self.controllers = self.get_parameter('controllers').value

        self.get_logger().info(f"Robot prefixes found: {self.robots_prefix}")
        self.total_no_efforts = self.no_robot * self.no_efforts
        self.get_logger().info(f"Total number of commands: {self.total_no_efforts}")

        self.robot_map = "robot_1_map"
        self.mocap_frame_id = "robot_1_mocap_link"

        # TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)

        # Robot objects
        self.robots: List[Robot] = []
        for k, (prefix, controller) in enumerate(list(zip(self.robots_prefix, self.controllers))):
            robot_k = Robot(self, self.tf_buffer, k, 4, prefix)
            if controller in robot_k.list_controllers():
                robot_k.set_controller(controller)
            else:
                self.get_logger().warn(
                    f"Unknown controller '{controller}' for {prefix}; using default {robot_k.controller_name}"
                )
            self.robots.append(robot_k)

        # Publisher
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )
        self.pub = self.create_publisher(RigidBodies, '/rigid_bodies', qos)

        # Frame counter
        self.frame_number = 0

        # Timer
        self.timer = self.create_timer(1 / 120.0, self.timer_callback)  # 120 Hz


    def timer_callback(self):
        msg = RigidBodies()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "optitrack_world"
        msg.frame_number = self.frame_number

        for idx, robot in enumerate(self.robots):
            state = robot.get_state()
            if state.get('status') != 'active':
                continue

            # state['pose'] is a list [x, y, z, roll, pitch, yaw] in NED, radians
            p_ned, rpy_ned = ned_list_to_point_rpy(state['pose'])

            # NED to ENU
            pose_enu = ned_to_enu_pose(p_ned=p_ned, rpy_ned_xyz=rpy_ned)

            try:
                ts = self.tf_buffer.lookup_transform(
                    target_frame=self.robot_map,
                    source_frame=self.mocap_frame_id,
                    time=rclpy.time.Time(),
                    timeout=Duration(seconds=0.1),
                )
                # self.get_logger().info(f'TF {self.mocap_frame_id} <- {self.robot_map}: {ts.transform}')
                # transform pose into map
                # pose_enu = apply_transform(pose_enu, ts, self.mocap_frame_id)
                pose_enu = apply_transform_pose(pose_enu, ts)


                            # ENU to OptiTrack
                pose_opt = transform_enu_to_optitrack(pose_enu)
                # pose_opt = pose_enu

                # Build RigidBody entry
                rb = RigidBody()
                name = self.robots_prefix[idx] if idx < len(self.robots_prefix) else f"rigid_body_{idx}"
                rb.rigid_body_name = name.rstrip('_')
                rb.markers = []  # empty, fill with Marker messages if available
                rb.pose = pose_opt

                msg.rigidbodies.append(rb)
            except TransformException as ex:
                self.get_logger().warn(f'Waiting for TF {self.mocap_frame_id} <- {self.robot_map}: {ex}')
                return


        self.pub.publish(msg)
        self.frame_number = (self.frame_number + 1) % (1 << 32)  # uint32 wrap


def main():
    rclpy.init()
    node = RigidBodiesPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
