# path_builder.py
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from nav_msgs.msg import Path
from mocap4r2_msgs.msg import RigidBodies
from rclpy.duration import Duration
import numpy as np
import tf2_ros
from tf2_ros import TransformException
from geometry_msgs.msg import TransformStamped

def apply_transform(ps_in: PoseStamped, ts: TransformStamped, target_frame: str) -> PoseStamped:
    # ts gives a transform that converts data in source into target
    t = ts.transform.translation
    q = ts.transform.rotation

    # position'
    p = np.array([ps_in.pose.position.x, ps_in.pose.position.y, ps_in.pose.position.z])
    Rt = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    p_out = Rt @ p + np.array([t.x, t.y, t.z])

    # orientation'
    q_out = quat_mul(q, ps_in.pose.orientation)

    ps_out = PoseStamped()
    ps_out.header.stamp = ps_in.header.stamp
    ps_out.header.frame_id = target_frame
    ps_out.pose.position.x, ps_out.pose.position.y, ps_out.pose.position.z = p_out.tolist()
    ps_out.pose.orientation = q_out
    return ps_out

# multiply two quaternions (Hamilton product)
def quat_mul(q1, q2):
    r1 = R.from_quat([q1.x, q1.y, q1.z, q1.w])
    r2 = R.from_quat([q2.x, q2.y, q2.z, q2.w])
    r_out = r1 * r2
    x, y, z, w = r_out.as_quat()
    return Quaternion(x=x, y=y, z=z, w=w)

# inverse of a quaternion
def quat_inv(q):
    r = R.from_quat([q.x, q.y, q.z, q.w])
    rinv = r.inv()
    x, y, z, w = rinv.as_quat()
    return Quaternion(x=x, y=y, z=z, w=w)

def transform_optitrack_to_enu(pose_opt: Pose) -> Pose:
    # position, [Xe, Yn, Zu] = [Zf, Xr, Yu]
    px, py, pz = pose_opt.position.x, pose_opt.position.y, pose_opt.position.z
    pos_enu = Point(x=pz, y=px, z=py)

    # orientation, pre multiply by q_perm = rotation that maps X->Y, Y->Z, Z->X
    q_perm = Quaternion(x=0.5, y=0.5, z=0.5, w=0.5)
    q_in = pose_opt.orientation
    q_enu = quat_mul(q_perm, q_in)

    return Pose(position=pos_enu, orientation=q_enu)


def is_valid_pose(p: Pose, pos_eps=1e-6) -> bool:
    # many OptiTrack drivers publish a zeroed first sample
    if abs(p.position.x) < pos_eps and abs(p.position.y) < pos_eps and abs(p.position.z) < pos_eps:
        return False
    # orientation should be near unit length
    nq = p.orientation.x**2 + p.orientation.y**2 + p.orientation.z**2 + p.orientation.w**2
    return 0.5 < nq < 1.5


class MocapPathBuilder(Node):
    def __init__(self):
        super().__init__('path_builder')

        self.declare_parameter('mocap_frame_id', 'robot_1_mocap_link')
        self.declare_parameter('max_points', 2000)

        self.declare_parameter('mocap_pose_topic', 'mocap_pose')
        self.declare_parameter('mocap_path_topic', 'mocap_path')

        self.declare_parameter('map_mocap_pose_topic', 'map_mocap_pose')
        self.declare_parameter('map_mocap_path_topic', 'map_mocap_path')

        self.declare_parameter('robot_map', 'robot_1_map')

        self.mocap_frame_id = self.get_parameter('mocap_frame_id').get_parameter_value().string_value
        self.max_points = self.get_parameter('max_points').get_parameter_value().integer_value
        self.mocap_pose_topic = self.get_parameter('mocap_pose_topic').get_parameter_value().string_value
        self.mocap_path_topic = self.get_parameter('mocap_path_topic').get_parameter_value().string_value
        self.map_pose_topic = self.get_parameter('map_mocap_pose_topic').get_parameter_value().string_value
        self.map_path_topic = self.get_parameter('map_mocap_path_topic').get_parameter_value().string_value
        self.robot_map = self.get_parameter('robot_map').get_parameter_value().string_value

        # Subscriber QoS for mocap
        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )

        # Publishers QoS for RViz
        pose_pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        path_pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )

        self.mocap_pose_pub = self.create_publisher(PoseStamped, self.mocap_pose_topic, pose_pub_qos) # this is the ground truth pose in ENU
        self.mocap_path_pub = self.create_publisher(Path, self.mocap_path_topic, path_pub_qos)

        self.map_pose_pub = self.create_publisher(PoseStamped, self.map_pose_topic, pose_pub_qos)
        self.map_path_pub = self.create_publisher(Path, self.map_path_topic, path_pub_qos)
        

        self.mocap_path_msg = Path()
        self.mocap_path_msg.header = Header(frame_id=self.mocap_frame_id)

        self.map_path_msg = Path()
        self.map_path_msg.header = Header(frame_id=self.mocap_frame_id)

        self.subscription = self.create_subscription(
            RigidBodies, '/rigid_bodies', self.cb_rigid_bodies, sub_qos
        )

        # store the first valid pose, in ENU, as offset
        self.origin_offset = None

        # TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)

    def cb_rigid_bodies(self, msg: RigidBodies):
        if not msg.rigidbodies:
            return

        rb = msg.rigidbodies[0]

        # convert OptiTrack pose into ENU first
        pose_enu = transform_optitrack_to_enu(rb.pose)

        # skip zeroed or invalid first samples
        if not is_valid_pose(pose_enu):
            return

        # record the first valid ENU pose as origin
        if self.origin_offset is None:
            self.origin_offset = pose_enu

        # Build pose relative to initial offset
        ps_mocap = PoseStamped()
        ps_mocap.header.stamp = msg.header.stamp
        ps_mocap.header.frame_id = self.mocap_frame_id

        ps_mocap.pose.position.x = pose_enu.position.x - self.origin_offset.position.x
        ps_mocap.pose.position.y = pose_enu.position.y - self.origin_offset.position.y
        ps_mocap.pose.position.z = pose_enu.position.z - self.origin_offset.position.z
        ps_mocap.pose.orientation = quat_mul(pose_enu.orientation, quat_inv(self.origin_offset.orientation))
        self.mocap_pose_pub.publish(ps_mocap)

        self.mocap_path_msg.header.stamp = msg.header.stamp
        self.mocap_path_msg.header.frame_id = self.mocap_frame_id
        self.mocap_path_msg.poses.append(ps_mocap)

        if self.max_points > 0 and len(self.mocap_path_msg.poses) > self.max_points:
            del self.mocap_path_msg.poses[0:len(self.mocap_path_msg.poses) - self.max_points]

        self.mocap_path_pub.publish(self.mocap_path_msg)

        # try:
        #     ts = self.tf_buffer.lookup_transform(
        #         target_frame=self.mocap_frame_id,
        #         source_frame=self.robot_map,
        #         time=rclpy.time.Time(),
        #         timeout=Duration(seconds=0.1),
        #     )
        #     # self.get_logger().info(f'TF {self.mocap_frame_id} <- {self.robot_map}: {ts.transform}')
        #     # transform pose into map
        #     ps_map = apply_transform(ps_mocap, ts, self.mocap_frame_id)

        #     # publish pose in map
        #     self.map_pose_pub.publish(ps_map)

        #     # append to path in map
        #     self.map_path_msg.header.stamp = msg.header.stamp
        #     self.map_path_msg.header.frame_id = self.mocap_frame_id
        #     self.map_path_msg.poses.append(ps_map)
        #     if self.max_points > 0 and len(self.map_path_msg.poses) > self.max_points:
        #         del self.map_path_msg.poses[0:len(self.map_path_msg.poses) - self.max_points]
        #     self.map_path_pub.publish(self.map_path_msg)

        # except TransformException as ex:
        #     self.get_logger().warn(f'Waiting for TF {self.mocap_frame_id} <- {self.robot_map}: {ex}')
        #     return
        
        
def main():
    rclpy.init()
    node = MocapPathBuilder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
