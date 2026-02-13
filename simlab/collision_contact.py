#!/usr/bin/env python3
# collision_contact.py
import numpy as np
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker
from mesh_utils import make_marker, color, collect_env_meshes
from fcl_checker import FCLWorld


class CollisionNode(Node):
    def __init__(self):
        super().__init__('mesh_collision_node')
        self.declare_parameter('robot_description', '')

        urdf_string = self.get_parameter('robot_description').get_parameter_value().string_value
        if not urdf_string:
            self.get_logger().error('robot_description param is empty. Did you load it into the param server in launch')
            raise RuntimeError('no robot_description')

        # optional log like your original
        robot_links, env_links, floor_depth = collect_env_meshes(urdf_string)
        self.get_logger().info(f'robot_links { [x["link"] for x in robot_links] }')
        self.get_logger().info(f'env_links { [x["link"] for x in env_links] }')

        # single world that mirrors your original data layout
        self.world = FCLWorld(urdf_string=urdf_string, world_frame='world', vehicle_radius=0.4)

        # TF and pub
        self.tf_buf = Buffer()
        self.tf = TransformListener(self.tf_buf, self)
        self.contact_pub = self.create_publisher(Marker, 'contact_markers', 10)

        # 20 Hz
        self.timer = self.create_timer(0.05, self.tick)

    def tick(self):
        ok = self.world.update_from_tf(self.tf_buf, rclpy.time.Time())
        if not ok:
            return

        # clear old markers
        clear = Marker()
        clear.header.frame_id = 'world'
        clear.action = Marker.DELETEALL
        self.contact_pub.publish(clear)

        # 1. collision, one marker per robot link vs env link
        pairs = self.world.robot_env_contacts_one_point_per_pair()

        CONTACT_MARKER_SIZE = 0.05
        red = color(r=1.0, g=0.1, b=0.1, a=1.0)
        for idx, (pair_key, p_world) in enumerate(pairs.items()):
            m = make_marker('contact', idx, 'world', CONTACT_MARKER_SIZE, p_world, red)
            m.lifetime.sec = 0
            m.lifetime.nanosec = int(0.1 * 1e9)
            self.contact_pub.publish(m)

        # 2. global clearance identical intent, now with nearest points enabled
        try:
            resp = self.world.global_clearance()
            if resp is not None:
                md, p_robot, p_env = resp
                # self.get_logger().info(f'global min dist {md:.4f} m')

                # optional nearest point markers
                # blue = color(r=0.1, g=0.1, b=0.95, a=1.0)
                green = color(r=0.1, g=0.95, b=0.1, a=1.0)
                # mr = make_marker('nearest_robot', 1001, 'world', 0.05, p_robot, blue)
                me = make_marker('nearest_env',   1002, 'world', 0.05, p_env,   green)
                # mr.lifetime.nanosec = int(0.1 * 1e9)
                me.lifetime.nanosec = int(0.1 * 1e9)
                # self.contact_pub.publish(mr)
                self.contact_pub.publish(me)
        except Exception as e:
            self.get_logger().warn(f'clearance failed, {e}')

def main():
    rclpy.init()
    node = CollisionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
