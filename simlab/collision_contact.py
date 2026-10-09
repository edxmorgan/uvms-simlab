#!/usr/bin/env python3
# collision_contact.py
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from simlab.dynamic_world import DynamicWorldModel
from simlab.utils.meshes import make_marker, color, collect_env_meshes
from simlab.fcl_checker import FCLWorld
from simlab.shutdown import shutdown_node, spin_until_shutdown


class CollisionNode(Node):
    def __init__(self):
        super().__init__('mesh_collision_node')
        self.declare_parameter('robot_description', '')
        self.declare_parameter('world_frame', 'world')

        urdf_string = self.get_parameter('robot_description').get_parameter_value().string_value
        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        if not urdf_string:
            self.get_logger().error('robot_description param is empty. Did you load it into the param server in launch')
            raise RuntimeError('no robot_description')

        robot_links, env_links, _ = collect_env_meshes(urdf_string)
        self.get_logger().info(f'robot_links { [x["link"] for x in robot_links] }')
        self.get_logger().info(f'env_links { [x["link"] for x in env_links] }')

        self.world = FCLWorld(urdf_string=urdf_string, world_frame=self.world_frame, vehicle_radius=0.4)
        self.dynamic_world = DynamicWorldModel(
            self, world_frame=self.world_frame,
            robot_radius_provider=lambda: self.world.vehicle_radius,
        )
        self._last_error = None

        # TF and pub
        self.tf_buf = Buffer()
        self.tf = TransformListener(self.tf_buf, self)
        self.contact_pub = self.create_publisher(Marker, 'contact_markers', 100)

        # 20 Hz
        self.timer = self.create_timer(0.05, self.tick)

    def tick(self):
        if not rclpy.ok():
            return
        # clear old markers
        clear = Marker()
        clear.header.frame_id = self.world_frame
        clear.action = Marker.DELETEALL
        try:
            self.contact_pub.publish(clear)
        except Exception:
            if rclpy.ok():
                raise
            return

        try:
            if self.dynamic_world.error:
                raise RuntimeError(self.dynamic_world.error)
            self.world.set_dynamic_bodies(self.dynamic_world.bodies)
            if not self.world.update_from_tf(self.tf_buf, rclpy.time.Time()):
                raise RuntimeError(f"missing TF: {', '.join(self.world.missing_frames)}")
            pairs = self.world.robot_env_contacts_one_point_per_pair()
            for idx, (pair, point) in enumerate(pairs.items()):
                marker = make_marker('contact/' + ' -> '.join(pair), idx,
                                     self.world_frame, 0.05, point, color(1, 0.1, 0.1, 1))
                self._publish_marker(marker)
            resp = self.world.global_clearance()
            if resp is None:
                raise RuntimeError('no valid nearest-point result for loaded geometry')
            distance, p_robot, p_env = resp
            pair = ' -> '.join(self.world.last_clearance_pair)
            if distance > 0 and not pairs:
                self._publish_marker(make_marker('nearest_robot', 1001, self.world_frame, 0.05, p_robot, color(0.1, 0.1, 0.95, 1)))
                self._publish_marker(make_marker('nearest_env', 1002, self.world_frame, 0.05, p_env, color(0.1, 0.95, 0.1, 1)))
                line = make_marker('clearance_line', 1003, self.world_frame, 0.01, [0, 0, 0], color(0.1, 0.95, 0.1, 1))
                line.type = Marker.LINE_LIST
                line.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in (p_robot, p_env)]
                self._publish_marker(line)
            contact_pair = ' -> '.join(next(iter(pairs))) if pairs else pair
            text = f'CONTACT: {contact_pair}' if pairs or distance <= 0 else f'{distance:.3f} m: {pair}'
            self._status(text, p_env, failed=bool(pairs or distance <= 0))
            self._last_error = None
        except Exception as e:
            if rclpy.ok():
                self._status(f'Collision information unavailable: {e}', [0, 0, 0], failed=True)
                if str(e) != self._last_error:
                    self.get_logger().warn(f'Collision information unavailable: {e}')
                    self._last_error = str(e)

    def _publish_marker(self, marker):
        marker.lifetime.sec = 1
        self.contact_pub.publish(marker)

    def _status(self, text, position, *, failed):
        marker = make_marker('collision_status', 1004, self.world_frame, 0.12,
                             position, color(1, 0.2, 0.1, 1) if failed else color(1, 1, 1, 1))
        marker.type = Marker.TEXT_VIEW_FACING
        marker.pose.position.z += 0.15
        marker.text = text
        self._publish_marker(marker)

    def destroy_node(self):
        self.dynamic_world.close()
        self.tf.unregister()
        return super().destroy_node()

def main():
    rclpy.init()
    node = CollisionNode()
    try:
        spin_until_shutdown(node)
    finally:
        shutdown_node(node)

if __name__ == '__main__':
    main()
