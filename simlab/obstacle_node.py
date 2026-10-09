"""Authoritative ROS adapter: one selected source, one motion integrator."""
import copy
import json

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from bringup.dynamic_obstacle_sim import DynamicObstacleTransport
from simlab.dynamic_obstacle_sources import obstacle_behavior_source_class
from simlab.obstacle_runtime import ObstacleRuntime, ObstacleRevisionConflict
from simlab.msg import ObstacleScene
from simlab.srv import EditDynamicObstacles


class ObstacleNode(DynamicObstacleTransport):
    def __init__(self):
        super().__init__()
        self.declare_parameter('dynamic_obstacle_source', 'scripted_motion')
        self.declare_parameter('source_config', '{}')
        self.declare_parameter('experiment_seed', 0)
        self.source_name = self.get_parameter('dynamic_obstacle_source').value
        source = obstacle_behavior_source_class(self.source_name)()
        self.runtime = ObstacleRuntime(source, self.obstacles,
            config=json.loads(self.get_parameter('source_config').value),
            seed=self.get_parameter('experiment_seed').value)
        self.fault = ''
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.scene_publisher = self.create_publisher(ObstacleScene, '/obstacle_scene', qos)
        self.edit_service = self.create_service(EditDynamicObstacles,
            '~/edit_dynamic_obstacles', self._edit)
        self.get_logger().info(f'Obstacle behavior source: {self.source_name}')
        self._publish_snapshot()

    def _scene(self):
        revision, obstacles = self.runtime.snapshot()
        if self.fault:
            # Collision and camera consumers predict from the published twist.
            # A held scene must not advertise continued obstacle motion.
            for obstacle in obstacles.obstacles:
                obstacle.twist = Twist()
        obstacles.header.stamp = self.get_clock().now().to_msg()
        return ObstacleScene(revision=revision, source=self.source_name,
                             fault=self.fault, obstacles=obstacles)

    def _publish_snapshot(self):
        state = self._scene()
        self.obstacles = state.obstacles
        super()._publish_snapshot()
        self.scene_publisher.publish(state)

    def _advance(self):
        now = self.get_clock().now()
        dt = max(0., (now - self._last_time).nanoseconds * 1e-9)
        self._last_time = now
        if self.fault:
            return
        try:
            self.runtime.step(dt)
        except Exception as exc:
            self.fault = str(exc)
            self.get_logger().error(f'Obstacle source fault; holding last scene until reset: {exc}')

    def _publish_callback(self):
        self._advance()
        self._publish_snapshot()

    def _edit(self, request, response):
        try:
            revision, current = self.runtime.snapshot()
            operation = request.operation.strip().lower()
            if request.check_revision and request.expected_revision != revision:
                raise ObstacleRevisionConflict(f'expected revision {request.expected_revision}, current {revision}')
            if operation == 'get':
                pass
            elif operation == 'reset':
                self._validate_obstacles_do_not_overlap_robots(self.runtime.initial_snapshot().obstacles)
                self.runtime.reset()
                self.fault = ''
                self._last_time = self.get_clock().now()
            else:
                items = copy.deepcopy(request.obstacles.obstacles)
                if request.obstacles.header.frame_id not in ('', self.world_frame):
                    raise ValueError(f'expected frame {self.world_frame}')
                existing = {o.id for o in current.obstacles}
                names = {o.id for o in items}
                remove = []
                if operation == 'add':
                    if existing & names:
                        raise ValueError('add requires new obstacle IDs')
                elif operation == 'update':
                    if names - existing:
                        raise ValueError('update requires existing obstacle IDs')
                elif operation == 'remove':
                    if items:
                        raise ValueError('remove accepts IDs only')
                    remove = list(request.ids)
                elif operation == 'clear':
                    if items or request.ids:
                        raise ValueError('clear accepts no obstacle payload')
                    remove = list(existing)
                elif operation == 'replace':
                    remove = list(existing - names)
                else:
                    raise ValueError(f'unknown obstacle operation {operation!r}')
                if operation in ('add', 'update', 'replace'):
                    if request.ids:
                        raise ValueError('use obstacle messages, not IDs, for this operation')
                    if items:
                        self._validate_obstacles_do_not_overlap_robots(items)
                self.runtime.edit(expected_revision=revision, upsert=items, remove=remove)
            response.success = True
            response.message = f'{operation} completed'
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        response.scene = self._scene()
        self._publish_snapshot()
        return response

    def destroy_node(self):
        if hasattr(self, 'runtime'):
            self.runtime.close()
        return super().destroy_node()


def main():
    rclpy.init()
    node = None
    try:
        node = ObstacleNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
