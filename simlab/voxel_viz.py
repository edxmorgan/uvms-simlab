#!/usr/bin/env python3
# Copyright (C) 2025 Edward Morgan
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Visualize shared static and dynamic collision surfaces in their current poses."""
import os
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener

from simlab.dynamic_world import DynamicWorldModel
from simlab.shutdown import shutdown_node, spin_until_shutdown
from simlab.utils.meshes import collect_env_meshes, points_to_cloud2, se3_from_rpy_xyz
from simlab.voxel_geometry import LocalVoxelCache, transform_centers


class VoxelVizNode(Node):
    def __init__(self):
        super().__init__('voxel_viz_node')
        self.declare_parameter('robot_description', '')
        self.declare_parameter('world_frame', 'world')
        self.world_frame = str(self.get_parameter('world_frame').value)
        urdf = self.get_parameter('robot_description').value
        if not urdf:
            raise RuntimeError('robot_description is empty')
        # Same environment selection and local mesh origins as FCLWorld.
        _, self.static_meshes, _ = collect_env_meshes(urdf)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.dynamic_world = DynamicWorldModel(
            self, world_frame=self.world_frame, robot_radius_provider=lambda: 0.0)
        self.voxel_size = 0.1
        cache_dir = Path(os.environ.get('ROS_HOME', str(Path.home() / '.ros'))) / 'collision_voxels'
        # A large bathymetry build must not starve dynamic obstacle voxelization.
        self.caches = {name: LocalVoxelCache(cache_dir, self.voxel_size) for name in ('static', 'dynamic')}
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.cloud_pub = self.create_publisher(PointCloud2, '/env_voxels_cloud', qos)
        self.dynamic_pub = self.create_publisher(PointCloud2, '/dynamic_voxels_cloud', qos)
        self._signatures = {}
        self._errors = {}
        self.timer = self.create_timer(0.1, self.tick)
        self.get_logger().info('Shared collision voxels build in background at 0.1 m')

    @staticmethod
    def _static_spec(info):
        return (4, (), info['uri'], tuple(info['scale']))

    @staticmethod
    def _dynamic_spec(state):
        return (state.collision_type, state.collision_dimensions,
                state.collision_mesh_resource, state.collision_mesh_scale)

    def _static_sources(self):
        sources = []
        for info in self.static_meshes:
            tf = self.tf_buffer.lookup_transform(self.world_frame, info['link'], rclpy.time.Time())
            q, p = tf.transform.rotation, tf.transform.translation
            link = np.eye(4)
            link[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            link[:3, 3] = [p.x, p.y, p.z]
            transform = link @ se3_from_rpy_xyz(info['rpy'], info['xyz'])
            sources.append((self._static_spec(info), transform[:3, :3], transform[:3, 3]))
        return sources

    def _dynamic_sources(self):
        if self.dynamic_world.error:
            raise RuntimeError(self.dynamic_world.error)
        return [(self._dynamic_spec(state), state.obstacle_to_world_rotation, state.center_world)
                for state in self.dynamic_world.obstacles.values()]

    def _publish_sources(self, name, publisher, sources):
        ready = []
        pending = False
        for spec, rotation, translation in sources:
            points = self.caches[name].request(spec)
            if points is None:
                pending = True
            else:
                ready.append((points, rotation, translation))
        if pending:
            raise RuntimeError('building local collision voxels')
        signature = tuple((spec, np.asarray(rotation).tobytes(), np.asarray(translation).tobytes())
                          for spec, rotation, translation in sources)
        if self._signatures.get(name) == signature:
            return
        points = np.concatenate([transform_centers(*item) for item in ready]) if ready else np.empty((0, 3))
        # Zero stamp: retained world-frame data remains valid for late subscribers.
        publisher.publish(points_to_cloud2(points, frame_id=self.world_frame))
        self._signatures[name] = signature

    def tick(self):
        if not rclpy.ok():
            return
        active_specs = {
            'static': {self._static_spec(info) for info in self.static_meshes},
            'dynamic': {self._dynamic_spec(s) for s in self.dynamic_world.obstacles.values()},
        }
        for name, publisher, get_sources in (
                ('static', self.cloud_pub, self._static_sources),
                ('dynamic', self.dynamic_pub, self._dynamic_sources)):
            try:
                self._publish_sources(name, publisher, get_sources())
                self._errors.pop(name, None)
            except Exception as exc:
                error = str(exc)
                if self._errors.get(name) != error:
                    self.get_logger().warn(f'{name} voxel visualization unavailable: {error}')
                    publisher.publish(points_to_cloud2(np.empty((0, 3)), frame_id=self.world_frame))
                    self._errors[name] = error
                self._signatures.pop(name, None)
        for name, cache in self.caches.items():
            cache.retain(active_specs[name])

    def destroy_node(self):
        for cache in self.caches.values():
            cache.close()
        self.dynamic_world.close()
        self.tf_listener.unregister()
        return super().destroy_node()


def main():
    rclpy.init()
    node = VoxelVizNode()
    try:
        spin_until_shutdown(node)
    finally:
        shutdown_node(node)


if __name__ == '__main__':
    main()
