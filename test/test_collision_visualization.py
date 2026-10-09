"""Check published visualization payloads against real collision results."""
from types import SimpleNamespace

import numpy as np
import pytest
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker

from simlab.collision_contact import CollisionNode
from simlab.voxel_viz import VoxelVizNode
from test_dynamic_collision_geometry import fcl_world, model, obstacle, update


@pytest.mark.parametrize('kind,dimensions', [(1, (1.,)), (2, (2., 2., 2.)), (3, (1., 2.))])
def test_contact_and_nearest_markers_follow_dynamic_scene(monkeypatch, kind, dimensions):
    monkeypatch.setattr('simlab.collision_contact.rclpy.ok', lambda: True)
    node = CollisionNode.__new__(CollisionNode)
    node.world_frame = 'world'
    node.world = fcl_world()
    node.world.update_from_tf = lambda *args: True
    node.dynamic_world = model()
    node.tf_buf = None
    node._last_error = None
    messages = []
    node.contact_pub = SimpleNamespace(publish=messages.append)
    item = obstacle(kind, dimensions)
    update(node.dynamic_world, item)
    node.tick()
    by_name = {m.ns: m for m in messages}
    assert messages[0].action == Marker.DELETEALL
    assert {'nearest_robot', 'nearest_env', 'clearance_line'} <= by_name.keys()
    assert 'dynamic/moving' in by_name['collision_status'].text
    line = by_name['clearance_line']
    a, b = line.points
    assert np.linalg.norm([a.x-b.x, a.y-b.y, a.z-b.z]) == pytest.approx(2.5)
    messages.clear()
    item.pose.position.x = 1.25
    update(node.dynamic_world, item)
    node.tick()
    assert any(m.ns.startswith('contact/') for m in messages)
    assert not any(m.ns == 'nearest_env' for m in messages)
    assert any(m.text.startswith('CONTACT:') for m in messages)
    assert all(m.lifetime.sec == 1 for m in messages[1:])


@pytest.mark.parametrize('layer', ['static', 'dynamic'])
def test_voxel_cloud_pose_updates_and_removal(layer):
    node = VoxelVizNode.__new__(VoxelVizNode)
    node.world_frame = 'world'
    node._signatures = {}
    node.caches = {layer: SimpleNamespace(request=lambda spec: np.array([[1., 0., 0.]]))}
    messages = []
    publisher = SimpleNamespace(publish=messages.append)
    source = ((1, (1.,), '', ()), np.eye(3), np.array([2., 3., 4.]))
    node._publish_sources(layer, publisher, [source])
    cloud = messages[-1]
    assert cloud.header.frame_id == 'world'
    assert cloud.header.stamp.sec == 0
    points = point_cloud2.read_points_numpy(cloud, field_names=('x', 'y', 'z'))
    np.testing.assert_allclose(points, [[3., 3., 4.]])
    node._publish_sources(layer, publisher, [source])
    assert len(messages) == 1  # Retained data; no unchanged full-cloud traffic.
    node._publish_sources(layer, publisher, [])
    assert messages[-1].width == 0  # Deleted objects must disappear.
