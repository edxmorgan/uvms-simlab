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
    node.scene_participant = SimpleNamespace(held=False)
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
    node.scene_participant = SimpleNamespace(held=False)
    node.world_frame = 'world'
    node._signatures = {}
    node.max_display_points = 250000
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


def test_contact_display_reports_unavailable_robot_snapshot(monkeypatch):
    from unittest.mock import Mock
    monkeypatch.setattr('simlab.collision_contact.rclpy.ok', lambda: True)
    messages = []
    node = SimpleNamespace(scene_participant=SimpleNamespace(held=False), world_frame='world',
        dynamic_world=SimpleNamespace(availability_error=lambda:'missing robot obstacle poses: robot_2_'),
        contact_pub=SimpleNamespace(publish=messages.append), _last_error=None,
        get_logger=lambda:Mock(), _status=Mock())
    CollisionNode.tick(node)
    assert messages[0].action == Marker.DELETEALL
    assert 'missing robot obstacle poses: robot_2_' in node._status.call_args.args[0]
    assert node._status.call_args.kwargs['failed']


def test_scene_switch_preserves_dynamic_obstacles_and_uses_detached_geometry(monkeypatch):
    from unittest.mock import Mock
    from simlab import environment_scene
    node = Mock()
    world = SimpleNamespace(vehicle_radius=.4, env_xyz_bounds=(-5.,5.,-5.,5.,-5.,0.))
    obstacles = [{'source':'robot'}, {'source':'scripted'}]
    backend = SimpleNamespace(node=node, world_frame='world', fcl_world=world, robots=[],
        dynamic_world=SimpleNamespace(bodies=obstacles), tf_buffer=Mock())
    build = Mock(return_value=world)
    monkeypatch.setattr(environment_scene, 'FCLWorld', build)
    environment_scene.attach_backend(backend)
    assert backend.scene_participant.prepare('<robot name="scene"/>') == (world, '<robot name="scene"/>')
    build.assert_called_once()
    assert backend.dynamic_world.bodies is obstacles
    assert backend.scene_participant.prepare_async


@pytest.mark.parametrize('simulated', [(), ('vehicle',), ('manipulator',), ('vehicle', 'manipulator')])
def test_scene_switch_does_not_reset_or_reauthorize_robots(monkeypatch, simulated):
    from unittest.mock import Mock
    from simlab import environment_scene
    node = Mock()
    world = SimpleNamespace(vehicle_radius=.4, env_xyz_bounds=(-5.,5.,-5.,5.,-5.,0.))
    robot = Mock()
    robot._active_controller_is_replay.return_value = False
    robot.k_robot = 0
    robot.simulated_components = simulated
    robot.get_state.return_value = {'pose':[1., 2., -1., .1, .2, .3]}
    backend = SimpleNamespace(node=node, world_frame='world', fcl_world=world, robots=[robot],
                              vehicle_waypoint_missions=[Mock()], tf_buffer=Mock())
    environment_scene.attach_backend(backend)
    backend.scene_participant.on_hold()
    assert robot.scene_switch_hold
    robot.reset_simulation_with_state.assert_not_called()
    robot.execution_gate.stop.assert_not_called()
    robot.execution_gate.authorize.assert_not_called()
    robot.abrupt_planner_stop.assert_not_called()
    assert robot.pose_command == [1., 2., -1., .1, .2, .3]
    backend.scene_participant.on_release()
    assert not robot.scene_switch_hold
    assert not backend.scene_transition


def test_scene_switch_rejects_teleop_before_changing_control(monkeypatch):
    from unittest.mock import Mock
    from simlab import environment_scene
    from simlab.robot import ControlMode
    robot = Mock(control_mode=ControlMode.TELEOP)
    backend = SimpleNamespace(node=Mock(), world_frame='world', robots=[robot],
        fcl_world=SimpleNamespace(env_xyz_bounds=(-5.,5.,-5.,5.,-5.,0.)))
    environment_scene.attach_backend(backend)
    with pytest.raises(ValueError, match='joystick'):
        backend.scene_participant.on_hold()
    assert not backend.scene_transition
    robot.hold_current_state_with_feedback.assert_not_called()
    robot._zero_teleop_commands.assert_not_called()
