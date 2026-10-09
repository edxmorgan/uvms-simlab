"""ROS adapter regression tests on a separate DDS domain; no robot commands."""
import time
from unittest.mock import Mock

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from bringup.obstacle_description import obstacle_from_config
from simlab.obstacle_node import ObstacleNode
from simlab.obstacle_editor import ObstacleEditor
from simlab.msg import ObstacleScene
from simlab.srv import EditDynamicObstacles


@pytest.fixture
def node(monkeypatch):
    monkeypatch.setenv('ROS_DOMAIN_ID', '95')
    rclpy.init(args=['--ros-args', '-p', 'reject_robot_overlaps:=false'])
    instance = ObstacleNode()
    try:
        yield instance
    finally:
        instance.destroy_node()
        rclpy.shutdown()


def edit(node, operation, *items, **kwargs):
    request = EditDynamicObstacles.Request(operation=operation, **kwargs)
    request.obstacles.header.frame_id = 'world'
    request.obstacles.obstacles = list(items)
    return node._edit(request, EditDynamicObstacles.Response())


def item(name='one'):
    return obstacle_from_config({'id': name, 'type': 'sphere', 'dimensions': [0.8],
                                 'position': [2., 0., -1.]}, 0)


def test_atomic_operations_and_no_path_requirement(node):
    assert edit(node, 'add', item()).success
    assert not edit(node, 'add', item()).success
    assert not edit(node, 'update', item('missing')).success
    assert edit(node, 'add', item('two')).success
    change = item()
    change.pose.position.y = 3.
    assert edit(node, 'update', change).success
    result = edit(node, 'get')
    assert len(result.scene.obstacles.obstacles) == 2
    assert result.scene.obstacles.obstacles[0].pose.position.y == 3.
    assert edit(node, 'remove', ids=['one']).success
    assert edit(node, 'clear').success
    assert not edit(node, 'get').scene.obstacles.obstacles


def test_stale_revision_and_invalid_mesh_preserve_world(node):
    first = edit(node, 'add', item())
    assert not edit(node, 'clear', check_revision=True, expected_revision=0).success
    invalid = item('mesh')
    invalid.collision_type = 4
    invalid.collision_mesh_resource = '/missing/mesh.obj'
    invalid.collision_mesh_scale = [1., 1., 1.]
    assert not edit(node, 'add', invalid).success
    assert edit(node, 'get').scene.obstacles.obstacles == first.scene.obstacles.obstacles
    assert edit(node, 'get').scene.revision == first.scene.revision


def test_replace_and_reset_use_typed_edit_service(node):
    result = edit(node, 'replace', item())
    assert result.success
    assert node.runtime.snapshot()[1].obstacles[0].id == 'one'
    assert edit(node, 'reset').success
    assert not node.runtime.snapshot()[1].obstacles
    services = dict(node.get_service_names_and_types())
    assert '/dynamic_obstacle_sim_node/set_dynamic_obstacles' not in services


def test_timer_calls_only_source_integration(node):
    obstacle = item()
    obstacle.twist.linear.x = 2.
    assert edit(node, 'add', obstacle).success
    now = node.get_clock().now()
    from rclpy.duration import Duration
    node._last_time = now - Duration(seconds=0.5)
    node._publish_callback()
    x = node.runtime.snapshot()[1].obstacles[0].pose.position.x
    assert 2.99 < x < 3.1
    assert not hasattr(node, '_integrate_obstacles')


def test_fault_is_published_and_reset_recovers(node):
    moving = item()
    moving.twist.linear.x = 1.
    assert edit(node, 'add', moving).success
    original = node.runtime._source.step
    def fail(*args, **kwargs):
        raise ValueError('policy failure')
    node.runtime._source.step = fail
    node._publish_callback()
    state = edit(node, 'get').scene
    assert 'policy failure' in state.fault
    assert len(state.obstacles.obstacles) == 1
    assert state.obstacles.obstacles[0].twist.linear.x == 0.
    node.runtime._source.step = original
    assert edit(node, 'reset').success
    node._publish_callback()
    assert not edit(node, 'get').scene.fault


def test_overlap_rejection_preserves_scene(node):
    assert edit(node, 'add', item()).success
    node.reject_robot_overlaps = True
    node.robot_base_frames = ['robot_test']
    node._current_robot_positions = lambda: {'robot_test': (2., 0., -1.)}
    result = edit(node, 'add', item('overlap'))
    assert not result.success
    assert 'overlaps' in result.message
    assert [o.id for o in result.scene.obstacles.obstacles] == ['one']


def test_mesh_marker_resource_is_uri(node, tmp_path):
    import trimesh
    path = tmp_path / 'mesh.obj'
    trimesh.creation.box().export(path)
    mesh = obstacle_from_config({'id': 'mesh', 'type': 'mesh',
        'collision_mesh_resource': str(path), 'collision_mesh_scale': [2., 3., 4.]}, 0)
    assert edit(node, 'add', mesh).success
    assert node._marker_mesh_resource(mesh) == path.as_uri()
    from visualization_msgs.msg import Marker
    marker = Marker()
    node._set_marker_scale(marker, mesh)
    assert (marker.scale.x, marker.scale.y, marker.scale.z) == (2., 3., 4.)


def test_real_service_and_late_scene_subscriber(node):
    client_node = rclpy.create_node('obstacle_test_client')
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(client_node)
    def until(predicate):
        deadline = time.monotonic() + 5.
        while not predicate() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
        assert predicate()
    try:
        client = client_node.create_client(EditDynamicObstacles,
            '/dynamic_obstacle_sim_node/edit_dynamic_obstacles')
        assert client.wait_for_service(timeout_sec=3.)
        request = EditDynamicObstacles.Request(operation='add')
        request.obstacles.obstacles = [item()]
        future = client.call_async(request)
        until(future.done)
        assert future.result().success
        messages = []
        sub = client_node.create_subscription(ObstacleScene, '/obstacle_scene', messages.append,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        until(lambda: bool(messages) and bool(messages[-1].obstacles.obstacles))
        assert messages[-1].source == 'scripted_motion'
        assert messages[-1].obstacles.obstacles[0].id == 'one'
    finally:
        executor.shutdown()
        client_node.destroy_node()


def test_editor_add_sends_only_draft_not_existing_scene():
    editor = ObstacleEditor.__new__(ObstacleEditor)
    editor.node = Mock()
    editor.client = Mock()
    editor.client.service_is_ready.return_value = True
    editor.frame = 'world'
    editor.selected = None
    editor.draft = item()
    editor.items = {'existing': item('existing')}
    editor._send('add')
    request = editor.client.call_async.call_args.args[0]
    assert request.operation == 'add'
    assert len(request.obstacles.obstacles) == 1
    assert request.obstacles.obstacles[0].pose == editor.draft.pose
    assert request.obstacles.obstacles[0].id != 'existing'
    assert editor.draft.id == 'one'


def test_editor_registers_real_interactive_marker_without_robot(node):
    from interactive_markers.interactive_marker_server import InteractiveMarkerServer
    server = InteractiveMarkerServer(node, 'obstacle_editor_test')
    editor = ObstacleEditor(node, server, 'world')
    marker = server.get('obstacle_editor')
    assert marker is not None
    assert marker.header.frame_id == 'world'
    assert len(marker.controls) == 7
    titles = {entry.title for entry in marker.menu_entries}
    assert {'Marine mammals', 'Sharks', 'Fish', 'Rays',
            '[ ] Blue Whale (2 m)', '[ ] Bluefin Tuna (1 m)'} <= titles
    assert {'Replanning', 'Enable', 'Disable', 'Scene / source',
            'Reset source and scene', 'Load world profile (replaces scene)',
            'Clear all obstacles', 'Select obstacle', 'Obstacle settings (preview)',
            'Add obstacle'} <= titles
    editor._shape('box')
    editor._size(2.)
    assert list(editor.draft.collision_dimensions) == [2., 2., 2.]
    server.clear()
    server.applyChanges()


@pytest.mark.parametrize('command', ['enable_dynamic_replanning',
    'disable_dynamic_replanning', 'dynamic_replanning_status', 'set_world_profile'])
def test_obstacle_menu_routes_world_commands(command):
    editor = ObstacleEditor.__new__(ObstacleEditor)
    editor.world_client = Mock()
    editor.world_client.service_is_ready.return_value = True
    editor._world_command(command, 'profile' if command == 'set_world_profile' else '')
    request = editor.world_client.call_async.call_args.args[0]
    assert request.command == command
    assert request.name == ('profile' if command == 'set_world_profile' else '')


def test_robot_menu_has_no_obstacle_controls():
    import inspect
    import simlab.interactive_control as interactive
    source = inspect.getsource(interactive)
    assert '"Dynamic Obstacles"' not in source
    assert 'clear_dynamic_obstacles_handle' not in source
    assert 'enable_dynamic_replanning_handle' not in source
