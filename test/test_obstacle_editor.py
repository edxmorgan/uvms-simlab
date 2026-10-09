import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import trimesh
from bringup.obstacle_description import obstacle_from_config
from interactive_markers.menu_handler import MenuHandler
from simlab.msg import ObstacleScene

from simlab.obstacle_editor import ObstacleEditor, load_mesh_catalog


def test_mesh_catalog_and_scale_keep_visual_collision_alignment(tmp_path):
    mesh = tmp_path / 'animal.stl'
    trimesh.creation.box().export(mesh)
    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps({'Animal': {'collision_mesh_resource': mesh.as_uri(),
        'collision_mesh_scale': [2., 1., .5], 'visual_dimensions': [4., 2., 1.]}}))
    catalog = load_mesh_catalog(path)
    editor = ObstacleEditor.__new__(ObstacleEditor)
    editor.draft = copy.deepcopy(catalog['Animal'])
    editor.draft.pose.position.x = 7.
    editor.draft.twist.linear.y = .2
    editor._render = lambda: None
    editor._mesh(catalog['Animal'])
    assert editor.draft.pose.position.x == 7.
    assert editor.draft.twist.linear.y == .2
    editor._size(1.)
    assert list(editor.draft.collision_mesh_scale) == [1., .5, .25]
    assert list(editor.draft.visual_dimensions) == [2., 1., .5]
    assert list(catalog['Animal'].collision_mesh_scale) == [2., 1., .5]


@pytest.mark.parametrize('content', ['[]', '{"bad": 3}', '{"": {}}', '{"missing": {}}'])
def test_invalid_catalog_is_rejected(tmp_path, content):
    path = tmp_path / 'catalog.json'
    path.write_text(content)
    with pytest.raises(ValueError):
        load_mesh_catalog(path)


def test_no_catalog_is_valid():
    assert load_mesh_catalog('') == {}


def test_reset_orientation_preserves_placement_and_motion(selection_editor):
    editor = selection_editor
    editor.draft.pose.position.z = -3.
    editor.draft.twist.linear.x = .2
    editor.draft.pose.orientation.x = 1.
    editor.draft.pose.orientation.w = 0.
    editor._reset_orientation()
    assert editor.draft.pose.orientation.w == 1.
    assert editor.draft.pose.orientation.x == 0.
    assert editor.draft.pose.position.z == -3.
    assert editor.draft.twist.linear.x == .2


def test_marine_preview_uses_embedded_materials(selection_editor):
    from ament_index_python.packages import get_package_share_directory
    from pathlib import Path
    catalog = load_mesh_catalog(Path(get_package_share_directory('simlab')) / 'obstacle_meshes/catalog.json')
    editor = selection_editor
    editor._mesh(catalog['Marine mammals / Blue Whale (2 m)'])
    marker = editor.server.marker.controls[0].markers[0]
    assert marker.mesh_use_embedded_materials
    assert marker.mesh_resource.endswith('.dae')
    assert [marker.color.r, marker.color.g, marker.color.b, marker.color.a] == [0.]*4


@pytest.fixture
def selection_editor(monkeypatch):
    class Server:
        def insert(self, marker, **kwargs):
            self.marker = copy.deepcopy(marker)

        def get(self, name):
            return copy.deepcopy(self.marker)

        def applyChanges(self):
            pass

    monkeypatch.setattr('simlab.obstacle_editor.list_world_profiles', lambda: [])
    node = Mock()
    node.has_parameter.return_value = True
    node.get_parameter.side_effect = lambda name: SimpleNamespace(value={
        'obstacle_preview_size': .8, 'obstacle_mesh_catalog': ''}[name])
    editor = ObstacleEditor(node, Server(), 'world')
    scene = ObstacleScene()
    scene.obstacles.obstacles = [obstacle_from_config(
        {'id': name, 'type': 'sphere', 'dimensions': [1.]}, i)
        for i, name in enumerate(('first', 'second'))]
    editor._snapshot(scene)
    return editor


def selection_titles(editor):
    handles = set(editor._selection_handles.values())
    return [entry.title for entry in editor.server.marker.menu_entries if entry.id in handles]


def test_selected_obstacle_checkbox_matches_robot_menu(selection_editor):
    editor = selection_editor
    assert selection_titles(editor) == ['[ ] first', '[ ] second']
    editor._select('first')
    assert selection_titles(editor) == ['[x] first', '[ ] second']
    editor._select('second')
    assert selection_titles(editor) == ['[ ] first', '[x] second']
    assert editor.draft.id == 'second'
    editor._reload_catalog()  # Rebuilding the menu must retain selection.
    assert selection_titles(editor) == ['[ ] first', '[x] second']
    assert editor.menu.getCheckState(editor._selection_handles['second']) == MenuHandler.CHECKED


@pytest.mark.parametrize('clear_all', [False, True])
def test_deleted_selection_does_not_leave_stale_checkbox(selection_editor, clear_all):
    editor = selection_editor
    editor._select('second')
    scene = ObstacleScene()
    if not clear_all:
        scene.obstacles.obstacles = [editor.items['first']]
    editor._snapshot(scene)
    assert editor.selected is None
    assert selection_titles(editor) == ([] if clear_all else ['[ ] first'])
    editor._select('second')  # Late feedback from a menu opened before deletion.
    assert editor.selected is None


def menu_titles(editor, parent=None):
    return [entry.title for entry in editor.server.marker.menu_entries
            if parent is None or entry.parent_id == parent]


def test_context_actions_and_new_preview(selection_editor):
    editor = selection_editor
    assert menu_titles(editor, 0)[:3] == ['Select obstacle', 'Obstacle settings (preview)', 'Add obstacle']
    assert 'Update selected obstacle' not in menu_titles(editor)
    editor._select('first')
    assert 'Add obstacle' not in menu_titles(editor)
    assert 'Update selected obstacle' in menu_titles(editor)
    editor.draft.pose.position.x = 12.
    editor._new()
    assert editor.selected is None
    assert editor.draft.pose.position.x == 12.
    assert '[x] New obstacle (preview)' in menu_titles(editor)
    assert selection_titles(editor) == ['[ ] first', '[ ] second']
    assert editor.items['first'].pose.position.x == 0.


def test_settings_checkboxes_follow_preview(selection_editor):
    editor = selection_editor
    assert '[x] Sphere' in menu_titles(editor)
    editor._velocity(.25)
    editor._shape('cylinder')
    assert editor.draft.twist.linear.x == .25  # Shape edits retain velocity.
    assert '[x] Cylinder' in menu_titles(editor)
    assert '[ ] Sphere' in menu_titles(editor)
    assert '[x] +X 0.25 m/s' in menu_titles(editor)
    assert 'Radius / height (m)' in menu_titles(editor)
    editor._size(.5)
    assert '[x] 0.5 / 1' in menu_titles(editor)
    assert list(editor.draft.collision_dimensions) == [.5, 1.]


def test_custom_values_do_not_claim_to_match_presets(selection_editor):
    editor = selection_editor
    editor.draft = obstacle_from_config({'type': 'box', 'dimensions': [1., 2., 3.],
                                        'linear_velocity': [0., .25, 0.]}, 0)
    editor._render()
    entries = editor.server.marker.menu_entries
    for title in ('Cube side length (m)', 'Initial velocity (world frame)'):
        parent = next(e.id for e in entries if e.title == title)
        assert all(t.startswith('[ ]') for t in menu_titles(editor, parent))


def test_delete_requires_confirmation_and_preserves_operation(selection_editor):
    from visualization_msgs.msg import InteractiveMarkerFeedback
    editor = selection_editor
    editor._select('first')
    editor._send = Mock()
    entries = editor.server.marker.menu_entries
    parent = next(e.id for e in entries if e.title == 'Delete selected obstacle')
    child = next(e for e in entries if e.parent_id == parent)
    assert child.title == 'Confirm delete'
    feedback = InteractiveMarkerFeedback(marker_name=editor.name,
        event_type=InteractiveMarkerFeedback.MENU_SELECT, menu_entry_id=parent)
    editor.menu.processFeedback(feedback)
    editor._send.assert_not_called()
    feedback.menu_entry_id = child.id
    editor.menu.processFeedback(feedback)
    editor._send.assert_called_once_with('remove')


@pytest.mark.parametrize('kind,dimensions,radius', [
    ('sphere', [8.], 8.), ('box', [6., 8., 0.2], (100.04 ** .5) / 2),
    ('cylinder', [3., 8.], 5.)])
def test_large_preview_is_draggable_with_exposed_axes(selection_editor, kind, dimensions, radius):
    from visualization_msgs.msg import InteractiveMarkerControl, InteractiveMarkerFeedback
    editor = selection_editor
    editor.draft = obstacle_from_config({'type': kind, 'dimensions': dimensions}, 0)
    editor._render()
    marker = editor.server.marker
    assert marker.scale > 2 * radius
    body = marker.controls[0]
    assert body.interaction_mode == InteractiveMarkerControl.MOVE_PLANE
    assert body.orientation_mode == InteractiveMarkerControl.VIEW_FACING
    assert body.independent_marker_orientation
    assert len(marker.controls) == 7
    assert marker.menu_entries  # The same right-click menu is still attached.
    feedback = InteractiveMarkerFeedback(event_type=InteractiveMarkerFeedback.POSE_UPDATE)
    feedback.pose.position.x = 10.
    editor._feedback(feedback)
    assert editor.draft.pose.position.x == 10.


def test_handles_enclose_visual_mesh_not_smaller_collision_proxy(selection_editor, tmp_path):
    mesh = trimesh.creation.box(extents=[2., 2., 2.])
    mesh.apply_translation([20., 0., 0.])
    path = tmp_path / 'offset_mesh.stl'
    mesh.export(path)
    editor = selection_editor
    editor.draft = obstacle_from_config({'type': 'sphere', 'dimensions': [.5],
        'visual_type': 'mesh', 'visual_mesh_resource': path.as_uri(),
        'visual_dimensions': [2., 1., 1.]}, 0)
    editor._render()
    assert editor.server.marker.scale > 84.
