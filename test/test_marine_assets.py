"""Offline checks for shipped assets, not network-dependent download tests."""
import hashlib
import io
import json
from pathlib import Path
import runpy
import struct
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest.mock import Mock

import fcl
import numpy as np
import pytest
import trimesh
from ament_index_python.packages import get_package_share_directory
from bringup.collision_geometry import collision_geometry, collision_surface_mesh, resolve_collision_mesh
from bringup.obstacle_description import visual_geometry
from simlab.obstacle_editor import load_mesh_catalog
from bringup.obstacle_description import obstacle_from_config
from bringup.dynamic_obstacle_sim import DynamicObstacleTransport
from bringup.sim_camera_renderer import read_triangle_mesh, open3d_to_polydata
from ros2_control_blue_reach_5.msg import DynamicObstacleArray


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'resource' / 'obstacle_meshes' / 'catalog.json'
ENTRIES = json.loads(CATALOG.read_text())


@pytest.mark.parametrize('name', sorted(ENTRIES))
def test_installed_animal_geometry_and_display_agree(name):
    config = ENTRIES[name]
    uri = config['collision_mesh_resource']
    scale = tuple(config['collision_mesh_scale'])
    mesh = collision_surface_mesh(4, (), uri, scale)
    assert len(mesh.faces) > 100
    assert np.isfinite(mesh.vertices).all()
    np.testing.assert_allclose(mesh.bounds.mean(axis=0), 0., atol=1e-5)
    expected = 1. if name.startswith('Fish / ') else 2.
    assert max(mesh.extents) == pytest.approx(expected, abs=1e-5)
    geometry, radius = collision_geometry(4, (), uri, scale)
    assert isinstance(geometry, fcl.BVHModel)
    assert radius > 0
    target = fcl.CollisionObject(geometry)
    probe = fcl.CollisionObject(fcl.Sphere(.01), fcl.Transform(mesh.vertices[0]))
    assert fcl.collide(probe, target, fcl.CollisionRequest(), fcl.CollisionResult()) > 0
    visual = collision_surface_mesh(4, (), config['visual_mesh_resource'], scale)
    # Compare complete triangle coordinates, not merely similar bounding boxes.
    def ordered_vertices(surface):
        vertices = np.round(surface.triangles.reshape(-1, 3), 5)
        return vertices[np.lexsort(vertices.T[::-1])]
    np.testing.assert_allclose(ordered_vertices(mesh), ordered_vertices(visual), atol=2e-5)
    root = ET.parse(resolve_collision_mesh(config['visual_mesh_resource'])).getroot()
    ns = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}
    assert root.find('c:asset/c:up_axis', ns).text == 'Z_UP'
    colours = {e.text for e in root.findall('.//c:diffuse/c:color', ns)}
    assert len(colours) >= 2


def test_default_catalog_is_installed_and_provenance_matches():
    directory = Path(get_package_share_directory('simlab')) / 'obstacle_meshes'
    assert json.loads((directory / 'catalog.json').read_text()) == ENTRIES
    catalog = load_mesh_catalog(directory / 'catalog.json')
    assert len(catalog) == 38
    for item in catalog.values():
        kind, dimensions, resource = visual_geometry(item)
        assert kind == item.collision_type == 4
        assert resource.endswith('.dae')
        assert item.collision_mesh_resource.endswith('.stl')
        assert dimensions == tuple(item.collision_mesh_scale)
    for entry in json.loads((directory / 'imported.json').read_text()):
        path = resolve_collision_mesh(f'package://simlab/obstacle_meshes/marine/{entry["name"]}.stl')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['stl_sha256']
        assert hashlib.sha256(path.with_suffix('.dae').read_bytes()).hexdigest() == entry['visual_sha256']
    assert (directory / 'LICENSE.txt').is_file()


def test_marine_markers_use_original_materials_not_orange():
    obstacle = obstacle_from_config({'id': 'whale', 'type': 'mesh', **ENTRIES['Marine mammals / Blue Whale (2 m)']}, 0)
    node = SimpleNamespace(obstacles=DynamicObstacleArray(obstacles=[obstacle]),
        marker_publisher=Mock(), _last_marker_count=0,
        _marker_type=DynamicObstacleTransport._marker_type,
        _set_marker_scale=DynamicObstacleTransport._set_marker_scale,
        _marker_mesh_resource=DynamicObstacleTransport._marker_mesh_resource)
    DynamicObstacleTransport._publish_markers(node)
    marker = node.marker_publisher.publish.call_args.args[0].markers[0]
    assert marker.mesh_use_embedded_materials
    assert marker.mesh_resource.endswith('.dae')
    assert [marker.color.r, marker.color.g, marker.color.b, marker.color.a] == [0.]*4
    assert marker.pose == obstacle.pose


def test_camera_preserves_marine_material_colours():
    path = ROOT / 'resource/obstacle_meshes/marine/blue-whale.dae'
    mesh = read_triangle_mesh(str(path), 0, preserve_materials=True)
    assert mesh.has_vertex_colors()
    colours = np.unique(np.asarray(mesh.vertex_colors), axis=0)
    assert len(colours) == 3
    assert np.ptp(colours[:, 2]) > .5
    poly = open3d_to_polydata(mesh)
    np.testing.assert_allclose(poly.point_data['asset_rgb'], mesh.vertex_colors)


def test_quantized_positions_are_decoded_before_import():
    decode = runpy.run_path(str(ROOT / 'tools' / 'import_marine_assets.py'))['decode_normalized_glb']
    positions = np.array([[0, 0, 0], [32767, 0, 0], [0, 32767, 0]], dtype='<i2')
    binary = positions.tobytes() + b'\0\0'
    document = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0]}],
        'nodes': [{'mesh': 0}], 'meshes': [{'primitives': [{'attributes': {'POSITION': 0}}]}],
        'buffers': [{'byteLength': len(binary)}],
        'bufferViews': [{'buffer': 0, 'byteOffset': 0, 'byteLength': positions.nbytes}],
        'accessors': [{'bufferView': 0, 'componentType': 5122, 'count': 3,
                       'type': 'VEC3', 'normalized': True, 'min': [0, 0, 0], 'max': [32767, 32767, 0]}]}
    encoded = json.dumps(document).encode()
    encoded += b' ' * (-len(encoded) % 4)
    glb = (struct.pack('<III', 0x46546C67, 2, 28+len(encoded)+len(binary))
           + struct.pack('<II', len(encoded), 0x4E4F534A) + encoded
           + struct.pack('<II', len(binary), 0x004E4942) + binary)
    scene = trimesh.load(io.BytesIO(decode(glb)), file_type='glb')
    np.testing.assert_allclose(scene.bounds, [[0, 0, 0], [1, 1, 0]])
