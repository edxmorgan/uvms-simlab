import time
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from bringup.collision_geometry import collision_surface_mesh
from simlab.voxel_geometry import LocalVoxelCache, transform_centers
from simlab.voxel_viz import VoxelVizNode


def wait_for(cache, spec):
    end = time.monotonic() + 10
    while time.monotonic() < end:
        points = cache.request(spec)
        if points is not None:
            return points
        time.sleep(0.01)
    raise AssertionError('voxel job did not finish')


@pytest.mark.parametrize('spec', [(1, (0.2,), '', ()), (2, (0.4, 0.6, 0.2), '', ()), (3, (0.2, 0.6), '', ())])
def test_primitive_collision_surfaces_have_local_voxels(tmp_path, spec):
    cache = LocalVoxelCache(tmp_path, 0.1)
    try:
        points = wait_for(cache, spec)
        assert len(points) > 0
        assert np.all(np.isfinite(points))
        bounds = collision_surface_mesh(*spec).bounds
        assert np.all(points >= bounds[0] - 0.1)
        assert np.all(points <= bounds[1] + 0.1)
        assert cache.request(spec) is points
    finally:
        cache.close()


def test_mesh_content_scale_and_resolution_are_cache_keys(tmp_path):
    path = tmp_path / 'mesh.stl'
    trimesh.creation.box(extents=[0.4, 0.4, 0.4]).export(path)
    directory = tmp_path / 'cache'
    spec = (4, (), str(path), (1., 1., 1.))
    cache = LocalVoxelCache(directory, 0.1)
    try:
        first = wait_for(cache, spec)
        stretched = wait_for(cache, (4, (), str(path), (2., 1., 1.)))
        assert stretched[:, 0].max() > first[:, 0].max()
        assert len(list(directory.glob('*.npy'))) == 2
        cache.retain([])
        assert not cache.jobs
    finally:
        cache.close()
    # A fresh process/asset reload must not reuse a stale resolution-only cache.
    trimesh.creation.box(extents=[0.8, 0.4, 0.4]).export(path)
    collision_surface_mesh.cache_clear()
    cache = LocalVoxelCache(directory, 0.2)
    try:
        changed = wait_for(cache, spec)
        assert changed[:, 0].max() > first[:, 0].max()
        assert len(list(directory.glob('*.npy'))) == 3
    finally:
        cache.close()


def test_missing_mesh_reports_failure(tmp_path):
    cache = LocalVoxelCache(tmp_path, 0.1)
    try:
        with pytest.raises(ValueError, match='does not exist'):
            wait_for(cache, (4, (), str(tmp_path / 'absent.stl'), (1., 1., 1.)))
    finally:
        cache.close()


def test_transform_does_not_change_local_cache():
    points = np.array([[1., 0., 0.]])
    points.setflags(write=False)
    result = transform_centers(points, Rotation.from_euler('z', np.pi / 2).as_matrix(), [10., 20., -3.])
    assert np.allclose(result, [[10., 21., -3.]])
    assert np.array_equal(points, [[1., 0., 0.]])


def test_static_transform_matches_fcl_link_and_mesh_origins():
    info = {'link': 'bathymetry_test', 'uri': '/mesh.stl', 'scale': [2., 2., 2.],
            'xyz': [1., 0., 0.], 'rpy': [0., 0., np.pi / 2]}
    q = Rotation.from_euler('z', np.pi / 2).as_quat()
    tf = SimpleNamespace(transform=SimpleNamespace(
        rotation=SimpleNamespace(x=q[0], y=q[1], z=q[2], w=q[3]),
        translation=SimpleNamespace(x=10., y=0., z=-8.)))
    node = SimpleNamespace(static_meshes=[info], world_frame='base_link',
                           tf_buffer=SimpleNamespace(lookup_transform=lambda *_: tf),
                           _static_spec=VoxelVizNode._static_spec)
    spec, rotation, translation = VoxelVizNode._static_sources(node)[0]
    assert spec[-1] == (2., 2., 2.)
    assert np.allclose(transform_centers(np.array([[2., 0., 0.]]), rotation, translation), [[8., 1., -8.]])


def test_cloud_pose_updates_and_deletion_without_revoxelization():
    points = np.array([[1., 0., 0.]])
    cache = SimpleNamespace(request=lambda _: points)
    node = SimpleNamespace(caches={'dynamic': cache}, _signatures={}, world_frame='base_link')
    messages = []
    publisher = SimpleNamespace(publish=messages.append)
    spec = (1, (1.,), '', ())
    sources = [(spec, np.eye(3), np.array([10., 0., 0.]))]
    VoxelVizNode._publish_sources(node, 'dynamic', publisher, sources)
    VoxelVizNode._publish_sources(node, 'dynamic', publisher, sources)
    assert len(messages) == 1  # unchanged retained cloud isn't resent
    assert messages[-1].header.frame_id == 'base_link'
    assert np.allclose(np.frombuffer(messages[-1].data, dtype='<f4'), [11., 0., 0.])
    sources[0] = (spec, np.eye(3), np.array([20., 0., 0.]))
    VoxelVizNode._publish_sources(node, 'dynamic', publisher, sources)
    assert np.allclose(np.frombuffer(messages[-1].data, dtype='<f4'), [21., 0., 0.])
    VoxelVizNode._publish_sources(node, 'dynamic', publisher, [])
    assert messages[-1].width == 0


def test_pending_jobs_do_not_publish_incomplete_cloud():
    cache = SimpleNamespace(request=lambda _: None)
    node = SimpleNamespace(caches={'dynamic': cache}, _signatures={}, world_frame='world')
    messages = []
    with pytest.raises(RuntimeError, match='building'):
        VoxelVizNode._publish_sources(node, 'dynamic', SimpleNamespace(publish=messages.append),
                                     [((1, (1.,), '', ()), np.eye(3), np.zeros(3))])
    assert messages == []
