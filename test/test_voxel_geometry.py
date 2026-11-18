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
    node = SimpleNamespace(max_display_points=250000, caches={'dynamic': cache}, _signatures={}, world_frame='base_link')
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
    node = SimpleNamespace(max_display_points=250000, caches={'dynamic': cache}, _signatures={}, world_frame='world')
    messages = []
    with pytest.raises(RuntimeError, match='building'):
        VoxelVizNode._publish_sources(node, 'dynamic', SimpleNamespace(publish=messages.append),
                                     [((1, (1.,), '', ()), np.eye(3), np.zeros(3))])
    assert messages == []


def _slow_voxel_worker(*args):
    time.sleep(30)


def test_shutdown_cancels_running_native_voxel_job(tmp_path, monkeypatch):
    import simlab.voxel_geometry as module
    monkeypatch.setattr(module, '_voxel_worker', _slow_voxel_worker)
    cache = LocalVoxelCache(tmp_path, 0.1)
    spec = (1, (1.,), '', ())
    cache.request(spec)
    time.sleep(.2)
    start = time.monotonic()
    cache.close()
    assert time.monotonic() - start < 3.
    assert cache.jobs[spec].done()
    with pytest.raises(RuntimeError, match='cancelled'):
        cache.jobs[spec].result()


def test_surface_rasterization_matches_triangle_box_collision():
    import fcl
    from simlab.voxel_geometry import surface_voxel_centers
    rng = np.random.default_rng(8)
    pitch = .2
    for triangle in [np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                     *rng.uniform(-.5, .5, (12, 3, 3))]:
        mesh = trimesh.Trimesh(vertices=triangle, faces=[[0, 1, 2]], process=False)
        actual = {tuple(x) for x in np.rint(surface_voxel_centers(mesh, pitch) / pitch).astype(int)}
        bvh = fcl.BVHModel(); bvh.beginModel(3, 1)
        bvh.addSubModel(triangle, np.array([[0, 1, 2]], dtype=np.int32)); bvh.endModel()
        obj = fcl.CollisionObject(bvh)
        lo = np.ceil(triangle.min(axis=0) / pitch - .5).astype(int)
        hi = np.floor(triangle.max(axis=0) / pitch + .5).astype(int)
        expected = set()
        from itertools import product
        for cell in product(*(range(a, b + 1) for a, b in zip(lo, hi))):
            cube = fcl.CollisionObject(# Include boundary contact despite FCL floating-point roundoff.
            fcl.Box(pitch + 1e-10, pitch + 1e-10, pitch + 1e-10),
                fcl.Transform(np.array(cell, dtype=float) * pitch))
            if fcl.collide(obj, cube, fcl.CollisionRequest(), fcl.CollisionResult()):
                expected.add(cell)
        assert actual == expected


def test_cloud_packing_preserves_xyz_for_noncontiguous_big_endian_arrays():
    import struct
    from simlab.utils.meshes import points_to_cloud2
    source = np.arange(30, dtype='>f8').reshape(10, 3)[::2]
    cloud = points_to_cloud2(source)
    expected = b''.join(struct.pack('<fff', *row) for row in source)
    assert bytes(cloud.data) == expected
    assert cloud.row_step == len(expected)
    assert not cloud.is_bigendian


def test_display_budget_preserves_small_objects_and_bounds_large_clouds():
    from simlab.voxel_geometry import display_point_subsets
    arrays = [np.arange(n * 3).reshape(n, 3) for n in (1000000, 5, 1000)]
    saved = [a.copy() for a in arrays]
    result = display_point_subsets(arrays, 205)
    assert [len(a) for a in result] == [100, 5, 100]
    assert result[1] is arrays[1]
    for source, subset in zip(arrays, result):
        np.testing.assert_array_equal(subset[0], source[0])
        np.testing.assert_array_equal(subset[-1], source[-1])
    for original, after in zip(saved, arrays):
        np.testing.assert_array_equal(original, after)
    assert display_point_subsets(arrays, 0) is arrays
    assert display_point_subsets(arrays, 2000000) is arrays
    assert sum(map(len, display_point_subsets(arrays, 2))) == 2
    assert display_point_subsets([], 100) == []
    with pytest.raises(ValueError):
        display_point_subsets(arrays, -1)


def test_voxels_use_one_shared_snapshot_for_robots_and_scripted_obstacles():
    from types import SimpleNamespace as NS
    from simlab.voxel_viz import VoxelVizNode
    robot = NS(source='robot', owner_robot='robot_1_', collision_type=1,
        collision_dimensions=(.35,), collision_mesh_resource='', collision_mesh_scale=(),
        obstacle_to_world_rotation=np.eye(3), center_world=np.array([1.,2.,3.]))
    scripted = NS(source='scripted', owner_robot='', collision_type=2,
        collision_dimensions=(1.,2.,3.), collision_mesh_resource='', collision_mesh_scale=(),
        obstacle_to_world_rotation=np.eye(3), center_world=np.array([4.,5.,6.]))
    # No robot description, TF or separate sphere generator is needed here.
    node = NS(dynamic_world=NS(availability_error=lambda:None, obstacles={'robot':robot,'box':scripted}),
              _dynamic_spec=VoxelVizNode._dynamic_spec)
    sources = VoxelVizNode._dynamic_sources(node)
    assert len(sources) == 2
    assert sources[0][0] == (1, (.35,), '', ())
    assert sources[1][0] == (2, (1.,2.,3.), '', ())
    np.testing.assert_array_equal(sources[0][2], robot.center_world)
    robot.center_world = np.array([7.,8.,9.])
    np.testing.assert_array_equal(VoxelVizNode._dynamic_sources(node)[0][2], [7.,8.,9.])
    node.dynamic_world.obstacles.pop('robot')
    assert len(VoxelVizNode._dynamic_sources(node)) == 1
    node.dynamic_world.availability_error = lambda:'missing robot obstacle poses: robot_1_'
    with pytest.raises(RuntimeError, match='missing robot obstacle poses'):
        VoxelVizNode._dynamic_sources(node)
