from types import SimpleNamespace

import fcl
import numpy as np
import pytest
import trimesh

from bringup.collision_geometry import collision_geometry
from ros2_control_blue_reach_5.msg import DynamicObstacle, DynamicObstacleArray
from simlab.dynamic_world import DynamicWorldModel
from simlab.fcl_checker import FCLWorld


class Node:
    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=0))

    def create_subscription(self, *args):
        return object()

    def get_logger(self):
        return SimpleNamespace(error=lambda message: None)


def model():
    return DynamicWorldModel(Node(), world_frame='world', robot_radius_provider=lambda: 0.5)


def obstacle(kind=1, dimensions=(1.0,), x=4.0, name='moving'):
    msg = DynamicObstacle()
    msg.id = name
    msg.source = 'scripted'
    msg.collision_type = kind
    msg.collision_dimensions = list(dimensions)
    msg.pose.position.x = x
    msg.pose.orientation.w = 1.0
    return msg


def update(world, *obstacles, frame='world'):
    msg = DynamicObstacleArray()
    msg.header.frame_id = frame
    msg.obstacles = list(obstacles)
    world.update_from_msg(msg)


def fcl_world():
    world = FCLWorld.__new__(FCLWorld)
    world.bodies_robot = []
    geom = fcl.Sphere(0.5)
    world.bodies_robot = [{'name': 'robot', 'geom': geom, 'fcl_obj': fcl.CollisionObject(geom)}]
    world.bodies_env = []
    world.bodies_dynamic = []
    world.manager_env = fcl.DynamicAABBTreeCollisionManager()
    world.manager_env.setup()
    return world


@pytest.mark.parametrize('kind,dimensions', [(1, (1.,)), (2, (2., 2., 2.)), (3, (1., 2.))])
def test_primitive_clearance_contacts_and_removal(kind, dimensions):
    dynamic = model()
    moving = obstacle(kind, dimensions)
    update(dynamic, moving)
    assert dynamic.error is None
    world = fcl_world()
    world.set_dynamic_bodies(dynamic.bodies)
    result = world.global_clearance()
    assert result[0] == pytest.approx(2.5, abs=1e-5)
    assert np.linalg.norm(result[1] - result[2]) == pytest.approx(result[0], abs=1e-5)
    assert world.last_clearance_pair == ('robot', 'dynamic/moving')
    assert not world.robot_env_contacts_one_point_per_pair()
    assert dynamic.min_clearance_xyz([0, 0, 0]).distance_m == pytest.approx(2.5)
    moving.pose.position.x = 1.25
    update(dynamic, moving)
    world.set_dynamic_bodies(dynamic.bodies)
    assert ('robot', 'dynamic/moving') in world.robot_env_contacts_one_point_per_pair()
    update(dynamic)
    world.set_dynamic_bodies(dynamic.bodies)
    assert world.global_clearance() is None
    assert not world.robot_env_contacts_one_point_per_pair()


def test_rotated_box_matches_primitive_distance():
    dynamic = model()
    moving = obstacle(2, (2., 4., 2.), x=5.)
    moving.pose.orientation.z = np.sin(np.pi / 4)
    moving.pose.orientation.w = np.cos(np.pi / 4)
    update(dynamic, moving)
    world = fcl_world()
    world.set_dynamic_bodies(dynamic.bodies)
    assert world.global_clearance()[0] == pytest.approx(2.5, abs=1e-5)
    assert dynamic.min_clearance_xyz([0, 0, 0]).distance_m == pytest.approx(2.5)


def test_real_mesh_scale_and_prediction_do_not_move_live_object(tmp_path):
    path = tmp_path / 'box.stl'
    trimesh.creation.box(extents=[2, 2, 2]).export(path)
    moving = obstacle(4, (), x=5.)
    moving.collision_mesh_resource = path.as_uri()
    moving.collision_mesh_scale = [2., 1., 1.]
    moving.twist.linear.x = -1.
    dynamic = model()
    update(dynamic, moving)
    assert dynamic.error is None
    world = fcl_world()
    world.set_dynamic_bodies(dynamic.bodies)
    assert world.global_clearance()[0] == pytest.approx(2.5, abs=1e-5)
    assert dynamic.min_clearance_xyz([0, 0, 0], t_offset=1.).distance_m == pytest.approx(1.5, abs=1e-5)
    _, robot_point, env_point = world.global_clearance()
    assert robot_point[0] == pytest.approx(0.5, abs=1e-5)
    assert env_point[0] == pytest.approx(3., abs=1e-5)
    assert world.global_clearance()[0] == pytest.approx(2.5, abs=1e-5)
    assert dynamic.min_clearance_xyz([0, 0, 0], t_offset=3.).distance_m < 0
    moving.pose.position.x = 2.
    update(dynamic, moving)
    world.set_dynamic_bodies(dynamic.bodies)
    assert ('robot', 'dynamic/moving') in world.robot_env_contacts_one_point_per_pair()


def test_replacement_and_shared_geometry_preserve_obstacle_identity():
    dynamic = model()
    update(dynamic, obstacle(x=1.), obstacle(x=-1., name='other'))
    world = fcl_world()
    world.set_dynamic_bodies(dynamic.bodies)
    assert set(world.robot_env_contacts_one_point_per_pair()) == {('robot', 'dynamic/moving'), ('robot', 'dynamic/other')}
    update(dynamic, obstacle(2, (1., 1., 1.), x=5.))
    world.set_dynamic_bodies(dynamic.bodies)
    assert not world.robot_env_contacts_one_point_per_pair()
    assert world.global_clearance()[0] == pytest.approx(4., abs=1e-5)


@pytest.mark.parametrize('bad', [obstacle(1, (-1.,)), obstacle(4, (1.,)), obstacle(99, (1.,)), obstacle(x=float('nan'))])
def test_invalid_geometry_fails_closed_and_recovers(bad):
    dynamic = model()
    update(dynamic, bad)
    assert dynamic.error
    assert dynamic.min_clearance_xyz([0, 0, 0]).distance_m == float('-inf')
    update(dynamic, obstacle())
    assert dynamic.error is None


def test_missing_snapshot_wrong_frame_and_duplicate_ids_fail_closed():
    dynamic = model()
    assert dynamic.min_clearance_xyz([0, 0, 0]).distance_m == float('-inf')
    update(dynamic, obstacle(), frame='wrong')
    assert dynamic.error
    update(dynamic, obstacle(), obstacle())
    assert dynamic.error
    update(dynamic)
    assert dynamic.error is None
    assert dynamic.min_clearance_xyz([0, 0, 0]) is None


def test_missing_mesh_is_not_a_bounding_proxy():
    with pytest.raises(ValueError, match='does not exist'):
        collision_geometry(4, (10.,), '/nonexistent/collision_mesh.stl', (1., 1., 1.))


@pytest.mark.parametrize('kind,dimensions', [(1, (1.,)), (2, (2., 2., 2.)), (3, (1., 2.)), (4, ())])
def test_mesh_robot_against_all_dynamic_geometries(tmp_path, kind, dimensions):
    path = tmp_path / 'box.stl'
    trimesh.creation.box(extents=[2, 2, 2]).export(path)
    geometry, _ = collision_geometry(4, (), str(path), (1., 1., 1.))
    world = fcl_world()
    world.bodies_robot = [{'name': 'robot', 'geom': geometry,
                           'fcl_obj': fcl.CollisionObject(geometry, fcl.Transform(np.array([10., 0., 0.])))}]
    moving = obstacle(kind, dimensions, x=14.)
    if kind == 4:
        moving.collision_mesh_resource = str(path)
        moving.collision_mesh_scale = [1., 1., 1.]
    dynamic = model()
    update(dynamic, moving)
    world.set_dynamic_bodies(dynamic.bodies)
    distance, robot_point, env_point = world.global_clearance()
    assert distance == pytest.approx(2., abs=1e-5)
    assert robot_point[0] == pytest.approx(11., abs=1e-5)
    assert env_point[0] == pytest.approx(13., abs=1e-5)
    moving.pose.position.x = 11.5
    update(dynamic, moving)
    world.set_dynamic_bodies(dynamic.bodies)
    assert ('robot', 'dynamic/moving') in world.robot_env_contacts_one_point_per_pair()


def test_static_and_dynamic_geometry_share_nearest_query():
    world = fcl_world()
    geometry = fcl.Sphere(1.)
    world.bodies_env = [{'name': 'bathymetry', 'geom': geometry,
                         'fcl_obj': fcl.CollisionObject(geometry, fcl.Transform(np.array([10., 0., 0.])))}]
    dynamic = model()
    update(dynamic, obstacle(x=4.))
    world.set_dynamic_bodies(dynamic.bodies)
    assert world.global_clearance()[0] == pytest.approx(2.5)
    assert world.last_clearance_pair[1] == 'dynamic/moving'
    update(dynamic)
    world.set_dynamic_bodies(dynamic.bodies)
    assert world.global_clearance()[0] == pytest.approx(8.5)
    assert world.last_clearance_pair[1] == 'bathymetry'


def test_mesh_profile_producers_agree(tmp_path):
    from bringup.obstacle_description import obstacle_from_config
    from simlab.world_profiles import dynamic_obstacles_from_world_profile
    path = tmp_path / 'box.stl'
    trimesh.creation.box().export(path)
    config = {'id': 'mesh', 'type': 'mesh', 'collision_mesh_resource': str(path),
              'collision_mesh_scale': [2., 3., 4.]}
    first = obstacle_from_config(config, 0)
    second = dynamic_obstacles_from_world_profile({'obstacles': [config]}, 'world').obstacles[0]
    for msg in (first, second):
        assert msg.collision_mesh_resource == str(path)
        assert list(msg.collision_mesh_scale) == [2., 3., 4.]
        assert msg.visual_mesh_resource == str(path)
        assert list(msg.visual_dimensions) == [2., 3., 4.]


def test_contact_visualization_publishes_pair_and_diagnostics(monkeypatch):
    from simlab.collision_contact import CollisionNode
    from visualization_msgs.msg import Marker
    from simlab import collision_contact
    monkeypatch.setattr(collision_contact.rclpy, 'ok', lambda: True)
    world = fcl_world()
    dynamic = model()
    update(dynamic, obstacle())
    world.update_from_tf = lambda *_: True
    markers = []
    node = SimpleNamespace(scene_participant=SimpleNamespace(held=False),
                           world=world, dynamic_world=dynamic, world_frame='world',
                           tf_buf=None, contact_pub=SimpleNamespace(publish=markers.append),
                           _last_error=None, get_logger=lambda: SimpleNamespace(warn=lambda *_: None))
    node._publish_marker = lambda m: CollisionNode._publish_marker(node, m)
    node._status = lambda *args, **kwargs: CollisionNode._status(node, *args, **kwargs)
    CollisionNode.tick(node)
    assert markers[0].action == Marker.DELETEALL
    assert {'nearest_robot', 'nearest_env', 'clearance_line', 'collision_status'} <= {m.ns for m in markers}
    assert 'dynamic/moving' in markers[-1].text
    dynamic.error = 'bad mesh'
    markers.clear()
    CollisionNode.tick(node)
    assert len(markers) == 2
    assert 'unavailable' in markers[-1].text


def test_rotated_mesh_sphere_endpoints_are_in_world_frame(tmp_path):
    from bringup.collision_geometry import distance_with_nearest
    from scipy.spatial.transform import Rotation
    path = tmp_path / 'rotated.stl'
    trimesh.creation.box(extents=[2, 2, 2]).export(path)
    geometry, _ = collision_geometry(4, (), str(path), (1., 1., 1.))
    rotation = Rotation.from_euler('z', np.pi / 4).as_matrix()
    origin = np.array([10., 5., -3.])
    center = origin + rotation @ np.array([4., 0., 0.])
    mesh = {'geom': geometry, 'fcl_obj': fcl.CollisionObject(geometry, fcl.Transform(rotation, origin))}
    sphere_geometry = fcl.Sphere(1.)
    sphere = {'geom': sphere_geometry, 'fcl_obj': fcl.CollisionObject(sphere_geometry, fcl.Transform(center))}
    distance, p_mesh, p_sphere = distance_with_nearest(mesh, sphere)
    assert distance == pytest.approx(2., abs=1e-5)
    assert np.allclose(p_mesh, origin + rotation @ [1., 0., 0.], atol=1e-5)
    assert np.allclose(p_sphere, origin + rotation @ [3., 0., 0.], atol=1e-5)
    reverse = distance_with_nearest(sphere, mesh)
    assert np.allclose(reverse[1], p_sphere)
    assert np.allclose(reverse[2], p_mesh)


def test_cached_mesh_bvh_keeps_world_transforms_independent(tmp_path):
    from simlab.utils.meshes import fcl_bvh_from_mesh
    path = tmp_path / 'box.obj'
    trimesh.creation.box().export(path)
    first, _, _ = fcl_bvh_from_mesh(str(path), [1., 1., 1.])
    second, _, _ = fcl_bvh_from_mesh(str(path), [1., 1., 1.])
    assert first is second
    a = fcl.CollisionObject(first, fcl.Transform())
    b = fcl.CollisionObject(second, fcl.Transform())
    b.setTranslation(np.array([5., 0., 0.]))
    np.testing.assert_allclose(a.getTranslation(), [0., 0., 0.])
    assert fcl.distance(a, b) == pytest.approx(4.)
    shifted, _, _ = fcl_bvh_from_mesh(str(path), [1., 1., 1.], xyz=(2., 0., 0.))
    assert shifted is not first


def test_static_world_updates_only_when_transform_changes():
    from unittest.mock import Mock
    world = FCLWorld.__new__(FCLWorld)
    world.bodies_robot = []
    world.world_frame = 'world'
    world.bodies_robot = []
    body = {'frame': 'terrain', 'fcl_obj': Mock()}
    world.bodies_env = [body]
    world.manager_env = Mock()
    position = SimpleNamespace(x=0., y=0., z=0.)
    rotation = SimpleNamespace(w=1., x=0., y=0., z=0.)
    tf = SimpleNamespace(lookup_transform=lambda *args: SimpleNamespace(
        transform=SimpleNamespace(translation=position, rotation=rotation)))
    assert world.update_from_tf(tf, None)
    assert world.update_from_tf(tf, None)
    assert world.manager_env.update.call_count == 1
    position.z = -2.
    assert world.update_from_tf(tf, None)
    assert world.manager_env.update.call_count == 2
    tf.lookup_transform = Mock(side_effect=RuntimeError('TF unavailable'))
    assert not world.update_from_tf(tf, None)
    assert world.missing_frames == ['terrain']


def test_threshold_clearance_prunes_only_separated_bounds(monkeypatch):
    from scipy.spatial.transform import Rotation
    world = fcl_world()
    world.bodies_robot[0]['local_bounds'] = np.array([[-.5]*3, [.5]*3])
    geom = fcl.Box(2., 4., 2.)
    obj = fcl.CollisionObject(geom)
    world.bodies_env = [dict(name='box', geom=geom, fcl_obj=obj,
        local_bounds=np.array([[-1., -2., -1.], [1., 2., 1.]]))]
    for angle in (0., .3, 1.2):
        for x in (1., 2., 3., 100.):
            obj.setTransform(fcl.Transform(Rotation.from_euler('z', angle).as_matrix(), [x, 0., 0.]))
            assert world.has_clearance(.01) == (world.global_clearance()[0] > .01)
    def unexpected(*args):
        raise AssertionError('Distant pair reached narrow-phase query')
    monkeypatch.setattr(fcl, 'distance', unexpected)
    assert world.has_clearance(.01)


def test_scene_requires_declared_collision_mesh():
    from simlab.utils.meshes import collect_env_meshes
    description = '<robot name="test"><link name="bathymetry_test">' +         '<visual><geometry><mesh filename="visual.obj"/></geometry></visual>' +         '<collision><origin xyz="1 2 3"/><geometry><mesh filename="collision.obj"/></geometry></collision>' +         '</link></robot>'
    _, meshes, _ = collect_env_meshes(description)
    assert meshes[0]['uri'] == 'collision.obj'
    assert meshes[0]['xyz'] == [1., 2., 3.]
    import xml.etree.ElementTree as ET
    root = ET.fromstring(description); link = root.find('link')
    link.remove(link.find('collision'))
    with pytest.raises(ValueError, match='requires explicit collision geometry: bathymetry_test'):
        collect_env_meshes(ET.tostring(root, encoding='unicode'))


def test_world_and_temporary_geometry_release_without_cyclic_collection(tmp_path):
    import gc
    import weakref
    from simlab.utils.meshes import environment_mesh_bounds
    path = tmp_path / 'box.stl'
    trimesh.creation.box().export(path)
    description = f'<robot name="test"><link name="world"/><link name="bathymetry_test"><collision><geometry><mesh filename="{path}"/></geometry></collision></link></robot>'
    # Prime immutable geometry caches before measuring per-scene ownership.
    first = FCLWorld(description); del first
    gc.collect()
    enabled = gc.isenabled(); gc.disable()
    try:
        world = FCLWorld(description)
        np.testing.assert_allclose(world.min_coords, [-.5]*3)
        np.testing.assert_allclose(world.max_coords, [.5]*3)
        reference = weakref.ref(world)
        del world
        assert reference() is None
    finally:
        if enabled: gc.enable()


def test_scene_bounds_include_all_fixed_ancestors(tmp_path):
    from simlab.utils.meshes import collect_env_meshes, environment_mesh_bounds
    path = tmp_path / 'box.stl'; trimesh.creation.box().export(path)
    description = f'<robot name="test"><link name="world"/><link name="mount"/>' +         f'<link name="bathymetry_test"><collision><geometry><mesh filename="{path}"/></geometry></collision></link>' +         '<joint name="a" type="fixed"><parent link="world"/><child link="mount"/><origin xyz="10 0 0"/></joint>' +         '<joint name="b" type="fixed"><parent link="mount"/><child link="bathymetry_test"/><origin xyz="0 2 0"/></joint></robot>'
    _, meshes, _ = collect_env_meshes(description)
    lower, upper = environment_mesh_bounds(meshes)
    np.testing.assert_allclose(lower, [9.5, 1.5, -.5])
    np.testing.assert_allclose(upper, [10.5, 2.5, .5])


def test_unsupported_environment_geometry_is_not_silently_omitted():
    from simlab.utils.meshes import collect_env_meshes
    with pytest.raises(ValueError, match='must be a mesh'):
        collect_env_meshes('<robot name="test"><link name="bathymetry_box"><collision><geometry><box size="1 1 1"/></geometry></collision></link></robot>')


def test_environment_grid_covers_offset_scenes_with_bounded_line_count():
    from simlab.utils.geometry import environment_grid_markers
    for bounds in [(-25.,25.,-25.,25.,-8.,0.), (-25.,131.28,-286.22,323.12,-8.,0.),
                   (300.,301.,-400.,-399.,-3.5,0.)]:
        result = environment_grid_markers(bounds, 'world')
        assert len(result.markers) == 2
        for i, marker in enumerate(result.markers):
            points = np.array([[p.x,p.y,p.z] for p in marker.points])
            assert len(points) <= 804
            assert points[:,0].min() <= bounds[0] and points[:,0].max() >= bounds[1]
            assert points[:,1].min() <= bounds[2] and points[:,1].max() >= bounds[3]
            assert np.all(points[:,2] == (0. if i == 0 else bounds[4]))
            assert marker.id == i and marker.header.frame_id == 'world'
            assert marker.pose.orientation.w == 1.
        np.testing.assert_array_equal(
            [[p.x,p.y] for p in result.markers[0].points],
            [[p.x,p.y] for p in result.markers[1].points])


def test_shared_robot_snapshot_owner_filtering_and_contact_pairs():
    dynamic = model()
    a, b = obstacle(x=0., name='robot/one'), obstacle(x=2., name='robot/two')
    for obj, owner in [(a,'robot_1_'), (b,'robot_2_')]:
        obj.source = 'robot'; obj.owner_robot = owner; obj.velocity_valid = True
    update(dynamic, a, b)
    assert dynamic.external_bodies == []
    assert dynamic.min_clearance_xyz([0,0,0], exclude_owner='robot_1_').distance_m > 0
    assert dynamic.min_clearance_xyz([0,0,0], exclude_owner='robot_2_').distance_m < 0
    world = fcl_world()
    world.set_dynamic_bodies(dynamic.bodies)
    assert len(list(world.collision_pairs())) == 1
    b.pose.position.x = .5
    update(dynamic, a, b)
    world.set_dynamic_bodies(dynamic.bodies)
    assert len(world.robot_env_contacts_one_point_per_pair()) == 1
    assert not world.has_clearance(.01)
    assert len(dynamic.bodies_for('robot_1_')) == 1


def test_missing_robot_and_stale_snapshot_do_not_look_like_free_space(monkeypatch):
    import simlab.dynamic_world as module
    clock = [10.]
    monkeypatch.setattr(module.time, 'monotonic', lambda:clock[0])
    dynamic = model()
    msg = DynamicObstacleArray()
    msg.header.frame_id = 'world'
    msg.unavailable_robot_owners = ['robot_1_']
    dynamic.update_from_msg(msg)
    assert dynamic.min_clearance_xyz([0,0,0], exclude_owner='robot_2_').distance_m == -float('inf')
    assert dynamic.min_clearance_xyz([0,0,0], exclude_owner='robot_1_') is None
    clock[0] += 2.
    assert dynamic.min_clearance_xyz([0,0,0], exclude_owner='robot_1_').distance_m == -float('inf')


def test_prediction_accounts_for_sample_age_and_requires_robot_velocity(monkeypatch):
    import simlab.dynamic_world as module
    monkeypatch.setattr(module.time, 'monotonic', lambda:10.)
    dynamic = model()
    obj = obstacle(x=2., dimensions=(.2,), name='robot/other')
    obj.source='robot'; obj.owner_robot='robot_2_'; obj.velocity_valid=True
    obj.twist.linear.x = -1.
    obj.stamp.sec = 3
    msg = DynamicObstacleArray(obstacles=[obj])
    msg.header.frame_id='world'; msg.header.stamp.sec=3; msg.header.stamp.nanosec=200000000
    dynamic.update_from_msg(msg)
    assert dynamic.min_clearance_xyz([0,0,0], exclude_owner='robot_1_', t_offset=.3).distance_m == pytest.approx(.8)
    obj.velocity_valid=False
    msg.obstacles=[obj]; dynamic.update_from_msg(msg)
    assert dynamic.min_clearance_xyz([0,0,0], exclude_owner='robot_1_', t_offset=.3).distance_m == -float('inf')


def test_planning_snapshot_is_admitted_once_while_live_world_stays_fresh(monkeypatch):
    import simlab.dynamic_world as module
    clock = [10.]
    monkeypatch.setattr(module.time, 'monotonic', lambda:clock[0])
    dynamic = model()
    obj = obstacle(x=4., dimensions=(.2,))
    obj.twist.linear.x = -1.
    msg = DynamicObstacleArray(obstacles=[obj]); msg.header.frame_id='world'; msg.header.stamp.sec=5
    dynamic.update_from_msg(msg)
    snapshot = dynamic.snapshot()
    clearance = snapshot.min_clearance_xyz([0,0,0]).distance_m
    clock[0] += 2.
    assert dynamic.error == 'stale dynamic obstacle snapshot'
    assert snapshot.min_clearance_xyz([0,0,0]).distance_m == clearance
    nested = snapshot.snapshot()
    assert nested._snapshot_at == snapshot._snapshot_at
    assert nested.min_clearance_xyz([0,0,0]).distance_m == clearance
    assert nested.availability_error() is None
    obj.pose.position.x=1.
    msg.obstacles=[obj]; dynamic.update_from_msg(msg)
    assert dynamic.min_clearance_xyz([0,0,0]).distance_m != clearance
    assert snapshot.min_clearance_xyz([0,0,0]).distance_m == clearance


def test_retained_old_message_is_not_fresh_at_new_subscriber():
    dynamic = model()
    dynamic.node.get_clock = lambda: SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=10_000_000_000))
    msg = DynamicObstacleArray(); msg.header.frame_id='world'; msg.header.stamp.sec=2
    dynamic.update_from_msg(msg)
    assert dynamic.error == 'stale dynamic obstacle snapshot'
    assert dynamic.min_clearance_xyz([0,0,0]).distance_m == -float('inf')


def test_snapshot_requires_explicit_source_metadata():
    dynamic = model()
    moving = obstacle()
    moving.source = ''
    update(dynamic, moving)
    assert 'invalid obstacle ownership/source' in dynamic.error
    assert dynamic.min_clearance_xyz([0, 0, 0]).distance_m == -np.inf


@pytest.mark.parametrize('invalid', ['frame', 'id', 'quaternion'])
def test_snapshot_rejects_missing_pose_identity_metadata(invalid):
    dynamic = model()
    obj = obstacle()
    if invalid == 'id':
        obj.id = ' '
    elif invalid == 'quaternion':
        obj.pose.orientation.w = 0.
    update(dynamic, obj, frame='' if invalid == 'frame' else 'world')
    assert dynamic.error
    assert dynamic.min_clearance_xyz([0.,0.,0.]).distance_m == -np.inf


def test_sample_freshness_shared_with_visualization_and_owner_exclusion():
    from simlab.voxel_viz import VoxelVizNode
    dynamic = model()
    obj = obstacle(name='robot/robot_1_')
    obj.source = 'robot'; obj.owner_robot = 'robot_1_'; obj.stamp.sec = 1
    msg = DynamicObstacleArray(obstacles=[obj])
    msg.header.frame_id = 'world'; msg.header.stamp.sec = 5
    dynamic.update_from_msg(msg)
    assert 'stale obstacle sample' in dynamic.availability_error()
    assert dynamic.availability_error(exclude_owner='robot_1_') is None
    assert dynamic.min_clearance_xyz([0,0,0]).distance_m == -np.inf
    assert dynamic.min_clearance_xyz([0,0,0], exclude_owner='robot_1_') is None
    node = SimpleNamespace(dynamic_world=dynamic, _dynamic_spec=VoxelVizNode._dynamic_spec)
    with pytest.raises(RuntimeError, match='stale obstacle sample'):
        VoxelVizNode._dynamic_sources(node)




def test_inflated_sphere_matches_distance_threshold_and_radius_updates():
    world = fcl_world()
    world.vehicle_radius = .582
    world._planner_obj = fcl.CollisionObject(fcl.Sphere(world.vehicle_radius))
    obstacle = fcl.CollisionObject(fcl.Box(2., 2., 2.))
    world.manager_env.registerObjects([obstacle])
    world.manager_env.setup()
    for radius in (.582, .3):
        world.vehicle_radius = radius
        world._planner_obj = fcl.CollisionObject(fcl.Sphere(radius))
        for margin in (.01, .1):
            for gap in (-.02, .005, .015, .09, .11, 1.):
                point = [1. + radius + gap, 0., 0.]
                assert world.planner_has_clearance_at_xyz(point, margin) == (world.min_distance_xyz(point) >= margin)
