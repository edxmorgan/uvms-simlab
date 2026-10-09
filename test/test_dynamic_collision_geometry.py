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
    def create_subscription(self, *args):
        return object()

    def get_logger(self):
        return SimpleNamespace(error=lambda message: None)


def model():
    return DynamicWorldModel(Node(), world_frame='world', robot_radius_provider=lambda: 0.5)


def obstacle(kind=1, dimensions=(1.0,), x=4.0, name='moving'):
    msg = DynamicObstacle()
    msg.id = name
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
    node = SimpleNamespace(world=world, dynamic_world=dynamic, world_frame='world',
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
