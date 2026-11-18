"""Vehicle task admission and execution use the same sphere as navigation."""
from types import SimpleNamespace
from unittest.mock import Mock
import fcl
import numpy as np
import pytest
from simlab.fcl_checker import FCLWorld
from simlab.planner_world import PlannerWorld
from simlab.uvms_backend import UVMSBackendCore


def world():
    w = FCLWorld.__new__(FCLWorld)
    w.robot_prefixes = {'robot_1_'}
    w.bodies_robot = []
    box = fcl.Box(.02, 2., 2.)
    w.bodies_env = [dict(name='wall', fcl_obj=fcl.CollisionObject(box), geom=box)]
    w.bodies_dynamic = []
    w.manager_env = fcl.DynamicAABBTreeCollisionManager()
    w.manager_env.registerObjects([w.bodies_env[0]['fcl_obj']]); w.manager_env.setup()
    w.env_xyz_bounds = (-5., 5., -5., 5., -5., 5.)
    w.set_robot_collision_radius(.574)
    return w


def test_overlap_distance_and_named_start_error():
    w = world()
    assert w.min_distance_xyz([0., 0., 0.]) == 0.
    assert 'wall' in PlannerWorld(fcl_world=w).invalid_position_reason([0., 0., 0.])
    assert w.min_distance_xyz([2., 0., 0.]) > 1.


def test_task_translation_cannot_cross_wall_with_clear_endpoints():
    p = PlannerWorld(fcl_world=world())
    with pytest.raises(ValueError, match='translation'):
        p.require_clear_segment([-2., 0., 0.], [2., 0., 0.])
    p.require_clear_segment([-2., -2., 0.], [2., -2., 0.])
    with pytest.raises(ValueError, match='target.*bounds'):
        p.require_clear_segment([-2., -2., 0.], [20., -2., 0.])


@pytest.mark.parametrize('planner_type', ['Bitstar', 'RRTstar', 'RRTConnect'])
def test_all_planners_reject_actual_overlap_before_search(planner_type):
    from simlab.motion_planning.planners.ompl import OmplPlanner
    w = world(); p = PlannerWorld(fcl_world=w)
    planner = OmplPlanner(SimpleNamespace(planner_world=p, get_logger=lambda: Mock()),
                          env_bounds=w.env_xyz_bounds)
    planner.ss = Mock(wraps=planner.ss)
    result = planner.plan_se3_path([0., 0., 0.], [1., 0., 0., 0.],
        [2., 0., 0.], [1., 0., 0., 0.], planner_type=planner_type)
    assert not result.is_success and 'wall' in result.message
    planner.ss.solve.assert_not_called()
    result = planner.plan_se3_path([-8., 0., 0.], [1., 0., 0., 0.],
        [2., 0., 0.], [1., 0., 0., 0.], planner_type=planner_type)
    assert not result.is_success and 'bounds' in result.message
    planner.ss.solve.assert_not_called()


def test_logged_ancient_ship_pose_is_blocked_and_reset_pose_is_clear():
    from pathlib import Path
    import xacro
    from builtin_interfaces.msg import Time
    from ament_index_python.packages import get_package_share_directory
    from bringup.scene_runtime import environment_transforms
    root = Path(get_package_share_directory('ros2_control_blue_reach_5'))
    doc = xacro.parse(f'''<robot name="scene" xmlns:xacro="http://www.ros.org/wiki/xacro">
      <xacro:include filename="{root}/xacro/world.xacro"/>
      <xacro:sim_world world_frame="world" scene_file="{root}/scenes/ancient_ship/world.xacro"/>
    </robot>''')
    xacro.process_doc(doc)
    description = doc.toxml()
    w = FCLWorld(description, vehicle_radius=.574)
    transforms = {t.child_frame_id:t.transform for t in environment_transforms(description, Time())}
    for body in w.bodies_env:
        t = transforms[body['frame']]; p, q = t.translation, t.rotation
        body['fcl_obj'].setTransform(fcl.Transform([q.w,q.x,q.y,q.z], [p.x,p.y,p.z]))
    w.manager_env.update()
    p = PlannerWorld(fcl_world=w)
    assert 'bathymetry_ship_1' in p.invalid_position_reason([-5.052,-.817,-3.195])
    # Default map-origin reset expressed in world, from the robot map transform.
    assert p.invalid_position_reason([-2.4516342946321807,-.4820780440595218,0.], safety_margin=.01) == ''


def test_boundary_tracking_allowance_only_applies_to_start_not_target_or_obstacles():
    w = world(); w.env_xyz_bounds = (-5., 5., -5., 5., -5., 0.)
    p = PlannerWorld(fcl_world=w)
    p.require_clear_segment([-2., 0., .001], [-2., 0., -.01], start_bounds_tolerance=.01)
    with pytest.raises(ValueError, match='start.*bounds'):
        p.require_clear_segment([-2., 0., .02], [-2., 0., -.01], start_bounds_tolerance=.01)
    with pytest.raises(ValueError, match='target.*bounds'):
        p.require_clear_segment([-2., 0., 0.], [-2., 0., .001], start_bounds_tolerance=.01)
    with pytest.raises(ValueError, match='start.*intersects'):
        p.require_clear_segment([0., 0., .001], [-2., 0., -.01], start_bounds_tolerance=.01)


def test_task_solver_bounds_follow_current_scene_and_reserve_tracking_allowance():
    w = world()
    backend = SimpleNamespace(fcl_world=w, task_boundary_tracking_tolerance=.01)
    np.testing.assert_allclose(UVMSBackendCore.task_vehicle_bounds(backend),
                               [[-4.99,4.99]]*3)
    w.env_xyz_bounds = (-2., 3., -4., 5., -8., 0.)
    np.testing.assert_allclose(UVMSBackendCore.task_vehicle_bounds(backend),
                               [[-1.99,2.99],[-3.99,4.99],[-7.99,-.01]])




def test_zero_margin_timed_curve_still_rejects_real_static_overlap():
    w = world()
    assert w.min_distance_xyz([0., 0., 0.]) == 0.
    assert w.planner_in_collision_at_xyz([0., 0., 0.])
    with pytest.raises(ValueError, match='intersects static geometry'):
        PlannerWorld(fcl_world=w).require_clear_timed_trajectory(
            [(0., [0., 0., 0.])], max_speed=0., safety_margin=0.)
    PlannerWorld(fcl_world=w).require_clear_timed_trajectory(
        [(0., [2., 0., 0.])], max_speed=0., safety_margin=0.)
