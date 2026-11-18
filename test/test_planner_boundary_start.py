from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from simlab.planner_world import PlannerWorld
from simlab.motion_planning.planners.ompl import OmplPlanner

BOUNDS = (-23., 23., -15., 15., -8., 0.)


def world(collision=lambda p: False, clearance=lambda p: float('inf')):
    return PlannerWorld(fcl_world=SimpleNamespace(
        env_xyz_bounds=np.array(BOUNDS).reshape(3, 2), vehicle_radius=.574,
        planner_in_collision_at_xyz=collision,
        planner_collision_obstacle=lambda p: 'test obstacle', min_distance_xyz=clearance))


@pytest.mark.parametrize('height', [.0004075006765029586, .000887167611807459])
@pytest.mark.parametrize('planner_type', ['Bitstar', 'RRTConnect', 'RRTstar'])
def test_real_planners_preserve_measured_start_and_checked_boundary_connector(height, planner_type):
    w = world()
    node = SimpleNamespace(planner_world=w, get_logger=lambda: Mock())
    planner = OmplPlanner(node, safety_margin=.01, env_bounds=BOUNDS)
    start = np.array([-2.572, -1.540, height]); goal = [-2.554, -1.543, -1.126]
    result = planner.plan_se3_path(start, [1., 0., 0., 0.], goal, [1., 0., 0., 0.],
                                   time_limit=.05, planner_type=planner_type)
    assert result.is_success, result.message
    np.testing.assert_allclose(result.xyz[0], start)
    np.testing.assert_allclose(result.xyz[1], [start[0], start[1], 0.])
    np.testing.assert_allclose(result.xyz[-1], goal)
    assert np.all(result.xyz[1:, 2] <= 0.)
    assert 'checked boundary connector' in result.message


def test_boundary_allowance_rejects_large_errors_and_real_collisions():
    with pytest.raises(ValueError, match='outside environment bounds'):
        world().planning_start([0., 0., .011], BOUNDS, .01)
    with pytest.raises(ValueError, match='intersects'):
        world(collision=lambda p: True).planning_start([0., 0., .0008], BOUNDS, .01)
    with pytest.raises(ValueError, match='connector lacks'):
        world(clearance=lambda p: .0001).planning_start([0., 0., .0008], BOUNDS, .01)
    with pytest.raises(ValueError, match='nonfinite'):
        world().planning_start([0., 0., np.nan], BOUNDS, .01)
    with pytest.raises(ValueError, match='distance to environment bounds'):
        world().planning_start([23.008, 15.008, 0.], BOUNDS, .01)


def test_goal_bounds_remain_strict():
    node = SimpleNamespace(planner_world=world(), get_logger=lambda: Mock())
    planner = OmplPlanner(node, env_bounds=BOUNDS)
    result = planner.plan_se3_path([0., 0., -.1], [1., 0., 0., 0.],
                                   [0., 0., .0008], [1., 0., 0., 0.])
    assert not result.is_success
    assert 'Goal rejected' in result.message and 'outside environment bounds' in result.message


def test_large_scene_search_is_local_and_collision_resolution_is_physical():
    node = SimpleNamespace(planner_world=world(), get_logger=lambda: Mock())
    planner = OmplPlanner(node, env_bounds=(-110., 110., -110., 110., -6., 0.))
    result = planner.plan_se3_path([0., 0., -1.], [1., 0., 0., 0.],
                                   [1., 0., -1.], [1., 0., 0., 0.], time_limit=.1)
    assert result.is_success, result.message
    bounds = planner.space.getBounds()
    assert bounds.low[0] == -1. and bounds.high[0] == 2.
    assert planner.space.getLongestValidSegmentLength() <= .050001
    planner._set_search_bounds(planner.env_bounds)
    planner.space.setup()
    assert planner.space.getLongestValidSegmentLength() <= .050001


def test_failed_local_search_retries_full_workspace(monkeypatch):
    node = SimpleNamespace(planner_world=world(), get_logger=lambda: Mock())
    planner = OmplPlanner(node, env_bounds=BOUNDS)
    original = planner.ss
    calls = []
    class SetupProxy:
        def __getattr__(self, name):
            return getattr(original, name)
        def solve(self, duration):
            calls.append((duration, planner.space.getBounds().low[0]))
            if len(calls) == 1:
                return False
            return original.solve(duration)
    planner.ss = SetupProxy()
    result = planner.plan_se3_path([0., 0., -1.], [1., 0., 0., 0.],
                                   [1., 0., -1.], [1., 0., 0., 0.], time_limit=.1)
    assert result.is_success, result.message
    assert calls == [(.05, -1.), (.05, -23.)]
