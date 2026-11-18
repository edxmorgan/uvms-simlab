import numpy as np

from simlab.dynamic_world import DynamicClearance
from simlab.planner_world import PlannerWorld


class FakeFclWorld:
    def __init__(self, *, static_distance=1.0, in_collision=False):
        self.static_distance = float(static_distance)
        self.in_collision = bool(in_collision)

    def min_distance_xyz(self, xyz):
        np.asarray(xyz, dtype=float).reshape(3)
        return self.static_distance

    def planner_in_collision_at_xyz(self, xyz):
        np.asarray(xyz, dtype=float).reshape(3)
        return self.in_collision


class FakeDynamicWorld:
    def __init__(self, clearance=None):
        self.clearance = clearance
        self.last_t_offset = None

    def min_clearance_xyz(self, xyz, *, t_offset=0.0, exclude_owner=""):
        np.asarray(xyz, dtype=float).reshape(3)
        self.last_t_offset = float(t_offset)
        self.last_exclude_owner = exclude_owner
        if self.clearance is None:
            return None
        return DynamicClearance("moving", float(self.clearance))


def test_planner_world_uses_static_world_without_dynamic_obstacles():
    world = PlannerWorld(fcl_world=FakeFclWorld(static_distance=0.5))

    clearance = world.min_clearance_xyz([0.0, 0.0, -1.0])

    assert clearance.source == "static"
    assert clearance.distance_m == 0.5
    assert world.is_state_valid_xyz([0.0, 0.0, -1.0], safety_margin=0.25)
    assert not world.is_state_valid_xyz([0.0, 0.0, -1.0], safety_margin=0.75)


def test_planner_world_combines_static_and_dynamic_clearance():
    dynamic_world = FakeDynamicWorld(clearance=0.2)
    world = PlannerWorld(
        fcl_world=FakeFclWorld(static_distance=1.0),
        dynamic_world=dynamic_world,
    )

    clearance = world.min_clearance_xyz([0.0, 0.0, -1.0], t_offset=3.0)

    assert clearance.source == "dynamic"
    assert clearance.obstacle_id == "moving"
    assert clearance.distance_m == 0.2
    assert dynamic_world.last_t_offset == 3.0
    assert not world.is_state_valid_xyz([0.0, 0.0, -1.0], safety_margin=0.25, t_offset=3.0)


def test_planner_world_collision_checks_static_and_dynamic_worlds():
    assert PlannerWorld(fcl_world=FakeFclWorld(in_collision=True)).in_collision_xyz([0.0, 0.0, -1.0])

    dynamic_world = FakeDynamicWorld(clearance=-0.1)
    world = PlannerWorld(
        fcl_world=FakeFclWorld(static_distance=1.0, in_collision=False),
        dynamic_world=dynamic_world,
    )

    assert world.in_collision_xyz([0.0, 0.0, -1.0], t_offset=2.0)
    assert dynamic_world.last_t_offset == 2.0


def test_planner_world_treats_near_zero_dynamic_clearance_as_tangent_not_collision():
    world = PlannerWorld(
        fcl_world=FakeFclWorld(static_distance=1.0, in_collision=False),
        dynamic_world=FakeDynamicWorld(clearance=-5e-5),
    )

    assert not world.in_collision_xyz([0.0, 0.0, -1.0])
    assert world.is_state_valid_xyz([0.0, 0.0, -1.0])


def test_planner_world_rejects_real_dynamic_penetration():
    world = PlannerWorld(
        fcl_world=FakeFclWorld(static_distance=1.0, in_collision=False),
        dynamic_world=FakeDynamicWorld(clearance=-5e-3),
    )

    assert world.in_collision_xyz([0.0, 0.0, -1.0])
    assert not world.is_state_valid_xyz([0.0, 0.0, -1.0])


def test_shared_world_queries_exclude_the_requesting_owner_only():
    dynamic = FakeDynamicWorld(clearance=.2)
    first = PlannerWorld(fcl_world=FakeFclWorld(), dynamic_world=dynamic, robot_prefix='robot_1_')
    second = PlannerWorld(fcl_world=FakeFclWorld(), dynamic_world=dynamic, robot_prefix='robot_2_')
    first.min_clearance_xyz([0.,0.,0.])
    assert dynamic.last_exclude_owner == 'robot_1_'
    second.in_collision_xyz([0.,0.,0.])
    assert dynamic.last_exclude_owner == 'robot_2_'
    first.is_state_valid_xyz([0.,0.,0.], safety_margin=.1)
    assert dynamic.last_exclude_owner == 'robot_1_'


def test_timed_curve_checks_static_clearance_between_samples():
    import pytest
    from simlab.planner_world import TimedTrajectoryViolation

    class NarrowObstacle(FakeFclWorld):
        def min_distance_xyz(self, xyz):
            return np.linalg.norm(np.asarray(xyz) - [.5, 0., 0.]) - .01

    world = PlannerWorld(fcl_world=NarrowObstacle())
    with pytest.raises(TimedTrajectoryViolation, match='static') as caught:
        world.require_clear_timed_trajectory([(0., [0., 0., 0.]), (1., [1., 0., 0.])],
                                             max_speed=1., safety_margin=0.)
    assert caught.value.obstacle_id == 'static'
    assert caught.value.t_offset_s == 0.


def test_timed_curve_allows_clear_curved_passage():
    class Sphere(FakeFclWorld):
        def min_distance_xyz(self, xyz):
            return np.linalg.norm(xyz) - .8

    world = PlannerWorld(fcl_world=Sphere())
    times = np.linspace(0., np.pi, 101)
    samples = [(t, [np.cos(t), np.sin(t), 0.]) for t in times]
    world.require_clear_timed_trajectory(samples, max_speed=1., safety_margin=.01)


def test_timed_curve_accounts_for_obstacle_motion_between_samples():
    import pytest
    from types import SimpleNamespace
    from simlab.planner_world import TimedTrajectoryViolation

    class Crossing(FakeDynamicWorld):
        def __init__(self):
            self.obstacles = {'crossing': SimpleNamespace(owner_robot='', velocity_valid=True,
                collision_type=1, linear_velocity_world=np.array([2., 0., 0.]),
                angular_velocity_world=np.zeros(3))}
        def snapshot(self):
            return self
        def availability_error(self, exclude_owner):
            return None
        def min_clearance_xyz(self, xyz, *, t_offset=0., exclude_owner=''):
            return DynamicClearance('crossing', abs(-1. + 2.*t_offset) - .1)

    world = PlannerWorld(fcl_world=FakeFclWorld(), dynamic_world=Crossing())
    # Both endpoints are clear; the obstacle crosses the stationary robot halfway.
    with pytest.raises(TimedTrajectoryViolation, match='crossing'):
        world.require_clear_timed_trajectory([(0., [0.]*3), (1., [0.]*3)],
                                             max_speed=0., safety_margin=0.)


def test_timed_curve_rejects_missing_prediction_velocity():
    import pytest
    from types import SimpleNamespace
    dynamic = FakeDynamicWorld()
    dynamic.snapshot = lambda: dynamic
    dynamic.availability_error = lambda owner: None
    dynamic.obstacles = {'other': SimpleNamespace(owner_robot='robot_2_',
                                                 velocity_valid=False, obstacle_id='other')}
    world = PlannerWorld(fcl_world=FakeFclWorld(), dynamic_world=dynamic, robot_prefix='robot_1_')
    with pytest.raises(ValueError, match='velocity unavailable'):
        world.require_clear_timed_trajectory([(0., [0.]*3)], max_speed=0.)
    world.robot_prefix = 'robot_2_'
    world.require_clear_timed_trajectory([(0., [0.]*3)], max_speed=0.)


def test_timed_curve_refines_sampling_uncertainty_without_relaxing_margin():
    import pytest
    world = PlannerWorld(fcl_world=FakeFclWorld(static_distance=.0148))
    samples = [(0., [0., 0., 0.]), (.05, [.01, 0., 0.])]
    with pytest.raises(ValueError, match='certificate unresolved'):
        world.require_clear_timed_trajectory(samples, max_speed=.224, safety_margin=.01)
    visited = []
    def exact_sample(t):
        visited.append(t)
        return [.2*t, 0., 0.]
    world.require_clear_timed_trajectory(samples, max_speed=.224, safety_margin=.01,
                                         sample_at_time=exact_sample)
    assert visited == [.025]


def test_timed_curve_refinement_finds_collision_inside_interval():
    import pytest
    class NarrowObstacle(FakeFclWorld):
        def min_distance_xyz(self, xyz):
            return np.linalg.norm(np.asarray(xyz) - [.5, 0., 0.]) - .01
    world = PlannerWorld(fcl_world=NarrowObstacle())
    with pytest.raises(ValueError, match="clearance to 'static' -0.0100"):
        world.require_clear_timed_trajectory([(0., [0., 0., 0.]), (1., [1., 0., 0.])],
            max_speed=1., safety_margin=0., sample_at_time=lambda t: [t, 0., 0.])
