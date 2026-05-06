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

    def min_clearance_xyz(self, xyz, *, t_offset=0.0):
        np.asarray(xyz, dtype=float).reshape(3)
        self.last_t_offset = float(t_offset)
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
