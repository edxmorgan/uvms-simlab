from types import SimpleNamespace

import numpy as np
import pytest

from simlab.dynamic_obstacle_sources import (
    PathSphereObstacleSource,
    dynamic_obstacle_source_class,
    visible_dynamic_obstacle_source_names,
)
from simlab.dynamic_obstacle_sources.base import DynamicObstacleSourceRequest
from simlab.utils import path_obstacles


class FakeColor:
    r = 0.0
    g = 0.0
    b = 0.0
    a = 0.0


class FakeOrientation:
    x = 0.0
    y = 0.0
    z = 0.0
    w = 0.0


class FakePosition:
    x = 0.0
    y = 0.0
    z = 0.0


class FakePose:
    def __init__(self):
        self.position = FakePosition()
        self.orientation = FakeOrientation()


class FakeDynamicObstacle:
    GEOMETRY_NONE = 0
    GEOMETRY_SPHERE = 1
    GEOMETRY_BOX = 2
    GEOMETRY_CYLINDER = 3
    GEOMETRY_MESH = 4

    def __init__(self):
        self.id = ""
        self.pose = FakePose()
        self.collision_type = self.GEOMETRY_NONE
        self.collision_dimensions = []
        self.visual_type = self.GEOMETRY_NONE
        self.visual_dimensions = []
        self.color = FakeColor()


class FakeDynamicObstacleArray:
    def __init__(self):
        self.header = SimpleNamespace(frame_id="")
        self.obstacles = []



class FakeRobot:
    def __init__(self):
        self.planner = SimpleNamespace(
            planned_result={
                "is_success": True,
                "xyz": np.array(
                    [
                        [0.0, 0.0, -1.0],
                        [2.0, 0.0, -1.0],
                        [4.0, 0.0, -1.0],
                    ],
                    dtype=float,
                ),
            }
        )

    def _pose_from_state_in_frame(self, frame):
        assert frame == "world"
        return SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0, z=-1.0))


class ShortPathFakeRobot(FakeRobot):
    def __init__(self):
        self.planner = SimpleNamespace(
            planned_result={
                "is_success": True,
                "xyz": np.array(
                    [
                        [0.0, 0.0, -1.0],
                        [1.0, 0.0, -1.0],
                    ],
                    dtype=float,
                ),
            }
        )


class TightPathFakeRobot(FakeRobot):
    def __init__(self):
        self.planner = SimpleNamespace(
            planned_result={
                "is_success": True,
                "xyz": np.array(
                    [
                        [0.0, 0.0, -1.0],
                        [1.2, 0.0, -1.0],
                    ],
                    dtype=float,
                ),
            }
        )


class TooTightForRobotFakeRobot(FakeRobot):
    def __init__(self):
        self.planner = SimpleNamespace(
            planned_result={
                "is_success": True,
                "xyz": np.array(
                    [
                        [0.0, 0.0, -1.0],
                        [1.0, 0.0, -1.0],
                    ],
                    dtype=float,
                ),
            }
        )


class ModerateShortPathFakeRobot(FakeRobot):
    def __init__(self):
        self.planner = SimpleNamespace(
            planned_result={
                "is_success": True,
                "xyz": np.array(
                    [
                        [0.0, 0.0, -1.0],
                        [2.85, 0.0, -1.0],
                    ],
                    dtype=float,
                ),
            }
        )


def test_dynamic_obstacle_source_registry_exposes_path_sphere():
    assert dynamic_obstacle_source_class("path_sphere") is PathSphereObstacleSource
    assert "path_sphere" in visible_dynamic_obstacle_source_names()
    with pytest.raises(ValueError, match="unknown dynamic obstacle source"):
        dynamic_obstacle_source_class("missing")


def test_path_sphere_source_places_obstacle_ahead_on_active_path(monkeypatch):
    monkeypatch.setattr(path_obstacles, "DynamicObstacle", FakeDynamicObstacle)
    obstacles = FakeDynamicObstacleArray()
    obstacles.header.frame_id = "world"
    source = PathSphereObstacleSource()

    result = source.create(
        DynamicObstacleSourceRequest(
            robot=FakeRobot(),
            existing_obstacles=obstacles,
            world_frame="world",
            name="test_obstacle",
            distance_ahead=3.0,
            radius=0.4,
        )
    )

    assert result is not None
    assert result.obstacle.id == "test_obstacle"
    assert result.obstacle.collision_type == FakeDynamicObstacle.GEOMETRY_SPHERE
    assert result.obstacle.collision_dimensions == [pytest.approx(0.4)]
    np.testing.assert_allclose(result.center_world, [3.0, 0.0, -1.0])
    assert result.detail_fields["path_ahead_m"] == pytest.approx(3.0)
    assert result.detail_fields["euclidean_from_robot_m"] == pytest.approx(3.0)
    assert result.detail_fields["remaining_path_m"] == pytest.approx(4.0)


def test_path_sphere_source_returns_none_without_active_path(monkeypatch):
    monkeypatch.setattr(path_obstacles, "DynamicObstacle", FakeDynamicObstacle)
    robot = FakeRobot()
    robot.planner.planned_result = {"is_success": False}
    source = PathSphereObstacleSource()

    result = source.create(
        DynamicObstacleSourceRequest(
            robot=robot,
            existing_obstacles=FakeDynamicObstacleArray(),
            world_frame="world",
        )
    )

    assert result is None


def test_path_sphere_source_clamps_before_goal_clearance(monkeypatch):
    monkeypatch.setattr(path_obstacles, "DynamicObstacle", FakeDynamicObstacle)
    source = PathSphereObstacleSource()

    result = source.create(
        DynamicObstacleSourceRequest(
            robot=FakeRobot(),
            existing_obstacles=FakeDynamicObstacleArray(),
            world_frame="world",
            distance_ahead=10.0,
            radius=0.4,
            robot_collision_radius=0.5,
        )
    )

    assert result is not None
    np.testing.assert_allclose(result.center_world, [3.1, 0.0, -1.0])
    assert result.detail_fields["path_ahead_m"] == pytest.approx(3.1)
    assert result.detail_fields["remaining_path_m"] == pytest.approx(4.0)
    assert result.detail_fields["goal_clearance_m"] == pytest.approx(0.9)


def test_path_sphere_source_rejects_when_path_too_short_for_goal_clearance(monkeypatch):
    monkeypatch.setattr(path_obstacles, "DynamicObstacle", FakeDynamicObstacle)
    source = PathSphereObstacleSource()

    result = source.create(
        DynamicObstacleSourceRequest(
            robot=ShortPathFakeRobot(),
            existing_obstacles=FakeDynamicObstacleArray(),
            world_frame="world",
            distance_ahead=0.5,
            radius=0.6,
            robot_collision_radius=0.5,
        )
    )

    assert result is None


def test_path_sphere_source_shrinks_obstacle_to_keep_short_goal_reachable(monkeypatch):
    monkeypatch.setattr(path_obstacles, "DynamicObstacle", FakeDynamicObstacle)
    source = PathSphereObstacleSource()

    result = source.create(
        DynamicObstacleSourceRequest(
            robot=TightPathFakeRobot(),
            existing_obstacles=FakeDynamicObstacleArray(),
            world_frame="world",
            distance_ahead=10.0,
            radius=0.6,
            robot_collision_radius=0.5,
        )
    )

    assert result is not None
    np.testing.assert_allclose(result.center_world, [0.65, 0.0, -1.0])
    assert result.obstacle.collision_dimensions == [pytest.approx(0.05)]
    assert result.detail_fields["path_ahead_m"] == pytest.approx(0.65)
    assert result.detail_fields["remaining_path_m"] == pytest.approx(1.2)
    assert result.detail_fields["goal_clearance_m"] == pytest.approx(0.55)


def test_path_sphere_source_keeps_obstacle_clear_of_current_robot(monkeypatch):
    monkeypatch.setattr(path_obstacles, "DynamicObstacle", FakeDynamicObstacle)
    source = PathSphereObstacleSource()

    result = source.create(
        DynamicObstacleSourceRequest(
            robot=FakeRobot(),
            existing_obstacles=FakeDynamicObstacleArray(),
            world_frame="world",
            distance_ahead=0.1,
            radius=0.05,
            robot_collision_radius=0.574,
            robot_clearance_margin=0.25,
        )
    )

    assert result is not None
    np.testing.assert_allclose(result.center_world, [0.874, 0.0, -1.0])
    assert result.detail_fields["euclidean_from_robot_m"] == pytest.approx(0.874)
    assert result.detail_fields["robot_clearance_m"] == pytest.approx(0.25)
    assert result.detail_fields["robot_clearance_margin_m"] == pytest.approx(0.25)


def test_path_sphere_source_rejects_if_current_and_goal_clearances_cannot_both_fit(monkeypatch):
    monkeypatch.setattr(path_obstacles, "DynamicObstacle", FakeDynamicObstacle)
    source = PathSphereObstacleSource()

    result = source.create(
        DynamicObstacleSourceRequest(
            robot=TooTightForRobotFakeRobot(),
            existing_obstacles=FakeDynamicObstacleArray(),
            world_frame="world",
            distance_ahead=10.0,
            radius=0.6,
            robot_collision_radius=0.574,
            robot_clearance_margin=0.25,
        )
    )

    assert result is None

def test_path_sphere_source_shrinks_large_obstacle_to_keep_robot_margin(monkeypatch):
    monkeypatch.setattr(path_obstacles, "DynamicObstacle", FakeDynamicObstacle)
    source = PathSphereObstacleSource()

    result = source.create(
        DynamicObstacleSourceRequest(
            robot=ModerateShortPathFakeRobot(),
            existing_obstacles=FakeDynamicObstacleArray(),
            world_frame="world",
            distance_ahead=10.0,
            radius=0.8,
            robot_collision_radius=0.574,
            robot_clearance_margin=0.25,
        )
    )

    assert result is not None
    np.testing.assert_allclose(result.center_world, [1.976, 0.0, -1.0])
    assert result.obstacle.collision_dimensions == [pytest.approx(0.05)]
    assert result.detail_fields["robot_clearance_m"] == pytest.approx(1.352)
    assert result.detail_fields["goal_clearance_m"] == pytest.approx(0.874)

def test_path_sphere_source_goal_clearance_matches_inflated_replan_radius(monkeypatch):
    monkeypatch.setattr(path_obstacles, "DynamicObstacle", FakeDynamicObstacle)
    source = PathSphereObstacleSource()

    result = source.create(
        DynamicObstacleSourceRequest(
            robot=FakeRobot(),
            existing_obstacles=FakeDynamicObstacleArray(),
            world_frame="world",
            distance_ahead=10.0,
            radius=0.8,
            robot_collision_radius=0.574,
            robot_clearance_margin=0.25,
        )
    )

    assert result is not None
    assert result.detail_fields["goal_clearance_m"] == pytest.approx(1.624)
    assert result.detail_fields["remaining_path_m"] - result.detail_fields["path_ahead_m"] >= 1.624 - 1e-9



def test_path_sphere_source_uses_euclidean_clearance_on_returning_path(monkeypatch):
    monkeypatch.setattr(path_obstacles, "DynamicObstacle", FakeDynamicObstacle)
    robot = FakeRobot()
    robot.planner.planned_result["xyz"] = np.array(
        [
            [0.0, 0.0, -1.0],
            [2.0, 0.0, -1.0],
            [0.1, 0.0, -1.0],
            [0.1, 2.0, -1.0],
        ],
        dtype=float,
    )
    source = PathSphereObstacleSource()

    result = source.create(
        DynamicObstacleSourceRequest(
            robot=robot,
            existing_obstacles=FakeDynamicObstacleArray(),
            world_frame="world",
            distance_ahead=3.8,
            radius=0.4,
            robot_collision_radius=0.5,
            robot_clearance_margin=0.1,
        )
    )

    assert result is not None
    obstacle_radius = result.obstacle.collision_dimensions[0]
    center = np.asarray(result.center_world, dtype=float)
    current = np.array([0.0, 0.0, -1.0])
    goal = robot.planner.planned_result["xyz"][-1]
    assert np.linalg.norm(center - current) - obstacle_radius - 0.5 >= 0.1 - 1e-9
    assert np.linalg.norm(center - goal) - obstacle_radius - 0.5 >= 0.1 - 1e-9
