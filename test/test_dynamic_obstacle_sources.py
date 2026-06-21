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
