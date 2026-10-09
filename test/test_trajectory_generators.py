import numpy as np
import pytest

from simlab.motion_planning.trajectory_generators import (
    RuckigVehicleTrajectoryGenerator,
    vehicle_trajectory_generator_class,
    visible_vehicle_trajectory_generator_names,
)
from simlab.motion_planning.trajectory_generators.base import VehicleTrajectoryGeneratorTemplate


def test_ruckig_vehicle_trajectory_generator_is_registered():
    assert vehicle_trajectory_generator_class("ruckig") is RuckigVehicleTrajectoryGenerator
    assert "ruckig" in visible_vehicle_trajectory_generator_names()
    assert issubclass(RuckigVehicleTrajectoryGenerator, VehicleTrajectoryGeneratorTemplate)


def test_unknown_vehicle_trajectory_generator_reports_known_names():
    with pytest.raises(KeyError, match="Known vehicle trajectory generators: ruckig"):
        vehicle_trajectory_generator_class("does_not_exist")


def test_ruckig_replan_velocity_projection_removes_sideways_notch_component():
    current = np.array([0.0, 0.0, -1.0])
    path = np.array([
        [0.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -1.0],
    ])
    velocity = np.array([0.3, 0.4, 0.0])

    projected = RuckigVehicleTrajectoryGenerator._path_aligned_initial_velocity(current, path, velocity)

    np.testing.assert_allclose(projected, [0.3, 0.0, 0.0])


def test_ruckig_replan_velocity_projection_drops_backward_velocity():
    current = np.array([0.0, 0.0, -1.0])
    path = np.array([
        [0.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
    ])
    velocity = np.array([-0.2, 0.0, 0.0])

    projected = RuckigVehicleTrajectoryGenerator._path_aligned_initial_velocity(current, path, velocity)

    np.testing.assert_allclose(projected, [0.0, 0.0, 0.0])


def test_approximate_solution_is_not_returned_as_success():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from simlab.motion_planning.planners.ompl import OmplPlanner
    node = SimpleNamespace(planner_world=Mock(), get_logger=lambda: Mock())
    planner = OmplPlanner(node, env_bounds=(-2., 2., -2., 2., -2., 2.))
    planner.ss = Mock(wraps=planner.ss)
    planner.ss.solve.return_value = True  # OMPL approximate status is truthy too.
    planner.ss.haveExactSolutionPath.return_value = False
    result = planner.plan_se3_path([0., 0., 0.], [1., 0., 0., 0.],
                                  [1., 0., 0.], [1., 0., 0., 0.])
    assert not result.is_success
    assert 'approximate' in result.message
    planner.ss.getSolutionPath.assert_not_called()
    planner.ss.simplifySolution.assert_not_called()
