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


@pytest.mark.parametrize('velocity', ([.3, .4, 0.], [-.2, 0., 0.]))
def test_ruckig_replan_preserves_measured_velocity(velocity):
    from unittest.mock import Mock
    generator = RuckigVehicleTrajectoryGenerator(Mock(), 3, .01, 100)
    generator.start_from_path([0., 0., -1.], [[0., 0., -1.], [1., 0., -1.]],
                              [1.]*3, [1.]*3, [1.]*3, current_velocity=velocity)
    np.testing.assert_allclose(generator.calculation_request()['current_velocity'], velocity)


def test_approximate_solution_is_not_returned_as_success():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from simlab.motion_planning.planners.ompl import OmplPlanner
    node = SimpleNamespace(planner_world=Mock(), get_logger=lambda: Mock())
    node.planner_world.planning_start.side_effect = lambda xyz, bounds, tolerance: xyz
    node.planner_world.invalid_position_reason.return_value = ''
    node.planner_world.is_state_valid_xyz.return_value = True
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


def test_transferred_profiles_match_native_ruckig_including_braking():
    from ruckig import InputParameter, Trajectory, Ruckig
    from simlab.motion_planning.trajectory_generators.ruckig import encode_trajectory, sample_trajectory
    rng = np.random.default_rng(3)
    for _ in range(20):
        inp = InputParameter(3)
        inp.current_position = rng.normal(size=3).tolist()
        inp.current_velocity = rng.normal(size=3).tolist()
        inp.current_acceleration = rng.normal(size=3).tolist()
        inp.target_position = rng.normal(size=3).tolist()
        inp.max_velocity = inp.max_acceleration = inp.max_jerk = [1.] * 3
        trajectory = Trajectory(3)
        Ruckig(3, .01).calculate(inp, trajectory)
        transferred = encode_trajectory(trajectory)
        for elapsed in np.linspace(0., trajectory.duration + .1, 100):
            np.testing.assert_allclose(sample_trajectory(transferred, elapsed),
                                       trajectory.at_time(elapsed), rtol=0., atol=1e-12)


def test_prepared_generator_preview_does_not_advance_execution():
    from unittest.mock import Mock
    from ruckig import InputParameter, Trajectory, Ruckig
    from simlab.motion_planning.trajectory_generators.ruckig import encode_trajectory
    inp = InputParameter(3)
    inp.target_position = [1., 2., 3.]
    inp.max_velocity = inp.max_acceleration = inp.max_jerk = [1.] * 3
    trajectory = Trajectory(3)
    Ruckig(3, .01).calculate(inp, trajectory)
    generator = RuckigVehicleTrajectoryGenerator(Mock(), 3, .01, 100)
    generator.install_calculation(encode_trajectory(trajectory))
    first = generator.update(0.)
    np.testing.assert_allclose(first[:3], trajectory.at_time(.01), atol=1e-12)
    for elapsed, position in generator.preview_samples(horizon=.5):
        np.testing.assert_allclose(position, trajectory.at_time(.01 + elapsed)[0], atol=1e-12)
    assert generator.elapsed == .01
    np.testing.assert_allclose(generator.current_reference(), trajectory.at_time(.01), atol=1e-12)
