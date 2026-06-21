import pytest

from simlab.trajectory_generators import (
    RuckigVehicleTrajectoryGenerator,
    vehicle_trajectory_generator_class,
    visible_vehicle_trajectory_generator_names,
)
from simlab.trajectory_generators.base import VehicleTrajectoryGeneratorTemplate


def test_ruckig_vehicle_trajectory_generator_is_registered():
    assert vehicle_trajectory_generator_class("ruckig") is RuckigVehicleTrajectoryGenerator
    assert "ruckig" in visible_vehicle_trajectory_generator_names()
    assert issubclass(RuckigVehicleTrajectoryGenerator, VehicleTrajectoryGeneratorTemplate)


def test_unknown_vehicle_trajectory_generator_reports_known_names():
    with pytest.raises(KeyError, match="Known vehicle trajectory generators: ruckig"):
        vehicle_trajectory_generator_class("does_not_exist")
