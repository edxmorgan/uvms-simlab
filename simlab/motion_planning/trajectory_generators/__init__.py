from __future__ import annotations

from simlab.motion_planning.trajectory_generators.base import VehicleTrajectoryGeneratorTemplate
from simlab.motion_planning.trajectory_generators.ruckig import RuckigVehicleTrajectoryGenerator

DEFAULT_VEHICLE_TRAJECTORY_GENERATOR_CLASSES = [RuckigVehicleTrajectoryGenerator]


def vehicle_trajectory_generator_class(name: str) -> type[VehicleTrajectoryGeneratorTemplate]:
    for generator_class in DEFAULT_VEHICLE_TRAJECTORY_GENERATOR_CLASSES:
        if generator_class.registry_name == name:
            return generator_class
    known = ", ".join(visible_vehicle_trajectory_generator_names())
    raise KeyError(f"unknown vehicle trajectory generator '{name}'. Known vehicle trajectory generators: {known}")


def visible_vehicle_trajectory_generator_names() -> list[str]:
    return [
        generator_class.registry_name
        for generator_class in DEFAULT_VEHICLE_TRAJECTORY_GENERATOR_CLASSES
        if getattr(generator_class, "visible", True)
    ]


__all__ = [
    "DEFAULT_VEHICLE_TRAJECTORY_GENERATOR_CLASSES",
    "RuckigVehicleTrajectoryGenerator",
    "VehicleTrajectoryGeneratorTemplate",
    "vehicle_trajectory_generator_class",
    "visible_vehicle_trajectory_generator_names",
]
