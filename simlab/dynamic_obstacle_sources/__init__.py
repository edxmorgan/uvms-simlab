from __future__ import annotations

from simlab.dynamic_obstacle_sources.behavior import ObstacleBehaviorContext, ObstacleBehaviorSource
from simlab.dynamic_obstacle_sources.scripted_motion import ScriptedMotionSource

DEFAULT_OBSTACLE_BEHAVIOR_CLASSES = (ScriptedMotionSource,)


def obstacle_behavior_source_class(name):
    for source_cls in DEFAULT_OBSTACLE_BEHAVIOR_CLASSES:
        if source_cls.registry_name == str(name).strip():
            return source_cls
    raise ValueError(f"unknown obstacle behavior source '{name}'")

__all__ = [
    "DEFAULT_OBSTACLE_BEHAVIOR_CLASSES",
    "ObstacleBehaviorContext",
    "ObstacleBehaviorSource",
    "ScriptedMotionSource",
    "obstacle_behavior_source_class",
]
