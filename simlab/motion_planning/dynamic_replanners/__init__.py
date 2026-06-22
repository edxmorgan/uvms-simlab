from __future__ import annotations

from simlab.motion_planning.dynamic_replanners.base import (
    DynamicReplannerTemplate,
    ReplanDecision,
    TimedPathSample,
)
from simlab.motion_planning.dynamic_replanners.clearance_hysteresis import ClearanceHysteresisReplanner

DEFAULT_DYNAMIC_REPLANNER_CLASSES = [ClearanceHysteresisReplanner]


def dynamic_replanner_class(name: str) -> type[DynamicReplannerTemplate]:
    for replanner_class in DEFAULT_DYNAMIC_REPLANNER_CLASSES:
        if replanner_class.registry_name == name:
            return replanner_class
    known = ", ".join(visible_dynamic_replanner_names())
    raise KeyError(f"unknown dynamic replanner '{name}'. Known dynamic replanners: {known}")


def visible_dynamic_replanner_names() -> list[str]:
    return [
        replanner_class.registry_name
        for replanner_class in DEFAULT_DYNAMIC_REPLANNER_CLASSES
        if getattr(replanner_class, "visible", True)
    ]


__all__ = [
    "DEFAULT_DYNAMIC_REPLANNER_CLASSES",
    "ClearanceHysteresisReplanner",
    "DynamicReplannerTemplate",
    "ReplanDecision",
    "TimedPathSample",
    "dynamic_replanner_class",
    "visible_dynamic_replanner_names",
]
