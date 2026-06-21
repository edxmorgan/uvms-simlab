from __future__ import annotations

from simlab.dynamic_obstacle_sources.base import (
    DynamicObstacleSourceRequest,
    DynamicObstacleSourceResult,
    DynamicObstacleSourceTemplate,
)
from simlab.dynamic_obstacle_sources.path_sphere import PathSphereObstacleSource

DEFAULT_DYNAMIC_OBSTACLE_SOURCE_CLASSES = (
    PathSphereObstacleSource,
)

_DYNAMIC_OBSTACLE_SOURCE_BY_NAME = {
    source_cls.registry_name: source_cls for source_cls in DEFAULT_DYNAMIC_OBSTACLE_SOURCE_CLASSES
}


def dynamic_obstacle_source_class(name: str) -> type[DynamicObstacleSourceTemplate]:
    source_name = str(name or "").strip()
    try:
        return _DYNAMIC_OBSTACLE_SOURCE_BY_NAME[source_name]
    except KeyError as exc:
        valid = ", ".join(sorted(_DYNAMIC_OBSTACLE_SOURCE_BY_NAME))
        raise ValueError(f"unknown dynamic obstacle source '{source_name}'. Valid sources: {valid}") from exc


def visible_dynamic_obstacle_source_names() -> list[str]:
    return [
        source_cls.registry_name
        for source_cls in DEFAULT_DYNAMIC_OBSTACLE_SOURCE_CLASSES
        if getattr(source_cls, "visible", True)
    ]


__all__ = [
    "DEFAULT_DYNAMIC_OBSTACLE_SOURCE_CLASSES",
    "DynamicObstacleSourceRequest",
    "DynamicObstacleSourceResult",
    "DynamicObstacleSourceTemplate",
    "PathSphereObstacleSource",
    "dynamic_obstacle_source_class",
    "visible_dynamic_obstacle_source_names",
]
