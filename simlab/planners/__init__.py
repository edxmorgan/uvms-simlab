from simlab.planners.base import PlannerTemplate
from simlab.planners.ompl import BitStarPlanner, RrtConnectPlanner, RrtStarPlanner


DEFAULT_PLANNER_CLASSES = [
    RrtStarPlanner,
    BitStarPlanner,
    RrtConnectPlanner,
]


def visible_planner_names() -> list[str]:
    return [
        planner_cls.registry_name
        for planner_cls in DEFAULT_PLANNER_CLASSES
        if getattr(planner_cls, "visible", True)
    ]


__all__ = [
    "BitStarPlanner",
    "DEFAULT_PLANNER_CLASSES",
    "PlannerTemplate",
    "RrtConnectPlanner",
    "RrtStarPlanner",
    "visible_planner_names",
]
