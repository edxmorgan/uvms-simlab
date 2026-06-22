"""Motion-planning algorithms and components.

Algorithms may be split into planner -> trajectory generator -> controller, or
integrated so one plugin returns a timed trajectory or direct control sequence.
The shared :class:`MotionPlanResult` contract describes which execution shape an
algorithm produced.
"""

from simlab.motion_planning.result import MotionPlanKind, MotionPlanResult

__all__ = [
    "MotionPlanKind",
    "MotionPlanResult",
]
