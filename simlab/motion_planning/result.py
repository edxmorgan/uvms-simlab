from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class MotionPlanKind:
    """Execution shape produced by a motion-planning algorithm."""

    PATH = "path"
    TIMED_TRAJECTORY = "timed_trajectory"
    CONTROL_SEQUENCE = "control_sequence"


@dataclass
class MotionPlanResult:
    """Common result contract for split and integrated motion planners.

    Split planners such as OMPL usually fill ``xyz`` and ``quat_wxyz`` and let a
    trajectory generator time-parameterize the path. Integrated planners such as
    CHOMP/GPMP/MPC can fill ``time_from_start``, derivatives, or controls while
    still passing through the same high-level planning boundary.
    """

    is_success: bool
    kind: str = MotionPlanKind.PATH
    xyz: np.ndarray | None = None
    quat_wxyz: np.ndarray | None = None
    time_from_start: np.ndarray | None = None
    velocity: np.ndarray | None = None
    acceleration: np.ndarray | None = None
    control_sequence: np.ndarray | None = None
    path_length_cost: float = float("nan")
    geom_length: float = float("nan")
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        xyz = self.normalized_xyz()
        quat = self.normalized_quat_wxyz()
        if xyz.shape[0] == 0:
            return 0
        if quat.shape[0] == 0:
            return int(xyz.shape[0])
        return int(min(xyz.shape[0], quat.shape[0]))

    @property
    def can_transport_over_plan_vehicle_action(self) -> bool:
        # The current ROS action transports geometric paths only. Integrated
        # results are valid Python results, but timed trajectories and direct
        # control sequences need a richer execution transport before they can be
        # sent to Robot through PlanVehicle without losing information.
        return self.kind == MotionPlanKind.PATH

    @property
    def needs_trajectory_generator(self) -> bool:
        return self.kind == MotionPlanKind.PATH

    def normalized_xyz(self) -> np.ndarray:
        return _array_or_empty(self.xyz, width=3)

    def normalized_quat_wxyz(self) -> np.ndarray:
        xyz_count = self.normalized_xyz().shape[0]
        quat = _array_or_empty(self.quat_wxyz, width=4)
        if quat.shape[0] == 0 and xyz_count > 0:
            quat = np.zeros((xyz_count, 4), dtype=float)
            quat[:, 0] = 1.0
        return quat

    def as_action_payload(self) -> dict[str, Any]:
        xyz = self.normalized_xyz()
        quat = self.normalized_quat_wxyz()
        count = min(int(xyz.shape[0]), int(quat.shape[0]))
        return {
            "is_success": bool(self.is_success),
            "success": bool(self.is_success),
            "kind": str(self.kind),
            "xyz": xyz[:count],
            "quat_wxyz": quat[:count],
            "count": int(count),
            "path_length_cost": float(self.path_length_cost),
            "geom_length": float(self.geom_length),
            "message": str(self.message),
            "metadata": dict(self.metadata),
        }





def _array_or_empty(value: Any, *, width: int) -> np.ndarray:
    if value is None:
        return np.empty((0, width), dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.empty((0, width), dtype=float)
    if arr.ndim == 1:
        if arr.size % width != 0:
            raise ValueError(f"array size {arr.size} is not divisible by {width}")
        arr = arr.reshape(-1, width)
    if arr.ndim != 2 or arr.shape[1] != width:
        raise ValueError(f"array must have shape (N, {width})")
    return arr
