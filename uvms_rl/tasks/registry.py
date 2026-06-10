"""Task registry for UVMS RL experiments."""

from __future__ import annotations

from uvms_rl.tasks.hover_vehicle import HoverVehicleTask

TASKS = {
    HoverVehicleTask.name: HoverVehicleTask,
}


def make_task(name: str, *, robot_count: int, config: dict | None = None, seed: int | None = None):
    try:
        task_cls = TASKS[name]
    except KeyError as exc:
        names = ", ".join(sorted(TASKS))
        raise ValueError(f"unknown task '{name}'. Available tasks: {names}") from exc
    return task_cls(robot_count=robot_count, config=config, seed=seed)
