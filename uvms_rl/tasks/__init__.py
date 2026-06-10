"""Task registry for UVMS RL experiments."""

from uvms_rl.tasks.base import TaskBase
from uvms_rl.tasks.registry import TASKS, make_task

__all__ = ["TaskBase", "TASKS", "make_task"]
