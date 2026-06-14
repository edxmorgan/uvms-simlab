"""Python training adapters for the UVMS batch simulator."""

from uvms_rl.config import Experiment, load_experiment
from uvms_rl.rl_env import UvmsBatchEnv, UvmsBatchInfo
from uvms_rl.task_base import TaskBase

__all__ = ["Experiment", "RslRlUvmsEnv", "TaskBase", "UvmsBatchEnv", "UvmsBatchInfo", "load_experiment"]


def __getattr__(name: str):
    if name == "RslRlUvmsEnv":
        from uvms_rl.rsl_adapter import RslRlUvmsEnv

        return RslRlUvmsEnv
    raise AttributeError(name)
