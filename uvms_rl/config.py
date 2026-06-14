"""Experiment loading helpers."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from uvms_rl.task_base import TaskBase


@dataclass(frozen=True)
class Experiment:
    """A self-contained RL experiment definition."""

    name: str
    config: dict[str, Any]
    task_cls: type[TaskBase]
    root: Path | None = None


def load_experiment(name_or_path: str) -> Experiment:
    """Load ``config.yaml`` and ``task.py`` from an experiment folder.

    ``name_or_path`` can be a packaged experiment name such as
    ``hover_vehicle``, a filesystem directory containing ``config.yaml`` and
    ``task.py``, or a direct path to ``config.yaml``.
    """

    path = Path(name_or_path).expanduser()
    if path.exists():
        return _load_filesystem_experiment(path)

    package = f"uvms_rl.experiments.{name_or_path}"
    config_text = resources.files(package).joinpath("config.yaml").read_text(encoding="utf-8")
    config = yaml.safe_load(config_text) or {}
    module = importlib.import_module(f"{package}.task")
    return Experiment(
        name=name_or_path,
        config=config,
        task_cls=_task_class_from_module(module, f"{package}.task"),
        root=None,
    )


def _load_filesystem_experiment(path: Path) -> Experiment:
    root = path if path.is_dir() else path.parent
    config_path = root / "config.yaml" if path.is_dir() else path
    task_path = root / "task.py"
    if not config_path.exists():
        raise FileNotFoundError(f"experiment config not found: {config_path}")
    if not task_path.exists():
        raise FileNotFoundError(f"experiment task module not found: {task_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    module_name = f"uvms_rl_user_experiment_{root.name}"
    spec = importlib.util.spec_from_file_location(module_name, task_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to import experiment task module: {task_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return Experiment(
        name=root.name,
        config=config,
        task_cls=_task_class_from_module(module, str(task_path)),
        root=root,
    )


def _task_class_from_module(module, source_name: str) -> type[TaskBase]:
    task_cls = getattr(module, "Task", None)
    if isinstance(task_cls, type) and issubclass(task_cls, TaskBase):
        return task_cls
    candidates = [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, TaskBase)
        and value is not TaskBase
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(f"{source_name} must define Task, a TaskBase subclass")
    raise ValueError(f"{source_name} defines multiple TaskBase subclasses; name the intended class Task")
