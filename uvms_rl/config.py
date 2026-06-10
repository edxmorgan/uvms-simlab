"""Experiment config loading helpers."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml


def load_experiment_config(name_or_path: str) -> dict[str, Any]:
    path = Path(name_or_path).expanduser()
    if path.exists():
        with path.open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream) or {}

    resource_name = name_or_path if name_or_path.endswith(".yaml") else f"{name_or_path}.yaml"
    data = resources.files("uvms_rl.experiments").joinpath(resource_name).read_text(encoding="utf-8")
    return yaml.safe_load(data) or {}
