import json
from pathlib import Path
import sys

import pytest

PACKAGE_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from simlab.world_profiles import dynamic_obstacles_from_world_profile


WORLD_PROFILE_ROOT = PACKAGE_ROOT / "resource" / "world_profiles"


def test_world_profile_json_files_convert_to_obstacle_messages():
    profile_paths = sorted(WORLD_PROFILE_ROOT.glob("*.json"))
    assert profile_paths, "expected at least one world profile JSON file"

    for profile_path in profile_paths:
        profile = json.loads(profile_path.read_text())
        msg = dynamic_obstacles_from_world_profile(profile, "world")
        assert msg.header.frame_id == profile.get("frame_id", "world")
        assert len(msg.obstacles) == len(profile.get("obstacles", []))


def test_duplicate_dynamic_obstacle_ids_are_rejected():
    profile = {
        "frame_id": "world",
        "obstacles": [
            {"id": "duplicate", "type": "sphere", "dimensions": [0.25]},
            {"id": "duplicate", "type": "sphere", "dimensions": [0.35]},
        ],
    }

    with pytest.raises(ValueError, match="duplicate dynamic obstacle id"):
        dynamic_obstacles_from_world_profile(profile, "world")
