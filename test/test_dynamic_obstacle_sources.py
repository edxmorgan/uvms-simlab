import pytest

from simlab.dynamic_obstacle_sources import (
    ScriptedMotionSource, obstacle_behavior_source_class,
)


def test_selected_source_uses_behavior_lifecycle():
    assert obstacle_behavior_source_class("scripted_motion") is ScriptedMotionSource


@pytest.mark.parametrize("name", ["path_sphere", "missing"])
def test_removed_and_unknown_sources_are_rejected(name):
    with pytest.raises(ValueError, match="unknown obstacle behavior"):
        obstacle_behavior_source_class(name)
