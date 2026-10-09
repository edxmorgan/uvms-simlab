"""Behavior-source contract, independent of robot controllers and planners.

The host owns authoritative geometry and IDs. A selected source owns behavior
for the entire scene, including coupled/multi-agent policies. Inputs are private
copies. Sources return a complete snapshot; they never publish or integrate a
second simulator themselves. Geometry/topology changes use host edit operations.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from ros2_control_blue_reach_5.msg import DynamicObstacleArray


@dataclass(frozen=True)
class ObstacleBehaviorContext:
    time: float
    dt: float
    seed: int
    observations: Mapping[str, Any] = field(default_factory=dict)


class ObstacleBehaviorSource(ABC):
    registry_name = ""

    def initialize(self, scene: DynamicObstacleArray, config: dict, seed: int) -> None:
        """Load policy/resources once. Scene pose/size come from the caller."""

    @abstractmethod
    def step(self, scene: DynamicObstacleArray,
             context: ObstacleBehaviorContext) -> DynamicObstacleArray:
        """Compute the whole scene's next poses/twists, once per host step."""

    def on_edit(self, scene: DynamicObstacleArray) -> None:
        """Observe a validated user edit; discard obsolete per-agent state."""

    def reset(self, scene: DynamicObstacleArray, seed: int) -> None:
        """Restore policy state/RNG for repeatable experiments."""

    def close(self) -> None:
        """Release policy resources. Must also work after a failed step."""
