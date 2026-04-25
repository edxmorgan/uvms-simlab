from abc import ABC, abstractmethod

import numpy as np
from rclpy.node import Node


class ControllerTemplate(ABC):
    """Base interface for UVMS vehicle and arm controllers."""

    name: str = ""
    registry_name: str = ""

    def __init__(self, node: Node, arm_dof: int = 4, robot_prefix: str = ""):
        self.node = node
        self.arm_dof = int(arm_dof)
        self.robot_prefix = str(robot_prefix)

    @staticmethod
    def vector(values, expected_size: int, label: str) -> np.ndarray:
        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.size != expected_size:
            raise ValueError(f"{label} must have {expected_size} elements, got {vector.size}")
        return vector

    def arm_vector(self, values, label: str) -> np.ndarray:
        return self.vector(values, self.arm_dof + 1, label)

    def reset_controller_state(self) -> None:
        """Clear controller memory before a new independent control segment."""
        return

    @abstractmethod
    def vehicle_controller(
        self,
        state: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_acc: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def arm_controller(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        q_ref: np.ndarray,
        dq_ref: np.ndarray,
        ddq_ref: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        raise NotImplementedError
