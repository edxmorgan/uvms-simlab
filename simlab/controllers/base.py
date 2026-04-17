from abc import ABC, abstractmethod

import numpy as np
from rclpy.node import Node


class ControllerTemplate(ABC):
    """Base interface for UVMS vehicle and arm controllers."""

    name: str = ""
    registry_name: str = ""
    arm_gain_profile: str = "tau"

    def __init__(self, node: Node, arm_dof: int = 4):
        self.node = node
        self.arm_dof = int(arm_dof)

    @staticmethod
    def vector(values, expected_size: int, label: str) -> np.ndarray:
        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.size != expected_size:
            raise ValueError(f"{label} must have {expected_size} elements, got {vector.size}")
        return vector

    def arm_vector(self, values, label: str) -> np.ndarray:
        return self.vector(values, self.arm_dof + 1, label)

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
        Kp: np.ndarray,
        Ki: np.ndarray,
        Kd: np.ndarray,
        dt: float,
        u_max: np.ndarray,
        u_min: np.ndarray,
        model_param: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


VEHICLE_MODEL_PARAMS = [
    3.72028553e+01,
    2.21828075e+01,
    6.61734807e+01,
    3.38909801e+00,
    6.41362046e-01,
    6.41362034e-01,
    3.38909800e+00,
    1.39646394e+00,
    4.98032205e-01,
    2.53118738e+00,
    1.05000000e+02,
    9.78296453e+01,
    8.27479545e-01,
    1.36822559e-01,
    4.25841171e+00,
    -7.36416666e+01,
    -3.36082112e+01,
    -8.94055107e+01,
    -2.98736214e+00,
    -1.57921531e+00,
    -3.39766499e+00,
    -1.47912104e-04,
    -5.16373030e-04,
    -9.85522538e+01,
    -3.05907788e-02,
    -1.27877517e-01,
    -1.63514832e+00,
]
