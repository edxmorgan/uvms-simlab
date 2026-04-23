import numpy as np

from simlab.controllers.base import ControllerTemplate


class MPCController(ControllerTemplate):
    registry_name = "MPC"

    def __init__(self, node, arm_dof=4):
        super().__init__(node, arm_dof)
        self.arm_kp = np.ones(self.arm_dof + 1, dtype=float)
        self.arm_u_max = np.ones(self.arm_dof + 1, dtype=float)
        self.arm_u_min = -self.arm_u_max

    def vehicle_controller(self, state, target_pos, target_vel, target_acc, dt) -> np.ndarray:
        state = self.vector(state, 12, "state")
        target_pos = self.vector(target_pos, 6, "target_pos")
        return np.zeros(6, dtype=float)

    def arm_controller(
        self,
        q,
        q_dot,
        q_ref,
        dq_ref,
        ddq_ref,
        dt,
    ) -> np.ndarray:
        q = self.arm_vector(q, "q")
        return np.zeros(self.arm_dof + 1, dtype=float)