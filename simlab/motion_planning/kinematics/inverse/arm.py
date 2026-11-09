"""Existing arm-only position IK adapter."""
from pathlib import Path
import threading
import casadi as ca

from .base import ManipulatorIKTemplate


class ManipulatorPositionIK(ManipulatorIKTemplate):
    """Adapter for the existing arm-only CasADi position IK, unchanged."""
    registry_name = 'manipulator_position'

    def __init__(self, resource_directory):
        self.solver = ca.Function.load(str(Path(resource_directory) / 'model_functions/arm/ik_eval.casadi'))
        self.last_report = {}
        self.lock = threading.Lock()

    def solve(self, target_position):
        # Keep the existing evaluation and output ordering exactly as before.
        with self.lock:
            result = self.solver(target_position).full().flatten().tolist()
        self.last_report = {'task': self.task}
        return result
