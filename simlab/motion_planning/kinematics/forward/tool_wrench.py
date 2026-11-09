"""Power-conjugate tool loads using the exported Floating-KinDyn geometric J.

J rows: tool-origin linear/angular velocity in vehicle FLU axes.
J columns at world=0: base FLU linear/angular velocity, then arm edcb rates.
The simulator accepts FRD base loads; arm coordinates are unchanged.
"""
from pathlib import Path
import casadi as ca
import numpy as np


class ToolWrenchMapper:
    def __init__(self, resource_directory, mounting, tip):
        self.evaluate = ca.Function.load(str(Path(resource_directory)/
            'model_functions/arm/tool_geometric_jacobian.casadi'))
        self.mounting = np.asarray(mounting, dtype=float)
        self.tip = np.asarray(tip, dtype=float)
        self.velocity_conversion = np.diag([1., -1., -1., 1., -1., -1., 1., 1., 1., 1.])

    def geometry(self, joints):
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (4,) or not np.all(np.isfinite(joints)):
            raise ValueError('Task disturbance needs four finite measured joint positions')
        jacobian, transform = self.evaluate(joints, self.mounting, np.zeros(6), self.tip)
        jacobian, transform = np.asarray(jacobian), np.asarray(transform)
        if not np.all(np.isfinite(jacobian)) or not np.all(np.isfinite(transform)):
            raise ValueError('Nonfinite tool-wrench kinematics')
        return jacobian @ self.velocity_conversion, transform

    def map(self, joints, wrench, frame='tool', world_from_body=None):
        return self.resolve(joints, wrench, frame, world_from_body)[0]

    def resolve(self, joints, wrench, frame='tool', world_from_body=None):
        """Return generalized loads and the equivalent tool-frame wrench."""
        wrench = np.asarray(wrench, dtype=float)
        if wrench.shape != (6,) or not np.all(np.isfinite(wrench)):
            raise ValueError('Tool wrench must contain six finite force/torque components')
        jacobian, transform = self.geometry(joints)
        if frame == 'tool':
            rotation = transform[:3, :3]
        elif frame == 'body':
            rotation = np.eye(3)  # URDF FLU axes, still about the tool origin.
        elif frame == 'world':
            rotation = np.asarray(world_from_body, dtype=float).T
            if (rotation.shape != (3, 3) or not np.all(np.isfinite(rotation))
                    or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
                    or not np.isclose(np.linalg.det(rotation), 1., atol=1e-8)):
                raise ValueError('World-frame wrench needs a valid world/body rotation')
        else:
            raise ValueError('Unknown tool wrench frame')
        body_wrench_at_tool = np.r_[rotation @ wrench[:3], rotation @ wrench[3:]]
        tool_from_body = transform[:3, :3].T
        tool_wrench = np.r_[tool_from_body @ body_wrench_at_tool[:3],
                            tool_from_body @ body_wrench_at_tool[3:]]
        loads = jacobian.T @ body_wrench_at_tool
        if not np.all(np.isfinite(loads)) or not np.all(np.isfinite(tool_wrench)):
            raise ValueError('Tool wrench mapping produced nonfinite loads')
        return loads, tool_wrench
