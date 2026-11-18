"""Load the existing exported forward model without altering its mathematics."""
from pathlib import Path
import casadi as ca
import numpy as np
from simlab.utils.frames import PoseX


def load_forward_kinematics(resource_directory):
    """Return FK(q, mounting XYZ/RPY, world XYZ/RPY, tip XYZ/RPY).

    The model returns link poses in world NWU as XYZ + quaternion WXYZ.
    The final output is the tool pose. Inputs use the exported joint order.
    Each call loads a separate model instance for its owner.
    """
    return ca.Function.load(str(Path(resource_directory) / 'model_functions/arm/fk_eval.casadi'))


def tool_pose_from_sample(sample, fk, mounting, tip):
    """Return tool XYZ/RPY in map NWU; never use a delayed TF observation.

    Incoming vehicle pose is dynamics NED/FRD. The exported arm model expects
    NWU/FLU. Both samples use the same map, mounting, and tip geometry.
    """
    pose = np.asarray(sample['pose'], dtype=float).reshape(-1)
    joints = np.asarray(sample['q'], dtype=float).reshape(-1)
    if pose.size != 6 or joints.size < 4 or not np.isfinite(np.r_[pose, joints]).all():
        raise ValueError('Invalid tool metric state')
    xyz, rpy = PoseX.from_pose(
        xyz=pose[:3], rot=pose[3:], rot_rep='euler_xyz', frame='NED',
    ).get_pose(frame='NWU', rot_rep='euler_xyz')
    tool = np.asarray(fk(joints[:4], mounting, np.r_[xyz, rpy], tip)[-1], dtype=float).reshape(-1)
    if tool.size != 7 or not np.isfinite(tool).all() or np.linalg.norm(tool[3:]) < 1e-12:
        raise ValueError('Invalid forward-kinematics tool pose')
    xyz, rpy = PoseX.from_pose(
        xyz=tool[:3], rot=tool[3:], rot_rep='quat_wxyz', frame='NWU',
    ).get_pose(frame='NWU', rot_rep='euler_xyz')
    return np.r_[xyz, rpy]
