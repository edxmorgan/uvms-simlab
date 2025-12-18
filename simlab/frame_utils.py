# frame_utils.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal, Tuple, Union
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose

Frame = Literal["NED", "NWU"]
RotRep = Literal["euler_xyz", "quat_wxyz", "quat_xyzw", "matrix"]

# Signed axis reflection matrices for frame changes
# NED to NWU flips Y and Z
_REFLECT = {
    "NED": np.diag([1.0, 1.0, 1.0]),
    "NWU": np.diag([1.0, -1.0, -1.0]),
}

def _S(frame: Frame) -> np.ndarray:
    return _REFLECT[frame]

def transform_point(xyz: np.ndarray, src: Frame, dst: Frame) -> np.ndarray:
    """
    Convert a point between frames using axis reflection.

    x_dst = Sdst * Ssrc * x_src
    since S is its own inverse.
    """
    xyz = np.asarray(xyz, dtype=float).reshape(3)
    Ssrc = _S(src)
    Sdst = _S(dst)
    return (Sdst @ Ssrc @ xyz).astype(float)

def transform_rotation(rot: R, src: Frame, dst: Frame) -> R:
    """
    Convert a rotation between frames using change of basis.

    R_dst = C * R_src * C^{-1}, with C = Sdst * Ssrc and C^{-1} = C.
    """
    Ssrc = _S(src)
    Sdst = _S(dst)
    C = Sdst @ Ssrc
    R_src = rot.as_matrix()
    R_dst = C @ R_src @ C
    return R.from_matrix(R_dst)

def to_rotation(obj: Union[R, Tuple, np.ndarray], rep: RotRep) -> R:
    """Build a scipy Rotation from various reps."""
    if isinstance(obj, R):
        return obj
    if rep == "euler_xyz":
        arr = np.asarray(obj, dtype=float).reshape(3)
        return R.from_euler("xyz", arr, degrees=False)
    if rep == "quat_wxyz":
        q = np.asarray(obj, dtype=float).reshape(4)
        return R.from_quat(q, scalar_first=True)
    if rep == "quat_xyzw":
        q = np.asarray(obj, dtype=float).reshape(4)
        return R.from_quat(q, scalar_first=False)
    if rep == "matrix":
        M = np.asarray(obj, dtype=float).reshape(3, 3)
        return R.from_matrix(M)
    raise ValueError(f"Unknown rep {rep}")

def from_rotation(rot: R, rep: RotRep) -> np.ndarray:
    """Export a scipy Rotation to the requested rep."""
    if rep == "euler_xyz":
        return rot.as_euler("xyz", degrees=False)
    if rep == "quat_wxyz":
        return rot.as_quat(scalar_first=True)
    if rep == "quat_xyzw":
        return rot.as_quat(scalar_first=False)
    if rep == "matrix":
        return rot.as_matrix()
    raise ValueError(f"Unknown rep {rep}")

@dataclass
class PoseX:
    """
    Getter setter for pose across frames and rotation reps.

    Internal storage uses a chosen frame tag stored in _frame, so data is
    kept consistent without forcing a fixed canonical frame.
    """
    _p_world: np.ndarray      # position in the internal frame
    _R_world: R               # rotation in the internal frame
    _frame: Frame             # which frame the internals correspond to

    @classmethod
    def from_pose(
        cls,
        xyz: np.ndarray,
        rot: Union[R, Tuple, np.ndarray],
        rot_rep: RotRep,
        frame: Frame,
    ) -> "PoseX":
        p = np.asarray(xyz, dtype=float).reshape(3)
        R_in = to_rotation(rot, rot_rep)
        return cls(_p_world=p, _R_world=R_in, _frame=frame)

    # Setters
    def set_pose(
        self,
        xyz: np.ndarray,
        rot: Union[R, Tuple, np.ndarray],
        rot_rep: RotRep,
        frame: Frame,
    ) -> None:
        p = np.asarray(xyz, dtype=float).reshape(3)
        R_in = to_rotation(rot, rot_rep)
        # Convert incoming to internal frame
        p_int = transform_point(p, src=frame, dst=self._frame)
        R_int = transform_rotation(R_in, src=frame, dst=self._frame)
        self._p_world = p_int
        self._R_world = R_int

    # Getters
    def get_pose(
        self,
        frame: Frame,
        rot_rep: RotRep = "quat_wxyz",
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Convert internal to requested frame
        p_out = transform_point(self._p_world, src=self._frame, dst=frame)
        R_out = transform_rotation(self._R_world, src=self._frame, dst=frame)
        return p_out, from_rotation(R_out, rot_rep)

    def get_pose_as_Pose_msg(
        self,
        frame: Frame = None,
    ) -> Pose:
        if frame is None:
            frame = self._frame
        # Convert internal to requested frame
        p_out = transform_point(self._p_world, src=self._frame, dst=frame)
        R_out = transform_rotation(self._R_world, src=self._frame, dst=frame)

        self.pose_msg = Pose()
        self.pose_msg.position.x = float(p_out[0])
        self.pose_msg.position.y = float(p_out[1])
        self.pose_msg.position.z = float(p_out[2])
        q_wxyz = from_rotation(R_out, "quat_wxyz")
        self.pose_msg.orientation.x = float(q_wxyz[1])
        self.pose_msg.orientation.y = float(q_wxyz[2])
        self.pose_msg.orientation.z = float(q_wxyz[3])
        self.pose_msg.orientation.w = float(q_wxyz[0])
        return self.pose_msg
    
    def get_xyz(self, frame: Frame) -> np.ndarray:
        return transform_point(self._p_world, src=self._frame, dst=frame)

    def get_rot(self, frame: Frame, rot_rep: RotRep) -> np.ndarray:
        R_out = transform_rotation(self._R_world, src=self._frame, dst=frame)
        return from_rotation(R_out, rot_rep)

__all__ = [
    "Frame",
    "RotRep",
    "transform_point",
    "transform_rotation",
    "to_rotation",
    "from_rotation",
    "PoseX",
]
