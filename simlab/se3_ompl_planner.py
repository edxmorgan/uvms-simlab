# Copyright (C) 2025 Edward Morgan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

#!/usr/bin/env python3
import numpy as np
from functools import partial
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from fcl_checker import FCLWorld
try:
    from ompl import base as ob
    from ompl import geometric as og
except Exception as e:
    raise RuntimeError(
        "OMPL Python bindings not found. Install ompl or ompl-python."
    ) from e


class OmplPlanner():
    def __init__(self, 
                 rclpy_node: Node,
                 safety_margin=0.0,
                 env_bounds=None,
                 max_abs_roll=np.deg2rad(11.0),   # 11 degrees
                 max_abs_pitch=np.deg2rad(11.0),  # 11 degrees
                 ):
        self.rclpy_node = rclpy_node
        fcl_world = self._resolve_fcl_world(self.rclpy_node)
        self.space = ob.SE3StateSpace()

        # Bounds from FCL world AABB plus padding
        x_min, x_max, y_min, y_max, z_min, z_max = env_bounds

        self.rclpy_node.get_logger().info(
            f"Planner bounds x[{x_min:.2f}, {x_max:.2f}], "
            f"y[{y_min:.2f}, {y_max:.2f}], z[{z_min:.2f}, {z_max:.2f}]"
        )

        bounds = ob.RealVectorBounds(3)
        bounds.setLow(0, x_min); bounds.setHigh(0, x_max)
        bounds.setLow(1, y_min); bounds.setHigh(1, y_max)
        bounds.setLow(2, z_min); bounds.setHigh(2, z_max)
        self.space.setBounds(bounds)

        self.ss = og.SimpleSetup(self.space)

        checker = ob.StateValidityCheckerFn(
            partial(
                self._valid_with_fcl,
                fcl_world,
                float(safety_margin),
                x_min,
                x_max,
                y_min,
                y_max,
                z_min,
                z_max,
                float(max_abs_roll),
                float(max_abs_pitch),
            )
        )
        self.ss.setStateValidityChecker(checker)
        self.start = ob.State(self.space)
        self.goal = ob.State(self.space)


    def _resample_by_distance(self, xyz_np, quat_wxyz_np, spacing_m=0.20, max_points=2000):
        if xyz_np.shape[0] <= 2 or spacing_m <= 0:
            return xyz_np, quat_wxyz_np

        diffs = np.linalg.norm(np.diff(xyz_np, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(diffs)])
        total = float(cum[-1])
        if total == 0.0:
            return xyz_np[:1], quat_wxyz_np[:1]

        targets = np.arange(0.0, total, spacing_m)
        if targets.size == 0 or targets[-1] < total:
            targets = np.concatenate([targets, [total]])

        if targets.size > max_points:
            stride = int(np.ceil(targets.size / max_points))
            targets = targets[::stride]
            if targets[-1] != total:
                targets = np.concatenate([targets, [total]])

        # segment index for each target
        j = np.searchsorted(cum, targets, side="right") - 1
        j = np.clip(j, 0, len(cum) - 2)

        # interpolation factor within segment
        seg_len = (cum[j + 1] - cum[j])
        w = np.zeros_like(targets)
        good = seg_len > 1e-12
        w[good] = (targets[good] - cum[j[good]]) / seg_len[good]
        w = w[:, None]

        # linear interpolate xyz
        xyz_rs = (1.0 - w) * xyz_np[j] + w * xyz_np[j + 1]

        # SLERP quats along normalized distance
        s = targets / total
        q0 = quat_wxyz_np[0]
        q1 = quat_wxyz_np[-1]
        q0_xyzw = np.array([q0[1], q0[2], q0[3], q0[0]], dtype=float)
        q1_xyzw = np.array([q1[1], q1[2], q1[3], q1[0]], dtype=float)

        from scipy.spatial.transform import Rotation as R, Slerp
        key_rots = R.from_quat(np.vstack([q0_xyzw, q1_xyzw]))
        slerp = Slerp([0.0, 1.0], key_rots)
        quat_xyzw = slerp(s).as_quat()
        quat_rs = np.column_stack([quat_xyzw[:, 3], quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2]])

        return xyz_rs, quat_rs


    def _valid_with_fcl(
        self,
        fcl_world: FCLWorld,
        safety_margin: float,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float,
        max_abs_roll: float,
        max_abs_pitch: float,
        state,
    ):
        x, y, z = state.getX(), state.getY(), state.getZ()
        if not (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max):
            return False

        # Orientation constraint, keep roll and pitch near zero
        q = state.rotation()
        # OMPL uses w x y z, scipy uses x y z w
        quat = [q.x, q.y, q.z, q.w]
        roll, pitch, yaw = R.from_quat(quat).as_euler("xyz", degrees=False)
        if abs(roll) > max_abs_roll or abs(pitch) > max_abs_pitch:
            return False
        pw = np.array([x, y, z], float)

        if safety_margin is not None and safety_margin > 0.0:
            d = fcl_world.min_distance_xyz(pw)
            return d >= safety_margin
        else:
            in_collision = fcl_world.planner_in_collision_at_xyz(pw)
            return not in_collision


    def _resolve_fcl_world(self, rclpy_node: Node) -> FCLWorld:
        if hasattr(rclpy_node, "fcl_world") and getattr(rclpy_node, "fcl_world") is not None:
            return rclpy_node.fcl_world

        # backend = getattr(rclpy_node, "uvms_backend", None)
        # if backend is not None and hasattr(backend, "fcl_world") and backend.fcl_world is not None:
        #     return backend.fcl_world

        raise RuntimeError(
            "FCL world is unavailable. Expected rclpy_node.fcl_world "
            "or rclpy_node.uvms_backend.fcl_world."
        )

    def _get_path_length_objective(self, si: ob.SpaceInformation, threshold: float | None = None):
        obj = ob.PathLengthOptimizationObjective(si)
        if threshold is not None:
            obj.setCostThreshold(ob.Cost(float(threshold)))
        return obj


    def plan_se3_path(
        self,
        start_xyz,
        start_quat_wxyz,
        goal_xyz,
        goal_quat_wxyz,
        time_limit=0.75,
        spacing_m=0.20,
        dense_interpolation=400,
        max_points=2000,
        planner_type: str = "BITstar",   # "BITstar" or "RRTstar"
        path_length_threshold: float | None = None,  # optional optimality threshold
    ):
        # Reset planning data for a clean solve invocation.
        self.ss.clear()

        # fill poses
        s = self.start()
        s.setX(float(start_xyz[0])); s.setY(float(start_xyz[1])); s.setZ(float(start_xyz[2]))
        sr = s.rotation()
        sr.w = float(start_quat_wxyz[0]); sr.x = float(start_quat_wxyz[1])
        sr.y = float(start_quat_wxyz[2]); sr.z = float(start_quat_wxyz[3])

        g = self.goal()
        g.setX(float(goal_xyz[0])); g.setY(float(goal_xyz[1])); g.setZ(float(goal_xyz[2]))
        gr = g.rotation()
        gr.w = float(goal_quat_wxyz[0]); gr.x = float(goal_quat_wxyz[1])
        gr.y = float(goal_quat_wxyz[2]); gr.z = float(goal_quat_wxyz[3])

        # Enforce bounds to normalize SO3 and clamp R3
        self.space.enforceBounds(s)
        self.space.enforceBounds(g)

        si = self.ss.getSpaceInformation()
        self.ss.setStartAndGoalStates(self.start, self.goal)

        if not si.satisfiesBounds(s):
            raise RuntimeError("Start violates bounds after enforceBounds")
        if not si.satisfiesBounds(g):
            raise RuntimeError("Goal violates bounds after enforceBounds")

        # Optimality objective, minimize path length, optional threshold
        objective = self._get_path_length_objective(si, path_length_threshold)
        self.ss.setOptimizationObjective(objective)

        planner = None
        # planner and resolution
        if planner_type == "RRTstar":
            planner = og.RRTstar(si)
        elif planner_type == "Bitstar":
            planner = og.BITstar(si)
        elif planner_type == "RRTConnect":
            planner = og.RRTConnect(si)

        if planner is None:
            return {
                "is_success":False,
                "message":"Planner missing. Select planner"
            }
        
        # planner and resolution
        self.ss.setPlanner(planner)

        self.space.setLongestValidSegmentFraction(0.01)
        si.setStateValidityCheckingResolution(0.01)

        if not self.ss.solve(time_limit):
            return {
                "is_success":False,
                "message":"Planner did not find a solution"
            }

        path = self.ss.getSolutionPath()
        # Densify first for a smoother arc length estimate
        path.interpolate(int(dense_interpolation))

        # Collect the dense path
        xyz_dense, quat_dense = [], []
        for k in range(path.getStateCount()):
            st = path.getState(k)
            q = st.rotation()
            xyz_dense.append([st.getX(), st.getY(), st.getZ()])
            quat_dense.append([q.w, q.x, q.y, q.z])

        xyz_dense = np.asarray(xyz_dense, float)
        quat_dense = np.asarray(quat_dense, float)

        # Resample at fixed spatial spacing to keep PID errors modest
        xyz_rs, quat_rs = self._resample_by_distance(
            xyz_dense, quat_dense,
            spacing_m=float(spacing_m),
            max_points=int(max_points)
        )

        # Compute final cost and raw geometric length of the solution path
        try:
            cost_obj = path.cost(objective)          # OMPL computes path cost here
            cost_val = float(cost_obj.value())      # unwrap Cost to a float
        except Exception:
            cost_val = None

        try:
            geom_length = float(path.length())      # Euclidean length in state space
        except Exception:
            geom_length = None


        return {
            "xyz": xyz_rs,
            "quat_wxyz": quat_rs,
            "count": int(xyz_rs.shape[0]),
            "is_success": True,
            "path_length_cost": cost_val,
            "geom_length": geom_length,
            "message": (
                f"Planner found solution with {xyz_rs.shape[0]} waypoints at ~{spacing_m} m spacing"
                + (f", cost {cost_val:.3f}" if cost_val is not None else "")
                + (f", geom length {geom_length:.3f}" if geom_length is not None else "")
            ),
        }
