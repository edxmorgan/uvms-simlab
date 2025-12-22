# se3_ompl_planner.py
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





def _resample_by_distance(xyz_np, quat_np, spacing_m=0.20, max_points=2000):
    """
    Resample path at approximately fixed translation spacing.
    xyz_np:  (N, 3)
    quat_np: (N, 4) wxyz
    Returns xyz_rs, quat_rs with first and last included.
    """
    if xyz_np.shape[0] <= 2 or spacing_m <= 0:
        return xyz_np, quat_np

    # cumulative translation distance
    diffs = np.linalg.norm(np.diff(xyz_np, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(diffs)])
    total = float(cum[-1])

    if total == 0.0:
        return xyz_np[:1], quat_np[:1]

    # target distances, always include 0 and total
    targets = np.arange(0.0, total, spacing_m)
    if targets.size == 0 or targets[-1] < total:
        targets = np.concatenate([targets, [total]])

    # map target distances to indices in the dense path
    idx = np.searchsorted(cum, targets, side="left")
    idx = np.clip(idx, 0, len(cum) - 1)

    # enforce uniqueness and max size
    idx = np.unique(idx)
    if idx.size > max_points:
        stride = int(np.ceil(idx.size / max_points))
        idx = idx[::stride]
        if idx[-1] != len(cum) - 1:
            idx = np.concatenate([idx, [len(cum) - 1]])

    return xyz_np[idx], quat_np[idx]

def _valid_with_fcl(
    rclpy_node: Node,
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
        d = rclpy_node.uvms_backend.fcl_world.min_distance_xyz(pw)
        return d >= safety_margin
    else:
        in_collision = rclpy_node.uvms_backend.fcl_world.planner_in_collision_at_xyz(pw)
        return not in_collision

def _get_path_length_objective(si: ob.SpaceInformation, threshold: float | None = None):
    obj = ob.PathLengthOptimizationObjective(si)
    if threshold is not None:
        obj.setCostThreshold(ob.Cost(float(threshold)))
    return obj


def plan_se3_path(
    rclpy_node: Node,
    start_xyz,
    start_quat_wxyz,
    goal_xyz,
    goal_quat_wxyz,
    time_limit=0.75,
    safety_margin=0.0,
    env_bounds=None,
    spacing_m=0.20,
    dense_interpolation=400,
    max_points=2000,
    max_abs_roll=np.deg2rad(11.0),   # 11 degrees
    max_abs_pitch=np.deg2rad(11.0),  # 11 degrees
    planner_type: str = "BITstar",   # "BITstar" or "RRTstar"
    path_length_threshold: float | None = None,  # optional optimality threshold
):
    space = ob.SE3StateSpace()

    # Bounds from FCL world AABB plus padding
    x_min, x_max, y_min, y_max, z_min, z_max = env_bounds

    rclpy_node.get_logger().info(
        f"Planner bounds x[{x_min:.2f}, {x_max:.2f}], "
        f"y[{y_min:.2f}, {y_max:.2f}], z[{z_min:.2f}, {z_max:.2f}]"
    )

    bounds = ob.RealVectorBounds(3)
    bounds.setLow(0, x_min); bounds.setHigh(0, x_max)
    bounds.setLow(1, y_min); bounds.setHigh(1, y_max)
    bounds.setLow(2, z_min); bounds.setHigh(2, z_max)
    space.setBounds(bounds)

    ss = og.SimpleSetup(space)

    checker = ob.StateValidityCheckerFn(
        partial(
            _valid_with_fcl,
            rclpy_node,
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
    ss.setStateValidityChecker(checker)
    start = ob.State(space)
    goal = ob.State(space)

    # fill poses
    s = start()
    s.setX(float(start_xyz[0])); s.setY(float(start_xyz[1])); s.setZ(float(start_xyz[2]))
    sr = s.rotation()
    sr.w = float(start_quat_wxyz[0]); sr.x = float(start_quat_wxyz[1])
    sr.y = float(start_quat_wxyz[2]); sr.z = float(start_quat_wxyz[3])

    g = goal()
    g.setX(float(goal_xyz[0])); g.setY(float(goal_xyz[1])); g.setZ(float(goal_xyz[2]))
    gr = g.rotation()
    gr.w = float(goal_quat_wxyz[0]); gr.x = float(goal_quat_wxyz[1])
    gr.y = float(goal_quat_wxyz[2]); gr.z = float(goal_quat_wxyz[3])

    # Enforce bounds to normalize SO3 and clamp R3
    space.enforceBounds(s)
    space.enforceBounds(g)

    si = ss.getSpaceInformation()
    ss.setStartAndGoalStates(start, goal)

    if not si.satisfiesBounds(s):
        raise RuntimeError("Start violates bounds after enforceBounds")
    if not si.satisfiesBounds(g):
        raise RuntimeError("Goal violates bounds after enforceBounds")

    # Optimality objective, minimize path length, optional threshold
    objective = _get_path_length_objective(si, path_length_threshold)
    ss.setOptimizationObjective(objective)


    # planner and resolution
    if planner_type == "RRTstar":
        planner = og.RRTstar(si)
    else:
        planner = og.BITstar(si)


    # planner and resolution
    ss.setPlanner(planner)

    space.setLongestValidSegmentFraction(0.01)
    si.setStateValidityCheckingResolution(0.01)

    if not ss.solve(time_limit):
        return {
            "is_success":False,
            "message":"Planner did not find a solution"
        }

    path = ss.getSolutionPath()
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
    xyz_rs, quat_rs = _resample_by_distance(
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
