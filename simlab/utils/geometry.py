from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=1)
def robot_obstacle_radius():
    """Shared vehicle/arm workspace sphere, centred at the vehicle base.

    Use the outermost workspace sample, not a random point-cloud quantile.
    This is the navigation approximation, not exact articulated geometry.
    """
    from pathlib import Path
    from ament_index_python.packages import get_package_share_directory
    path = Path(get_package_share_directory('simlab')) / 'model_functions/arm/workspace.npy'
    points = np.load(path)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or not len(points):
        raise ValueError('Invalid arm workspace for robot obstacle sphere')
    return max(.3, float(np.linalg.norm(points, axis=1).max())) + .03


def makeBox(fixed, scale, marker_type, initial_pose):
    marker = Marker()
    marker.type = marker_type
    marker.pose = initial_pose
    marker.scale.x = scale * 0.25
    marker.scale.y = scale * 0.25
    marker.scale.z = scale * 0.25

    if fixed:
        marker.color.r = 1.0 
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
    else:
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.color.a = 1.0
    return marker

def is_point_valid(workspace_hull, vehicle_body_hull, point):
    """
    Returns True if 'point' is in the workspace hull but *not* in the vehicle hull.
    Equivalently, we want:  point ∈ (Workspace \\ Vehicle) = Workspace ∩ (Vehicle)^c
    """
    inside_workspace = np.all(
        np.dot(workspace_hull.equations[:, :-1], point) + workspace_hull.equations[:, -1] <= 0
    )
    inside_vehicle = np.all(
        np.dot(vehicle_body_hull.equations[:, :-1], point) + vehicle_body_hull.equations[:, -1] <= 0
    )
    # accept the point if it is inside the workspace and *not* inside the vehicle hull.
    return inside_workspace and not inside_vehicle


def generate_rov_ellipsoid(a=0.5, b=0.3, c=0.2, num_points=10000):
    points = []
    while len(points) < num_points:
        pt = np.random.uniform(-1, 1, 3)
        if (pt[0]/a)**2 + (pt[1]/b)**2 + (pt[2]/c)**2 <= 1:
            points.append(pt)
    return points

def get_broadcast_tf(stamp, pose, parent_frame, child_frame):
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent_frame
    t.child_frame_id = child_frame
    t.transform.translation.x = pose.position.x
    t.transform.translation.y = pose.position.y
    t.transform.translation.z = pose.position.z
    t.transform.rotation = pose.orientation
    return t

def visualize_min_max_coords(min_coords, max_coords, bottom_z, world_frame):
    # Fallback if TF never came in
    pose_min = Pose()
    pose_min.position.x, pose_min.position.y, _ = min_coords
    pose_min.position.z = bottom_z
    pose_min.orientation.w = 1.0

    pose_max = Pose()
    pose_max.position.x, pose_max.position.y, _ = max_coords
    pose_max.position.z = 0.0
    pose_max.orientation.w = 1.0

    min_marker = makeBox(
        fixed=True,
        scale=1.2,
        marker_type=Marker.CUBE,
        initial_pose=pose_min,
    )
    max_marker = makeBox(
        fixed=False,
        scale=1.2,
        marker_type=Marker.CUBE,
        initial_pose=pose_max,
    )

    min_marker.header.frame_id = world_frame
    max_marker.header.frame_id = world_frame

    min_marker.ns = "fcl_extents"
    max_marker.ns = "fcl_extents"
    min_marker.id = 0
    max_marker.id = 1

    min_marker.action = Marker.ADD
    max_marker.action = Marker.ADD

    return min_marker, max_marker


def environment_grid_markers(bounds, frame):
    """Two retained square grids covering scene XY bounds, at surface and floor."""
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import MarkerArray
    import math
    xmin, xmax, ymin, ymax, bottom, _ = map(float, bounds)
    if not all(map(math.isfinite, (xmin, xmax, ymin, ymax, bottom))) or xmin > xmax or ymin > ymax:
        raise ValueError('Invalid environment grid bounds')
    span = max(50., xmax - xmin, ymax - ymin)
    # 1/2/5 metre progression; at most 200 cells across either grid.
    required = max(1., span / 198.)
    decade = 10. ** math.floor(math.log10(required))
    step = next(m * decade for m in (1., 2., 5., 10.) if m * decade >= required)
    cx, cy = (xmin + xmax) / 2., (ymin + ymax) / 2.
    x0 = math.floor((cx - span / 2.) / step) * step
    y0 = math.floor((cy - span / 2.) / step) * step
    cells = math.ceil(max(xmax - x0, ymax - y0, span) / step)
    result = MarkerArray()
    for index, (z, color) in enumerate(((0., (1., 1., 1.)), (bottom, (139/255., 69/255., 19/255.)))):
        marker = Marker()
        marker.header.frame_id = frame
        marker.ns = 'environment_grid'
        marker.id = index
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.
        marker.scale.x = .03
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = .5
        for i in range(cells + 1):
            x, y = x0 + i * step, y0 + i * step
            marker.points.extend([Point(x=x, y=y0, z=z), Point(x=x, y=y0 + cells * step, z=z),
                                  Point(x=x0, y=y, z=z), Point(x=x0 + cells * step, y=y, z=z)])
        result.markers.append(marker)
    return result
