from tf_transformations import quaternion_matrix, quaternion_from_matrix
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose
import numpy as np

def compute_bounding_sphere_radius(points, quantile=0.995, pad=0.03):
    """
    Robust radius from the origin for a cloud, using a high quantile,
    then add a small pad.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.size == 0:
        return 0.4  # fallback
    r = np.linalg.norm(pts, axis=1)
    return float(np.quantile(r, quantile) + pad)


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