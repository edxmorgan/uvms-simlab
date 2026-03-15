import numpy as np
import trimesh, fcl
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from scipy.spatial.transform import Rotation as R
import re
from urdf_parser_py.urdf import URDF, Mesh as URDFMesh
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

import struct

def color(r, g, b, a):
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


def se3_from_rpy_xyz(rpy, xyz):
    """Return a 4x4 homogeneous transform from rpy and xyz."""
    roll, pitch, yaw = rpy
    tx, ty, tz = xyz
    T = np.eye(4, dtype=float)
    T[:3, :3] = R.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()
    T[:3, 3] = [tx, ty, tz]
    return T

def fcl_bvh_from_mesh(path, scale=[1.0, 1.0, 1.0], rpy=(0.0, 0.0, 0.0), xyz=(0.0, 0.0, 0.0)):
    scene_or_mesh = trimesh.load(path, force='scene')
    mesh = scene_or_mesh.dump(concatenate=True) if isinstance(scene_or_mesh, trimesh.Scene) else scene_or_mesh
    if scale != 1.0:
        mesh.apply_scale(np.array(scale))
    if any(abs(v) > 1e-12 for v in rpy) or any(abs(v) > 1e-12 for v in xyz):
        T = se3_from_rpy_xyz(rpy, xyz)  # 4x4
        mesh.apply_transform(T)

    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    if V.size == 0 or F.size == 0:
        raise RuntimeError(f'Empty mesh after transforms, check file, {path}')
    bvh = fcl.BVHModel()
    bvh.beginModel(V.shape[0], F.shape[0])
    bvh.addSubModel(V, F)
    bvh.endModel()
    return bvh, V.shape[0], F.shape[0]

def make_marker(ns, mid, frame, scale_xyz, pos_world, color):
    m = Marker()
    m.header.frame_id = frame
    m.ns = ns
    m.id = mid
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.scale.x = m.scale.y = m.scale.z = scale_xyz
    m.color = color
    m.pose.position.x = float(pos_world[0])
    m.pose.position.y = float(pos_world[1])
    m.pose.position.z = float(pos_world[2])
    return m


def _parse_urdf_no_ros2_control(urdf_string: str) -> URDF:
    _ROS2_CTRL_RE = re.compile(r"<\s*ros2_control[\s\S]*?</\s*ros2_control\s*>", re.MULTILINE)
    urdf_clean = _ROS2_CTRL_RE.sub("", urdf_string)
    return URDF.from_xml_string(urdf_clean)

def _strip_file_prefix(uri: str) -> str:
    # Remove leading file:// if present
    if uri.startswith("file://"):
        return uri[len("file://"):]
    return uri

def collect_env_meshes(urdf_string: str):
    """
    Parse the URDF and return 2 lists:
      robot_out: meshes from links whose name starts with 'robot_'
      env_out:   meshes from links whose name starts with 'bathymetry_shipwreck'

    Each mesh dict now also carries info about the joint that attaches
    that link to its parent, so you can climb the tree later.

    Added fields:
      parent_link
      parent_to_link_xyz
      parent_to_link_rpy
      joint_type
      joint_name
    """

    model = _parse_urdf_no_ros2_control(urdf_string)

    robot_out = []
    env_out = []

    child_to_parent = {}
    floor_depth = 0.0

    for j in model.joints or []:
        child_name = j.child if hasattr(j, "child") else None
        parent_name = j.parent if hasattr(j, "parent") else None

        if child_name is None or parent_name is None:
            continue

        j_origin = getattr(j, "origin", None)

        if j_origin is not None:
            j_xyz = list(getattr(j_origin, "xyz", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])
            j_rpy = list(getattr(j_origin, "rpy", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])
        else:
            j_xyz = [0.0, 0.0, 0.0]
            j_rpy = [0.0, 0.0, 0.0]

        child_to_parent[child_name] = {
            "joint_name": j.name,
            "joint_type": j.type,
            "parent_link": parent_name,
            "parent_to_link_xyz": j_xyz,
            "parent_to_link_rpy": j_rpy,
            }


    for link in model.links or []:
        link_name = link.name

        # only care about robot_* and bathymetry_shipwreck*
        is_robot_link = link_name.startswith("robot_")
        is_env_link   = link_name.startswith("bathymetry_")
        is_floor_link = link_name.startswith("world_bottom")

        if not (is_robot_link or is_env_link or is_floor_link):
            continue
                
        # get the parent joint info for this link, if any
        joint_info = child_to_parent.get(link_name, None)

        if is_floor_link:
            # assume only one floor link, take the z from its first visual
            floor_depth = joint_info["parent_to_link_xyz"][2]

        # walk visuals
        for vis in getattr(link, "visuals", []) or []:
            geom = getattr(vis, "geometry", None)
            if not isinstance(geom, URDFMesh):
                continue  # skip non-mesh visuals

            origin = getattr(vis, "origin", None)

            xyz_local = list(
                getattr(origin, "xyz", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]
            )
            rpy_local = list(
                getattr(origin, "rpy", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]
            )
            scale = list(
                getattr(geom, "scale", [1.0, 1.0, 1.0]) or [1.0, 1.0, 1.0]
            )

            raw_uri = getattr(geom, "filename", "")
            uri = _strip_file_prefix(raw_uri)

            entry = {
                "link":  link_name,
                "uri":   uri,
                "scale": scale,
                "xyz":   xyz_local,   # mesh pose in link frame
                "rpy":   rpy_local,   # mesh pose in link frame
            }

            # attach parent joint info so you can recurse upward
            if joint_info is not None:
                entry["joint_name"] = joint_info["joint_name"]
                entry["joint_type"] = joint_info["joint_type"]
                entry["parent_link"] = joint_info["parent_link"]
                entry["parent_to_link_xyz"] = joint_info["parent_to_link_xyz"]
                entry["parent_to_link_rpy"] = joint_info["parent_to_link_rpy"]
            else:
                # root link case
                entry["joint_name"] = None
                entry["joint_type"] = None
                entry["parent_link"] = None
                entry["parent_to_link_xyz"] = [0.0, 0.0, 0.0]
                entry["parent_to_link_rpy"] = [0.0, 0.0, 0.0]

            # push to the appropriate list
            if is_robot_link:
                robot_out.append(entry)
            if is_env_link:
                env_out.append(entry)

    return robot_out, env_out, floor_depth


def getAABB_OBB(mesh):
    # Get the axis-aligned bounding box (AABB)
    min_coords = mesh.bounds[0]
    max_coords = mesh.bounds[1]

    # get the oriented bounding box (OBB)
    obb = mesh.bounding_box_oriented
    obb_corners = trimesh.bounds.corners(obb.bounds)

    AABB = min_coords, max_coords
    OBB = obb_corners
    return AABB, OBB

def rpy_xyz_to_mat(rpy, xyz):
    T = trimesh.transformations.euler_matrix(
        rpy[0], rpy[1], rpy[2], axes="sxyz"
    )
    T[0:3, 3] = xyz
    return T

def conc_env_trimesh(env_mesh_infos):
    meshes = []

    for info in env_mesh_infos:
        path_abs = info["uri"]
        scale_xyz = np.array(info["scale"], dtype=float)

        # mesh pose in its own link frame
        xyz_local = np.array(info["xyz"], dtype=float)
        rpy_local = np.array(info["rpy"], dtype=float)

        # link pose in its parent frame
        xyz_parent = np.array(info["parent_to_link_xyz"], dtype=float)
        rpy_parent = np.array(info["parent_to_link_rpy"], dtype=float)

        scene_or_mesh = trimesh.load(path_abs, force='scene')
        mesh = (
            scene_or_mesh.dump(concatenate=True)
            if isinstance(scene_or_mesh, trimesh.Scene)
            else scene_or_mesh
        )

        # 1. scale
        mesh.apply_scale(scale_xyz)

        # 2. place mesh inside the link frame
        T_local = rpy_xyz_to_mat(rpy_local, xyz_local)
        mesh.apply_transform(T_local)

        # 3. place that link frame into the parent frame
        T_parent = rpy_xyz_to_mat(rpy_parent, xyz_parent)
        mesh.apply_transform(T_parent)

        meshes.append(mesh)

    if len(meshes) == 0:
        return None
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def points_to_cloud2(points_xyz: np.ndarray,
                     frame_id: str = "world",
                     stamp=None) -> PointCloud2:
    """
    Build a PointCloud2 from an (N,3) float32 array in meters.
    No colors, just XYZ.
    """
    assert points_xyz.ndim == 2 and points_xyz.shape[1] == 3, \
        "points_xyz must be (N,3)"

    # Prepare header
    header = Header()
    header.frame_id = frame_id
    if stamp is not None:
        header.stamp = stamp

    # Define fields for x, y, z, each float32
    fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]

    # Pack the data into bytes
    # points_xyz.astype(np.float32).tobytes() would be fastest,
    # but ROS PointCloud2 layout requires little-endian float32 triplets,
    # which matches struct.pack("<fff") repeated.
    # We will build a bytearray for clarity.
    buff = bytearray()
    for x, y, z in points_xyz.astype(np.float32):
        buff += struct.pack('<fff', x, y, z)

    cloud = PointCloud2()
    cloud.header = header
    cloud.height = 1
    cloud.width = points_xyz.shape[0]
    cloud.fields = fields
    cloud.is_bigendian = False
    cloud.point_step = 12          # 3 floats * 4 bytes
    cloud.row_step = cloud.point_step * cloud.width
    cloud.is_dense = True          # no invalid points
    cloud.data = bytes(buff)

    return cloud