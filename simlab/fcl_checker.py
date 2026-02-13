# fcl_world.py
import numpy as np
import fcl
from typing import Dict, Tuple
from mesh_utils import fcl_bvh_from_mesh, collect_env_meshes, conc_env_trimesh, getAABB_OBB

class FCLWorld:
    """
    Mirror of your original structure, but owned inside one class.
    Exposes bodies_robot, bodies_env, manager_robot, manager_env
    Provides TF updates, collision contacts, and global clearance
    """

    def __init__(self, urdf_string: str, world_frame: str = "world", vehicle_radius: float = 0.4):
        if not urdf_string:
            raise ValueError("URDF string is empty")

        self.world_frame = world_frame
        self.vehicle_radius = float(vehicle_radius)

        # parse URDF
        robot_mesh_infos, env_mesh_infos, floor_depth = collect_env_meshes(urdf_string)
        self.floor_depth = floor_depth

        # merge meshes into one Trimesh in world frame
        env_mesh = conc_env_trimesh(env_mesh_infos)
        AABB, OBB = getAABB_OBB(env_mesh)
        self.min_coords, self.max_coords = AABB
        self.obb_corners = OBB

        # build FCL bodies identical to your original structure
        self.bodies_robot = self._build_fcl_bodies(robot_mesh_infos, "robot")
        self.bodies_env   = self._build_fcl_bodies(env_mesh_infos,   "env")

        # managers
        self.manager_robot = fcl.DynamicAABBTreeCollisionManager()
        self.manager_env   = fcl.DynamicAABBTreeCollisionManager()

        self.manager_robot.registerObjects([b["fcl_obj"] for b in self.bodies_robot])
        self.manager_env.registerObjects([b["fcl_obj"] for b in self.bodies_env])

        self.manager_robot.setup()
        self.manager_env.setup()

        # planner sphere helper, optional
        self._planner_geom = fcl.Sphere(self.vehicle_radius)
        self._planner_obj  = fcl.CollisionObject(self._planner_geom, fcl.Transform())

        self.env_xyz_bounds = self._compute_env_bounds_from_fcl(z_min=self.floor_depth, pad_xy=0.0, pad_z=0.0)

    def _compute_env_bounds_from_fcl(self, z_min, pad_xy=0.5, pad_z=1e-4):
        """
        Compute planner bounds from FCL world's AABB, with a small padding.
        If fcl_world is None or does not have min_coords, fall back to a large box.
        """
        min_c = np.asarray(self.min_coords, float)
        max_c = np.asarray(self.max_coords, float)

        x_min = float(min_c[0] - pad_xy)
        x_max = float(max_c[0] + pad_xy)
        y_min = float(min_c[1] - pad_xy)
        y_max = float(max_c[1] + pad_xy)

        z_max = 0.0 + pad_z

        return x_min, x_max, y_min, y_max, z_min, z_max

    # --------------- structure helpers ---------------

    def _build_fcl_bodies(self, link_list, kind: str):
        out = []
        for m in link_list:
            path_abs  = m["uri"]
            scale_vec = m["scale"]
            xyz       = tuple(m["xyz"])
            rpy       = tuple(m["rpy"])

            bvh, _, _ = fcl_bvh_from_mesh(path_abs, scale_vec, rpy, xyz)
            obj = fcl.CollisionObject(bvh, fcl.Transform())

            out.append({
                "name":   m["link"],
                "frame":  m["link"],
                "fcl_obj": obj,
                "geom":   bvh,
            })
        return out

    # --------------- TF update ---------------

    def update_from_tf(self, tf_buffer, time_obj) -> bool:
        """
        Update transforms for robot and env bodies from TF
        Returns True only if all lookups succeed
        """
        ok_all = True
        for body in self.bodies_robot + self.bodies_env:
            try:
                t = tf_buffer.lookup_transform(self.world_frame, body["frame"], time_obj)
                q = t.transform.rotation
                p = t.transform.translation
                body["fcl_obj"].setTransform(
                    fcl.Transform([q.w, q.x, q.y, q.z], [p.x, p.y, p.z])
                )
            except Exception:
                ok_all = False

        self.manager_robot.update()
        self.manager_env.update()
        return ok_all

    # --------------- collision contacts ---------------

    def robot_env_contacts_one_point_per_pair(self) -> Dict[Tuple[str, str], np.ndarray]:
        """
        Many to many collision, identical pattern to your original code
        Returns map (name_robot, name_env) -> one representative world point
        """
        req = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
        cdata = fcl.CollisionData(request=req)
        self.manager_robot.collide(self.manager_env, cdata, fcl.defaultCollisionCallback)

        # map id(geom) to name like your original
        geom_id_to_name = {}
        for b in self.bodies_robot + self.bodies_env:
            geom_id_to_name[id(b["geom"])] = b["name"]

        pair_to_point: Dict[Tuple[str, str], np.ndarray] = {}
        for contact in cdata.result.contacts:
            n0 = geom_id_to_name.get(id(contact.o1), "unknown")
            n1 = geom_id_to_name.get(id(contact.o2), "unknown")

            # keep robot vs env pairing order for readability
            if n0 in [b["name"] for b in self.bodies_robot] and n1 in [e["name"] for e in self.bodies_env]:
                key = (n0, n1)
            elif n1 in [b["name"] for b in self.bodies_robot] and n0 in [e["name"] for e in self.bodies_env]:
                key = (n1, n0)
            else:
                # unknown pairing, still sort to avoid duplication
                key = tuple(sorted([n0, n1]))

            if key not in pair_to_point:
                pair_to_point[key] = np.array(contact.pos, dtype=float)

        return pair_to_point

    # --------------- global clearance ---------------

    def global_clearance(self):
        """
        Manager to manager distance with nearest points enabled
        Returns (min_dist, nearest_point_on_robot, nearest_point_on_env)
        """
        req = fcl.DistanceRequest(enable_nearest_points=True)
        ddata = fcl.DistanceData(request=req)
        self.manager_robot.distance(self.manager_env, ddata, fcl.defaultDistanceCallback)

        # nearest_points is two 3D tuples
        if ddata.result is not None and ddata.result.nearest_points is not None and len(ddata.result.nearest_points) == 2:
            md = float(ddata.result.min_distance)
            pr = np.array(ddata.result.nearest_points[0], dtype=float)
            pe = np.array(ddata.result.nearest_points[1], dtype=float)
            resp = md, pr, pe
        else:
            resp = None
        return resp

    # --------------- optional planner sphere helpers ---------------
    def set_robot_collision_radius(self, r: float):
        self.vehicle_radius = float(r)
        self._planner_geom = fcl.Sphere(self.vehicle_radius)
        self._planner_obj  = fcl.CollisionObject(self._planner_geom, fcl.Transform())

    def enforce_bounds(self, xyz, *, for_sphere: bool = True):
        x_min, x_max, y_min, y_max, z_min, z_max = self.env_xyz_bounds

        r = float(self.vehicle_radius) if for_sphere else 0.0

        # shrink bounds so the sphere stays fully inside
        x_min_c = x_min + r
        x_max_c = x_max - r
        y_min_c = y_min + r
        y_max_c = y_max - r
        z_min_c = z_min + r
        z_max_c = z_max

        # handle degenerate case, sphere too big for the box
        if x_min_c > x_max_c or y_min_c > y_max_c or z_min_c > z_max_c:
            # safest behavior: clamp to the box center and let collision checker handle it
            cx = 0.5 * (x_min + x_max)
            cy = 0.5 * (y_min + y_max)
            cz = 0.5 * (z_min + z_max)
            return np.array([cx, cy, cz], dtype=float)

        mins = np.array([x_min_c, y_min_c, z_min_c], dtype=float)
        maxs = np.array([x_max_c, y_max_c, z_max_c], dtype=float)

        xyz = np.array([float(x) for x in xyz], dtype=float)
        return np.clip(xyz, mins, maxs)

    
    def planner_in_collision_at_xyz(self, xyz) -> bool:
        """
        Check collision between the planner sphere at xyz and the environment.
        Returns False if the planner sphere or env is not initialized.
        """
        if self._planner_obj is None or self.manager_env is None:
            return False
        self._planner_obj.setTransform(fcl.Transform([1.0, 0.0, 0.0, 0.0], [float(x) for x in xyz]))
        req = fcl.CollisionRequest(num_max_contacts=1, enable_contact=False)
        cdata = fcl.CollisionData(request=req)
        self.manager_env.collide(self._planner_obj, cdata, fcl.defaultCollisionCallback)
        return bool(cdata.result.is_collision)
    
    def min_distance_xyz(self, xyz) -> float:
        """
        Return minimum distance between planner sphere at xyz and the environment.
        Returns +inf if the planner sphere or env is not initialized.
        """
        if self._planner_obj is None or self.manager_env is None:
            return float("inf")
        self._planner_obj.setTransform(fcl.Transform([1.0, 0.0, 0.0, 0.0], [float(x) for x in xyz]))
        req = fcl.DistanceRequest(enable_nearest_points=True)
        ddata = fcl.DistanceData(request=req)
        self.manager_env.distance(self._planner_obj, ddata, fcl.defaultDistanceCallback)
        md = float(ddata.result.min_distance)
        return md
