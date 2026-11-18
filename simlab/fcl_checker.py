from bringup.scene_runtime import robot_link_owners
# fcl_world.py
import numpy as np
import fcl
from bringup.collision_geometry import distance_with_nearest, collision_surface_mesh
from typing import Dict, Tuple
from simlab.utils.meshes import fcl_bvh_from_mesh, collect_env_meshes, environment_mesh_bounds, se3_from_rpy_xyz

class FCLWorld:
    """
    Mirror of your original structure, but owned inside one class.
    Exposes bodies_robot, bodies_env, and the environment query manager
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

        self.min_coords, self.max_coords = environment_mesh_bounds(env_mesh_infos)

        # build FCL bodies identical to your original structure
        self.bodies_robot = self._build_fcl_bodies(robot_mesh_infos, "robot")
        self.bodies_env   = self._build_fcl_bodies(env_mesh_infos,   "env")
        self.bodies_dynamic = []
        self.robot_prefixes = set(robot_link_owners(urdf_string).values())
        self.missing_frames = []
        self.last_clearance_pair = None

        # managers
        self.manager_env   = fcl.DynamicAABBTreeCollisionManager()

        self.manager_env.registerObjects([b["fcl_obj"] for b in self.bodies_env])

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

            mesh = collision_surface_mesh(4, (), path_abs, tuple(scale_vec))
            transform = se3_from_rpy_xyz(rpy, xyz)
            vertices = np.asarray(mesh.vertices) @ transform[:3, :3].T + transform[:3, 3]
            bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
            out.append({
                "name":   m["link"],
                "frame":  m["link"],
                "owner": m.get("owner"),
                "fcl_obj": obj,
                "geom":   bvh,
                "local_bounds": bounds,
            })
        return out

    # --------------- TF update ---------------

    def collision_pairs(self):
        """Detailed robot/environment contacts and one pair per robot proxy.

        Robot proxies share the dynamic snapshot; never collide a robot mesh
        against its own proxy or also report the same robot pair as mesh pairs.
        """
        scripted = [b for b in self.bodies_dynamic if b['source'] != 'robot']
        proxies = [b for b in self.bodies_dynamic if b['source'] == 'robot']
        for robot in self.bodies_robot:
            for obstacle in self.bodies_env + scripted:
                yield robot, obstacle
        for index, robot in enumerate(proxies):
            for other in proxies[index + 1:]:
                if robot['owner'] != other['owner']:
                    yield robot, other

    def set_dynamic_bodies(self, bodies):
        """Synchronize current dynamic objects into the environment broad phase.

        Static objects remain TF-driven. Dynamic poses come from obstacle messages.
        Keep geometry/object ownership alive while registered with FCL.
        """
        old = {id(b['fcl_obj']): b for b in self.bodies_dynamic}
        new = {id(b['fcl_obj']): b for b in bodies}
        for key in old.keys() - new.keys():
            self.manager_env.unregisterObject(old[key]['fcl_obj'])
        for key in new.keys() - old.keys():
            self.manager_env.registerObjects([new[key]['fcl_obj']])
        self.bodies_dynamic = list(bodies)
        self.manager_env.update()

    def update_from_tf(self, tf_buffer, time_obj) -> bool:
        """
        Update transforms for robot and env bodies from TF
        Returns True only if all lookups succeed
        """
        ok_all = True
        self.missing_frames = []
        environment_changed = False
        environment_ids = {id(body) for body in self.bodies_env}
        for body in self.bodies_robot + self.bodies_env:
            try:
                t = tf_buffer.lookup_transform(self.world_frame, body["frame"], time_obj)
                q = t.transform.rotation
                p = t.transform.translation
                pose = (q.w, q.x, q.y, q.z, p.x, p.y, p.z)
                if body.get('_last_tf_pose') != pose:
                    body["fcl_obj"].setTransform(
                        fcl.Transform(list(pose[:4]), list(pose[4:]))
                    )
                    body['_last_tf_pose'] = pose
                    environment_changed |= id(body) in environment_ids
            except Exception:
                ok_all = False
                self.missing_frames.append(body['frame'])

        if environment_changed:
            self.manager_env.update()
        return ok_all

    # --------------- collision contacts ---------------

    def robot_env_contacts_one_point_per_pair(self) -> Dict[Tuple[str, str], np.ndarray]:
        """
        Many to many collision, identical pattern to your original code
        Returns map (name_robot, name_env) -> one representative world point
        """
        pair_to_point: Dict[Tuple[str, str], np.ndarray] = {}
        # Per-pair requests avoid a global contact cap hiding other obstacles.
        # Direct object ownership also disambiguates obstacles sharing cached geometry.
        for robot, env in self.collision_pairs():
            result = fcl.CollisionResult()
            fcl.collide(robot['fcl_obj'], env['fcl_obj'],
                        fcl.CollisionRequest(num_max_contacts=1, enable_contact=True), result)
            if result.contacts:
                point = np.asarray(result.contacts[0].pos, dtype=float)
                if np.all(np.isfinite(point)):
                    pair_to_point[(robot['name'], env['name'])] = point

        return pair_to_point

    # --------------- global clearance ---------------

    def global_clearance(self):
        """
        Manager to manager distance with nearest points enabled
        Returns (min_dist, nearest_point_on_robot, nearest_point_on_env)
        """
        best = None
        self.last_clearance_pair = None
        for robot, env in self.collision_pairs():
            distance, robot_point, env_point = distance_with_nearest(robot, env)
            if best is None or distance < best[0]:
                best = (distance, robot_point, env_point)
                self.last_clearance_pair = (robot['name'], env['name'])
        return best

    def has_clearance(self, margin):
        """Exact threshold query, pruning pairs by conservative world AABBs."""
        if not np.isfinite(margin) or margin < 0:
            raise ValueError('Clearance margin must be finite and nonnegative')
        self.last_clearance_pair = None
        def bounds(body):
            local = body.get('local_bounds')
            if local is None:
                return None
            obj = body['fcl_obj']
            center = obj.getRotation() @ local.mean(axis=0) + obj.getTranslation()
            extent = np.abs(obj.getRotation()) @ ((local[1] - local[0]) * .5)
            return center - extent, center + extent
        for robot, env in self.collision_pairs():
            a, b = bounds(robot), bounds(env)
            if a is not None and b is not None:
                separation = np.maximum(np.maximum(a[0] - b[1], b[0] - a[1]), 0.)
                if np.linalg.norm(separation) > margin + 1e-9:
                    continue
            result = fcl.DistanceResult()
            distance = float(fcl.distance(robot['fcl_obj'], env['fcl_obj'],
                fcl.DistanceRequest(enable_nearest_points=False), result))
            if not np.isfinite(distance) or distance <= margin:
                self.last_clearance_pair = (robot['name'], env['name'])
                return False
        return bool(self.bodies_robot and (self.bodies_env or self.bodies_dynamic))

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

    
    def planner_has_clearance_at_xyz(self, xyz, margin):
        """Threshold clearance query using an inflated sphere, without nearest points."""
        radius = self.vehicle_radius + max(0.0, float(margin))
        cached = getattr(self, '_clearance_sphere', None)
        if cached is None or cached[0] != radius:
            cached = (radius, fcl.CollisionObject(fcl.Sphere(radius)))
            self._clearance_sphere = cached
        obj = cached[1]
        obj.setTransform(fcl.Transform([1., 0., 0., 0.], list(map(float, xyz))))
        data = fcl.CollisionData(request=fcl.CollisionRequest(num_max_contacts=1, enable_contact=False))
        self.manager_env.collide(obj, data, fcl.defaultCollisionCallback)
        return not data.result.is_collision

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
    
    def planner_collision_obstacle(self, xyz):
        """Name the intersecting body for a rejected request (not a hot-path query)."""
        self._planner_obj.setTransform(fcl.Transform([1., 0., 0., 0.], list(map(float, xyz))))
        for body in self.bodies_env + self.bodies_dynamic:
            result = fcl.CollisionResult()
            fcl.collide(self._planner_obj, body['fcl_obj'], fcl.CollisionRequest(), result)
            if result.is_collision:
                return body['name']
        return ''

    def min_distance_xyz(self, xyz) -> float:
        """
        Return minimum distance between planner sphere at xyz and the environment.
        Returns +inf if the planner sphere or env is not initialized.
        """
        if self._planner_obj is None or self.manager_env is None:
            return float("inf")
        # FCL mesh/sphere distance is not a signed penetration query. Some
        # bindings return an uninitialized tiny positive distance on overlap.
        if self.planner_in_collision_at_xyz(xyz):
            return 0.0
        req = fcl.DistanceRequest(enable_nearest_points=True)
        ddata = fcl.DistanceData(request=req)
        self.manager_env.distance(self._planner_obj, ddata, fcl.defaultDistanceCallback)
        md = float(ddata.result.min_distance)
        return md
