# path_markers.py
from dataclasses import dataclass
import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose
import numpy as np
from scipy.spatial.transform import Rotation as R

@dataclass
class Colors:
    wp  : tuple = (0.1, 0.6, 0.95, 1.0)
    goal: tuple = (0.95, 0.2, 0.2, 1.0)
    target: tuple = (0.1, 0.9, 0.2, 1.0)

class PathPlanner:
    """
    Minimal publisher for a SPHERE_LIST of waypoints and a single goal SPHERE.
    """

    def __init__(self, pub, ns="planner", base_id=9001):
        self.planned_result = None
        self.ns = ns
        self.base_id = int(base_id)
        self.colors = Colors()
        self._last_arr = None
        self._last_wp_count = 0
        self._last_path_t_ns = 0
        self._path_refresh_period_ns = int(0.5 * 1e9)

        self._last_target_t_ns = 0
        self._last_target_xyz = None


        self.pub = pub

    def clear_path(self, stamp, frame_id):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = self.ns
        m.id = self.base_id
        m.action = Marker.DELETE
        self.pub.publish(m)
        m.id = self.base_id + 1
        self.pub.publish(m)
        self._last_arr = None
        self._last_wp_count = 0
        self._last_path_t_ns = 0

    def update_path_viz(self,
               stamp,
               frame_id,
               xyz_np,
               step=3,
               wp_size=0.08,
               goal_size=0.14):
        """
        xyz_np is Nx3, in frame_id coordinates. If None or empty, markers are cleared.
        """

        if xyz_np is None or len(xyz_np) == 0:
            self.clear_path(stamp, frame_id)
            return

        t_ns = int(stamp.sec * 1_000_000_000 + stamp.nanosec)
        arr = np.asarray(xyz_np, dtype=float)
        changed = (
            self._last_arr is None
            or self._last_arr.shape != arr.shape
            or np.max(np.abs(self._last_arr - arr)) > 1e-4
        )
        refresh_due = (t_ns - self._last_path_t_ns) >= self._path_refresh_period_ns
        if not changed and not refresh_due:
            return

        if changed:
            self._last_arr = arr.copy()
        self._last_path_t_ns = t_ns

        # subsample for waypoints, but keep the true last point for the goal
        step = max(1, int(step))
        pts_vis = arr[::step]

        # waypoint list
        wp = Marker()
        wp.header.frame_id = frame_id
        wp.header.stamp = stamp
        wp.ns = self.ns
        wp.id = self.base_id
        wp.type = Marker.SPHERE_LIST
        wp.action = Marker.ADD
        wp.scale.x = float(wp_size)
        wp.scale.y = float(wp_size)
        wp.scale.z = float(wp_size)
        wp.pose.orientation.w = 1.0
        wp.color.r, wp.color.g, wp.color.b, wp.color.a = self.colors.wp
        wp.frame_locked = True
        wp.points = []
        if len(pts_vis) > 1:
            for p in pts_vis[:-1]:
                pt = Point()
                pt.x, pt.y, pt.z = float(p[0]), float(p[1]), float(p[2])
                wp.points.append(pt)

        # clear old list if subsampling collapsed to zero
        wp_count = len(wp.points)
        if self._last_wp_count > 0 and wp_count == 0:
            clear = Marker()
            clear.header.frame_id = frame_id
            clear.header.stamp = stamp
            clear.ns = self.ns
            clear.id = self.base_id
            clear.action = Marker.DELETE
            self.pub.publish(clear)
        self._last_wp_count = wp_count

        # goal sphere from the true last element
        gx, gy, gz = map(float, arr[-1])
        goal = Marker()
        goal.header.frame_id = frame_id
        goal.header.stamp = stamp
        goal.ns = self.ns
        goal.id = self.base_id + 1
        goal.type = Marker.SPHERE
        goal.action = Marker.ADD
        goal.scale.x = float(goal_size)
        goal.scale.y = float(goal_size)
        goal.scale.z = float(goal_size)
        goal.pose.position.x = gx
        goal.pose.position.y = gy
        goal.pose.position.z = gz
        goal.pose.orientation.w = 1.0
        goal.color.r, goal.color.g, goal.color.b, goal.color.a = self.colors.goal
        goal.frame_locked = True

        if wp_count > 0:
            self.pub.publish(wp)
        self.pub.publish(goal)

    def clear_target(self, stamp, frame_id, marker_id_offset=100):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = self.ns
        m.id = self.base_id + int(marker_id_offset)
        m.action = Marker.DELETE
        self.pub.publish(m)

        self._last_target_xyz = None
        self._last_target_t_ns = 0

    def update_target_viz(
        self,
        stamp,
        frame_id,
        xyz,
        quat_wxyz=None,
        marker_id_offset=100,
        size=0.08,
        ttl_sec=0.0,
        rate_hz=30.0,
        eps=1e-4,
        as_arrow=False,
    ):
        """
        Publish a single marker for the current 'target' pose.
        - xyz: (3,) position in frame_id coordinates
        - quat_wxyz: optional (4,) orientation, defaults to identity
        - marker_id_offset: keeps ids separate from waypoint ids
        - ttl_sec: if > 0, marker auto expires in RViz if publishing stops
        - rate_hz: throttle publishing
        - eps: minimum position change to republish
        - as_arrow: if True, publish ARROW, otherwise SPHERE
        """

        if xyz is None:
            self.clear_target(stamp, frame_id, marker_id_offset)
            return

        # throttle
        t_ns = int(stamp.sec * 1_000_000_000 + stamp.nanosec)
        period_ns = int(1e9 / max(1e-6, float(rate_hz)))
        if (t_ns - self._last_target_t_ns) < period_ns:
            return

        xyz = np.asarray(xyz, dtype=float).reshape(3)
        if self._last_target_xyz is not None:
            if np.max(np.abs(xyz - self._last_target_xyz)) < float(eps):
                return

        self._last_target_t_ns = t_ns
        self._last_target_xyz = xyz.copy()

        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = self.ns
        m.id = self.base_id + int(marker_id_offset)
        m.action = Marker.ADD
        m.type = Marker.ARROW if as_arrow else Marker.SPHERE

        m.pose.position.x = float(xyz[0])
        m.pose.position.y = float(xyz[1])
        m.pose.position.z = float(xyz[2])

        if quat_wxyz is None:
            m.pose.orientation.w = 1.0
        else:
            q = np.asarray(quat_wxyz, dtype=float).reshape(4)
            m.pose.orientation.w = float(q[0])
            m.pose.orientation.x = float(q[1])
            m.pose.orientation.y = float(q[2])
            m.pose.orientation.z = float(q[3])

        m.scale.x = float(size)
        m.scale.y = float(size)
        m.scale.z = float(size)

        if as_arrow:
            # nicer arrow proportions
            m.scale.x = float(size * 2.8)  # shaft length
            m.scale.y = float(size * 0.15) # shaft diameter
            m.scale.z = float(size * 0.15) # head diameter

        m.color.r, m.color.g, m.color.b, m.color.a = self.colors.target
        m.frame_locked = True

        if ttl_sec and ttl_sec > 0.0:
            m.lifetime = Duration(sec=int(ttl_sec), nanosec=int((ttl_sec % 1.0) * 1e9))

        self.pub.publish(m)


    def quat_wxyz_from_x_to_vec_scipy(self, v):
        v = np.asarray(v, dtype=float).reshape(3)
        n = np.linalg.norm(v)
        if n < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0])  # wxyz identity

        d = v / n

        # Align +X to direction d
        r, _ = R.align_vectors([d], [[1.0, 0.0, 0.0]])

        # SciPy returns quaternion as (x, y, z, w)
        q_xyzw = r.as_quat()
        return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)  # wxyz
