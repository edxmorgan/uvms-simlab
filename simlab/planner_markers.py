# path_markers.py
from dataclasses import dataclass
import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

@dataclass
class Colors:
    wp  : tuple = (0.1, 0.6, 0.95, 1.0)
    goal: tuple = (0.95, 0.2, 0.2, 1.0)

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

        self.pub = pub

    def clear(self, stamp, frame_id):
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

    def update(self,
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
            self.clear(stamp, frame_id)
            return

        arr = np.asarray(xyz_np, dtype=float)
        changed = (
            self._last_arr is None
            or self._last_arr.shape != arr.shape
            or np.max(np.abs(self._last_arr - arr)) > 1e-4
        )
        if not changed:
            return

        self._last_arr = arr.copy()

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
