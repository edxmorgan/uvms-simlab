# Copyright (C) 2026 Edward Morgan
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
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Pose
from visualization_msgs.msg import Marker


def clone_pose(pose: Pose) -> Pose:
    return deepcopy(pose)


@dataclass
class VehicleWaypointMission:
    robot_prefix: str
    position_tolerance_m: float = 0.35
    speed_tolerance_mps: float = 0.02
    yaw_tolerance_rad: float = float(np.deg2rad(5.0))
    waypoints: List[Pose] = field(default_factory=list)
    executing: bool = False
    current_index: int = 0
    active_index: Optional[int] = None
    state: str = "idle"

    def add_waypoint(self, pose: Pose) -> int:
        self.waypoints.append(clone_pose(pose))
        return len(self.waypoints)

    def pop_last_waypoint(self) -> Optional[Pose]:
        if not self.waypoints:
            return None

        removed = self.waypoints.pop()
        if self.current_index > len(self.waypoints):
            self.current_index = len(self.waypoints)
        if self.active_index is not None and self.active_index >= len(self.waypoints):
            self.active_index = None
            self.state = "idle"
            self.executing = False
        return removed

    def pop_waypoint(self, index: int) -> Optional[Pose]:
        if index < 0 or index >= len(self.waypoints):
            return None

        removed = self.waypoints.pop(index)

        if self.active_index is not None:
            if index < self.active_index:
                self.active_index -= 1
            elif index == self.active_index:
                self.active_index = None
                self.state = "idle"

        if index < self.current_index:
            self.current_index -= 1
        elif self.current_index > len(self.waypoints):
            self.current_index = len(self.waypoints)

        if not self.waypoints:
            self.executing = False
            self.current_index = 0
            self.active_index = None
            self.state = "idle"

        return removed

    def clear(self) -> None:
        self.waypoints.clear()
        self.executing = False
        self.current_index = 0
        self.active_index = None
        self.state = "idle"

    def start(self) -> bool:
        if not self.waypoints:
            return False
        self.executing = True
        self.current_index = 0
        self.active_index = None
        self.state = "idle"
        return True

    def stop(self) -> None:
        self.executing = False
        self.active_index = None
        self.state = "idle"

    def current_waypoint(self) -> Optional[Pose]:
        if self.current_index >= len(self.waypoints):
            return None
        return clone_pose(self.waypoints[self.current_index])

    def active_waypoint(self) -> Optional[Pose]:
        if self.active_index is None or self.active_index >= len(self.waypoints):
            return None
        return clone_pose(self.waypoints[self.active_index])

    def mark_planning(self) -> bool:
        if not self.executing or self.current_index >= len(self.waypoints):
            return False
        self.active_index = self.current_index
        self.state = "planning"
        return True

    def mark_tracking(self) -> None:
        if self.executing and self.active_index is not None:
            self.state = "tracking"

    def advance(self) -> bool:
        if self.active_index is None:
            return False
        self.current_index = self.active_index + 1
        self.active_index = None
        self.state = "idle"
        if self.current_index >= len(self.waypoints):
            self.executing = False
            return False
        return True

    def active_display_index(self) -> Optional[int]:
        if self.active_index is not None:
            return self.active_index
        if self.executing and self.current_index < len(self.waypoints):
            return self.current_index
        return None


class VehicleWaypointViz:
    def __init__(self, pub, ns: str, base_id: int):
        self.pub = pub
        self.ns = ns
        self.base_id = int(base_id)

    def clear(self, stamp, frame_id: str) -> None:
        for offset in range(3):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = self.ns
            marker.id = self.base_id + offset
            marker.action = Marker.DELETE
            self.pub.publish(marker)

    def update(
        self,
        *,
        stamp,
        frame_id: str,
        waypoints: List[Pose],
        active_index: Optional[int],
    ) -> None:
        if not waypoints:
            self.clear(stamp, frame_id)
            return

        points = []
        for pose in waypoints:
            pt = Point()
            pt.x = float(pose.position.x)
            pt.y = float(pose.position.y)
            pt.z = float(pose.position.z)
            points.append(pt)

        path_marker = Marker()
        path_marker.header.frame_id = frame_id
        path_marker.header.stamp = stamp
        path_marker.ns = self.ns
        path_marker.id = self.base_id
        path_marker.action = Marker.DELETE
        self.pub.publish(path_marker)

        spheres = Marker()
        spheres.header.frame_id = frame_id
        spheres.header.stamp = stamp
        spheres.ns = self.ns
        spheres.id = self.base_id + 1
        spheres.type = Marker.SPHERE_LIST
        spheres.action = Marker.ADD
        spheres.scale.x = 0.12
        spheres.scale.y = 0.12
        spheres.scale.z = 0.12
        spheres.pose.orientation.w = 1.0
        spheres.color.r = 0.08
        spheres.color.g = 0.55
        spheres.color.b = 0.95
        spheres.color.a = 0.95
        spheres.frame_locked = True
        spheres.points = points
        self.pub.publish(spheres)

        highlight = Marker()
        highlight.header.frame_id = frame_id
        highlight.header.stamp = stamp
        highlight.ns = self.ns
        highlight.id = self.base_id + 2
        if active_index is None or active_index >= len(waypoints):
            highlight.action = Marker.DELETE
            self.pub.publish(highlight)
            return

        pose = waypoints[active_index]
        highlight.type = Marker.SPHERE
        highlight.action = Marker.ADD
        highlight.pose = clone_pose(pose)
        highlight.scale.x = 0.18
        highlight.scale.y = 0.18
        highlight.scale.z = 0.18
        highlight.color.r = 0.98
        highlight.color.g = 0.80
        highlight.color.b = 0.12
        highlight.color.a = 0.98
        highlight.frame_locked = True
        highlight.lifetime = Duration(sec=0, nanosec=0)
        self.pub.publish(highlight)


def pose_position_distance(a: Pose, b: Pose) -> float:
    return float(
        np.linalg.norm(
            np.array(
                [
                    a.position.x - b.position.x,
                    a.position.y - b.position.y,
                    a.position.z - b.position.z,
                ],
                dtype=float,
            )
        )
    )
