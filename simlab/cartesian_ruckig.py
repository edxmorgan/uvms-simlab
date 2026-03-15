# cartesian_ruckig.py

# Copyright (C) 2025 Edward Morgan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY, without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

#!/usr/bin/env python3

# cartesian_ruckig.py

from ruckig import Ruckig, InputParameter, OutputParameter, Result
import numpy as np
from rclpy.node import Node

class VehicleCartesianRuckig:
    def __init__(self, rclpy_node: Node, dofs: int, control_dt: float, max_waypoints: int):
        if dofs != 3:
            raise ValueError("CartesianRuckig is intended for 3 DoF position (x, y, z)")
        self.rclpy_node = rclpy_node
        self.otg = Ruckig(dofs, control_dt, max_waypoints)
        self.inp = InputParameter(dofs)
        self.out = OutputParameter(dofs, max_waypoints)
        self.active = False
        self.yaw_finish_threshold = 0.98


    def start_from_path(
        self,
        current_position,
        path_xyz,
        max_vel,
        max_acc,
        max_jerk,
    ):
        path_xyz = np.asarray(path_xyz, dtype=float)
        if path_xyz.ndim != 2 or path_xyz.shape[1] != 3:
            raise ValueError("path_xyz must be an array of shape (N, 3)")

        current_position = np.asarray(current_position, dtype=float)
        if current_position.shape != (3,):
            raise ValueError("current_position must be length 3")

        max_vel = np.asarray(max_vel, dtype=float)
        max_acc = np.asarray(max_acc, dtype=float)
        max_jerk = np.asarray(max_jerk, dtype=float)

        self.inp.current_position = current_position.tolist()
        self.inp.current_velocity = [0.0, 0.0, 0.0]
        self.inp.current_acceleration = [0.0, 0.0, 0.0]

        # All inner waypoints as intermediate positions
        if path_xyz.shape[0] > 2:
            intermediate = [row.tolist() for row in path_xyz[1:-1]]
        else:
            intermediate = []

        self.inp.intermediate_positions = intermediate
        self.inp.target_position = path_xyz[-1].tolist()
        self.inp.target_velocity = [0.0, 0.0, 0.0]
        self.inp.target_acceleration = [0.0, 0.0, 0.0]

        self.inp.max_velocity = max_vel.tolist()
        self.inp.max_acceleration = max_acc.tolist()
        self.inp.max_jerk = max_jerk.tolist()

        self.active = True

    def update(self, yaw_blend_factor):
        """Advance one control step along the current trajectory."""
        if not self.active:
            return None, None, None, Result.Error

        res = self.otg.update(self.inp, self.out)
        pos = list(self.out.new_position)
        vel = list(self.out.new_velocity)
        acc = list(self.out.new_acceleration)

        self.out.pass_to_input(self.inp)

        if self.out.new_calculation:
            self.rclpy_node.get_logger().info(
                f"Ruckig new trajectory, calculation {self.out.calculation_duration:0.1f} µs, "
                f"duration {self.out.trajectory.duration:0.4f} s"
            )

        if res == Result.Finished and yaw_blend_factor >= self.yaw_finish_threshold:
            self.rclpy_node.get_logger().info("Ruckig trajectory finished")
            self.active = False

        return pos, vel, acc, res

