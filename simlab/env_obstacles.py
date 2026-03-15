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
import os
import numpy as np
import rclpy
from rclpy.node import Node


class EnvObstacleNode(Node):
    def __init__(self):
        super().__init__("env_obstacle_node")

        # timer
        self.timer = self.create_timer(0.1, self.obstacles)

    def obstacles(self):
        pass
        # self.get_logger().warn(
        #         'obstacles in the wayyyyyy!!!!'
        #     )


def main():
    rclpy.init()
    node = EnvObstacleNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
