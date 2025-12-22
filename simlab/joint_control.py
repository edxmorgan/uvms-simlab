# Copyright (C) 2025 Edward Morgan
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
import rclpy
from rclpy.node import Node
# Import your robot class
from robot import Robot


class Dof_Control_Node(Node):
    def __init__(self):
        super().__init__('dof_control_node',
                         automatically_declare_parameters_from_overrides=True)

        # Retrieve parameters (e.g. number of robots, efforts, and robot prefixes).
        self.no_robot = self.get_parameter('no_robot').value
        self.no_efforts = self.get_parameter('no_efforts').value
        self.robots_prefix = self.get_parameter('robots_prefix').value
        self.controllers = self.get_parameter('controllers').value

        self.get_logger().info(f"Robot prefixes found: {self.robots_prefix}")
        self.total_no_efforts = self.no_robot * self.no_efforts
        self.get_logger().info(f"Total number of commands: {self.total_no_efforts}")

        self.robots = [Robot(self, k, 4, prefix) for k, prefix in enumerate(self.robots_prefix)]

        # Create a timer callback to publish commands at 1000 Hz.
        frequency = 1000  # Hz
        self.timer = self.create_timer(1.0 / frequency, self.timer_callback)


    def timer_callback(self):
        pass
        # command_msg = Command()
        # command_msg.command_type = self.controllers
        # command_msg.acceleration.data = [0.0]*11
        # command_msg.twist.data = [0.0]*11
        # command_msg.pose.data = []
        # for robot in self.robots:
        #     robot.publish_robot_path()
        #     ref1= [2.0, 3.0, 2.0, 0.0, 0.0, 0.0, 4.0, 3.0, 3.0, 4.0, 0.0]

        #     ref = ref1
        #     command_msg.pose.data.extend(ref)
        # # Publish the command
        # # self.get_logger().info(f'{command_msg.pose.data}')
        # self.uvms_publisher_.publish(command_msg)


###############################################################################
# Main entry point.
###############################################################################
def main(args=None):
    rclpy.init(args=args)
    dof_control_node = Dof_Control_Node()
    try:
        rclpy.spin(dof_control_node)
    except KeyboardInterrupt:
        dof_control_node.get_logger().info('dof_control node stopped by KeyboardInterrupt.')
    finally:
        dof_control_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
