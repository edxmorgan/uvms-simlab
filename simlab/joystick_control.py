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
from robot import Robot
from typing import List
import tf2_ros
###############################################################################
# ROS2 Node that uses the PS4 controller for ROV teleoperation.
#
# The ROV command is built as follows:
#   - ROV Command (6 elements): [surge, sway, heave, roll, pitch, yaw]
#       surge  = - (left stick vertical)   (inverted so that pushing forward is positive)
#       sway   = left stick horizontal
#       heave  = analog value from triggers
#       roll   = 0.0 (unused)
#       pitch  = right stick vertical
#       yaw    = right stick horizontal
#
#   - Manipulator Command (5 elements): all zeros.
#
# Total command for each robot is 11 elements.
###############################################################################

class PS4TeleopNode(Node):
    def __init__(self):
        super().__init__('ps4_teleop_node',
                         automatically_declare_parameters_from_overrides=True)

        # Retrieve parameters (e.g. number of robots, efforts, and robot prefixes).
        self.no_robot = self.get_parameter('no_robot').value
        self.no_efforts = self.get_parameter('no_efforts').value
        self.robots_prefix = self.get_parameter('robots_prefix').value
        self.controllers = self.get_parameter('controllers').value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info(f"Robot prefixes found: {self.robots_prefix}")
        self.total_no_efforts = self.no_robot * self.no_efforts
        self.get_logger().info(f"Total number of commands: {self.total_no_efforts}")

        self.robots:List[Robot] = []
        
        for k, (prefix, controller) in enumerate(list(zip(self.robots_prefix, self.controllers))):
            robot_k = Robot(self,  self.tf_buffer, k, 4, prefix, controller)
            self.robots.append(robot_k)

        # Create a timer callback to publish commands at 1000 Hz.
        frequency = 1000  # Hz
        self.timer = self.create_timer(1.0 / frequency, self.timer_callback)    

    def timer_callback(self):
        for robot in self.robots:
            [surge, sway, heave, roll, pitch, yaw] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            [e_joint, d_joint, c_joint, b_joint, a_joint] = [0.0, 0.0, 0.0, 0.0, 0.0]

            if robot.has_joystick_interface:
                # Safely acquire the latest controller values.
                with robot.controller_lock:
                    surge = robot.rov_surge
                    sway = robot.rov_sway
                    heave = robot.rov_z
                    roll = robot.rov_roll
                    pitch = robot.rov_pitch
                    yaw = robot.rov_yaw

                    e_joint= robot.jointe
                    d_joint= robot.jointd
                    c_joint= robot.jointc
                    b_joint= robot.jointb
                    a_joint= robot.jointa
            wrench_body_6 = [surge, sway, heave, roll, pitch, yaw]
            arm_effort_5 = [e_joint, d_joint, c_joint, b_joint, a_joint]
            # self.get_logger().info(f"ROV Command: {wrench_body_6}, Arm Command: {arm_effort_5}")
            robot.publish_commands(wrench_body_6, arm_effort_5)

def main(args=None):
    rclpy.init(args=args)
    teleop_node = PS4TeleopNode()
    try:
        rclpy.spin(teleop_node)
    except KeyboardInterrupt:
        teleop_node.get_logger().info('PS4 Teleop node stopped by KeyboardInterrupt.')
    finally:
        teleop_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
