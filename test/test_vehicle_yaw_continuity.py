import numpy as np

from scipy.spatial.transform import Rotation as R

from simlab.robot import Robot, yaw_only_quat_wxyz


def test_vehicle_command_yaw_unwraps_against_previous_command():
    robot = Robot.__new__(Robot)
    robot.ned_pose = [0.0, 0.0, 0.0, 0.0, 0.0, -2.10]
    robot._last_vehicle_cmd_yaw = None
    robot._last_vehicle_target_yaw = None
    robot._last_vehicle_cmd_yaw_step = 0.0

    max_step = 0.02
    first = robot.continuous_vehicle_command_yaw(0.852, fallback_yaw=robot.ned_pose[5], max_step=max_step)
    second = robot.continuous_vehicle_command_yaw(-5.046, fallback_yaw=robot.ned_pose[5], max_step=max_step)
    third = robot.continuous_vehicle_command_yaw(0.877, fallback_yaw=robot.ned_pose[5], max_step=max_step)

    assert abs(first - robot.ned_pose[5]) <= max_step + 1e-12
    assert abs(second - first) <= max_step + 1e-12
    assert abs(third - second) <= max_step + 1e-12
    assert first > robot.ned_pose[5]
    assert second > first
    assert third > second


def test_vehicle_reference_pose_unwraps_before_controller_dispatch():
    robot = Robot.__new__(Robot)
    robot.ned_pose = [0.0, 0.0, 0.0, 0.0, 0.0, -2.107]
    robot._last_vehicle_cmd_yaw = None
    robot._last_vehicle_target_yaw = None
    robot._last_vehicle_cmd_yaw_step = 0.0
    positive_branch = np.zeros(6)
    negative_branch = np.zeros(6)
    positive_branch[5] = 0.852
    negative_branch[5] = -5.046

    first = robot.continuous_vehicle_reference_pose(positive_branch, fallback_yaw=robot.ned_pose[5])
    second = robot.continuous_vehicle_reference_pose(negative_branch, fallback_yaw=robot.ned_pose[5])

    assert abs((first[5] - robot.ned_pose[5]) - 2.959) < 1e-6
    assert abs(second[5] - first[5]) < 0.5
    assert second[5] > 0.0
    assert negative_branch[5] == -5.046


def test_vehicle_command_yaw_keeps_shortest_turn_when_direction_changes():
    robot = Robot.__new__(Robot)
    robot.ned_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 2.0]
    robot._last_vehicle_cmd_yaw = 2.0
    robot._last_vehicle_target_yaw = 2.0
    robot._last_vehicle_cmd_yaw_step = 0.02

    command = robot.continuous_vehicle_command_yaw(0.0, fallback_yaw=robot.ned_pose[5], max_step=0.02)

    assert command < 2.0
    assert abs(command - 1.98) < 1e-12
    assert robot._last_vehicle_cmd_yaw_step < 0.0


def test_vehicle_path_heading_yaw_uses_ned_direction():
    robot = Robot.__new__(Robot)

    north_yaw = robot._yaw_from_ned_direction([1.0, 0.0, 0.0], fallback_yaw=1.0)
    east_yaw = robot._yaw_from_ned_direction([0.0, 1.0, 0.0], fallback_yaw=0.0)

    assert north_yaw == 0.0
    assert east_yaw == np.pi / 2.0


def test_vehicle_path_heading_yaw_holds_when_direction_is_too_small():
    robot = Robot.__new__(Robot)

    yaw = robot._yaw_from_ned_direction([0.001, 0.0, 0.0], fallback_yaw=1.25, speed_threshold=0.03)

    assert yaw == 1.25


def test_planner_yaw_only_quaternion_removes_transient_roll_pitch():
    quat_xyzw = R.from_euler("xyz", [0.2, -0.25, 1.1]).as_quat()
    quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

    projected = yaw_only_quat_wxyz(quat_wxyz)
    projected_xyzw = [projected[1], projected[2], projected[3], projected[0]]
    roll, pitch, yaw = R.from_quat(projected_xyzw).as_euler("xyz", degrees=False)

    assert abs(roll) < 1e-12
    assert abs(pitch) < 1e-12
    assert abs(yaw - 1.1) < 1e-12
