import sys
import types

import numpy as np

control_msgs_msg = sys.modules.get("control_msgs.msg")
if control_msgs_msg is not None:
    if not hasattr(control_msgs_msg, "DynamicInterfaceGroupValues"):
        control_msgs_msg.DynamicInterfaceGroupValues = object
    if not hasattr(control_msgs_msg, "DynamicJointState"):
        control_msgs_msg.DynamicJointState = object
    if not hasattr(control_msgs_msg, "InterfaceValue"):
        control_msgs_msg.InterfaceValue = object

simlab_msg = sys.modules.get("simlab.msg")
if simlab_msg is None:
    simlab_msg = types.ModuleType("simlab.msg")
    sys.modules["simlab.msg"] = simlab_msg
if not hasattr(simlab_msg, "ControllerPerformance"):
    simlab_msg.ControllerPerformance = object
if not hasattr(simlab_msg, "ReferenceTargets"):
    simlab_msg.ReferenceTargets = object

simlab_srv = sys.modules.get("simlab.srv")
if simlab_srv is None:
    simlab_srv = types.ModuleType("simlab.srv")
    sys.modules["simlab.srv"] = simlab_srv
simlab_srv.ResetSimVehicle = object
simlab_srv.ResetSimManipulator = object
simlab_srv.ResetSimRobotState = object

simlab_action = sys.modules.get("simlab.action")
if simlab_action is None:
    simlab_action = types.ModuleType("simlab.action")
    sys.modules["simlab.action"] = simlab_action
simlab_action.PlanVehicle = object

from simlab.controllers.pid import LowLevelPidController
from simlab.robot import Robot


def test_vehicle_command_yaw_unwraps_against_previous_command():
    robot = Robot.__new__(Robot)
    robot.ned_pose = [0.0, 0.0, 0.0, 0.0, 0.0, -2.10]
    robot._last_vehicle_cmd_yaw = None
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


def test_pid_yaw_guard_preserves_target_continuity_across_equivalent_branches():
    controller = object.__new__(LowLevelPidController)
    controller._last_vehicle_pid_target_yaw = None
    state = np.zeros(12)
    state[5] = -2.107
    positive_branch = np.zeros(6)
    negative_branch = np.zeros(6)
    positive_branch[5] = 0.852
    negative_branch[5] = -5.046

    first = controller._normalize_target_yaw_for_pid(state, positive_branch)
    second = controller._normalize_target_yaw_for_pid(state, negative_branch)

    assert abs((first[5] - state[5]) - 2.959) < 1e-6
    assert abs(second[5] - first[5]) < 0.5
    assert second[5] > 0.0
