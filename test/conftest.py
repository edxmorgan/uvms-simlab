"""Shared pytest stubs for ROS interfaces absent in some test environments."""

import importlib
import sys
import types


def _module(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        parent_name, _, child_name = name.rpartition(".")
        if parent_name:
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, child_name, mod)
        return mod


def _ensure_attr(mod, name: str):
    if not hasattr(mod, name):
        setattr(mod, name, object)


def _ensure_srv_attr(mod, name: str):
    if not hasattr(mod, name):
        setattr(mod, name, type(name, (), {"Request": object, "Response": object}))


control_msgs_msg = _module("control_msgs.msg")
for _name in ("DynamicInterfaceGroupValues", "DynamicJointState", "InterfaceValue"):
    _ensure_attr(control_msgs_msg, _name)

simlab_msg = _module("simlab.msg")
for _name in ("ControllerPerformance", "ReferenceTargets"):
    _ensure_attr(simlab_msg, _name)

simlab_srv = _module("simlab.srv")
for _name in (
    "BackendPoseCommand",
    "BackendRobotCommand",
    "BackendWaypointCommand",
    "BackendWorldCommand",
    "ResetSimVehicle",
    "ResetSimManipulator",
    "ResetSimRobotState",
):
    _ensure_srv_attr(simlab_srv, _name)

simlab_action = _module("simlab.action")
_ensure_attr(simlab_action, "PlanVehicle")

ros2_control_msg = _module("ros2_control_blue_reach_5.msg")
for _name in (
    "DynamicObstacle",
    "DynamicObstacleArray",
    "SimCameraDynamics",
    "SimManipulatorDynamics",
    "SimVehicleDynamics",
):
    _ensure_attr(ros2_control_msg, _name)

ros2_control_srv = _module("ros2_control_blue_reach_5.srv")
for _name in ("ResetSimUvms", "SetDynamicObstacles", "SetSimCameraDynamics", "SetSimDynamics"):
    _ensure_srv_attr(ros2_control_srv, _name)
