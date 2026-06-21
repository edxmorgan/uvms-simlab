from types import SimpleNamespace

import numpy as np
import pytest

from simlab.dynamic_replanners import DEFAULT_DYNAMIC_REPLANNER_CLASSES, visible_dynamic_replanner_names
from simlab.dynamic_replanners.base import ReplanDecision
from simlab.dynamic_replanners.clearance_hysteresis import ClearanceHysteresisReplanner
from simlab.dynamic_world import DynamicClearance
from simlab.robot import ControlMode


def test_dynamic_replanner_registry_exposes_default_strategy():
    assert ClearanceHysteresisReplanner in DEFAULT_DYNAMIC_REPLANNER_CLASSES
    assert "clearance_hysteresis" in visible_dynamic_replanner_names()


class FakeDynamicWorld:
    def __init__(self, close_clearance=0.1):
        self.obstacles = {"moving": object()}
        self.t_offsets = []
        self.close_clearance = float(close_clearance)

    def min_clearance_xyz(self, xyz, *, t_offset=0.0):
        np.asarray(xyz, dtype=float).reshape(3)
        self.t_offsets.append(float(t_offset))
        if t_offset >= 3.0:
            return DynamicClearance("moving", self.close_clearance)
        return DynamicClearance("moving", 2.0)


class FakeRobot:
    prefix = "robot_1_"
    control_mode = ControlMode.PLANNER
    sim_reset_hold = False
    task_based_controller = False
    max_traj_vel = [0.5, 0.0, 0.0]

    def __init__(self):
        self.planner_action_client = SimpleNamespace(busy=False)
        self.vehicle_cart_traj = SimpleNamespace(active=True)
        self.plan_calls = 0
        self.last_plan_kwargs = None
        self.planner = SimpleNamespace(
            planned_result={
                "is_success": True,
                "xyz": np.array(
                    [
                        [0.0, 0.0, -2.0],
                        [1.0, 0.0, -2.0],
                        [2.0, 0.0, -2.0],
                    ]
                ),
            }
        )

    def _pose_from_state_in_frame(self, frame):
        return SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=-2.0),
        )

    def plan_vehicle_trajectory_action(self, **kwargs):
        self.plan_calls += 1
        self.last_plan_kwargs = dict(kwargs)


class FakeLogger:
    def info(self, msg):
        pass

    def warn(self, msg):
        pass

    def error(self, msg):
        pass



class CountingLogger(FakeLogger):
    def __init__(self):
        self.info_messages = []

    def info(self, msg):
        self.info_messages.append(msg)


def test_dynamic_replanner_hysteresis_suppression_log_is_throttled():
    logger = CountingLogger()
    replanner = ClearanceHysteresisReplanner.__new__(ClearanceHysteresisReplanner)
    replanner.node = SimpleNamespace(get_logger=lambda: logger)
    replanner.robot = SimpleNamespace(prefix="robot_1_")
    replanner.replan_hysteresis_m = 0.10
    replanner._last_replan_obstacle_id = "moving"
    replanner._last_replan_clearance_m = -0.07
    replanner._last_hysteresis_suppression_log_time = 0.0
    decision = ReplanDecision("replan", "blocked", "moving", 0.02)

    assert replanner._is_hysteresis_suppressed(decision)
    assert replanner._is_hysteresis_suppressed(decision)
    assert len(logger.info_messages) == 1

    replanner._last_hysteresis_suppression_log_time -= 2.0
    assert replanner._is_hysteresis_suppressed(decision)
    assert len(logger.info_messages) == 2

def test_reset_selected_simulation_clears_dynamic_obstacles_before_robot_reset():
    from simlab.uvms_backend import UVMSBackendCore

    events = []
    robot = SimpleNamespace(
        prefix="robot_1_",
        k_robot=0,
        reset_simulation=lambda: events.append("reset_robot"),
    )
    backend = UVMSBackendCore.__new__(UVMSBackendCore)
    backend.robot_selected = robot
    backend.node = SimpleNamespace(get_logger=lambda: FakeLogger())
    backend.clear_vehicle_waypoints_for_robot = lambda robot_k: events.append(("clear_waypoints", robot_k))
    backend.clear_dynamic_obstacles = lambda: (events.append("clear_obstacles") or (True, "cleared"))

    assert backend.reset_selected_simulation()
    assert events == [("clear_waypoints", 0), "clear_obstacles", "reset_robot"]



def test_dynamic_replanner_uses_predicted_obstacle_time_offsets():
    dynamic_world = FakeDynamicWorld()
    replanner = ClearanceHysteresisReplanner.__new__(ClearanceHysteresisReplanner)
    replanner.backend = SimpleNamespace(
        world_frame="world",
        dynamic_world=dynamic_world,
    )
    replanner.robot = FakeRobot()
    replanner.lookahead_time_s = 10.0
    replanner.safety_margin_m = 0.25
    replanner.max_samples = 8

    decision = replanner.evaluate()

    assert decision.should_replan
    assert "t+" in decision.reason
    assert max(dynamic_world.t_offsets) >= 3.0


def test_dynamic_replanner_suppresses_repeat_replan_with_hysteresis():
    robot = FakeRobot()
    backend = SimpleNamespace(
        node=SimpleNamespace(get_logger=lambda: FakeLogger()),
        world_frame="world",
        dynamic_world=FakeDynamicWorld(close_clearance=0.12),
        fcl_world=SimpleNamespace(vehicle_radius=0.574),
    )
    mission = SimpleNamespace(
        executing=True,
        active_index=0,
        active_waypoint=lambda: object(),
    )
    replanner = ClearanceHysteresisReplanner(
        backend,
        robot,
        mission,
        cooldown_s=0.0,
        lookahead_time_s=10.0,
        safety_margin_m=0.25,
        replan_hysteresis_m=0.10,
    )

    replanner.tick()
    replanner.tick()

    assert robot.plan_calls == 1
    assert robot.last_plan_kwargs is not None
    assert robot.last_plan_kwargs["robot_collision_radius"] == pytest.approx(0.574)
    assert robot.last_plan_kwargs["dynamic_obstacle_prediction_speed"] == pytest.approx(0.5)


def test_dynamic_replanner_replans_active_plan_execute_trajectory_without_waypoint_mission():
    robot = FakeRobot()
    goal_pose = object()
    robot.last_vehicle_goal_pose_world = goal_pose
    backend = SimpleNamespace(
        node=SimpleNamespace(get_logger=lambda: FakeLogger()),
        world_frame="world",
        dynamic_world=FakeDynamicWorld(close_clearance=0.12),
        fcl_world=SimpleNamespace(vehicle_radius=0.574),
    )
    backend.dynamic_world.min_clearance_xyz = lambda xyz, t_offset=0.0: DynamicClearance("moving", 0.12)
    mission = SimpleNamespace(
        executing=False,
        active_index=None,
        active_waypoint=lambda: (_ for _ in ()).throw(AssertionError("waypoint goal should not be used")),
    )
    replanner = ClearanceHysteresisReplanner(
        backend,
        robot,
        mission,
        cooldown_s=0.0,
        lookahead_time_s=10.0,
        safety_margin_m=0.25,
        replan_hysteresis_m=0.10,
    )

    replanner.tick()

    assert robot.plan_calls == 1
    assert robot.last_plan_kwargs is not None
    assert robot.last_plan_kwargs["goal_pose"] is not None
    assert robot.last_plan_kwargs["preempt_current"] is False



class FakePlannerWorld:
    def __init__(self):
        self.calls = []

    def is_state_valid_xyz(self, xyz, *, safety_margin=0.0, t_offset=0.0):
        self.calls.append((np.asarray(xyz, dtype=float), float(safety_margin), float(t_offset)))
        return True


class FakeOmplRotation:
    x = 0.0
    y = 0.0
    z = 0.0
    w = 1.0


class FakeOmplState:
    def __init__(self, xyz):
        self._xyz = tuple(float(v) for v in xyz)

    def getX(self):
        return self._xyz[0]

    def getY(self):
        return self._xyz[1]

    def getZ(self):
        return self._xyz[2]

    def rotation(self):
        return FakeOmplRotation()


def test_ompl_validity_checker_time_indexes_dynamic_obstacles():
    try:
        from simlab.planners.ompl import OmplPlanner
    except RuntimeError as exc:
        pytest.skip(str(exc))

    planner = OmplPlanner.__new__(OmplPlanner)
    planner_world = FakePlannerWorld()

    assert planner._valid_with_fcl(
        planner_world,
        0.25,
        -10.0,
        10.0,
        -10.0,
        10.0,
        -10.0,
        10.0,
        np.deg2rad(11.0),
        np.deg2rad(11.0),
        np.array([0.0, 0.0, 0.0]),
        5.0,
        FakeOmplState([3.0, 4.0, 0.0]),
    )

    assert len(planner_world.calls) == 1
    xyz, safety_margin, t_offset = planner_world.calls[0]
    np.testing.assert_allclose(xyz, [3.0, 4.0, 0.0])
    assert safety_margin == pytest.approx(0.25)
    assert t_offset == pytest.approx(1.0)


def test_ompl_validity_checker_keeps_current_snapshot_when_prediction_speed_disabled():
    try:
        from simlab.planners.ompl import OmplPlanner
    except RuntimeError as exc:
        pytest.skip(str(exc))

    planner = OmplPlanner.__new__(OmplPlanner)
    planner_world = FakePlannerWorld()

    assert planner._valid_with_fcl(
        planner_world,
        0.0,
        -10.0,
        10.0,
        -10.0,
        10.0,
        -10.0,
        10.0,
        np.deg2rad(11.0),
        np.deg2rad(11.0),
        np.array([0.0, 0.0, 0.0]),
        0.0,
        FakeOmplState([3.0, 4.0, 0.0]),
    )

    assert planner_world.calls[0][2] == pytest.approx(0.0)
