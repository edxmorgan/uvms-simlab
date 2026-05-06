from types import SimpleNamespace

import numpy as np

from simlab.dynamic_replanner import DynamicReplanner
from simlab.dynamic_world import DynamicClearance
from simlab.robot import ControlMode


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


class FakeLogger:
    def info(self, msg):
        pass

    def warn(self, msg):
        pass

    def error(self, msg):
        pass


def test_dynamic_replanner_uses_predicted_obstacle_time_offsets():
    dynamic_world = FakeDynamicWorld()
    replanner = DynamicReplanner.__new__(DynamicReplanner)
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
    replanner = DynamicReplanner(
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
