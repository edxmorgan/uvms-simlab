from __future__ import annotations

import copy
import hashlib
import time
from typing import TYPE_CHECKING

import numpy as np

from simlab.motion_planning.dynamic_replanners.base import (
    DynamicReplannerTemplate,
    ReplanDecision,
)
from simlab.planner_world import PlannerWorld
from simlab.robot import ControlMode

if TYPE_CHECKING:
    from simlab.robot import Robot
    from simlab.uvms_backend import UVMSBackendCore
    from simlab.vehicle_waypoint_mission import VehicleWaypointMission


class ClearanceHysteresisReplanner(DynamicReplannerTemplate):
    """Per-robot event-triggered replanning supervisor."""

    registry_name = "clearance_hysteresis"

    def __init__(
        self,
        backend: "UVMSBackendCore",
        robot: "Robot",
        mission: "VehicleWaypointMission",
        *,
        cooldown_s: float,
        lookahead_time_s: float,
        safety_margin_m: float,
        replan_hysteresis_m: float = 0.10,
    ):
        self.backend = backend
        self.node = backend.node
        self.robot = robot
        self.mission = mission
        self.cooldown_s = max(0.0, float(cooldown_s))
        self.lookahead_time_s = max(0.0, float(lookahead_time_s))
        self.safety_margin_m = max(0.0, float(safety_margin_m))
        self.replan_hysteresis_m = max(0.0, float(replan_hysteresis_m))
        self._last_replan_time: float | None = None
        self._last_replan_obstacle_id = ""
        self._last_replan_clearance_m: float | None = None
        self._last_replan_path_signature = ""
        self._last_same_path_suppression_log_time = 0.0
        self._last_hysteresis_suppression_log_time = 0.0
        self.replan_count = 0
        self.last_replan_reason = ""

    def configure(
        self,
        *,
        cooldown_s: float | None = None,
        lookahead_time_s: float | None = None,
        safety_margin_m: float | None = None,
        replan_hysteresis_m: float | None = None,
    ) -> None:
        if cooldown_s is not None:
            self.cooldown_s = max(0.0, float(cooldown_s))
        if lookahead_time_s is not None:
            self.lookahead_time_s = max(0.0, float(lookahead_time_s))
        if safety_margin_m is not None:
            self.safety_margin_m = max(0.0, float(safety_margin_m))
        if replan_hysteresis_m is not None:
            self.replan_hysteresis_m = max(0.0, float(replan_hysteresis_m))

    def tick(self) -> None:
        robot = self.robot
        if (robot.sim_reset_hold or robot.task_based_controller
                or robot.control_mode in (ControlMode.REPLAY, ControlMode.REPLAY_SETTLE)
                or not self.backend.dynamic_replanning_enabled):
            return
        if robot.vehicle_cart_traj is None or not robot.vehicle_cart_traj.active:
            return
        decision = self._remaining_path_decision(robot)
        if not decision.should_replan or robot.planner_action_client.busy:
            return

        world_signature = self._world_signature()
        context = (decision.path_signature, world_signature)
        if context != getattr(self, '_attempt_context', None):
            previous_attempt_time = self._last_replan_time
            self.reset_history()
            self._last_replan_time = previous_attempt_time
            self._attempt_context = context
        if not self._cooldown_ready():
            return
        if self._is_same_path_attempt_suppressed(decision) or self._is_hysteresis_suppressed(decision):
            return
        goal_pose = self._active_goal_pose()
        if goal_pose is None:
            return
        self._last_replan_time = time.monotonic()
        self._last_replan_obstacle_id = decision.obstacle_id
        self._last_replan_clearance_m = decision.clearance_m
        self._last_replan_path_signature = decision.path_signature
        self.replan_count += 1
        self.last_replan_reason = decision.reason
        self.robot.plan_vehicle_trajectory_action(
            goal_pose=goal_pose, time_limit=1.0,
            robot_collision_radius=float(self.backend.fcl_world.vehicle_radius),
            preempt_current=False,
            dynamic_obstacle_prediction_speed=self._nominal_vehicle_speed(robot))


    def _planner_world(self):
        return PlannerWorld(fcl_world=self.backend.fcl_world,
                            dynamic_world=self.backend.dynamic_world,
                            robot_prefix=self.robot.prefix)


    def _world_signature(self):
        world = getattr(self.backend, 'dynamic_world', None)
        obstacles = getattr(world, 'obstacles', {})
        records = sorted((name, repr(value)) for name, value in obstacles.items()
                         if not value.owner_robot or value.owner_robot != self.robot.prefix)
        unavailable = sorted(set(world.unavailable_robot_owners) - {self.robot.prefix}) if world is not None else []
        return hashlib.sha256(repr((records, unavailable, getattr(world, 'error', None))).encode()).hexdigest()


    def _active_goal_pose(self):
        if getattr(self.mission, 'active_index', None) is not None:
            return self.mission.active_waypoint()
        goal_pose = getattr(self.robot, "last_vehicle_goal_pose_world", None)
        if goal_pose is None:
            return None
        return copy.deepcopy(goal_pose)

    def evaluate(self) -> ReplanDecision:
        robot = self.robot
        if robot.control_mode in (ControlMode.REPLAY, ControlMode.REPLAY_SETTLE):
            return ReplanDecision("keep", "replay active")
        if robot.sim_reset_hold or robot.task_based_controller:
            return ReplanDecision("keep", "robot unavailable")
        if robot.planner_action_client.busy:
            return ReplanDecision("keep", "planner busy")
        if robot.vehicle_cart_traj is None or not robot.vehicle_cart_traj.active:
            return ReplanDecision("keep", "no active trajectory")

        return self._remaining_path_decision(robot)

    def status_summary(self) -> str:
        last_clearance = (
            "none"
            if self._last_replan_clearance_m is None
            else f"{self._last_replan_clearance_m:.3f} m"
        )
        return (
            f"{self.robot.prefix}: state=monitoring, replans={self.replan_count}, "
            f"last_obstacle='{self._last_replan_obstacle_id or 'none'}', "
            f"last_clearance={last_clearance}, "
            f"last_reason='{self.last_replan_reason or 'none'}'"
        )

    def reset_history(self, *, reset_count: bool = False) -> None:
        self._last_replan_time = None
        self._last_replan_obstacle_id = ""
        self._last_replan_clearance_m = None
        self._last_replan_path_signature = ""
        self._last_same_path_suppression_log_time = 0.0
        self._last_hysteresis_suppression_log_time = 0.0
        self.last_replan_reason = ""
        if reset_count:
            self.replan_count = 0

    def _cooldown_ready(self) -> bool:
        if self._last_replan_time is None:
            return True
        return (time.monotonic() - self._last_replan_time) >= self.cooldown_s

    def _is_hysteresis_suppressed(self, decision: ReplanDecision) -> bool:
        if self._last_replan_clearance_m is None or decision.clearance_m is None:
            return False
        if decision.obstacle_id != self._last_replan_obstacle_id:
            return False
        if decision.path_signature and decision.path_signature != getattr(self, '_last_replan_path_signature', ''):
            return False
        improvement_threshold = self._last_replan_clearance_m - self.replan_hysteresis_m
        if decision.clearance_m < improvement_threshold:
            return False

        now = time.monotonic()
        if now - self._last_hysteresis_suppression_log_time >= 2.0:
            self._last_hysteresis_suppression_log_time = now
            self.node.get_logger().info(
                f"[DynamicReplanner] suppressing repeat replan for {self.robot.prefix}: "
                f"clearance to '{decision.obstacle_id}' is {decision.clearance_m:.3f} m, "
                f"last trigger was {self._last_replan_clearance_m:.3f} m "
                f"(hysteresis {self.replan_hysteresis_m:.3f} m)"
            )
        return True

    def _is_same_path_attempt_suppressed(self, decision: ReplanDecision) -> bool:
        if not decision.path_signature:
            return False
        if decision.obstacle_id != self._last_replan_obstacle_id:
            return False
        if decision.path_signature != self._last_replan_path_signature:
            return False

        now = time.monotonic()
        if now - self._last_same_path_suppression_log_time >= 2.0:
            self._last_same_path_suppression_log_time = now
            self.node.get_logger().info(
                f"[DynamicReplanner] suppressing repeat replan for {self.robot.prefix}: "
                f"obstacle '{decision.obstacle_id}' is still blocking the same active path. "
                "Waiting for a new path or obstacle update."
            )
        return True


    def _remaining_path_decision(self, robot: "Robot") -> ReplanDecision:
        preview = getattr(robot.vehicle_cart_traj, 'preview_samples', None)
        if not callable(preview):
            return ReplanDecision('replan', 'trajectory generator must provide timed previews',
                                  'unavailable', float('-inf'), '', 0.0)
        signature = ''
        try:
            if self.backend.dynamic_world is None:
                raise ValueError('dynamic world unavailable')
            horizon = self.lookahead_time_s
            horizon += max(0., float(getattr(self.backend, 'dynamic_replanning_latency_budget', 1.5)))
            samples = preview(horizon=horizon, sample_dt=0.05)
            if not samples:
                return ReplanDecision('pending', 'trajectory preview not yet available')
            planned = getattr(robot.planner, 'planned_result', None) or {}
            signature = self._path_signature(np.asarray(planned.get('xyz', [])))
            self._planner_world().require_clear_timed_trajectory(
                samples, max_speed=robot.vehicle_cart_traj.speed_bound(), safety_margin=self.safety_margin_m,
                sample_at_time=robot.vehicle_cart_traj.preview_position)
            return ReplanDecision('keep', 'timed trajectory valid')
        except Exception as exc:
            return ReplanDecision('replan', f'timed trajectory unavailable or obstructed: {exc}',
                                  getattr(exc, 'obstacle_id', 'unavailable'),
                                  getattr(exc, 'distance_m', float('-inf')), signature,
                                  getattr(exc, 't_offset_s', 0.0))

    @staticmethod
    def _path_signature(path_xyz: np.ndarray) -> str:
        path = np.asarray(path_xyz, dtype=float).reshape(-1, 3)
        if path.shape[0] == 0:
            return ""
        return hashlib.sha256(np.round(path, 5).tobytes()).hexdigest()

    @staticmethod
    def _nominal_vehicle_speed(robot: "Robot") -> float:
        max_traj_vel = np.asarray(getattr(robot, "max_traj_vel", [0.15, 0.15, 0.10]), dtype=float)
        if max_traj_vel.size == 0:
            return 0.15
        speed = float(np.linalg.norm(max_traj_vel.reshape(-1)))
        return max(speed, 0.05)
