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
from simlab.planner_world import DYNAMIC_CLEARANCE_TOLERANCE_M
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
        collision_stop_enabled: bool = True,
        collision_stop_margin_m: float = 0.0,
        replan_hysteresis_m: float = 0.10,
    ):
        self.backend = backend
        self.node = backend.node
        self.robot = robot
        self.mission = mission
        self.cooldown_s = max(0.0, float(cooldown_s))
        self.lookahead_time_s = max(0.0, float(lookahead_time_s))
        self.safety_margin_m = max(0.0, float(safety_margin_m))
        self.collision_stop_enabled = bool(collision_stop_enabled)
        self.collision_stop_margin_m = max(0.0, float(collision_stop_margin_m))
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
        collision_stop_enabled: bool | None = None,
        collision_stop_margin_m: float | None = None,
        replan_hysteresis_m: float | None = None,
    ) -> None:
        if cooldown_s is not None:
            self.cooldown_s = max(0.0, float(cooldown_s))
        if lookahead_time_s is not None:
            self.lookahead_time_s = max(0.0, float(lookahead_time_s))
        if safety_margin_m is not None:
            self.safety_margin_m = max(0.0, float(safety_margin_m))
        if collision_stop_enabled is not None:
            self.collision_stop_enabled = bool(collision_stop_enabled)
        if collision_stop_margin_m is not None:
            self.collision_stop_margin_m = max(0.0, float(collision_stop_margin_m))
        if replan_hysteresis_m is not None:
            self.replan_hysteresis_m = max(0.0, float(replan_hysteresis_m))

    def tick(self) -> None:
        robot = self.robot
        if robot.sim_reset_hold or robot.task_based_controller or robot.control_mode in (ControlMode.REPLAY, ControlMode.REPLAY_SETTLE):
            self._paused_goal = None
            return
        blocked = getattr(robot, 'navigation_blocked_goal', None)
        if blocked is not None and getattr(robot, 'navigation_blocked_epoch', -1) == getattr(robot, '_navigation_epoch', 0):
            self._paused_goal = copy.deepcopy(blocked)
            self._pause_epoch = getattr(robot, '_navigation_epoch', 0)
            self.mission.pause()
            robot.navigation_blocked_goal = None
        if getattr(self, '_paused_goal', None) is not None:
            if getattr(robot, '_navigation_epoch', 0) != self._pause_epoch:
                self._paused_goal = None  # Explicit stop, new goal, or controller change.
            else:
                self._try_resume()
                return
        if self._stop_if_current_dynamic_collision():
            return

        # Safety decisions run even while a planner action is busy, and before
        # cooldown/hysteresis can suppress another planning request.
        if robot.vehicle_cart_traj is None or not robot.vehicle_cart_traj.active:
            return
        decision = self._remaining_path_decision(robot)
        if self._braking_required() or (decision.should_replan and self._should_preempt_for_near_conflict(decision)):
            self._pause_navigation(decision.reason or 'braking clearance exhausted')
            self._try_resume()
            return
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
            robot_collision_radius=float(self.backend.fcl_world.vehicle_radius) + self.safety_margin_m,
            preempt_current=False,
            dynamic_obstacle_prediction_speed=self._nominal_vehicle_speed(robot))

    def _pause_navigation(self, reason):
        if getattr(self, '_safety_held', False) and not getattr(self.robot.vehicle_cart_traj, 'active', False):
            return
        self._paused_goal = self._active_goal_pose()
        self.mission.pause()
        self.robot.abrupt_planner_stop(publish_zero=False)
        self.robot.hold_current_state_with_feedback()
        self._pause_epoch = getattr(self.robot, '_navigation_epoch', 0)
        self._safety_held = True
        self._last_recovery_attempt = None
        self.node.get_logger().warn(f'[DynamicReplanner] holding {self.robot.prefix}; goal retained: {reason}')

    def _try_resume(self):
        if getattr(self, '_paused_goal', None) is None:
            return
        # Hardware recovery requires explicit operator control.
        if getattr(self.backend, 'use_vehicle_hardware', True):
            return
        if self.robot.planner_action_client.busy:
            return
        if getattr(self.robot.vehicle_cart_traj, 'active', False) and getattr(self, '_recovery_sent', False):
            decision = self._remaining_path_decision(self.robot)
            if self._braking_required() or (decision.should_replan and self._should_preempt_for_near_conflict(decision)):
                self._pause_navigation('replacement trajectory is no longer clear')
                return
            if decision.action == 'pending':
                return
            self.mission.resume()
            self._paused_goal = None
            self._safety_held = False
            self._recovery_sent = False
            return
        pose = self.robot._pose_from_state_in_frame(self.backend.world_frame)
        if pose is None:
            return
        xyz = [pose.position.x, pose.position.y, pose.position.z]
        from simlab.planner_world import PlannerWorld
        world = PlannerWorld(fcl_world=self.backend.fcl_world, dynamic_world=self.backend.dynamic_world)
        if not world.is_state_valid_xyz(xyz, safety_margin=self.safety_margin_m):
            return
        if np.linalg.norm(self.robot._current_vehicle_velocity_world_nwu()) > 0.02:
            return
        now = time.monotonic()
        last = getattr(self, '_last_recovery_attempt', None)
        if last is not None and now - last < max(2.0, self.cooldown_s):
            return
        self._last_recovery_attempt = now
        sent = self.robot.plan_vehicle_trajectory_action(
            goal_pose=copy.deepcopy(self._paused_goal), time_limit=1.0,
            robot_collision_radius=float(self.backend.fcl_world.vehicle_radius) + self.safety_margin_m,
            preempt_current=True, dynamic_obstacle_prediction_speed=self._nominal_vehicle_speed(self.robot))
        self._pause_epoch = getattr(self.robot, '_navigation_epoch', 0)
        self._recovery_sent = bool(sent)

    def _braking_required(self):
        from simlab.motion_planning.navigation_safety import stopping_distance
        world = getattr(self.backend, 'dynamic_world', None)
        if world is None:
            return False
        pose = self.robot._pose_from_state_in_frame(self.backend.world_frame)
        if pose is None:
            return True
        xyz = np.array([pose.position.x, pose.position.y, pose.position.z])
        clearance = world.min_clearance_xyz(xyz)
        if clearance is None:
            return False
        speed = float(np.linalg.norm(self.robot._current_vehicle_velocity_world_nwu()))
        acc = np.asarray(getattr(self.robot, 'max_traj_acc', [0.1]*3), dtype=float)
        decel = min(float(getattr(self.backend, 'dynamic_braking_deceleration', 0.1)), float(np.min(acc)))
        reaction = self._reaction_time_s()
        tracking = 0.0
        out = getattr(self.robot.vehicle_cart_traj, 'out', None)
        if out is not None:
            tracking = float(np.linalg.norm(np.asarray(out.new_position) - xyz))
        try:
            distance = stopping_distance(speed, decel, reaction, tracking)
        except ValueError:
            return True
        return clearance.distance_m < max(self.safety_margin_m, self.collision_stop_margin_m + distance)

    def _world_signature(self):
        world = getattr(self.backend, 'dynamic_world', None)
        obstacles = getattr(world, 'obstacles', {})
        return hashlib.sha256(repr(sorted((name, repr(value)) for name, value in obstacles.items())).encode()).hexdigest()

    def _stop_if_current_dynamic_collision(self) -> bool:
        if not self.collision_stop_enabled:
            return False

        dynamic_world = getattr(self.backend, "dynamic_world", None)
        if dynamic_world is None or (not dynamic_world.obstacles and not getattr(dynamic_world, 'error', None)):
            return False

        pose_now = self.robot._pose_from_state_in_frame(self.backend.world_frame)
        if pose_now is None:
            return False

        current_xyz = np.array(
            [pose_now.position.x, pose_now.position.y, pose_now.position.z],
            dtype=float,
        )
        clearance = dynamic_world.min_clearance_xyz(current_xyz)
        stop_threshold = self.collision_stop_margin_m - DYNAMIC_CLEARANCE_TOLERANCE_M
        if clearance is None or clearance.distance_m >= stop_threshold:
            return False

        self._pause_navigation(
            f"current clearance to '{clearance.obstacle_id}' is {clearance.distance_m:.3f} m")
        return True

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
            f"{self.robot.prefix}: state={'blocked' if getattr(self, '_paused_goal', None) is not None else 'monitoring'}, replans={self.replan_count}, "
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

    def _stop_window_s(self) -> float:
        speed = float(np.linalg.norm(self.robot._current_vehicle_velocity_world_nwu()))
        acc = np.asarray(getattr(self.robot, 'max_traj_acc', [0.1]*3), dtype=float)
        decel = max(1e-3, min(float(getattr(self.backend, 'dynamic_braking_deceleration', 0.1)), float(np.min(acc))))
        return max(2.0, speed / decel + self._reaction_time_s())

    def _reaction_time_s(self):
        return (1.0 / max(0.1, float(getattr(self.backend, 'dynamic_replanning_rate', 3.0)))
                + 0.25 + max(0.0, float(getattr(self.backend, 'dynamic_replanning_latency_budget', 1.5))))

    def _should_preempt_for_near_conflict(self, decision: ReplanDecision) -> bool:
        if decision.t_offset_s is None:
            return False
        return decision.t_offset_s <= self._stop_window_s()

    def _remaining_path_decision(self, robot: "Robot") -> ReplanDecision:
        preview = getattr(robot.vehicle_cart_traj, 'preview_samples', None)
        if callable(preview):
            try:
                samples = preview(horizon=max(self.lookahead_time_s, self._stop_window_s()), sample_dt=0.05)
                world = getattr(self.backend, 'dynamic_world', None)
                if world is None:
                    return ReplanDecision('replan', 'dynamic world unavailable',
                                          'unavailable', float('-inf'), '', 0.0)
                if not samples:
                    # Incremental Ruckig has no preview before its first update.
                    # This is not evidence that the trajectory is collision-free.
                    return ReplanDecision('pending', 'trajectory preview not yet available')
                planned = getattr(robot.planner, 'planned_result', None) or {}
                signature = self._path_signature(np.asarray(planned.get('xyz', [])))
                for offset, point in samples:
                    clearance = world.min_clearance_xyz(point, t_offset=offset)
                    if clearance is not None and clearance.distance_m < self.safety_margin_m:
                        return ReplanDecision('replan', 'timed trajectory violates dynamic clearance',
                            clearance.obstacle_id, clearance.distance_m, signature, offset)
                return ReplanDecision('keep', 'timed trajectory valid')
            except Exception as exc:
                return ReplanDecision('replan', f'trajectory preview unavailable: {exc}', 'unavailable',
                                      float('-inf'), '', 0.0)
        return ReplanDecision('replan', 'trajectory generator must provide timed previews',
                              'unavailable', float('-inf'), '', 0.0)

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
