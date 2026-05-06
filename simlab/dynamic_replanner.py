from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from simlab.robot import ControlMode

if TYPE_CHECKING:
    from simlab.robot import Robot
    from simlab.uvms_backend import UVMSBackendCore
    from simlab.vehicle_waypoint_mission import VehicleWaypointMission


@dataclass(frozen=True)
class ReplanDecision:
    action: str
    reason: str = ""
    obstacle_id: str = ""
    clearance_m: float | None = None
    path_signature: str = ""
    t_offset_s: float | None = None

    @property
    def should_replan(self) -> bool:
        return self.action == "replan"


@dataclass(frozen=True)
class TimedPathSample:
    xyz: np.ndarray
    t_offset: float


class DynamicReplanner:
    """Per-robot event-triggered replanning supervisor."""

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
        max_samples: int = 24,
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
        self.max_samples = max(2, int(max_samples))
        self.replan_hysteresis_m = max(0.0, float(replan_hysteresis_m))
        self._last_replan_time: float | None = None
        self._last_replan_obstacle_id = ""
        self._last_replan_clearance_m: float | None = None
        self._last_replan_path_signature = ""
        self._last_same_path_suppression_log_time = 0.0
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
        max_samples: int | None = None,
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
        if max_samples is not None:
            self.max_samples = max(2, int(max_samples))
        if replan_hysteresis_m is not None:
            self.replan_hysteresis_m = max(0.0, float(replan_hysteresis_m))

    def tick(self) -> None:
        if not self.mission.executing or self.mission.active_index is None:
            return

        if self._stop_if_current_dynamic_collision():
            return

        decision = self.evaluate()
        if not decision.should_replan:
            return

        if not self._cooldown_ready():
            return

        if self._is_same_path_attempt_suppressed(decision):
            return

        if self._is_hysteresis_suppressed(decision):
            return

        goal_pose = self.mission.active_waypoint()
        if goal_pose is None:
            return

        self._last_replan_time = time.monotonic()
        self._last_replan_obstacle_id = decision.obstacle_id
        self._last_replan_clearance_m = decision.clearance_m
        self._last_replan_path_signature = decision.path_signature
        self.replan_count += 1
        self.last_replan_reason = decision.reason
        self.node.get_logger().warn(
            f"[DynamicReplanner] replanning {self.robot.prefix}: {decision.reason}"
        )
        planner_radius = float(self.backend.fcl_world.vehicle_radius) + self.safety_margin_m
        self.robot.plan_vehicle_trajectory_action(
            goal_pose=goal_pose,
            time_limit=1.0,
            robot_collision_radius=planner_radius,
            preempt_current=False,
        )

    def _stop_if_current_dynamic_collision(self) -> bool:
        if not self.collision_stop_enabled:
            return False

        dynamic_world = getattr(self.backend, "dynamic_world", None)
        if dynamic_world is None or not dynamic_world.obstacles:
            return False

        pose_now = self.robot._pose_from_state_in_frame(self.backend.world_frame)
        if pose_now is None:
            return False

        current_xyz = np.array(
            [pose_now.position.x, pose_now.position.y, pose_now.position.z],
            dtype=float,
        )
        clearance = dynamic_world.min_clearance_xyz(current_xyz)
        if clearance is None or clearance.distance_m > self.collision_stop_margin_m:
            return False

        self.mission.stop()
        self.robot.abrupt_planner_stop(publish_zero=False)
        self.robot.hold_current_state_with_feedback()
        self.node.get_logger().error(
            f"[DynamicReplanner] emergency stop {self.robot.prefix}: "
            f"current clearance to dynamic obstacle '{clearance.obstacle_id}' is "
            f"{clearance.distance_m:.3f} m <= stop margin {self.collision_stop_margin_m:.3f} m"
        )
        return True

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
            f"{self.robot.prefix}: replans={self.replan_count}, "
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
        improvement_threshold = self._last_replan_clearance_m - self.replan_hysteresis_m
        if decision.clearance_m < improvement_threshold:
            return False

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

        if self._should_stop_for_unresolved_blocked_path(decision):
            stop_mission = getattr(self.mission, "stop", None)
            if callable(stop_mission):
                stop_mission()
            stop_robot = getattr(self.robot, "abrupt_planner_stop", None)
            if callable(stop_robot):
                stop_robot(publish_zero=False)
            hold_robot = getattr(self.robot, "hold_current_state_with_feedback", None)
            if callable(hold_robot):
                hold_robot()
            self.node.get_logger().error(
                f"[DynamicReplanner] stopping {self.robot.prefix}: obstacle "
                f"'{decision.obstacle_id}' still blocks the active path after a replan "
                f"attempt and the predicted conflict is now t+{decision.t_offset_s:.1f}s."
            )
            return True

        now = time.monotonic()
        if now - self._last_same_path_suppression_log_time >= 2.0:
            self._last_same_path_suppression_log_time = now
            self.node.get_logger().info(
                f"[DynamicReplanner] suppressing repeat replan for {self.robot.prefix}: "
                f"obstacle '{decision.obstacle_id}' is still blocking the same active path. "
                "Waiting for a new path or obstacle update."
            )
        return True

    def _should_stop_for_unresolved_blocked_path(self, decision: ReplanDecision) -> bool:
        if decision.t_offset_s is None:
            return False
        stop_time = max(2.0, min(4.0, 0.4 * self.lookahead_time_s))
        return decision.t_offset_s <= stop_time

    def _remaining_path_decision(self, robot: "Robot") -> ReplanDecision:
        planned = getattr(robot.planner, "planned_result", None)
        if not planned or not planned.get("is_success", False):
            return ReplanDecision("keep", "no successful path to validate")

        try:
            path_xyz = np.asarray(planned.get("xyz", []), dtype=float).reshape(-1, 3)
        except Exception:
            return ReplanDecision("keep", "path unavailable")
        if path_xyz.shape[0] < 2:
            return ReplanDecision("keep", "path too short")
        path_signature = self._path_signature(path_xyz)

        pose_now = robot._pose_from_state_in_frame(self.backend.world_frame)
        if pose_now is None:
            return ReplanDecision("keep", "current pose unavailable")

        current_xyz = np.array(
            [pose_now.position.x, pose_now.position.y, pose_now.position.z],
            dtype=float,
        )
        nearest_idx = int(np.argmin(np.linalg.norm(path_xyz - current_xyz, axis=1)))
        samples = self._sample_lookahead_path(robot, path_xyz, nearest_idx, current_xyz)
        if not samples:
            return ReplanDecision("keep", "no lookahead samples")

        dynamic_world = getattr(self.backend, "dynamic_world", None)
        if dynamic_world is None or not dynamic_world.obstacles:
            return ReplanDecision("keep", "no dynamic obstacles")

        for sample in samples:
            dynamic_clearance = dynamic_world.min_clearance_xyz(
                sample.xyz,
                t_offset=sample.t_offset,
            )
            if dynamic_clearance is not None and dynamic_clearance.distance_m < self.safety_margin_m:
                return ReplanDecision(
                    "replan",
                    f"path clearance to dynamic obstacle '{dynamic_clearance.obstacle_id}' "
                    f"is {dynamic_clearance.distance_m:.3f} m below margin "
                    f"{self.safety_margin_m:.3f} m at t+{sample.t_offset:.1f}s",
                    dynamic_clearance.obstacle_id,
                    dynamic_clearance.distance_m,
                    path_signature,
                    sample.t_offset,
                )

        return ReplanDecision("keep", "remaining path valid")

    def _sample_lookahead_path(
        self,
        robot: "Robot",
        path_xyz: np.ndarray,
        start_idx: int,
        current_xyz: np.ndarray,
    ) -> list[TimedPathSample]:
        remaining = path_xyz[max(0, start_idx):]
        if remaining.shape[0] == 0:
            return []

        nominal_speed = self._nominal_vehicle_speed(robot)
        lookahead_distance = nominal_speed * self.lookahead_time_s
        if lookahead_distance > 0.0:
            deltas = np.linalg.norm(np.diff(remaining, axis=0), axis=1)
            cumulative = np.concatenate(([0.0], np.cumsum(deltas)))
            end_idx = int(np.searchsorted(cumulative, lookahead_distance, side="right"))
            remaining = remaining[: max(2, min(end_idx + 1, remaining.shape[0]))]

        if remaining.shape[0] <= self.max_samples:
            sampled = remaining
        else:
            indices = np.linspace(0, remaining.shape[0] - 1, self.max_samples, dtype=int)
            sampled = remaining[indices]

        return self._timed_path_samples(sampled, current_xyz, nominal_speed)

    @staticmethod
    def _path_signature(path_xyz: np.ndarray) -> str:
        path = np.asarray(path_xyz, dtype=float).reshape(-1, 3)
        if path.shape[0] == 0:
            return ""
        indices = sorted({0, path.shape[0] // 2, path.shape[0] - 1})
        key_points = np.round(path[indices], 3).reshape(-1)
        return f"{path.shape[0]}:" + ",".join(f"{value:.3f}" for value in key_points)

    @staticmethod
    def _nominal_vehicle_speed(robot: "Robot") -> float:
        max_traj_vel = np.asarray(getattr(robot, "max_traj_vel", [0.15, 0.15, 0.10]), dtype=float)
        if max_traj_vel.size == 0:
            return 0.15
        speed = float(np.linalg.norm(max_traj_vel.reshape(-1)))
        return max(speed, 0.05)

    @staticmethod
    def _timed_path_samples(
        sampled_xyz: np.ndarray,
        current_xyz: np.ndarray,
        nominal_speed: float,
    ) -> list[TimedPathSample]:
        if sampled_xyz.size == 0:
            return []

        samples = np.asarray(sampled_xyz, dtype=float).reshape(-1, 3)
        current = np.asarray(current_xyz, dtype=float).reshape(3)
        distances = [float(np.linalg.norm(samples[0] - current))]
        if samples.shape[0] > 1:
            segment_lengths = np.linalg.norm(np.diff(samples, axis=0), axis=1)
            distances.extend((distances[0] + np.cumsum(segment_lengths)).tolist())

        speed = max(float(nominal_speed), 0.05)
        return [
            TimedPathSample(xyz=xyz, t_offset=max(0.0, distance / speed))
            for xyz, distance in zip(samples, distances)
        ]
