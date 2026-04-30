# Copyright (C) 2026 Edward Morgan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass
class ControllerPerformanceMetrics:
    """Normalized per-robot tracking metrics for controller comparison."""

    vehicle_length_scale_m: float = 0.4
    along_track_time_scale_sec: float = 1.0
    min_speed_scale_mps: float = 0.1
    min_accel_scale_mps2: float = 0.1
    arm_position_scale_rad: float = np.pi
    min_arm_velocity_scale_radps: float = 0.1
    min_arm_acceleration_scale_radps2: float = 0.1
    vehicle_force_scale_n: float = 100.0
    vehicle_torque_scale_nm: float = 10.0
    arm_effort_scale_nm: float = 10.0
    settling_score_tolerance: float = 0.1
    _count: int = 0
    _sum_sq: dict[str, float] = field(default_factory=dict)
    _last_reference_position: np.ndarray | None = None
    _last_actual_position: np.ndarray | None = None
    _last_snapshot: dict[str, float] = field(default_factory=dict)
    _start_sim_time: float | None = None
    _start_energy_j: float | None = None
    _distance_traveled_m: float = 0.0
    _max_tracking_score: float = 0.0
    _time_to_tolerance_sec: float = -1.0
    _within_tolerance: bool = False
    _was_active: bool = False

    @staticmethod
    def fixed_array(values, size: int) -> np.ndarray:
        result = np.zeros(size, dtype=float)
        values = np.asarray(values, dtype=float).reshape(-1)
        n = min(size, values.size)
        if n:
            result[:n] = values[:n]
        return result

    @staticmethod
    def wrap_angle_error(error) -> np.ndarray:
        return (np.asarray(error, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi

    def reset(self) -> None:
        self._count = 0
        self._sum_sq.clear()
        self._last_reference_position = None
        self._last_actual_position = None
        self._last_snapshot.clear()
        self._start_sim_time = None
        self._start_energy_j = None
        self._distance_traveled_m = 0.0
        self._max_tracking_score = 0.0
        self._time_to_tolerance_sec = -1.0
        self._within_tolerance = False

    def update(
        self,
        *,
        current: Mapping,
        reference: Mapping,
        active: bool,
    ) -> dict[str, float]:
        if not active:
            self._last_snapshot = {"active": 0.0}
            self._last_reference_position = None
            self._last_actual_position = None
            self._was_active = False
            return dict(self._last_snapshot)
        if not self._was_active:
            self.reset()
            self._was_active = True

        pose = self.fixed_array(current.get("pose", []), 6)
        pose_ref = self.fixed_array(reference.get("pose", []), 6)
        body_vel = self.fixed_array(current.get("body_vel", []), 6)
        body_vel_ref = self.fixed_array(reference.get("body_vel", []), 6)
        body_acc = self.fixed_array(current.get("body_acc", []), 6)
        body_acc_ref = self.fixed_array(reference.get("body_acc", []), 6)

        q = self.fixed_array(current.get("q", []), 4)
        q_ref = self.fixed_array(reference.get("q", []), 4)
        dq = self.fixed_array(current.get("dq", []), 4)
        dq_ref = self.fixed_array(reference.get("dq", []), 4)
        ddq = self.fixed_array(current.get("ddq", []), 4)
        ddq_ref = self.fixed_array(reference.get("ddq", []), 4)

        pos_error = pose_ref[:3] - pose[:3]
        direction = self.fixed_array(reference.get("trajectory_direction", []), 3)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-6 and self._last_reference_position is not None:
            direction = pose_ref[:3] - self._last_reference_position
            direction_norm = float(np.linalg.norm(direction))
        self._last_reference_position = pose_ref[:3].copy()

        position_scale = max(float(self.vehicle_length_scale_m), 1e-6)
        if direction_norm > 1e-6:
            tangent = direction / direction_norm
            along_track_m = float(np.dot(pos_error, tangent))
            cross_track_vec = pos_error - along_track_m * tangent
            cross_track_m = float(np.linalg.norm(cross_track_vec))
        else:
            along_track_m = 0.0
            cross_track_m = float(np.linalg.norm(pos_error))

        target_speed = float(np.linalg.norm(body_vel_ref[:3]))
        velocity_scale = max(target_speed, self.min_speed_scale_mps)
        angular_velocity_scale = max(float(np.linalg.norm(body_vel_ref[3:6])), self.min_speed_scale_mps)
        accel_scale = max(float(np.linalg.norm(body_acc_ref[:3])), self.min_accel_scale_mps2)
        angular_accel_scale = max(float(np.linalg.norm(body_acc_ref[3:6])), self.min_accel_scale_mps2)
        along_track_scale = max(
            target_speed * self.along_track_time_scale_sec,
            position_scale,
        )
        arm_velocity_scale = max(float(np.sqrt(np.mean(dq_ref * dq_ref))), self.min_arm_velocity_scale_radps)
        arm_acceleration_scale = max(float(np.sqrt(np.mean(ddq_ref * ddq_ref))), self.min_arm_acceleration_scale_radps2)

        attitude_error = self.wrap_angle_error(pose_ref[3:6] - pose[3:6])
        q_error = q_ref - q
        dq_error = dq_ref - dq
        ddq_error = ddq_ref - ddq

        sim_time = float(current.get("sim_time", 0.0) or 0.0)
        if self._start_sim_time is None:
            self._start_sim_time = sim_time
        elapsed_sec = max(sim_time - self._start_sim_time, 0.0)
        if self._last_actual_position is not None:
            self._distance_traveled_m += float(np.linalg.norm(pose[:3] - self._last_actual_position))
        self._last_actual_position = pose[:3].copy()

        body_forces = self.fixed_array(current.get("body_forces", []), 6)
        arm_effort = self.fixed_array(current.get("arm_effort", []), 4)
        normalized_vehicle_effort = float(
            np.linalg.norm(body_forces[:3] / max(self.vehicle_force_scale_n, 1e-6))
            + np.linalg.norm(body_forces[3:6] / max(self.vehicle_torque_scale_nm, 1e-6))
        )
        normalized_arm_effort = float(np.linalg.norm(arm_effort / max(self.arm_effort_scale_nm, 1e-6)))
        total_energy_j = float(current.get("vehicle_control_energy_abs", 0.0) or 0.0) + float(
            current.get("arm_control_energy_abs", 0.0) or 0.0
        )
        if self._start_energy_j is None:
            self._start_energy_j = total_energy_j
        active_energy_j = max(total_energy_j - self._start_energy_j, 0.0)

        instant = {
            "active": 1.0,
            "vehicle_cross_track_m": cross_track_m,
            "vehicle_along_track_m": along_track_m,
            "vehicle_n_cross_track": cross_track_m / position_scale,
            "vehicle_n_along_track": abs(along_track_m) / along_track_scale,
            "vehicle_n_position": float(np.linalg.norm(pos_error)) / position_scale,
            "vehicle_n_attitude": float(np.linalg.norm(attitude_error)) / np.pi,
            "vehicle_n_linear_velocity": float(np.linalg.norm(body_vel_ref[:3] - body_vel[:3])) / velocity_scale,
            "vehicle_n_angular_velocity": float(np.linalg.norm(body_vel_ref[3:6] - body_vel[3:6])) / angular_velocity_scale,
            "vehicle_n_linear_acceleration": float(np.linalg.norm(body_acc_ref[:3] - body_acc[:3])) / accel_scale,
            "vehicle_n_angular_acceleration": float(np.linalg.norm(body_acc_ref[3:6] - body_acc[3:6])) / angular_accel_scale,
            "arm_n_position": float(np.sqrt(np.mean(q_error * q_error))) / self.arm_position_scale_rad,
            "arm_n_velocity": float(np.sqrt(np.mean(dq_error * dq_error))) / arm_velocity_scale,
            "arm_n_acceleration": float(np.sqrt(np.mean(ddq_error * ddq_error))) / arm_acceleration_scale,
            "normalized_control_effort": normalized_vehicle_effort + normalized_arm_effort,
            "energy_per_meter": active_energy_j / max(self._distance_traveled_m, self.vehicle_length_scale_m),
            "energy_per_second": active_energy_j / max(elapsed_sec, 1e-6),
        }

        score_terms = [
            instant["vehicle_n_cross_track"],
            instant["vehicle_n_along_track"],
            instant["vehicle_n_attitude"],
            instant["vehicle_n_linear_velocity"],
            instant["vehicle_n_linear_acceleration"],
            instant["arm_n_position"],
            instant["arm_n_velocity"],
            instant["arm_n_acceleration"],
        ]
        instant["tracking_score"] = float(np.sqrt(np.mean(np.square(score_terms))))
        instant["effort_per_tracking_score"] = instant["normalized_control_effort"] / max(
            instant["tracking_score"],
            1e-6,
        )
        self._max_tracking_score = max(self._max_tracking_score, instant["tracking_score"])
        if not self._within_tolerance:
            if instant["tracking_score"] <= self.settling_score_tolerance:
                self._time_to_tolerance_sec = elapsed_sec
                self._within_tolerance = True
        instant["time_to_tolerance_sec"] = self._time_to_tolerance_sec
        instant["peak_tracking_score"] = self._max_tracking_score

        self._count += 1
        for key, value in instant.items():
            if key == "active":
                continue
            self._sum_sq[key] = self._sum_sq.get(key, 0.0) + float(value) * float(value)

        snapshot = dict(instant)
        for key, sum_sq in self._sum_sq.items():
            snapshot[f"{key}_rms"] = float(np.sqrt(sum_sq / max(self._count, 1)))
        snapshot["sample_count"] = float(self._count)
        self._last_snapshot = snapshot
        return dict(snapshot)

    def snapshot(self) -> dict[str, float]:
        return dict(self._last_snapshot)
