# Copyright (C) 2026 Edward Morgan
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Versioned live tracking metrics, integrated over fresh simulation time."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class ControllerPerformanceMetrics:
    vehicle_length_scale_m: float = 0.4
    along_track_time_scale_sec: float = 1.0
    min_speed_scale_mps: float = 0.1
    min_accel_scale_mps2: float = 0.1
    min_angular_speed_scale_radps: float = 0.1
    min_angular_accel_scale_radps2: float = 0.1
    arm_position_scale_rad: float = np.pi
    min_arm_velocity_scale_radps: float = 0.1
    min_arm_acceleration_scale_radps2: float = 0.1
    vehicle_force_scale_n: float = 100.0
    vehicle_torque_scale_nm: float = 10.0
    arm_effort_scale_nm: float = 10.0
    settling_score_tolerance: float = 0.1
    settling_dwell_sec: float = 1.0
    max_sample_gap_sec: float = 0.5
    _count: int = field(default=0, init=False)
    _integrals: dict = field(default_factory=dict, init=False)
    _durations: dict = field(default_factory=dict, init=False)
    _last_snapshot: dict = field(default_factory=dict, init=False)
    _previous: dict | None = field(default=None, init=False)
    _last_seen_time: float | None = field(default=None, init=False)
    _start_sim_time: float | None = field(default=None, init=False)
    _current_time: float | None = field(default=None, init=False)
    _feedback_age: float = field(default=float('nan'), init=False)
    _valid_duration: float = field(default=0., init=False)
    _distance_traveled_m: float = field(default=0., init=False)
    _energy_j: float = field(default=0., init=False)
    _energy_duration: float = field(default=0., init=False)
    _energy_distance: float = field(default=0., init=False)
    _max_tracking_score: float = field(default=0., init=False)
    _time_to_tolerance_sec: float = field(default=-1., init=False)
    _in_band_since: float | None = field(default=None, init=False)
    _was_active: bool = field(default=False, init=False)

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if name.startswith('_'):
                continue
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')

    @staticmethod
    def fixed_array(values, size: int) -> np.ndarray:
        """Legacy shape helper for callers; metric validation never pads inputs."""
        result = np.zeros(size, dtype=float)
        values = np.asarray(values, dtype=float).reshape(-1)
        result[:min(size, values.size)] = values[:size]
        return result

    @staticmethod
    def _array(data, key, size, allow_extra=False):
        values = np.asarray(data[key], dtype=float).reshape(-1)
        if (values.size < size if allow_extra else values.size != size) or not np.isfinite(values).all():
            raise ValueError(f'{key} requires {size} finite components')
        return values[:size]

    def reset(self):
        # Preserve user configuration, discard all state of the previous window.
        fresh = type(self)(**{k: getattr(self, k) for k, f in self.__dataclass_fields__.items() if f.init})
        self.__dict__.update({k: getattr(fresh, k) for k in self.__dataclass_fields__})

    def _unavailable(self, active, status):
        self._previous = None
        self._in_band_since = None
        elapsed = max(self._current_time-self._start_sim_time, 0.) if self._current_time is not None and self._start_sim_time is not None else float('nan')
        self._last_snapshot = dict(active=bool(active), valid=False, status=status,
            metric_version=2., sample_count=float(self._count), valid_duration_sec=self._valid_duration,
            elapsed_sec=elapsed, coverage_fraction=self._valid_duration/elapsed if elapsed > 0 else float('nan'),
            feedback_age_sec=self._feedback_age,
            settled=False, settling_time_sec=-1., within_tolerance_duration_sec=0.,
            effort_valid=False, energy_valid=False, tool_valid=False)
        return dict(self._last_snapshot)

    def update(self, *, current: Mapping, reference: Mapping, active: bool) -> dict:
        if not active:
            self._was_active = False
            return self._unavailable(False, 'inactive')
        if not self._was_active:
            self.reset()
            self._was_active = True
        try:
            t = float(current['sim_time'])
            if not np.isfinite(t):
                raise ValueError('nonfinite sim_time')
            duplicate = self._last_seen_time is not None and t == self._last_seen_time
            if self._last_seen_time is not None and t < self._last_seen_time:
                self.reset()
                self._was_active = True
                self._last_seen_time = t
                self._current_time = self._start_sim_time = t
                self._feedback_age = float(current.get('feedback_age_sec', float('nan')))
                return self._unavailable(True, 'clock moved backwards; window reset')
            self._last_seen_time = t
            self._current_time = t
            if self._start_sim_time is None:
                self._start_sim_time = t
            self._feedback_age = float(current.get('feedback_age_sec', float('nan')))
            feedback_valid = current.get('feedback_valid', True)
            if not isinstance(feedback_valid, (bool, np.bool_, int, float)) or feedback_valid not in (True, False) or not feedback_valid:
                raise ValueError('feedback unavailable')
            if 'feedback_age_sec' in current:
                age = float(current['feedback_age_sec'])
                if not np.isfinite(age) or age < 0 or age > self.max_sample_gap_sec:
                    raise ValueError('stale feedback')
            state = {k: self._array(current, k, n, n == 4) for k, n in
                     [('pose', 6), ('body_vel', 6), ('body_acc', 6), ('q', 4), ('dq', 4), ('ddq', 4)]}
            target = {k: self._array(reference, k, n, n == 4) for k, n in
                      [('pose', 6), ('body_vel', 6), ('body_acc', 6), ('q', 4), ('dq', 4), ('ddq', 4)]}
        except (KeyError, TypeError, ValueError) as exc:
            return self._unavailable(True, f'invalid input: {exc}')
        if duplicate:
            # A repeated fresh sample has zero measure, but its receive age and
            # status still need publishing through snapshot()-based consumers.
            self._last_snapshot = dict(self._last_snapshot,
                status='duplicate timestamp; no new sample', feedback_age_sec=self._feedback_age)
            return dict(self._last_snapshot)
        elapsed = t - self._start_sim_time
        previous = self._previous
        dt = t - previous['time'] if previous else 0.
        if dt > self.max_sample_gap_sec:
            previous = None
            dt = 0.
            self._in_band_since = None

        pose, pose_ref = state['pose'], target['pose']
        v, vd = state['body_vel'], target['body_vel']
        a, ad = state['body_acc'], target['body_acc']
        R, Rd = Rotation.from_euler('xyz', pose[3:]), Rotation.from_euler('xyz', pose_ref[3:])
        pos_error = pose_ref[:3] - pose[:3]
        angle = float((Rd.inv() * R).magnitude())
        linear_velocity_error = np.linalg.norm(R.apply(v[:3]) - Rd.apply(vd[:3]))
        angular_velocity_error = np.linalg.norm(R.apply(v[3:]) - Rd.apply(vd[3:]))
        # body_acc is d(body_vel)/dt, not inertial acceleration expressed in body.
        world_acc = R.apply(a[:3] + np.cross(v[3:], v[:3]))
        world_acc_ref = Rd.apply(ad[:3] + np.cross(vd[3:], vd[:3]))
        linear_acceleration_error = np.linalg.norm(world_acc - world_acc_ref)
        angular_acceleration_error = np.linalg.norm(R.apply(a[3:]) - Rd.apply(ad[3:]))
        try:
            direction = self._array(reference, 'trajectory_direction', 3) if 'trajectory_direction' in reference else Rd.apply(vd[:3])
        except (TypeError, ValueError):
            return self._unavailable(True, 'invalid trajectory direction')
        if np.linalg.norm(direction) <= 1e-6 and previous:
            direction = pose_ref[:3] - previous['reference_position']
        if np.linalg.norm(direction) > 1e-6:
            tangent = direction / np.linalg.norm(direction)
            along = float(pos_error @ tangent)
            cross = float(np.linalg.norm(pos_error - along * tangent))
        else:
            along, cross = 0., float(np.linalg.norm(pos_error))
        qe, dve, dae = (target[k] - state[k] for k in ('q', 'dq', 'ddq'))
        arm_rms = lambda x: float(np.sqrt(np.mean(x*x)))
        instant = dict(
            vehicle_cross_track_m=cross, vehicle_along_track_m=along,
            vehicle_position_error_m=float(np.linalg.norm(pos_error)), vehicle_attitude_error_rad=angle,
            vehicle_linear_velocity_error_mps=float(linear_velocity_error),
            vehicle_angular_velocity_error_radps=float(angular_velocity_error),
            vehicle_linear_acceleration_error_mps2=float(linear_acceleration_error),
            vehicle_angular_acceleration_error_radps2=float(angular_acceleration_error),
            vehicle_n_cross_track=cross/self.vehicle_length_scale_m,
            vehicle_n_along_track=abs(along)/max(np.linalg.norm(vd[:3])*self.along_track_time_scale_sec, self.vehicle_length_scale_m),
            vehicle_n_position=float(np.linalg.norm(pos_error))/self.vehicle_length_scale_m,
            vehicle_n_attitude=angle/np.pi,
            vehicle_n_linear_velocity=float(linear_velocity_error)/max(np.linalg.norm(vd[:3]), self.min_speed_scale_mps),
            vehicle_n_angular_velocity=float(angular_velocity_error)/max(np.linalg.norm(vd[3:]), self.min_angular_speed_scale_radps),
            vehicle_n_linear_acceleration=float(linear_acceleration_error)/max(np.linalg.norm(world_acc_ref), self.min_accel_scale_mps2),
            vehicle_n_angular_acceleration=float(angular_acceleration_error)/max(np.linalg.norm(ad[3:]), self.min_angular_accel_scale_radps2),
            arm_n_position=arm_rms(qe)/self.arm_position_scale_rad,
            arm_n_velocity=arm_rms(dve)/max(arm_rms(target['dq']), self.min_arm_velocity_scale_radps),
            arm_n_acceleration=arm_rms(dae)/max(arm_rms(target['ddq']), self.min_arm_acceleration_scale_radps2),
            arm_position_rms_rad=arm_rms(qe), arm_max_position_error_rad=float(np.max(abs(qe))))
        score_keys = ('vehicle_n_cross_track','vehicle_n_along_track','vehicle_n_attitude',
                      'vehicle_n_linear_velocity','vehicle_n_linear_acceleration',
                      'arm_n_position','arm_n_velocity','arm_n_acceleration')
        instant['tracking_score'] = float(np.sqrt(np.mean([instant[k]**2 for k in score_keys])))
        if not all(np.isfinite(value) for value in instant.values()):
            return self._unavailable(True, 'nonfinite derived tracking metric')
        effort_valid = False
        instant['normalized_control_effort'] = float('nan')
        try:
            wrench = self._array(current, 'body_forces', 6)
            effort = self._array(current, 'arm_effort', 4, True)
            instant['normalized_control_effort'] = float(np.linalg.norm(wrench[:3]/self.vehicle_force_scale_n)
                + np.linalg.norm(wrench[3:]/self.vehicle_torque_scale_nm) + np.linalg.norm(effort/self.arm_effort_scale_nm))
            effort_valid = bool(np.isfinite(instant['normalized_control_effort']))
            if not effort_valid:
                instant['normalized_control_effort'] = float('nan')
        except (KeyError, TypeError, ValueError):
            pass
        tool_valid = False
        instant['tool_position_error_m'] = instant['tool_attitude_error_rad'] = float('nan')
        try:
            tool = self._array(current, 'tool_pose', 6)
            tool_ref = self._array(reference, 'tool_pose', 6)
            instant['tool_position_error_m'] = float(np.linalg.norm(tool[:3]-tool_ref[:3]))
            instant['tool_attitude_error_rad'] = float((Rotation.from_euler('xyz',tool_ref[3:]).inv()*Rotation.from_euler('xyz',tool[3:])).magnitude())
            tool_valid = bool(np.isfinite(instant['tool_position_error_m']) and np.isfinite(instant['tool_attitude_error_rad']))
            if not tool_valid:
                instant['tool_position_error_m'] = instant['tool_attitude_error_rad'] = float('nan')
        except (KeyError, TypeError, ValueError):
            pass
        energy = None
        try:
            energy = np.array([float(current['vehicle_control_energy_abs']), float(current['arm_control_energy_abs'])])
            if not np.isfinite(energy).all() or np.any(energy < 0):
                energy = None
        except (KeyError, TypeError, ValueError):
            pass
        distance = float(np.linalg.norm(pose[:3]-previous['position'])) if previous else 0.
        energy_valid = bool(previous and energy is not None and previous['energy'] is not None and np.all(energy >= previous['energy']))
        if previous:
            self._valid_duration += dt
            self._distance_traveled_m += distance
            for k, value in instant.items():
                old = previous['instant'][k]
                if np.isfinite(value) and np.isfinite(old):
                    self._integrals[k] = self._integrals.get(k, 0.) + .5*(value*value + old*old)*dt
                    self._durations[k] = self._durations.get(k, 0.) + dt
            if energy_valid:
                self._energy_j += float(np.sum(energy-previous['energy']))
                self._energy_duration += dt
                self._energy_distance += distance
        self._count += 1
        self._max_tracking_score = max(self._max_tracking_score, instant['tracking_score'])
        if instant['tracking_score'] <= self.settling_score_tolerance:
            if self._time_to_tolerance_sec < 0:
                self._time_to_tolerance_sec = elapsed
            if self._in_band_since is None:
                self._in_band_since = t
        else:
            self._in_band_since = None
        dwell = t-self._in_band_since if self._in_band_since is not None else 0.
        settled = self._in_band_since is not None and dwell >= self.settling_dwell_sec
        snapshot = dict(instant)
        for k in instant:
            duration = self._durations.get(k, 0.)
            snapshot[k+'_rms'] = float(np.sqrt(self._integrals[k]/duration)) if duration > 0 else float('nan')
        snapshot.update(active=True, valid=True, status='valid', metric_version=2.,
            effort_valid=effort_valid, energy_valid=energy_valid, tool_valid=tool_valid, feedback_age_sec=self._feedback_age,
            arm_position_error_rad=qe.tolist(), arm_velocity_error_radps=dve.tolist(), arm_acceleration_error_radps2=dae.tolist(),
            # Historical diagnostic, not an efficiency objective; never RMS this ratio.
            effort_per_tracking_score=instant['normalized_control_effort']/max(instant['tracking_score'],1e-6),
            energy_per_meter=self._energy_j/self._energy_distance if energy_valid and self._energy_distance > 0 else float('nan'),
            energy_per_second=self._energy_j/self._energy_duration if energy_valid and self._energy_duration > 0 else float('nan'),
            mechanical_work_j=self._energy_j if energy_valid else float('nan'),
            time_to_tolerance_sec=self._time_to_tolerance_sec,
            peak_tracking_score=self._max_tracking_score,
            sample_count=float(self._count), elapsed_sec=elapsed, valid_duration_sec=self._valid_duration,
            coverage_fraction=self._valid_duration/elapsed if elapsed > 0 else float('nan'),
            settled=bool(settled), settling_time_sec=self._in_band_since-self._start_sim_time if settled else -1.,
            within_tolerance_duration_sec=dwell, settling_dwell_sec=self.settling_dwell_sec)
        self._previous = dict(time=t, position=pose[:3].copy(), reference_position=pose_ref[:3].copy(), energy=energy, instant=instant)
        self._last_snapshot = snapshot
        return dict(snapshot)

    def snapshot(self) -> dict:
        return dict(self._last_snapshot)
