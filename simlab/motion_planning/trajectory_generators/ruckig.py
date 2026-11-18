from __future__ import annotations

import numpy as np
from rclpy.node import Node
from ruckig import InputParameter, OutputParameter, Result, Ruckig, RuckigError, Trajectory
from bisect import bisect_right

from simlab.motion_planning.trajectory_generators.base import VehicleTrajectoryGeneratorTemplate


def encode_trajectory(trajectory):
    """Transfer Ruckig's exact constant-jerk coefficients, without resampling."""
    profiles = []
    for section in trajectory.profiles:
        encoded = []
        for profile in section:
            item = {name: list(getattr(profile, name)) for name in ('t', 'p', 'v', 'a', 'j')}
            item['ends'] = np.cumsum(item['t']).tolist()
            item['brake'] = {name: list(getattr(profile.brake, name))
                             for name in ('t', 'p', 'v', 'a', 'j')}
            item['brake']['duration'] = profile.brake.duration
            encoded.append(item)
        profiles.append(encoded)
    return dict(duration=trajectory.duration, ends=list(trajectory.intermediate_durations), profiles=profiles)


def sample_trajectory(data, elapsed):
    """Evaluate the exported profiles using Ruckig's at_time convention."""
    section = bisect_right(data['ends'], elapsed)
    finished = elapsed >= data['duration']
    if finished:
        section = len(data['profiles']) - 1
    local = elapsed - (data['ends'][section - 1] if section else 0.)
    result = []
    for profile in data['profiles'][section]:
        t = local
        brake = profile['brake']
        if section == 0:
            t -= brake['duration']
        if not finished and section == 0 and t < 0.:
            t = local
            index = int(t >= brake['t'][0])
            if index:
                t -= brake['t'][0]
            source = brake
            jerk = source['j'][index]
        elif finished or t >= profile['ends'][-1]:
            t -= profile['ends'][-1]
            source, index, jerk = profile, -1, 0.
        else:
            index = bisect_right(profile['ends'], t)
            if index:
                t -= profile['ends'][index - 1]
            source, jerk = profile, profile['j'][index]
        position, velocity, acceleration = (source[name][index] for name in ('p', 'v', 'a'))
        result.append((position + t*(velocity + t*(acceleration/2. + t*jerk/6.)),
                       velocity + t*(acceleration + t*jerk/2.), acceleration + t*jerk))
    return tuple(map(list, zip(*result)))


def calculate_trajectory(settings):
    """Run only in the calculation process: waypoint solves may use the cloud."""
    inp = InputParameter(3)
    for name, value in settings.items():
        setattr(inp, name, value)
    trajectory = Trajectory(3, len(inp.intermediate_positions))
    otg = Ruckig(3, .01, len(inp.intermediate_positions))
    result = otg.calculate(inp, trajectory)
    if result not in (Result.Working, Result.Finished):
        raise RuntimeError(f'Ruckig calculation failed: {result}')
    return encode_trajectory(trajectory)


def trajectory_speed_bound(data):
    """Bound Euclidean speed using extrema of each constant-jerk phase."""
    maxima = np.zeros(3)
    for section in data['profiles']:
        for axis, profile in enumerate(section):
            for phase in (profile, profile['brake']):
                for duration, velocity, acceleration, jerk in zip(
                        phase['t'], phase['v'], phase['a'], phase['j']):
                    times = [0., duration]
                    if jerk and 0. < -acceleration / jerk < duration:
                        times.append(-acceleration / jerk)
                    maxima[axis] = max(maxima[axis], *(abs(velocity + acceleration*t + .5*jerk*t*t)
                                                       for t in times))
    if not np.isfinite(maxima).all():
        raise ValueError('trajectory speed bound is nonfinite')
    return float(np.linalg.norm(maxima))


class RuckigVehicleTrajectoryGenerator(VehicleTrajectoryGeneratorTemplate):
    registry_name = "ruckig"

    def __init__(self, rclpy_node: Node, dofs: int, control_dt: float, max_waypoints: int):
        if dofs != 3:
            raise ValueError("RuckigVehicleTrajectoryGenerator is intended for 3 DoF position (x, y, z)")
        self.max_waypoints = max_waypoints
        self.control_dt = float(control_dt)
        self.prepared = None
        self.elapsed = 0.
        self.rclpy_node = rclpy_node
        self.otg = Ruckig(dofs, control_dt, max_waypoints)
        self.inp = InputParameter(dofs)
        self.out = OutputParameter(dofs, max_waypoints)
        self.active = False
        self.yaw_finish_threshold = 0.95
        self.last_result = None
        self.last_yaw_blend_factor = 0.0

    def start_from_path(
        self,
        current_position,
        path_xyz,
        max_vel,
        max_acc,
        max_jerk,
        current_velocity=None,
    ):
        path_xyz = np.asarray(path_xyz, dtype=float)
        if path_xyz.ndim != 2 or path_xyz.shape[1] != 3:
            raise ValueError("path_xyz must be an array of shape (N, 3)")

        current_position = np.asarray(current_position, dtype=float)
        if current_position.shape != (3,):
            raise ValueError("current_position must be length 3")

        max_vel = np.asarray(max_vel, dtype=float)
        max_acc = np.asarray(max_acc, dtype=float)
        max_jerk = np.asarray(max_jerk, dtype=float)

        if current_velocity is None:
            current_velocity = np.zeros(3, dtype=float)
        else:
            current_velocity = np.asarray(current_velocity, dtype=float)
            if current_velocity.shape != (3,) or not np.all(np.isfinite(current_velocity)):
                raise ValueError("current_velocity must be length 3 with finite values")

        self.prepared = None
        self.elapsed = 0.
        self.inp.current_position = current_position.tolist()
        self.inp.current_velocity = current_velocity.tolist()
        self.inp.current_acceleration = [0.0, 0.0, 0.0]

        if path_xyz.shape[0] > 2:
            intermediate = [row.tolist() for row in path_xyz[1:-1]]
        else:
            intermediate = []

        self.inp.intermediate_positions = intermediate
        self.inp.target_position = path_xyz[-1].tolist()
        self.inp.target_velocity = [0.0, 0.0, 0.0]
        self.inp.target_acceleration = [0.0, 0.0, 0.0]

        self.inp.max_velocity = max_vel.tolist()
        self.inp.max_acceleration = max_acc.tolist()
        self.inp.max_jerk = max_jerk.tolist()
        self.active = True
        self.last_result = None
        self.last_yaw_blend_factor = 0.0
        speed = float(np.linalg.norm(current_velocity))
        if speed > 1e-4:
            self.rclpy_node.get_logger().info(
                f"Ruckig trajectory seeded with current velocity {current_velocity.round(3).tolist()} m/s"
            )


    def calculation_request(self):
        return {name: list(getattr(self.inp, name)) for name in (
            'current_position', 'current_velocity', 'current_acceleration',
            'target_position', 'target_velocity', 'target_acceleration',
            'intermediate_positions', 'max_velocity', 'max_acceleration', 'max_jerk')}

    def install_calculation(self, prepared):
        self.prepared = prepared
        self.elapsed = 0.
        self.active = True

    def speed_bound(self):
        if self.prepared is not None:
            return trajectory_speed_bound(self.prepared)
        if self.last_result is None:
            raise ValueError('speed bound requires a calculated trajectory')
        return trajectory_speed_bound(encode_trajectory(self.out.trajectory))

    def update(self, yaw_blend_factor):
        """Advance one control step along the current trajectory."""
        if not self.active:
            return None, None, None, Result.Error

        if self.prepared is not None:
            self.elapsed += self.control_dt
            pos, vel, acc = sample_trajectory(self.prepared, self.elapsed)
            self.last_result = (Result.Finished if self.elapsed >= self.prepared['duration'] else Result.Working)
            self.last_yaw_blend_factor = float(yaw_blend_factor)
            if self.last_result == Result.Finished and yaw_blend_factor >= self.yaw_finish_threshold:
                self.active = False
            return pos, vel, acc, self.last_result

        try:
            res = self.otg.update(self.inp, self.out)
        except RuckigError as exc:
            self.rclpy_node.get_logger().error(f"Ruckig update failed: {exc}")
            self.active = False
            self.last_result = Result.Error
            return None, None, None, Result.Error

        pos = list(self.out.new_position)
        vel = list(self.out.new_velocity)
        acc = list(self.out.new_acceleration)
        self.last_result = res
        self.last_yaw_blend_factor = float(yaw_blend_factor)

        self.out.pass_to_input(self.inp)

        if self.out.new_calculation:
            self.rclpy_node.get_logger().info(
                f"Ruckig new trajectory, calculation {self.out.calculation_duration:0.1f} us, "
                f"duration {self.out.trajectory.duration:0.4f} s"
            )

        if res == Result.Finished and yaw_blend_factor >= self.yaw_finish_threshold:
            self.rclpy_node.get_logger().info("Ruckig trajectory finished")
            self.active = False

        return pos, vel, acc, res

    def current_reference(self):
        if not self.active:
            return None
        if self.prepared is not None:
            return tuple(np.asarray(v) for v in sample_trajectory(self.prepared, self.elapsed))
        source = self.inp if self.last_result is None else self.out
        prefix = 'current_' if self.last_result is None else 'new_'
        return tuple(np.array(getattr(source, prefix + field), dtype=float)
                     for field in ('position', 'velocity', 'acceleration'))

    def preview_position(self, t_offset):
        """Evaluate a future reference position without advancing execution."""
        if not np.isfinite(t_offset) or t_offset < 0:
            raise ValueError('preview offset must be finite and nonnegative')
        if self.prepared is not None:
            elapsed = min(self.prepared['duration'], self.elapsed + t_offset)
            return np.asarray(sample_trajectory(self.prepared, elapsed)[0])
        if self.last_result is None:
            raise ValueError('preview requires a calculated trajectory')
        elapsed = min(float(self.out.trajectory.duration), float(self.out.time) + t_offset)
        return np.asarray(self.out.trajectory.at_time(elapsed)[0])

    def preview_samples(self, *, horizon=None, sample_dt=0.05):
        """Read the active Ruckig output; never calculate or advance execution."""
        if not self.active:
            return []
        if not np.isfinite(sample_dt) or sample_dt <= 0:
            raise ValueError('sample period must be positive and finite')
        if self.prepared is not None:
            remaining = max(0., self.prepared['duration'] - self.elapsed)
            duration = remaining if horizon is None else min(remaining, max(0., float(horizon)))
            count = 1 if duration == 0 else int(np.ceil(duration / sample_dt)) + 1
            if count > 20000:
                raise ValueError('trajectory exceeds preview sample budget')
            return [(float(t), self.preview_position(float(t))) for t in np.linspace(0., duration, count)]
        if self.last_result is None:
            return []  # The original update loop has not calculated a trajectory yet.
        elapsed = float(self.out.time)
        remaining = max(0.0, float(self.out.trajectory.duration) - elapsed)
        duration = remaining if horizon is None else min(remaining, max(0.0, float(horizon)))
        count = 1 if duration == 0 else int(np.ceil(duration / sample_dt)) + 1
        if count > 20000:
            raise ValueError('trajectory exceeds preview sample budget')
        return [(float(t), self.preview_position(float(t))) for t in np.linspace(0., duration, count)]

    def close(self):
        self.active = False
        self.out = None
        self.inp = None
        self.otg = None
        self.rclpy_node = None
