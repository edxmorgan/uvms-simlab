from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simlab.dynamic_world import DynamicClearance, DynamicWorldModel
from simlab.fcl_checker import FCLWorld


DYNAMIC_CLEARANCE_TOLERANCE_M = 1e-4
DEFAULT_BOUNDARY_TRACKING_TOLERANCE_M = 0.01


@dataclass(frozen=True)
class PlannerWorldClearance:
    source: str
    distance_m: float
    obstacle_id: str = ""


class TimedTrajectoryViolation(ValueError):
    def __init__(self, message, *, t_offset_s, obstacle_id, distance_m):
        super().__init__(message)
        self.t_offset_s = t_offset_s
        self.obstacle_id = obstacle_id
        self.distance_m = distance_m


class PlannerWorld:
    """Common static/dynamic world query API for planner implementations."""

    def __init__(
        self,
        *,
        fcl_world: FCLWorld,
        dynamic_world: DynamicWorldModel | None = None,
        robot_prefix: str = "",
    ):
        self.fcl_world = fcl_world
        self.dynamic_world = dynamic_world
        self.robot_prefix = robot_prefix

    def min_clearance_xyz(
        self,
        xyz,
        *,
        t_offset: float = 0.0,
    ) -> PlannerWorldClearance:
        point = np.asarray(xyz, dtype=float).reshape(3)
        static_distance = float(self.fcl_world.min_distance_xyz(point))
        best = PlannerWorldClearance(source="static", distance_m=static_distance)

        dynamic_clearance = self._dynamic_clearance(point, t_offset=t_offset)
        if dynamic_clearance is not None and dynamic_clearance.distance_m < best.distance_m:
            return PlannerWorldClearance(
                source="dynamic",
                distance_m=float(dynamic_clearance.distance_m),
                obstacle_id=dynamic_clearance.obstacle_id,
            )
        return best

    def in_collision_xyz(self, xyz, *, t_offset: float = 0.0) -> bool:
        point = np.asarray(xyz, dtype=float).reshape(3)
        if self.fcl_world.planner_in_collision_at_xyz(point):
            return True
        dynamic_clearance = self._dynamic_clearance(point, t_offset=t_offset)
        return (
            dynamic_clearance is not None
            and dynamic_clearance.distance_m < -DYNAMIC_CLEARANCE_TOLERANCE_M
        )

    def is_state_valid_xyz(
        self,
        xyz,
        *,
        safety_margin: float = 0.0,
        t_offset: float = 0.0,
    ) -> bool:
        margin = max(0.0, float(safety_margin))
        if margin > 0.0:
            required = max(0.0, margin - DYNAMIC_CLEARANCE_TOLERANCE_M)
            threshold_query = getattr(self.fcl_world, 'planner_has_clearance_at_xyz', None)
            if threshold_query is not None:
                if not threshold_query(xyz, required):
                    return False
                dynamic = self._dynamic_clearance(np.asarray(xyz, dtype=float), t_offset=t_offset)
                return dynamic is None or dynamic.distance_m >= required
            return self.min_clearance_xyz(xyz, t_offset=t_offset).distance_m >= required
        return not self.in_collision_xyz(xyz, t_offset=t_offset)

    def invalid_position_reason(self, xyz, *, safety_margin=0.0, bounds=None, bounds_tolerance=0.):
        point = np.asarray(xyz, dtype=float).reshape(3)
        if not np.isfinite(point).all():
            return 'position is nonfinite'
        bounds = np.asarray(self.fcl_world.env_xyz_bounds if bounds is None else bounds).reshape(3, 2)
        if not np.isfinite(bounds_tolerance) or bounds_tolerance < 0:
            raise ValueError("Bounds tolerance must be finite and nonnegative")
        if np.any(point < bounds[:, 0]-bounds_tolerance) or np.any(point > bounds[:, 1]+bounds_tolerance):
            return (f"position is outside environment bounds: xyz={point.tolist()}, "
                    f"bounds={bounds.tolist()}, tolerance={bounds_tolerance:g} m")
        if self.fcl_world.planner_in_collision_at_xyz(point):
            obstacle = self.fcl_world.planner_collision_obstacle(point)
            return (f'planning sphere (radius {self.fcl_world.vehicle_radius:.3f} m) '
                    f'intersects {obstacle or "static geometry"}')
        clearance = self.min_clearance_xyz(point)
        if clearance.distance_m < max(0., safety_margin) - DYNAMIC_CLEARANCE_TOLERANCE_M:
            return (f'{clearance.source} clearance {clearance.distance_m:.3f} m '
                    f'is below {safety_margin:.3f} m')
        return ''

    def _obstacle_motion_speed_bound(self, dynamic):
        from bringup.collision_geometry import collision_geometry
        obstacle_speed = 0.
        if dynamic is not None:
            error = dynamic.availability_error(self.robot_prefix)
            if error:
                raise ValueError('trajectory world unavailable: ' + error)
            for obstacle in dynamic.obstacles.values():
                if self.robot_prefix and obstacle.owner_robot == self.robot_prefix:
                    continue
                if not obstacle.velocity_valid:
                    raise ValueError('trajectory obstacle velocity unavailable: ' + obstacle.obstacle_id)
                # Rotation does not change a sphere. For other shapes bound every
                # surface point by the shared geometry's enclosing radius.
                angular_speed = float(np.linalg.norm(obstacle.angular_velocity_world))
                rotational_speed = 0.
                if obstacle.collision_type != 1 and angular_speed:
                    _, radius = collision_geometry(obstacle.collision_type, obstacle.collision_dimensions,
                        obstacle.collision_mesh_resource, obstacle.collision_mesh_scale)
                    rotational_speed = angular_speed * radius
                obstacle_speed = max(obstacle_speed,
                    float(np.linalg.norm(obstacle.linear_velocity_world)) + rotational_speed)
        return obstacle_speed

    def require_clear_timed_trajectory(self, samples, *, max_speed, safety_margin=0.01,
                                       sample_at_time=None):
        """Check a timed curve using clearance bounds between adjacent samples.

        This certifies the reference against the snapshot's constant-twist
        prediction, not uncertain future obstacle behavior or tracking error.
        max_speed must bound the curve's speed over the complete checked interval.
        An exact sample_at_time callback permits local refinement of intervals
        whose sampling allowance alone prevents a clearance certificate.
        """
        samples = list(samples)
        if not samples:
            raise ValueError('trajectory has no samples')
        times = np.asarray([sample[0] for sample in samples], dtype=float)
        points = np.asarray([sample[1] for sample in samples], dtype=float)
        if (points.shape != (len(samples), 3) or not np.isfinite(points).all()
                or not np.isfinite(times).all() or times[0] < 0
                or (len(times) > 1 and np.any(np.diff(times) <= 0))):
            raise ValueError('trajectory samples must be finite and strictly time ordered')
        if not np.isfinite(max_speed) or max_speed < 0 or not np.isfinite(safety_margin) or safety_margin < 0:
            raise ValueError('trajectory speed and margin must be finite and nonnegative')
        dynamic = self.dynamic_world.snapshot() if self.dynamic_world is not None else None
        obstacle_speed = self._obstacle_motion_speed_bound(dynamic)
        world = PlannerWorld(fcl_world=self.fcl_world, dynamic_world=dynamic, robot_prefix=self.robot_prefix)
        cache = {}

        def violation(offset, obstacle_id, distance, required, *, uncertain=False):
            label = 'clearance certificate unresolved' if uncertain else 'clearance'
            return TimedTrajectoryViolation(f"trajectory {label} to '{obstacle_id}' {distance:.4f} m < "
                f'{required:.4f} m at +{offset:.3f} s', t_offset_s=float(offset),
                obstacle_id=obstacle_id, distance_m=distance)

        def query(offset, point):
            if len(cache) >= 20000:
                raise ValueError('trajectory clearance refinement exceeds query budget')
            point = np.asarray(point, dtype=float)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError('trajectory refinement returned invalid position')
            static = float(self.fcl_world.min_distance_xyz(point))
            if not np.isfinite(static) and static != float('inf'):
                raise ValueError('trajectory static clearance is invalid')
            if static <= DYNAMIC_CLEARANCE_TOLERANCE_M and self.fcl_world.planner_in_collision_at_xyz(point):
                raise TimedTrajectoryViolation(
                    f'trajectory intersects static geometry at +{offset:.3f} s',
                    t_offset_s=float(offset), obstacle_id='static', distance_m=static)
            dynamic_check = world._dynamic_clearance(point, t_offset=float(offset))
            records = [('static', static, float(max_speed))]
            if dynamic_check is not None:
                records.append((dynamic_check.obstacle_id, dynamic_check.distance_m,
                                float(max_speed) + obstacle_speed))
            for obstacle_id, distance, _ in records:
                if distance < safety_margin - DYNAMIC_CLEARANCE_TOLERANCE_M:
                    raise violation(offset, obstacle_id, distance, safety_margin)
            cache[offset] = records

        for offset, point in zip(times, points):
            query(float(offset), point)

        def certify(left, right, depth=0):
            half_interval = (right - left) / 2.
            unresolved = None
            for offset in (left, right):
                for obstacle_id, distance, speed in cache[offset]:
                    required = safety_margin + speed * half_interval
                    if distance < required - DYNAMIC_CLEARANCE_TOLERANCE_M:
                        unresolved = (offset, obstacle_id, distance, required)
                        break
                if unresolved:
                    break
            if unresolved is None:
                return
            if sample_at_time is None or depth >= 12:
                raise violation(*unresolved, uncertain=True)
            middle = (left + right) / 2.
            query(middle, sample_at_time(middle))
            certify(left, middle, depth + 1)
            certify(middle, right, depth + 1)

        for left, right in zip(times[:-1], times[1:]):
            certify(float(left), float(right))


    def require_clear_segment(self, start, goal, *, safety_margin=0.01, start_bounds_tolerance=0.):
        """Conservatively certify a straight translation with a distance bound.

        Distance to static geometry is 1-Lipschitz: adding half the sampling
        interval to the margin also protects the spaces between samples.
        This is an admission check, not a guarantee about controller tracking.
        """
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        for label, point in (('start', start), ('target', goal)):
            reason = self.invalid_position_reason(point, safety_margin=safety_margin,
                bounds_tolerance=start_bounds_tolerance if label == "start" else 0.)
            if reason:
                raise ValueError(f'{label}: {reason}')
        length = float(np.linalg.norm(goal-start))
        count = max(1, int(np.ceil(length / 0.05)))
        if count > 4000:
            raise ValueError('direct task move exceeds collision-check budget; use vehicle planning')
        margin = safety_margin + length / (2*count)
        for point in np.linspace(start, goal, count+1):
            if not self.is_state_valid_xyz(point, safety_margin=margin):
                raise ValueError('direct vehicle translation lacks collision clearance; use vehicle planning')

    def planning_start(self, measured, bounds, tolerance):
        """Return an in-bounds start with a checked, short boundary connector.

        The caller must retain the measured pose and connector in its result.
        This allowance is for coordinate-boundary tracking error, not obstacles.
        """
        measured = np.asarray(measured, dtype=float).reshape(3)
        bounds = np.asarray(bounds, dtype=float).reshape(3, 2)
        reason = self.invalid_position_reason(measured, bounds=bounds,
                                              bounds_tolerance=tolerance)
        if reason:
            raise ValueError(reason)
        projected = np.clip(measured, bounds[:, 0], bounds[:, 1])
        distance = float(np.linalg.norm(projected-measured))
        if distance > tolerance:
            raise ValueError(f'distance to environment bounds {distance:g} m exceeds {tolerance:g} m')
        if distance:
            reason = self.invalid_position_reason(projected, bounds=bounds)
            if reason:
                raise ValueError(reason)
            # A midpoint clearance exceeding half the connector length covers
            # the complete segment, not merely its endpoints.
            clearance = self.min_clearance_xyz((measured+projected)/2).distance_m
            if clearance <= distance/2:
                raise ValueError('boundary connector lacks collision clearance')
        return projected

    def _dynamic_clearance(self, xyz: np.ndarray, *, t_offset: float) -> DynamicClearance | None:
        if self.dynamic_world is None:
            return None
        return self.dynamic_world.min_clearance_xyz(xyz, t_offset=t_offset, exclude_owner=self.robot_prefix)
