"""Planner/controller-independent timed trajectory and braking checks."""
import math


def stopping_distance(speed, deceleration, reaction_time, tracking_error=0.0):
    values = (speed, deceleration, reaction_time, tracking_error)
    if not all(math.isfinite(float(v)) for v in values) or deceleration <= 0:
        raise ValueError('braking parameters must be finite; deceleration must be positive')
    if min(speed, reaction_time, tracking_error) < 0:
        raise ValueError('braking parameters must be nonnegative')
    return speed * reaction_time + speed * speed / (2.0 * deceleration) + tracking_error
