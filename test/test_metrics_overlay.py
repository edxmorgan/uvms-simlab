"""Metric UI formatting tests; no ROS initialization or robot commands."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from simlab.uvms_backend import UVMSBackendCore


def overlay(tracking, energy=None, state=None):
    if energy is None:
        energy = dict(prefix='robot_1_', arm_payload_mass=0., arm_gravity=9.81,
                      vehicle_control_energy_abs=12., arm_control_energy_abs=3.,
                      vehicle_control_power_abs=2., arm_control_power_abs=1.)
    robot = NS(prefix='robot_1_', k_robot=0, controller_name='OGES', sim_reset_hold=False,
               get_energy_metrics=lambda: energy,
               get_controller_performance_metrics=lambda: tracking,
               get_state=lambda: {'ned_vel': np.zeros(6)} if state is None else state)
    backend = NS(robots=[robot], selected_robot=robot,
                 vehicle_waypoint_missions={0: NS(executing=False, waypoints=[])})
    return UVMSBackendCore.format_robot_metrics_overlay_text(backend)


def test_inactive_tracking_never_looks_like_zero_error():
    text = overlay({'active': False})
    assert 'Tracking: inactive' in text
    assert 'score 0.000' not in text
    assert 'nAtt' not in text
    assert 'Absolute mechanical work 15.00 J' in text
    assert 'absolute mechanical power 3.00 W (not electrical)' in text


@pytest.mark.parametrize('tracking', [
    {'active': True},
    {'active': True, 'valid': False, 'status': 'missing pose'},
])
def test_invalid_tracking_reports_unavailable(tracking):
    text = overlay(tracking)
    assert 'Tracking: unavailable' in text
    assert 'score 0.000' not in text
    if 'status' in tracking:
        assert tracking['status'] in text


def test_valid_tracking_explains_geometry_time_and_dwell():
    values = dict(active=True, valid=True, tracking_score=.2, tracking_score_rms=.3,
                  vehicle_n_cross_track=.1, vehicle_n_along_track=.2,
                  vehicle_n_linear_velocity=.3, vehicle_n_attitude=.4,
                  arm_n_position=.5, valid_duration_sec=2., coverage_fraction=.8,
                  settled=True, within_tolerance_duration_sec=1.2, settling_dwell_sec=1.)
    text = overlay(values)
    assert 'score 0.200 (time RMS 0.300)' in text
    assert 'nAtt SO(3)/π 0.400' in text
    assert 'Valid duration 2.000 s | coverage 0.800' in text
    assert 'score band: dwell confirmed (1.200 / 1.000 s)' in text


def test_missing_or_nonfinite_optional_values_are_unavailable():
    text = overlay({'active': True, 'valid': True}, energy={'prefix': 'robot_1_'}, state={})
    assert 'time RMS unavailable' in text
    assert 'work unavailable J' in text
    assert 'power unavailable W' in text
    assert 'v unavailable m/s' in text
    assert 'nan' not in text


def test_tool_metrics_and_receive_age_have_explicit_units():
    text = overlay(dict(active=True, valid=True, tool_valid=True,
                        tool_position_error_m=.012, tool_position_error_m_rms=.02,
                        tool_attitude_error_rad=.03, tool_attitude_error_rad_rms=.04,
                        feedback_age_sec=.015))
    assert 'Tool reference tracking: position 0.012 m (time RMS 0.020)' in text
    assert 'orientation SO(3) 0.030 rad (time RMS 0.040)' in text
    assert 'Feedback receive age 0.015 s (not sensor age)' in text


def test_invalid_tracking_hides_stale_energy_and_speed():
    text = overlay(dict(active=True, valid=False, status='stale feedback'))
    assert 'work unavailable J' in text
    assert 'power unavailable W' in text
    assert 'v unavailable m/s' in text


def test_overlay_wraps_and_sizes_all_lines():
    from unittest.mock import Mock
    backend = NS(format_robot_metrics_overlay_text=lambda: '\n'.join(['x' * 300] * 12),
                 robot_metrics_overlay_pub=Mock())
    UVMSBackendCore.publish_robot_metrics_overlay_callback(backend)
    message = backend.robot_metrics_overlay_pub.publish.call_args.args[0]
    assert max(map(len, message.text.splitlines())) <= 128
    assert message.height >= len(message.text.splitlines()) * message.text_size * 1.6


def test_inactive_stale_feedback_does_not_display_old_power_or_speed():
    text = overlay({'active': False, 'feedback_valid': False})
    assert 'Tracking: inactive' in text
    assert 'power unavailable W' in text
    assert 'v unavailable m/s' in text
