import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from simlab.performance_metrics import ControllerPerformanceMetrics


def sample(t=0., **changes):
    result = dict(pose=np.zeros(6), body_vel=np.zeros(6), body_acc=np.zeros(6),
                  q=np.zeros(5), dq=np.zeros(5), ddq=np.zeros(5), sim_time=t,
                  body_forces=np.zeros(6), arm_effort=np.zeros(5),
                  vehicle_control_energy_abs=0., arm_control_energy_abs=0.)
    return result | changes


def attitude_metric(actual, desired):
    metrics = ControllerPerformanceMetrics()
    ref = sample(pose=np.r_[np.zeros(3), desired])
    metrics.update(current=sample(pose=np.r_[np.zeros(3), actual]), reference=ref, active=True)
    return metrics.update(current=sample(.1, pose=np.r_[np.zeros(3), actual]), reference=ref, active=True)


def test_equivalent_euler_representations_have_zero_attitude_error():
    first = [.2, .4, .6]
    equivalent = [.2 + np.pi, np.pi - .4, .6 + np.pi]
    result = attitude_metric(first, equivalent)
    assert result['vehicle_n_attitude'] == pytest.approx(0, abs=1e-14)
    assert result['tracking_score'] == pytest.approx(0, abs=1e-14)


@pytest.mark.parametrize('angle', [0., 1e-9, .7, np.pi - 1e-9, np.pi])
def test_principal_angle_for_arbitrary_axis_and_common_frame(angle):
    axis = np.array([1., 2., -3.]) / np.sqrt(14)
    common = Rotation.from_euler('xyz', [.8, -.4, 1.1])
    actual = common * Rotation.from_rotvec(axis * angle)
    result = attitude_metric(actual.as_euler('xyz'), common.as_euler('xyz'))
    assert result['vehicle_n_attitude'] == pytest.approx(angle / np.pi, abs=1e-14)
    assert result['vehicle_n_attitude_rms'] == pytest.approx(angle / np.pi, abs=1e-14)


def test_yaw_wrap_uses_shortest_rotation():
    result = attitude_metric([0, 0, np.pi - .01], [0, 0, -np.pi + .01])
    assert result['vehicle_n_attitude'] == pytest.approx(.02 / np.pi)


def test_equivalent_gimbal_lock_orientations():
    result = attitude_metric([.2, np.pi / 2, .6], [.9, np.pi / 2, 1.3])
    assert result['vehicle_n_attitude'] == pytest.approx(0, abs=1e-14)


def test_velocity_and_acceleration_compare_physical_world_vectors():
    # Same world velocity/acceleration despite different orientations/body rates.
    m = ControllerPerformanceMetrics()
    state = sample(pose=[0,0,0,0,0,np.pi/2], body_vel=[0,-1,0,0,0,2],
                   body_acc=[-2,-3,0,0,0,.5])
    ref = sample(body_vel=[1,0,0,0,0,2], body_acc=[3,-2,0,0,0,.5])
    result = m.update(current=state, reference=ref, active=True)
    for key in ['vehicle_linear_velocity_error_mps','vehicle_angular_velocity_error_radps',
                'vehicle_linear_acceleration_error_mps2','vehicle_angular_acceleration_error_radps2']:
        assert result[key] == pytest.approx(0., abs=1e-14)


def test_rms_integrates_elapsed_time_not_callback_count():
    m = ControllerPerformanceMetrics(max_sample_gap_sec=2.)
    ref = sample()
    for t, x in [(0.,0.),(.1,1.),(1.,1.)]:
        r = m.update(current=sample(t, pose=[x,0,0,0,0,0]), reference=ref, active=True)
    assert r['vehicle_position_error_m_rms'] == pytest.approx(np.sqrt(.95))
    assert r['valid_duration_sec'] == pytest.approx(1.)
    assert r['coverage_fraction'] == 1.
    assert 'time_to_tolerance_sec_rms' not in r
    assert 'energy_per_meter_rms' not in r


@pytest.mark.parametrize('bad', [{}, sample(pose=[0,0]), sample(q=[0,np.nan,0,0]),
                                   sample(feedback_valid=False), sample(feedback_age_sec=1.)])
def test_bad_or_missing_inputs_are_unavailable(bad):
    r = ControllerPerformanceMetrics().update(current=bad, reference=sample(), active=True)
    assert not r['valid']
    assert 'tracking_score' not in r


def test_invalid_duplicate_and_gaps_break_integration_and_dwell():
    m = ControllerPerformanceMetrics(settling_dwell_sec=.2)
    ref = sample()
    m.update(current=sample(), reference=ref, active=True)
    r = m.update(current=sample(.2), reference=ref, active=True)
    assert r['settled']
    r = m.update(current=sample(.2), reference=ref, active=True)
    assert r['valid'] and r['settled'] and r['sample_count'] == 2.
    r = m.update(current=sample(.3), reference=ref, active=True)
    assert r['valid_duration_sec'] == pytest.approx(.3)
    assert r['settled']
    r = m.update(current=sample(1.), reference=ref, active=True)
    assert r['valid_duration_sec'] == pytest.approx(.3)
    assert r['within_tolerance_duration_sec'] == 0.
    m.update(current={}, reference=ref, active=True)
    r = m.update(current=sample(1.1), reference=ref, active=True)
    assert r['valid_duration_sec'] == pytest.approx(.3)


def test_settling_revokes_after_departure_but_first_entry_is_retained():
    m = ControllerPerformanceMetrics(settling_dwell_sec=.2)
    ref = sample()
    for t in [0,.1,.2]:
        r = m.update(current=sample(t), reference=ref, active=True)
    assert r['settled'] and r['settling_time_sec'] == 0.
    r = m.update(current=sample(.3, pose=[1,0,0,0,0,0]), reference=ref, active=True)
    assert not r['settled'] and r['settling_time_sec'] == -1.
    assert r['time_to_tolerance_sec'] == 0.


def test_time_rollback_resets_window_and_requires_new_anchor():
    m = ControllerPerformanceMetrics()
    for t in [10.,10.1]:
        m.update(current=sample(t), reference=sample(), active=True)
    r = m.update(current=sample(0.), reference=sample(), active=True)
    assert not r['valid'] and r['sample_count'] == 0.
    r = m.update(current=sample(.1), reference=sample(), active=True)
    assert r['valid'] and r['valid_duration_sec'] == 0.


def test_optional_metrics_never_become_zero_when_absent():
    m = ControllerPerformanceMetrics()
    state=sample();state.pop('body_forces');state.pop('vehicle_control_energy_abs')
    r=m.update(current=state,reference=sample(),active=True)
    assert r['valid'] and not r['effort_valid'] and not r['energy_valid'] and not r['tool_valid']
    assert np.isnan(r['normalized_control_effort']) and np.isnan(r['tool_position_error_m'])
    r=m.update(current=sample(.1,tool_pose=[1,0,0,0,0,0]),reference=sample(tool_pose=[0,0,0,0,0,np.pi]),active=True)
    assert r['tool_valid'] and r['tool_position_error_m']==1.
    assert r['tool_attitude_error_rad']==pytest.approx(np.pi)


@pytest.mark.parametrize('setting', ['vehicle_length_scale_m','settling_dwell_sec','max_sample_gap_sec'])
@pytest.mark.parametrize('value',[0.,-1.,float('nan')])
def test_configuration_must_be_finite_and_positive(setting,value):
    with pytest.raises(ValueError):ControllerPerformanceMetrics(**{setting:value})


def test_energy_work_is_not_fabricated_across_gap_or_counter_reset():
    m=ControllerPerformanceMetrics()
    m.update(current=sample(),reference=sample(),active=True)
    r=m.update(current=sample(.1,vehicle_control_energy_abs=2.),reference=sample(),active=True)
    assert r['mechanical_work_j']==2. and r['energy_per_second']==20.
    assert np.isnan(r['energy_per_meter'])  # no travel, not an artificial denominator
    r=m.update(current=sample(1.,vehicle_control_energy_abs=100.),reference=sample(),active=True)
    assert not r['energy_valid']
    r=m.update(current=sample(1.1,vehicle_control_energy_abs=1.),reference=sample(),active=True)
    assert not r['energy_valid']


def test_coverage_includes_initial_invalid_and_later_missing_intervals():
    m=ControllerPerformanceMetrics()
    r=m.update(current={'sim_time':0.},reference=sample(),active=True)
    assert not r['valid'] and r['elapsed_sec']==0.
    m.update(current=sample(1.),reference=sample(),active=True)
    r=m.update(current=sample(1.1),reference=sample(),active=True)
    assert r['coverage_fraction']==pytest.approx(.1/1.1)
    r=m.update(current={'sim_time':2.,'feedback_age_sec':1.},reference=sample(),active=True)
    assert not r['valid'] and r['elapsed_sec']==2.
    assert r['coverage_fraction']==pytest.approx(.05)
    assert r['feedback_age_sec']==1.


@pytest.mark.parametrize('flag',[float('nan'),2.,'true'])
def test_invalid_feedback_flag_is_not_truthy(flag):
    r=ControllerPerformanceMetrics().update(current=sample(feedback_valid=flag),reference=sample(),active=True)
    assert not r['valid']


def test_energy_counter_reset_cannot_be_hidden_by_other_counter_growth():
    m=ControllerPerformanceMetrics()
    m.update(current=sample(vehicle_control_energy_abs=10.,arm_control_energy_abs=10.),reference=sample(),active=True)
    r=m.update(current=sample(.1,vehicle_control_energy_abs=0.,arm_control_energy_abs=30.),reference=sample(),active=True)
    assert not r['energy_valid']


def test_duplicate_refreshes_cached_age_and_status_without_integrating():
    m=ControllerPerformanceMetrics()
    m.update(current=sample(feedback_age_sec=.01),reference=sample(),active=True)
    r=m.update(current=sample(feedback_age_sec=.2),reference=sample(),active=True)
    assert m.snapshot()['feedback_age_sec']==.2
    assert m.snapshot()['status']==r['status']=='duplicate timestamp; no new sample'
    assert m.snapshot()['sample_count']==1.


def test_invalid_sample_clock_rollback_resets_before_required_field_validation():
    m=ControllerPerformanceMetrics()
    for t in [10.,10.2]:m.update(current=sample(t),reference=sample(),active=True)
    r=m.update(current={'sim_time':0.,'feedback_valid':False},reference={},active=True)
    assert not r['valid'] and r['valid_duration_sec']==0.
    assert r['status']=='clock moved backwards; window reset'
    r=m.update(current=sample(.1),reference=sample(),active=True)
    assert r['valid'] and r['coverage_fraction']==0.
    r=m.update(current=sample(.2),reference=sample(),active=True)
    assert r['coverage_fraction']==pytest.approx(.5)


def test_optional_derived_overflow_is_unavailable_not_valid_infinity():
    with np.errstate(over='ignore',invalid='ignore'):
        r=ControllerPerformanceMetrics().update(
            current=sample(body_forces=np.full(6,1e308),tool_pose=[1e308,0,0,0,0,0]),
            reference=sample(tool_pose=[-1e308,0,0,0,0,0]),active=True)
    assert r['valid'] and not r['tool_valid'] and not r['effort_valid']
    assert np.isnan(r['tool_position_error_m']) and np.isnan(r['normalized_control_effort'])
