"""Waveform and whole-body routing contract for the disturbance injection node."""
import math
from pathlib import Path

import pytest

from simlab.disturbance_waveforms import load_vector, parse_args, waveform_value


def test_combined_axes_phase_delay_and_full_duration():
    args = parse_args(['--dof', 'surge', 'yaw', '--amplitude', '3', '.1',
                       '--frequency', '.1', '--cycles', '3',
                       '--phase', '0', '-90', '--delay', '0', '2'])
    values = load_vector(args.channels, 2., .1, 30.)
    assert values[0] == pytest.approx(3 * math.sin(.4 * math.pi))
    assert values[5] == pytest.approx(-.1)
    assert all(values[i] == 0. for i in range(10) if i not in (0, 5))
    assert load_vector(args.channels, 1., .1, 30.)[5] == 0.
    assert load_vector(args.channels, 31., .1, 30.)[0] == 0.
    assert load_vector(args.channels, 31., .1, 30.)[5] != 0.
    assert load_vector(args.channels, 32., .1, 30.) == [0.] * 10


@pytest.mark.parametrize('kind', ['sine', 'step', 'square', 'triangle', 'chirp', 'gaussian'])
def test_waveform_gates(kind):
    assert waveform_value(kind, -.01, .1, 30., .2) == 0.
    assert waveform_value(kind, 30., .1, 30., .2) == 0.


def test_chirp_integrates_frequency():
    assert waveform_value('chirp', 5., .1, 20., .3) == pytest.approx(-math.sqrt(.5))


@pytest.mark.parametrize('argv', [
    ['--dof', 'surge', 'yaw', '--amplitude', '3'],
    ['--delay', '-1'], ['--phase', 'nan'],
    ['--waveform', 'step', '--phase', '90'],
    ['--waveform', 'gaussian', '--phase', '90'],
    ['--waveform', 'gaussian', '--amplitude', '-1'],
    ['--seed', '-1'], ['--seed', '4294967296'],
    ['--frequency', '5'], ['--dof', 'surge', 'surge'],
])
def test_invalid_configuration_rejected_before_ros(argv):
    with pytest.raises(SystemExit) as error:
        parse_args(argv)
    assert error.value.code == 2


def test_backend_does_not_forward_injection():
    source = (Path(__file__).parents[1]/'simlab'/'uvms_backend.py').read_text()
    assert '/disturbance_injection/wholebody' not in source
    assert '_setup_demo_wrench' not in source


@pytest.mark.parametrize('changes', [
    {'dofs': []}, {'dofs': ['not_a_dof']}, {'waveforms': ['not_a_waveform']},
    {'amplitudes': [float('nan')]}, {'phases_deg': []}, {'delays_s': [-1.]},
])
def test_action_validation_has_no_cli_side_effects(changes, capsys):
    from simlab.action import RunDisturbance
    from simlab.disturbance_injection_node import goal_options
    values = dict(dofs=['surge'], amplitudes=[3.], waveforms=['sine'],
                  phases_deg=[0.], delays_s=[0.], frequency_hz=.1, cycles=3)
    values.update(changes)
    with pytest.raises(ValueError):
        goal_options(RunDisturbance.Goal(**values))
    assert capsys.readouterr().err == ''


def test_cli_and_action_use_identical_waveform_validation():
    from simlab.action import RunDisturbance
    from simlab.disturbance_injection_node import goal_options
    cli = parse_args(['--dof', 'surge', 'yaw', '--amplitude', '3', '.1',
                      '--waveform', 'sine', 'triangle', '--phase', '0', '-90',
                      '--delay', '0', '2', '--frequency', '.1', '--cycles', '3'])
    action = goal_options(RunDisturbance.Goal(
        dofs=['surge', 'yaw'], amplitudes=[3., .1], waveforms=['sine', 'triangle'],
        phases_deg=[0., -90.], delays_s=[0., 2.], frequency_hz=.1, cycles=3))
    assert [vars(c) for c in cli.channels] == [vars(c) for c in action.channels]
    assert cli.period == action.period
    assert cli.max_delay == action.max_delay


@pytest.mark.parametrize('age', [float('nan'), float('inf'), -.1, .5])
def test_invalid_receipt_age_never_marks_estimate_valid(age):
    from simlab.disturbance_plot import estimate_status
    value, status = estimate_status(
        {'estimator_valid': 1., 'estimated_disturbance.force.x': 2.}, 0, 'force.x', age)
    assert math.isnan(value)
    assert status == 'estimate stale/unavailable'


def test_gaussian_samples_are_repeatable_held_and_axis_independent():
    args = parse_args(['--dof', 'surge', 'yaw', '--amplitude', '3', '.1',
                       '--waveform', 'gaussian', '--seed', '42'])
    def sample(t, seed=42, channels=args.channels):
        return load_vector(channels, t, .1, 30., seed=seed)
    assert sample(.101) == sample(.149)  # Same simulation-time bin, including pause.
    assert sample(.101) != sample(.151)
    assert sample(.101) != sample(.101, seed=43)
    assert sample(.101) == sample(.101, channels=list(reversed(args.channels)))
    assert sample(.101)[0] / 3 != sample(.101)[5] / .1
    sample(20.)  # Skipped bins or other calls cannot change an earlier sample.
    assert sample(.101) == sample(.149)
    assert sample(-1.) == sample(30.) == [0.] * 10
    assert all(sample(.101)[i] == 0. for i in range(10) if i not in (0, 5))
    args.channels[1].delay_s = 2.
    assert sample(1.)[5] == 0.
    assert sample(32.) == [0.] * 10


def test_gaussian_mean_and_standard_deviation_semantics():
    import numpy as np
    args = parse_args(['--waveform', 'gaussian', '--amplitude', '3'])
    samples = np.array([load_vector(args.channels, (i+.5)/20., .1, 300., seed=42)[0]
                        for i in range(4096)])
    assert abs(samples.mean()) < .15
    assert samples.std() == pytest.approx(3., rel=.05)
    assert np.max(np.abs(samples)) > 6.  # Standard deviation is not a clipping bound.


def test_gaussian_action_preserves_seed_and_matches_cli():
    from simlab.action import RunDisturbance
    from simlab.disturbance_injection_node import goal_options
    cli = parse_args(['--waveform', 'gaussian', '--amplitude', '3', '--seed', '42',
                      '--frequency', '.1', '--cycles', '3'])
    action = goal_options(RunDisturbance.Goal(
        dofs=['surge'], amplitudes=[3.], waveforms=['gaussian'],
        phases_deg=[0.], delays_s=[0.], frequency_hz=.1, cycles=3, seed=42))
    assert action.seed == cli.seed == 42
    assert action.period * action.cycles == 30.
    assert load_vector(action.channels, .123, action.frequency, 30., seed=action.seed) == \
        load_vector(cli.channels, .123, cli.frequency, 30., seed=cli.seed)


@pytest.mark.parametrize('key', ['force.x', 'torque.z', 'torque.d'])
def test_uncertainty_uses_published_variance(key):
    from simlab.disturbance_plot import estimate_sigma
    assert estimate_sigma({'estimated_disturbance.variance.'+key: .25}, key, 1.) == .5
    assert estimate_sigma({'estimated_disturbance.variance.'+key: 0.}, key, 1.) == 0.
    for variance in [-1., math.nan, math.inf]:
        assert math.isnan(estimate_sigma({'estimated_disturbance.variance.'+key: variance}, key, 1.))
    assert math.isnan(estimate_sigma({}, key, 1.))
    assert math.isnan(estimate_sigma({'estimated_disturbance.variance.'+key: .25}, key, math.nan))


def test_uncertainty_band_preserves_missing_data_gaps():
    import numpy as np
    from simlab.disturbance_plot import uncertainty_polygons
    bands = uncertainty_polygons([0., 1., 2., 3., 4.], [1.] * 5,
                                 [.5, .5, math.nan, .25, .25])
    assert len(bands) == 2
    np.testing.assert_array_equal(bands[0], [[0., 0.], [1., 0.], [1., 2.], [0., 2.]])
    np.testing.assert_array_equal(bands[1], [[3., .5], [4., .5], [4., 1.5], [3., 1.5]])
    assert uncertainty_polygons([0.], [math.nan], [.5]) == []


def test_external_waveform_discovery_and_shared_gates(monkeypatch):
    from types import SimpleNamespace
    import simlab.disturbances as registry
    from simlab.disturbances.base import DisturbanceWaveform

    class UserForm(DisturbanceWaveform):
        registry_name = 'user_form'

        @staticmethod
        def sample(c):
            return c.elapsed / c.duration

    registry._registry.cache_clear()
    monkeypatch.setattr(registry, 'entry_points', lambda **kw: [
        SimpleNamespace(name='user_form', load=lambda: UserForm)])
    try:
        args = parse_args(['--waveform', 'user_form', '--amplitude', '2'])
        assert load_vector(args.channels, 1., .1, 4.)[0] == .5
        assert load_vector(args.channels, 4., .1, 4.) == [0.] * 10
        monkeypatch.setattr(UserForm, 'sample', lambda c: math.nan)
        with pytest.raises(ValueError, match='nonfinite'):
            load_vector(args.channels, 1., .1, 4.)
    finally:
        registry._registry.cache_clear()


def test_local_registry_rejects_duplicate_builtin(monkeypatch):
    import simlab.disturbances as registry
    from simlab.disturbances.builtin import Sine
    registry._registry.cache_clear()
    monkeypatch.setattr(registry, 'LOCAL_WAVEFORMS', (Sine,))
    try:
        with pytest.raises(ValueError, match='Duplicate'):
            registry.waveform_names()
    finally:
        registry._registry.cache_clear()


def test_waveform_identity_and_display_metadata():
    from simlab.disturbances import waveform_identity, waveform_class
    args = parse_args(['--waveform', 'gaussian'])
    cls = waveform_class('gaussian')
    assert 'σ' in cls.plot_title(args.channels[0], args)
    assert cls.plot_style == 'steps-post'
    identity = waveform_identity('gaussian')
    assert len(identity['source_sha256']) == 64
    assert identity['metadata']['sample_rate_hz'] == 20.
    assert waveform_class('sine').plot_style is None


def test_correlated_gaussian_stationary_covariance_and_channels():
    import numpy as np
    # Ensemble test crosses dyadic anchor boundaries and bridge samples.
    times = [0., .05, .95, 1., 1.55, 1.6, 1.65, 2.]
    samples = np.array([[waveform_value('correlated_gaussian', t, .1, 30.,
                         seed=seed, correlation_time=1.) for t in times]
                        for seed in range(3000)])
    expected = np.exp(-np.abs(np.subtract.outer(times, times)))
    np.testing.assert_allclose(samples.mean(axis=0), 0., atol=.065)
    np.testing.assert_allclose(np.cov(samples, rowvar=False), expected, atol=.085)
    other = [waveform_value('correlated_gaussian', 1., .1, 30., seed=seed,
                           channel_index=1, correlation_time=1.) for seed in range(3000)]
    assert abs(np.corrcoef(samples[:, 3], other)[0, 1]) < .065


def test_correlated_gaussian_random_access_and_routing():
    from simlab.disturbances.builtin import _correlated_sample
    a = parse_args(['--dof', 'surge', 'arm_d', '--waveform', 'correlated_gaussian',
                    '--amplitude', '2', '.1', '--correlation-time', '3', '--seed', '9'])
    times = [0., .05, .7, 2., 30., 999.]
    expected = [waveform_value('correlated_gaussian', t, .1, 2000., seed=9,
                              correlation_time=3.) for t in times]
    _correlated_sample.cache_clear()
    actual = [waveform_value('correlated_gaussian', t, .1, 2000., seed=9,
                            correlation_time=3.) for t in reversed(times)]
    assert actual == expected[::-1]
    values = load_vector(a.channels, .7, a.frequency, 30., seed=a.seed,
                         correlation_time=a.correlation_time)
    assert values[0] == 2*expected[2]
    assert all(values[i] == 0. for i in range(10) if i not in (0, 7))
    assert waveform_value('correlated_gaussian', -.1, .1, 30.) == 0.
    assert waveform_value('correlated_gaussian', 30., .1, 30.) == 0.


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf'])
def test_invalid_correlation_time(value):
    with pytest.raises(SystemExit):
        parse_args(['--waveform', 'correlated_gaussian', '--correlation-time', value])


def test_correlated_gaussian_action_configuration():
    from simlab.action import RunDisturbance
    from simlab.disturbance_injection_node import goal_options
    goal = RunDisturbance.Goal(dofs=['surge'], amplitudes=[2.],
        waveforms=['correlated_gaussian'], phases_deg=[0.], delays_s=[0.],
        frequency_hz=.1, cycles=3, seed=17, correlation_time_s=4.)
    from rclpy.serialization import serialize_message, deserialize_message
    goal = deserialize_message(serialize_message(goal), RunDisturbance.Goal)
    options = goal_options(goal)
    assert options.correlation_time == 4.
    assert options.channels[0].amplitude == 2.
    goal.correlation_time_s = -1.
    with pytest.raises(ValueError):
        goal_options(goal)
