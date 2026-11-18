"""Both DR paths consume native EKF evidence, without differentiating feedback."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from simlab.controllers.dr_oges import (
    DistributionallyRobustOgesController, decode_observation,
)
from simlab.dynamics_profiles import NominalModel
from simlab.dynamics_profiles import load_robot_dynamics_profile


@pytest.fixture(scope="module", autouse=True)
def private_controller_dependency():
    """Public installs do not require the private controller implementation."""
    global ResidualBuffer
    ResidualBuffer = pytest.importorskip(
        "namor.controls.dr_clf_oges", reason="optional private controller dependency"
    ).ResidualBuffer


def message(subsystem, step=1, generation=0):
    axes = (('force.x', 'force.y', 'force.z', 'torque.x', 'torque.y', 'torque.z')
            if subsystem == 'vehicle' else tuple('torque.' + a for a in 'edcb'))
    count = 33 if subsystem == 'vehicle' else 81
    values = {'estimator_valid': 1., 'estimator_step': float(step),
              'sim_generation': float(generation)}
    values.update({f'estimated_disturbance.{a}': i + .1 for i, a in enumerate(axes)})
    values.update({f'estimated_disturbance.variance.{a}': .02 for a in axes})
    values.update({f'estimator_nominal_parameter.{i}': float(i) for i in range(count)})
    if subsystem == 'arm':
        values.update({f'estimator_joint_locked.{axis}': 0. for axis in 'edcb'})
        values.update({f'estimator_disturbance_observable.{axis}': 1. for axis in 'edcb'})
    group = 'robot_1_IOs' if subsystem == 'vehicle' else 'robot_1__arm_IOs'
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=step, nanosec=0)),
        joint_names=[group],
        interface_values=[SimpleNamespace(interface_names=list(values), values=list(values.values()))])


def controller():
    obj = DistributionallyRobustOgesController.__new__(DistributionallyRobustOgesController)
    obj.node = Mock()
    obj.blue = SimpleNamespace(sim_p=np.arange(33))
    obj.alpha_params = SimpleNamespace(step_model_params=np.arange(81))
    obj._observations = {'vehicle': None, 'arm': None}
    obj.vehicle_residual_buffer = ResidualBuffer(6)
    obj.arm_residual_buffer = ResidualBuffer(4)
    return obj


def test_locked_arm_is_not_fresh_dr_evidence():
    msg = message('arm')
    interfaces = msg.interface_values[0]
    interfaces.values[interfaces.interface_names.index('estimator_joint_locked.c')] = 1.
    with pytest.raises(ValueError, match='locked'):
        decode_observation(msg, 'robot_1_', 'arm')


def test_joint_stop_is_not_fresh_dr_evidence():
    msg=message('arm')
    interfaces=msg.interface_values[0]
    interfaces.values[interfaces.interface_names.index('estimator_disturbance_observable.c')]=0.
    with pytest.raises(ValueError,match='joint-limit'):
        decode_observation(msg,'robot_1_','arm')


def test_simulation_default_model_matches_native_nominal_profile():
    from simlab.controllers.oges import OgesModelbasedController
    nominal = NominalModel.from_profile(load_robot_dynamics_profile('dory_alpha'))
    for cls in (OgesModelbasedController, DistributionallyRobustOgesController):
        obj = cls(Mock(), robot_prefix='robot_1_')
        np.testing.assert_array_equal(obj.alpha_params.step_model_params, nominal.arm_oges)
        np.testing.assert_array_equal(obj.blue.sim_p, nominal.vehicle_oges)
        assert obj.model_profile_sha256 == nominal.sha256


@pytest.mark.parametrize('subsystem,count', [('vehicle', 6), ('arm', 4)])
def test_native_posterior_consumption_and_rejections(subsystem, count):
    obj = controller()
    obs = decode_observation(message(subsystem), 'robot_1_', subsystem)
    obj.observe_dynamics(subsystem, obs)
    buffer = getattr(obj, subsystem + '_residual_buffer')
    np.testing.assert_array_equal(buffer.as_array()[0], np.arange(count) + .1)
    obj.observe_dynamics(subsystem, obs)  # duplicate must not count twice
    assert len(buffer) == 1
    # Native EKF runs at the hardware rate: a skipped broadcast is fine.
    newer = decode_observation(message(subsystem, 5), 'robot_1_', subsystem)
    obj.observe_dynamics(subsystem, newer)
    assert len(buffer) == 1  # >0.5 seconds gap clears historical evidence
    obj.observe_dynamics(subsystem, obs)  # delayed feedback ignored
    assert obj._observations[subsystem] is newer
    bad = decode_observation(message(subsystem, 6), 'robot_1_', subsystem)
    bad.model[0] += 1.
    obj.observe_dynamics(subsystem, bad)
    assert len(buffer) == 0
    obj.observe_dynamics(subsystem, newer)
    obj.observe_dynamics(subsystem, None)
    assert len(buffer) == 0


@pytest.mark.parametrize('subsystem', ['vehicle', 'arm'])
def test_reset_staleness_and_invalid_telemetry(subsystem):
    obj = controller()
    obj.observe_dynamics(subsystem, decode_observation(message(subsystem, 20), 'robot_1_', subsystem))
    reset = decode_observation(message(subsystem, 1, 1), 'robot_1_', subsystem)
    # Source message clock still moves forward across a simulator reset.
    reset = type(reset)(1, 1, 21., reset.period, reset.force, reset.variance, reset.model)
    obj.observe_dynamics(subsystem, reset)
    assert obj._observations[subsystem] is reset
    obj.node.get_clock().now().nanoseconds = int(22e9)
    obj._expire_disturbance(subsystem)
    assert obj._observations[subsystem] is None
    for name, value in [('estimator_valid', 0.), ('estimator_step', .5),
                        ('sim_generation', -1.)]:
        msg = message(subsystem)
        fields = msg.interface_values[0]
        fields.values[fields.interface_names.index(name)] = value
        with pytest.raises(ValueError):
            decode_observation(msg, 'robot_1_', subsystem)


@pytest.fixture(params=['robot_1_', 'robot_real_'])
def paired_oges(request):
    from simlab.controllers.oges import OgesModelbasedController
    node = Mock()
    node.get_clock().now().nanoseconds = 0
    return (OgesModelbasedController(node, robot_prefix=request.param),
            DistributionallyRobustOgesController(node, robot_prefix=request.param))


@pytest.mark.parametrize('evidence', ['empty', 'zero'])
def test_inactive_dr_matches_oges_over_moving_sequence(paired_oges, evidence):
    nominal, robust = paired_oges
    for c in paired_oges:
        c.reset_controller_state()
    rng = np.random.default_rng(421)
    for k in range(40):
        if k == 20:
            # A new segment clears residual history.
            for c in paired_oges:
                c.reset_controller_state()
        for subsystem, n in [('vehicle', 6), ('arm', 4)]:
            buffer = getattr(robust, subsystem + '_residual_buffer')
            buffer.clear()
            if evidence != 'empty':
                buffer.extend(np.zeros((60, n)), stamps=np.linspace(9.4,10.,60))
        q = np.r_[np.array([3.1, .7, .4, 2.1]) + rng.normal(0, .04, 4), .01]
        dq = rng.normal(0, .01, 5)
        qr = q + rng.normal(0, .03, 5)
        dqr = rng.normal(0, .02, 5)
        ddqr = rng.normal(0, .02, 5)
        # Large attitudes exercise the common geometric transport, not only hover.
        state = rng.normal(0, .15, 12)
        state[2] = 2.
        state[3:6] = [.6, (-1 if k % 2 else 1)*1.45, 2.7]
        target = state[:6] + rng.normal(0, .05, 6)
        vr, ar = rng.normal(0, .04, (2, 6))
        dt = [.005, .02, .05, .12][k % 4]
        np.testing.assert_array_equal(
            nominal.vehicle_controller(state, target, vr, ar, dt),
            robust.vehicle_controller(state, target, vr, ar, dt))
        np.testing.assert_array_equal(
            nominal.arm_controller(q, dq, qr, dqr, ddqr, dt),
            robust.arm_controller(q, dq, qr, dqr, ddqr, dt))
        assert robust.last_arm_dr_result.robust_margin == 0
        assert robust.last_vehicle_dr_result.robust_margin == 0


def test_shared_model_update_resets_nominal_and_dr_state(paired_oges):
    model = NominalModel.from_profile(load_robot_dynamics_profile('dory_alpha'))
    nominal, robust = paired_oges
    for subsystem, n in [('vehicle', 6), ('arm', 4)]:
        observe_native(robust, subsystem, np.ones(n), 1)
        assert len(getattr(robust, subsystem + '_residual_buffer')) == 1
    robust.last_arm_dr_result = object()
    robust.last_vehicle_dr_result = object()
    for c in paired_oges:
        c.set_model_profile(model)
        np.testing.assert_array_equal(c.blue.sim_p, model.vehicle_oges)
        np.testing.assert_array_equal(c.alpha_params.step_model_params, model.arm_oges)
    assert robust._observations == {'vehicle': None, 'arm': None}
    assert robust.last_arm_dr_result is robust.last_vehicle_dr_result is None
    assert len(robust.arm_residual_buffer) == len(robust.vehicle_residual_buffer) == 0
    # Updated models must also generate identical commands, not just zero memory.
    test_inactive_dr_matches_oges_over_moving_sequence(paired_oges, 'empty')


@pytest.mark.parametrize('dt', [0., -.01, float('nan'), float('inf')])
def test_shared_invalid_interval_rejected_before_mutation(paired_oges, dt):
    for c in paired_oges:
        c.reset_controller_state()
        with pytest.raises(ValueError, match='dt must be finite and positive'):
            c.arm_controller(np.zeros(5), np.zeros(5), np.zeros(5),
                             np.zeros(5), np.zeros(5), dt)
        with pytest.raises(ValueError, match='dt must be finite and positive'):
            c.vehicle_controller(np.zeros(12), np.zeros(6), np.zeros(6), np.zeros(6), dt)


def test_active_dr_projects_shared_command_with_physical_model(paired_oges, monkeypatch):
    nominal, robust = paired_oges
    for c in paired_oges:
        c.reset_controller_state()
    original = robust.arm_dr_projector.project
    seen = {}
    def capture(**kwargs):
        seen.update(kwargs)
        # Deliberately harmful observed load along the current CLF direction.
        robust.arm_residual_buffer.extend(np.tile(kwargs['H'] @ kwargs['A'] * 1e5, (60, 1)), stamps=np.linspace(9.4,10.,60))
        return original(**kwargs)
    monkeypatch.setattr(robust.arm_dr_projector, 'project', capture)
    q = np.array([3.1, .7, .4, 2.1, .01]); zero = np.zeros(5)
    target = q + [.03, -.04, .02, .01, 0]
    expected = nominal.arm_controller(q, zero, target, zero, zero, .02)
    actual = robust.arm_controller(q, zero, target, zero, zero, .02)
    np.testing.assert_array_equal(seen['u_nominal'], expected[:4])
    physical_bias = np.asarray(robust.Cqot_arm_vector(q[:4], zero[:4], robust.alpha_params.step_model_params)
        + robust.g_arm_vec(q[:4], robust.alpha_params.step_model_params)
        + robust.B_arm_vec(zero[:4], robust.alpha_params.step_model_params)).ravel()
    np.testing.assert_array_equal(seen['F'], physical_bias)
    assert 'nominal_compensation' not in seen and 'control_metric' not in seen
    assert robust.last_arm_dr_result.robust_margin > 0
    assert not np.array_equal(actual[:4], expected[:4])
    assert np.all(actual >= robust.arm_u_min) and np.all(actual <= robust.arm_u_max)
    assert actual[-1] == expected[-1]


def test_active_vehicle_dr_uses_shared_geometric_constraint(paired_oges, monkeypatch):
    from scipy.spatial.transform import Rotation
    nominal, robust = paired_oges
    for c in paired_oges:
        c.reset_controller_state()
    original = robust.vehicle_dr_projector.project
    seen = {}
    def capture(**kwargs):
        seen.update(kwargs)
        robust.vehicle_residual_buffer.extend(np.tile(kwargs['H'] @ kwargs['A'] * 1e5, (60, 1)), stamps=np.linspace(9.4,10.,60))
        return original(**kwargs)
    monkeypatch.setattr(robust.vehicle_dr_projector, 'project', capture)
    state = np.array([.1, -.2, 2., .7, 1.2, -2., .02, -.03, .01, -.04, .02, .01])
    target = state[:6] + [.04, -.03, .02, -.05, .06, .04]
    vel = np.array([.03, .01, -.01, .02, -.01, .03]); acc = vel*.1
    expected = nominal.vehicle_controller(state, target, vel, acc, .02)
    actual = robust.vehicle_controller(state, target, vel, acc, .02)
    np.testing.assert_array_equal(seen['u_nominal'], expected)
    R = Rotation.from_euler('xyz', state[3:6]).as_matrix()
    Rd = Rotation.from_euler('xyz', target[3:6]).as_matrix()
    J, Jd = np.kron(np.eye(2), R), np.kron(np.eye(2), Rd)
    A, b, _ = robust.uv_oges.A_b_V_func(state[:6], state[6:], target, vel, Jd,
                                      acc, robust.vehicle_weights, J, robust.vehicle_w_scale)
    np.testing.assert_allclose(seen['A'], np.asarray(A).ravel(), atol=1e-13)
    assert seen['b'] == pytest.approx(float(b))
    assert 'nominal_compensation' not in seen and 'control_metric' not in seen
    assert robust.last_vehicle_dr_result.robust_margin > 0
    assert not np.array_equal(actual, expected)
    assert np.all(actual >= robust.blue.u_min) and np.all(actual <= robust.blue.u_max)


def observe_native(controller, subsystem, force, step):
    """Feed the actual decoder a fresh posterior with the matching nominal model."""
    msg = message(subsystem, step)
    msg.header.stamp.sec = 10
    msg.header.stamp.nanosec = step * 10_000_000
    fields = msg.interface_values[0]
    model = (controller.blue.sim_p if subsystem == 'vehicle'
             else controller.alpha_params.step_model_params)
    for i, name in enumerate(fields.interface_names):
        if name.startswith('estimator_nominal_parameter.'):
            fields.values[i] = float(model[int(name.rsplit('.', 1)[1])])
        elif name.startswith('estimated_disturbance.') and '.variance.' not in name:
            axes = (('force.x', 'force.y', 'force.z', 'torque.x', 'torque.y', 'torque.z')
                    if subsystem == 'vehicle' else tuple('torque.' + a for a in 'edcb'))
            fields.values[i] = float(force[axes.index(name[len('estimated_disturbance.'):])])
    observation = decode_observation(msg, 'robot_1_', subsystem)
    controller.node.get_clock().now().nanoseconds = 10_000_000_000 + step*10_000_000
    controller.observe_dynamics(subsystem, observation)
    return observation


def control_case(controller, subsystem):
    if subsystem == 'arm':
        q = np.array([3.1, .7, .4, 2.1, .01]); z = np.zeros(5)
        return controller.arm_controller(q, z, q + [.03, -.04, .02, .01, 0], z, z, .02)
    x = np.array([.1, -.2, 2., .7, 1.2, -2., .02, -.03, .01, -.04, .02, .01])
    return controller.vehicle_controller(x, x[:6] + [.04, -.03, .02, -.05, .06, .04],
                                         np.zeros(6), np.zeros(6), .02)


@pytest.mark.parametrize('subsystem', ['arm', 'vehicle'])
@pytest.mark.parametrize('harmful', [False, True], ids=['beneficial', 'harmful'])
def test_native_evidence_threshold_and_active_expiry(paired_oges, monkeypatch, subsystem, harmful):
    nominal, robust = paired_oges
    projector = getattr(robust, subsystem + '_dr_projector')
    original = projector.project
    seen = {}
    def capture(**kwargs):
        seen.update(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(projector, 'project', capture)
    control_case(robust, subsystem)
    A, H = seen['A'], seen['H']
    assert A @ A > 0
    # Constant signed CLF effect, not zero evidence: the elapsed-time gate matters.
    force = (1 if harmful else -1) * (H @ A) * (1000 / (A @ A))
    for c in paired_oges:
        c.reset_controller_state()
    for step in range(1, 51):
        observe_native(robust, subsystem, force, step)
    expected = control_case(nominal, subsystem)
    np.testing.assert_array_equal(control_case(robust, subsystem), expected)
    result = getattr(robust, 'last_' + subsystem + '_dr_result')
    assert result.residual_sample_count == 50 and not result.robust_active
    observe_native(robust, subsystem, force, 51)
    expected = control_case(nominal, subsystem)
    actual = control_case(robust, subsystem)
    result = getattr(robust, 'last_' + subsystem + '_dr_result')
    assert result.residual_sample_count == 51
    assert result.robust_active == harmful
    if harmful:
        assert result.residual_effect_mean > projector.effect_deadband
        assert not np.array_equal(actual, expected)
        assert result.clf_residual <= result.slack + 1e-8
    else:
        assert result.residual_effect_mean < 0
        np.testing.assert_array_equal(actual, expected)
    robust.node.get_clock().now().nanoseconds += 510_000_000
    np.testing.assert_array_equal(control_case(robust, subsystem), control_case(nominal, subsystem))
    assert robust._observations[subsystem] is None
    assert len(getattr(robust, subsystem + '_residual_buffer')) == 0
    assert not getattr(robust, 'last_' + subsystem + '_dr_result').robust_active
    observe_native(robust, subsystem, force, 102)
    np.testing.assert_array_equal(control_case(robust, subsystem), control_case(nominal, subsystem))
    assert getattr(robust, 'last_' + subsystem + '_dr_result').residual_sample_count == 1
