"""Both DR paths consume native EKF evidence, without differentiating feedback."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from simlab.controllers.dr_oges.controller import DistributionallyRobustOgesController
from simlab.controllers.dr_oges.feedback import decode_observation
from simlab.controllers.model_profile import NominalModel
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
    obj._diagnostics = Mock()
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


def test_simulation_default_model_matches_native_nominal_profile(monkeypatch):
    def initialize(obj, node, arm_dof, robot_prefix):
        obj.node, obj.arm_dof, obj.robot_prefix = node, arm_dof, robot_prefix
        obj.blue = SimpleNamespace(sim_p=())
        obj.alpha_params = SimpleNamespace(step_model_params=())
    monkeypatch.setattr(
        'simlab.controllers.dr_oges.controller.OgesModelbasedController.__init__', initialize)
    obj = DistributionallyRobustOgesController(Mock(), robot_prefix='robot_1_')
    nominal = NominalModel.from_profile(load_robot_dynamics_profile('dory_alpha'))
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
