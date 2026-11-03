"""Controller model configuration must be explicit and independent of the plant."""
from types import SimpleNamespace
from unittest.mock import Mock
import json
import numpy as np
import pytest
from simlab.controllers.pid import LowLevelPidController
from simlab.controllers.invDyn import LowLevelInvDynController


@pytest.mark.parametrize('controller_type', [LowLevelPidController, LowLevelInvDynController])
def test_vehicle_integral_accumulates_and_reset_repeats_initial_command(controller_type):
    controller = controller_type(Mock(), robot_prefix='robot_1_')
    controller.kp = np.ones(6)
    controller.kd = np.ones(6)
    controller.ki = np.ones(6)
    state = np.zeros(12)
    target = np.array([.01, 0., 0., 0., 0., 0.])
    args = (state, target, np.zeros(6), np.zeros(6), .01)
    first = controller.vehicle_controller(*args)
    first_integral = controller.vehicle_pid_i_buffer.copy()
    second = controller.vehicle_controller(*args)
    assert np.linalg.norm(controller.vehicle_pid_i_buffer) > np.linalg.norm(first_integral)
    assert not np.array_equal(first, second)
    controller.reset_controller_state()
    np.testing.assert_allclose(controller.vehicle_controller(*args), first)


def test_oges_reset_clears_only_command_memory():
    from simlab.controllers.oges import OgesModelbasedController
    controller = OgesModelbasedController.__new__(OgesModelbasedController)
    controller.arm_dof = 4
    controller.vehicle_u_prev = np.ones(6)
    controller.arm_u_prev = np.ones(4)
    controller.vehicle_weights = np.arange(6.)
    controller.reset_controller_state()
    np.testing.assert_array_equal(controller.vehicle_u_prev, np.zeros(6))
    np.testing.assert_array_equal(controller.arm_u_prev, np.zeros(4))
    np.testing.assert_array_equal(controller.vehicle_weights, np.arange(6.))

