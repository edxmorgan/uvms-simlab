"""Controller model configuration must be explicit and independent of the plant."""
from types import SimpleNamespace
from unittest.mock import Mock
import json
import numpy as np
import pytest
from simlab.controller_configuration import controller_configuration
from simlab.controllers.pid import LowLevelPidController
from simlab.controllers.invDyn import LowLevelInvDynController


@pytest.mark.parametrize('prefix,status,hold,replay_pending', [
    ('robot_real_', 'inactive', False, False),
    ('robot_1_', 'active', False, False),
    ('robot_1_', 'inactive', False, True),
])
def test_model_changes_rejected_outside_allowed_state(prefix, status, hold, replay_pending):
    controller = Mock()
    robot = SimpleNamespace(prefix=prefix, status=status, sim_reset_hold=hold,
        _replay_reset_pending=replay_pending, controller_instance=lambda name: controller,
        _controller_is_replay=lambda _: False)
    request = SimpleNamespace(controller_name='PID', operation='set_model', model_profile='unused')
    with pytest.raises(ValueError):
        controller_configuration(robot, request)
    controller.set_model_profile.assert_not_called()


def test_get_is_read_only_and_nonfinite_snapshot_rejected():
    controller = Mock()
    controller.configuration_snapshot.return_value = {'parameters': {'gain': 1.}}
    robot = SimpleNamespace(controller_instance=lambda _: controller, _controller_is_replay=lambda _: False)
    request = SimpleNamespace(controller_name='PID', operation='get')
    assert json.loads(controller_configuration(robot, request))['parameters']['gain'] == 1.
    controller.set_model_profile.assert_not_called()
    controller.configuration_snapshot.return_value = {'gain': float('nan')}
    with pytest.raises(ValueError):
        controller_configuration(robot, request)
