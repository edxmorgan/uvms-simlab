"""Hardware guard uses native feedback, independently of identification exports."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from simlab.robot import Robot

@pytest.mark.parametrize('sampling', [False, True])
@pytest.mark.parametrize('valid', [0., 1.])
def test_hardware_validity_without_optional_sampling(sampling, valid):
    robot = Robot.__new__(Robot)
    robot.prefix = 'robot_real_'
    robot.node = Mock()
    robot.node.get_clock().now.return_value.nanoseconds = 123
    robot.update_state = Mock()
    names = ['estimator_valid'] + (['identification.valid'] if sampling else [])
    msg = NS(joint_names=['robot_real_IOs'], interface_values=[NS(interface_names=names, values=[valid, 1.])])
    robot.listener_callback(msg)
    assert robot._hardware_estimator_valid is (valid == 1.)
    assert robot._hardware_estimator_received == 123
    robot.update_state.assert_called_once_with(msg)

def test_simulated_step_does_not_enable_hardware_guard():
    robot = Robot.__new__(Robot)
    robot.prefix = 'robot_1_'
    robot.update_state = Mock()
    robot.listener_callback(NS(joint_names=['robot_1_IOs'], interface_values=[
        NS(interface_names=['estimator_valid', 'sim_step'], values=[0., 1.])]))
    assert not hasattr(robot, '_hardware_estimator_valid')
