from unittest.mock import Mock

import pytest
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float64MultiArray
from simlab.uvms_backend import UVMSBackendCore


def backend(monkeypatch):
    core = UVMSBackendCore.__new__(UVMSBackendCore)
    core.node = Mock()
    core.node.get_clock().now().to_msg.return_value = WrenchStamped().header.stamp
    core._demo_wrench_frame = 'robot_1_base_link'
    core._demo_wrench_command = None
    core._demo_wrench_pub = Mock()
    core._demo_arm_command = None
    core._demo_arm_pub = Mock()
    monkeypatch.setattr('simlab.uvms_backend.time.monotonic', lambda: 10.0)
    return core


def output(core):
    core._publish_demo_wrench()
    return core._demo_wrench_pub.publish.call_args.args[0]


def test_zero_injection_and_expiry(monkeypatch):
    core = backend(monkeypatch)
    assert output(core).wrench.force.x == 0.0
    msg = WrenchStamped()
    msg.wrench.force.x = 1.0
    msg.wrench.torque.z = 0.2
    core._receive_demo_wrench(msg)
    msg.wrench.force.x = 99.0
    assert output(core).wrench.force.x == 1.0
    assert output(core).wrench.torque.z == 0.2
    assert output(core).header.frame_id == 'robot_1_base_link'
    monkeypatch.setattr('simlab.uvms_backend.time.monotonic', lambda: 11.0)
    assert output(core).wrench.force.x == 0.0
    assert output(core).wrench.torque.z == 0.0


@pytest.mark.parametrize('frame,value', [('world', 1.0), ('', float('nan')),
                                          ('', float('inf'))])
def test_invalid_input_rejected(monkeypatch, frame, value):
    core = backend(monkeypatch)
    msg = WrenchStamped()
    msg.header.frame_id = frame
    msg.wrench.force.x = value
    core._receive_demo_wrench(msg)
    assert core._demo_wrench_command is None
    assert output(core).wrench.force.x == 0.0


def test_wholebody_mapping_copy_and_expiry(monkeypatch):
    core = backend(monkeypatch)
    msg = Float64MultiArray(data=[float(i) for i in range(10)])
    core._receive_demo_wholebody(msg)
    msg.data[6] = 99.
    result = output(core)
    assert result.wrench.force.y == 1.
    assert result.wrench.torque.z == 5.
    assert list(core._demo_arm_pub.publish.call_args.args[0].data) == [6., 7., 8., 9.]
    monkeypatch.setattr('simlab.uvms_backend.time.monotonic', lambda: 11.)
    assert output(core).wrench.torque.z == 0.
    assert list(core._demo_arm_pub.publish.call_args.args[0].data) == [0.] * 4


@pytest.mark.parametrize('values', [[0.] * 9, [0.] * 11, [float('nan')] * 10])
def test_bad_wholebody_command(monkeypatch, values):
    core = backend(monkeypatch)
    core._receive_demo_wholebody(Float64MultiArray(data=values))
    assert core._demo_wrench_command is None
    assert core._demo_arm_command is None


@pytest.mark.parametrize('hardware,prefixes', [(True, ['robot_1_']),
                                              (False, ['robot_1_', 'robot_2_'])])
def test_disabled_for_hardware_and_multi_robot(hardware, prefixes):
    core = UVMSBackendCore.__new__(UVMSBackendCore)
    core.node = Mock()
    core.use_vehicle_hardware = hardware
    core.robots_prefix = prefixes
    core._setup_demo_wrench()
    core.node.create_publisher.assert_not_called()
