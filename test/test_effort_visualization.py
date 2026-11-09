import math
from unittest.mock import Mock

import pytest
from builtin_interfaces.msg import Time
from control_msgs.msg import DynamicJointState, InterfaceValue
from simlab.effort_visualization import (
    AXES, EffortVisualization, arm_estimate, joint_geometry, joint_wrench, vehicle_wrench,
)


def description(prefix='robot_1_'):
    return '<robot name="test">' + ''.join(
        f'<joint name="{prefix}_axis_{a}" type="{kind}"><child link="{prefix}{a}"/>'
        f'<axis xyz="{v}"/></joint>' for a, kind, v in zip(AXES,
        ['revolute']*4+['prismatic'], ['0 0 1', '0 1 0', '0 1 0', '0 0 -1', '0 0 1'])) + '</robot>'


def test_urdf_axis_sign_and_force_torque_units():
    geometry = joint_geometry(description(), 'robot_1_')
    assert joint_wrench(0.2, *geometry['b'][1:]) == [0., 0., 0., 0., 0., -0.2]
    assert joint_wrench(-0.2, *geometry['b'][1:]) == [0., 0., 0., 0., 0., 0.2]
    assert joint_wrench(2., *geometry['a'][1:]) == [0., 0., 2., 0., 0., 0.]
    assert geometry['d'][0] == 'robot_1_d'
    with pytest.raises(ValueError):
        joint_geometry(description(), 'wrong_')


def test_vehicle_proper_rotation_preserves_norm_and_is_involutory():
    assert vehicle_wrench([1., 2., 3., 4., 5., 6.]) == [1., -2., -3., 4., -5., -6.]
    assert vehicle_wrench(vehicle_wrench([1., 2., 3., 4., 5., 6.])) == [1., 2., 3., 4., 5., 6.]


@pytest.mark.parametrize('flag,status', [('estimator_joint_locked.b', 'locked'),
    ('estimator_joint_at_limit.b', 'joint stop')])
def test_locked_or_contact_not_shown_as_valid_estimate(flag, status):
    row = {'estimator_valid': 1., 'estimator_disturbance_observable.b': 1.,
           'estimated_disturbance.torque.b': .2, flag: 1.}
    value, reason = arm_estimate(row, 'b')
    assert value is None and status in reason


def test_missing_or_nonfinite_estimates_not_zero():
    assert arm_estimate({}, 'e')[0] is None
    row = {'estimator_valid': 1., 'estimator_disturbance_observable.e': 1.,
           'estimated_disturbance.torque.e': math.nan}
    assert arm_estimate(row, 'e')[0] is None
    row['estimated_disturbance.torque.e'] = -.04
    assert arm_estimate(row, 'e') == (-.04, '')
    assert arm_estimate(row, 'a') == (None, 'not estimated')


def test_ros_messages_frames_timestamp_clear_stale_and_multi_robot(monkeypatch):
    # Exercise the real receive/render code with ROS message types, no robot commands.
    node = Mock()
    node.prefixes = ['robot_1_', 'robot_2_']
    node.geometry = {p: joint_geometry(description(p), p) for p in node.prefixes}
    node.publishers_by_key = {(p, part, source): Mock() for p in node.prefixes
                             for part in ('vehicle', *AXES)
                             for source in ('command', 'estimated_disturbance')}
    node.marker_publishers = {key: Mock() for key in node.publishers_by_key}
    node.get_parameter.return_value.value = .5
    node.get_clock.return_value.now.return_value.to_msg.return_value = Time(sec=124)
    node.emit = lambda *args: EffortVisualization.emit(node, *args)
    msg = DynamicJointState()
    msg.header.stamp = Time(sec=123)
    msg.joint_names = ['robot_1__axis_b', 'robot_1__arm_IOs']
    msg.interface_values = [InterfaceValue(interface_names=['effort', 'computed_effort'], values=[.2, 9.0]),
                           InterfaceValue(interface_names=['estimator_valid',
                           'estimator_disturbance_observable.b', 'estimated_disturbance.torque.b'],
                           values=[1., 1., -.04])]
    msg.joint_names.append('robot_1_IOs')
    components = ['force.x', 'force.y', 'force.z', 'torque.x', 'torque.y', 'torque.z']
    msg.interface_values.append(InterfaceValue(
        interface_names=components + ['estimated_disturbance.' + k for k in components] + ['estimator_valid'],
        values=[1., -2., 3., -4., 5., -6., .1, -.2, .3, -.4, .5, -.6, 1.]))
    monkeypatch.setattr('simlab.effort_visualization.time.monotonic', lambda: 10.)
    EffortVisualization.receive(node, msg)
    EffortVisualization.publish(node)
    overlay = node.overlay.publish.call_args.args[0].text
    assert overlay.startswith('DOF')
    assert f'{"Sway":<7}{"N":<6}{"-2.000":>13}  {"-0.200":>27}' in overlay
    assert f'{"Yaw":<7}{"Nm":<6}{"-6.000":>13}  {"-0.600":>27}' in overlay
    output = node.publishers_by_key['robot_1_', 'b', 'command'].publish.call_args.args[0]
    assert output.header.frame_id == 'robot_1_b'
    assert output.header.stamp.sec == 123
    assert output.wrench.torque.z == -.2
    output = node.publishers_by_key['robot_1_', 'b', 'estimated_disturbance'].publish.call_args.args[0]
    assert output.wrench.torque.z == .04
    output = node.publishers_by_key['robot_2_', 'b', 'command'].publish.call_args.args[0]
    assert math.isnan(output.wrench.torque.z)
    monkeypatch.setattr('simlab.effort_visualization.time.monotonic', lambda: 11.)
    EffortVisualization.publish(node)
    output = node.publishers_by_key['robot_1_', 'b', 'command'].publish.call_args.args[0]
    assert math.isnan(output.wrench.torque.z)
    assert output.header.stamp.sec == 124
    assert all('STALE' in m.text for m in node.labels.publish.call_args.args[0].markers)
    overlay = node.overlay.publish.call_args.args[0].text
    assert '-6.000' not in overlay
    assert f'{"Yaw":<7}{"Nm":<6}{"unavailable":>13}  {"Unavailable":>27}' in overlay


@pytest.mark.parametrize('axis', [(1.,0.,0.), (0.,1.,0.), (0.,0.,-1.)])
@pytest.mark.parametrize('value', [-100., -.02, .00001, .02, 100.])
def test_linear_arm_arc_thickness_and_right_hand_direction(axis, value):
    from simlab.effort_visualization import arm_effort_markers
    from visualization_msgs.msg import Marker
    arc, head = arm_effort_markers('joint', Time(sec=7), 'revolute', axis, value, 'command')
    assert arc.action == head.action == Marker.ADD
    assert arc.scale.x == .012
    for point in arc.points:
        xyz=(point.x,point.y,point.z)
        axial=sum(a*b for a,b in zip(xyz,axis))
        radius=math.sqrt(sum((xyz[i]-axial*axis[i])**2 for i in range(3)))
        assert radius == pytest.approx(.25*abs(value), abs=1e-12)
    a,b=arc.points[:2]
    # Remove centre offset before checking angular direction around the URDF axis.
    a=[getattr(a,k)+.025*axis[i] for i,k in enumerate(('x','y','z'))]
    b=[getattr(b,k)+.025*axis[i] for i,k in enumerate(('x','y','z'))]
    cross=(a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    assert sum(x*y for x,y in zip(cross,axis))*value > 0
    assert arc.lifetime.sec == 1
    assert arc.frame_locked and head.frame_locked
    assert arc.header.stamp == Time()
    assert head.header.stamp == Time()


def test_linear_arm_missing_zero_and_gripper():
    from simlab.effort_visualization import arm_effort_markers
    from visualization_msgs.msg import Marker
    for value in (None, math.nan, math.inf, 0.):
        assert all(m.action==Marker.DELETE for m in arm_effort_markers('joint',Time(),'revolute',(0.,0.,1.),value,'command'))
    arrow, unused = arm_effort_markers('joint',Time(),'prismatic',(0.,0.,1.),-100.,'command')
    assert arrow.points[-1].z == -10.
    assert unused.action == Marker.DELETE
