import pytest
from simlab.utils.joint_limits import position_endpoints


def description(lo='.0004', hi='.0137', prefix='robot_1_'):
    return f'''<robot name="test">
    <joint name="{prefix}_axis_a" type="prismatic"><limit lower="0" upper=".015"/></joint>
    <ros2_control name="arm" type="system"><joint name="{prefix}_axis_a">
    <param name="has_position_limits">1</param>
    <param name="min_position">{lo}</param><param name="max_position">{hi}</param>
    </joint></ros2_control></robot>'''


def test_effective_endpoints_and_zero_proportional_error():
    closed, opened = position_endpoints(description(), 'robot_1__axis_a')
    assert (closed, opened) == (.0004, .0137)
    for target, stop in [(closed, .0004), (opened, .0137)]:
        assert 1000 * (target - stop) == 0


def test_limits_follow_description_and_robot_prefix():
    assert position_endpoints(description('.001', '.012', 'robot_real_'),
                              'robot_real__axis_a') == (.001, .012)
    assert position_endpoints(description('-.1', '.1'), 'robot_1__axis_a') == (0, .015)


@pytest.mark.parametrize('lo,hi', [('.02','.03'), ('nan','.014'), ('.01','.001')])
def test_invalid_limits_rejected(lo, hi):
    with pytest.raises(ValueError):
        position_endpoints(description(lo, hi), 'robot_1__axis_a')
