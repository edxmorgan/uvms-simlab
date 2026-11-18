from pathlib import Path
from unittest.mock import Mock
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from simlab.motion_planning.kinematics.forward import tool_pose_from_sample
from simlab.motion_planning.kinematics.forward import load_forward_kinematics
from simlab.uvms_parameters import ReachParams


@pytest.fixture
def fk():
    return load_forward_kinematics(Path(__file__).parents[1] / 'resource')


def tool(sample, fk):
    return tool_pose_from_sample(sample, fk, ReachParams.base_T0_new, ReachParams.tipOffset)


def test_vehicle_translation_maps_ned_to_nwu_without_changing_tool_rotation(fk):
    sample = dict(pose=np.array([1., 2., 3., .1, -.2, .4]), q=np.array([3.1, .7, .4, 2.1]))
    a = tool(sample, fk)
    moved = sample['pose'].copy()
    moved[:3] += [2., 3., 4.]
    b = tool(sample | {'pose': moved}, fk)
    np.testing.assert_allclose(b[:3] - a[:3], [2., -3., -4.], atol=1e-12)
    assert (Rotation.from_euler('xyz', a[3:]).inv() * Rotation.from_euler('xyz', b[3:])).magnitude() < 1e-12


def test_tool_detects_arm_motion_with_stationary_vehicle(fk):
    sample = dict(pose=np.zeros(6), q=np.array([3.1, .7, .4, 2.1]))
    a = tool(sample, fk)
    b = tool(sample | {'q': sample['q'] + [0., .1, 0., 0.]}, fk)
    assert np.linalg.norm(a[:3] - b[:3]) > .001


@pytest.mark.parametrize('sample', [dict(pose=[0]*5, q=[0]*4), dict(pose=[0]*6, q=[0]*3), dict(pose=[float('nan')]*6, q=[0]*4)])
def test_invalid_samples_do_not_call_fk(sample):
    fk = Mock()
    with pytest.raises(ValueError):
        tool(sample, fk)
    fk.assert_not_called()
