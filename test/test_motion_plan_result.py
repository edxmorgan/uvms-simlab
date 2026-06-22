import numpy as np
import pytest

from simlab.motion_planning.result import MotionPlanKind, MotionPlanResult


def test_path_result_action_payload_normalizes_arrays():
    result = MotionPlanResult(
        is_success=True,
        xyz=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
        quat_wxyz=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        path_length_cost=2.0,
        geom_length=3.0,
        message="ok",
    )

    payload = result.as_action_payload()

    assert payload["kind"] == MotionPlanKind.PATH
    assert payload["count"] == 2
    assert payload["is_success"] is True
    assert payload["xyz"].shape == (2, 3)
    assert payload["quat_wxyz"].shape == (2, 4)
    assert result.needs_trajectory_generator


def test_timed_trajectory_result_is_valid_but_not_plan_vehicle_transportable():
    result = MotionPlanResult(
        is_success=True,
        kind=MotionPlanKind.TIMED_TRAJECTORY,
        xyz=np.zeros((4, 3)),
        quat_wxyz=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (4, 1)),
        time_from_start=np.linspace(0.0, 1.0, 4),
    )

    assert not result.can_transport_over_plan_vehicle_action
    assert not result.needs_trajectory_generator


def test_control_sequence_result_is_valid_but_not_plan_vehicle_transportable():
    result = MotionPlanResult(
        is_success=True,
        kind=MotionPlanKind.CONTROL_SEQUENCE,
        control_sequence=np.zeros((4, 6)),
    )

    assert not result.can_transport_over_plan_vehicle_action
    assert not result.needs_trajectory_generator


def test_invalid_xyz_shape_fails_fast():
    result = MotionPlanResult(is_success=True, xyz=[1.0, 2.0])

    with pytest.raises(ValueError, match="divisible by 3"):
        result.as_action_payload()
