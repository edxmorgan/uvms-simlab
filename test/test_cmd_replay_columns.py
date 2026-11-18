import csv
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

if "rosbag2_py" not in sys.modules:
    sys.modules["rosbag2_py"] = types.ModuleType("rosbag2_py")
if "control_msgs" not in sys.modules:
    sys.modules["control_msgs"] = types.ModuleType("control_msgs")
if "control_msgs.msg" not in sys.modules:
    control_msgs_msg = types.ModuleType("control_msgs.msg")
    control_msgs_msg.DynamicJointState = object
    sys.modules["control_msgs.msg"] = control_msgs_msg
if "simlab.msg" not in sys.modules:
    simlab_msg = types.ModuleType("simlab.msg")
    simlab_msg.ReferenceTargets = object
    sys.modules["simlab.msg"] = simlab_msg
if "rclpy.serialization" not in sys.modules:
    rclpy_serialization = types.ModuleType("rclpy.serialization")
    rclpy_serialization.deserialize_message = lambda *_args, **_kwargs: None
    sys.modules["rclpy.serialization"] = rclpy_serialization

from simlab.controllers.cmd_replay import CmdReplayController
from simlab.mcap_to_replay_profile import _replay_manifest
from simlab import mcap_to_replay_profile as converter


def test_reference_alignment_never_uses_future_data():
    samples = [(1.0, {'value': 1}), (1.1, {'value': 2})]
    assert converter._reference_at_or_before(samples, .99, .2) is None
    assert converter._reference_at_or_before(samples, 1.09, .2) == {'value': 1}
    assert converter._reference_at_or_before(samples, 1.1, .2) == {'value': 2}
    assert converter._reference_at_or_before(samples, 1.5, .2) is None
    assert converter._target_row(samples, time_sec=0, stamp_sec=.9, max_age_sec=.2) == {'time_sec': 0}


@pytest.mark.parametrize('values', [[1.0], [float('nan'), 2], [float('inf'), 2]])
def test_reference_vectors_cannot_be_padded_or_cleaned(values):
    with pytest.raises(ValueError):
        converter._fixed_vector(values, 2)


def test_reference_extraction_preserves_clock_and_derivatives(monkeypatch, tmp_path):
    reference = {key: list(np.arange(size) + index * .1)
                 for index, (key, size) in enumerate([
                     ('target_ned_pose', 6), ('target_body_vel', 6), ('target_body_acc', 6),
                     ('arm_ref', 5), ('arm_dref', 5), ('arm_ddref', 5)])}
    samples = [(10.0, reference), (10.013, reference), (10.031, reference)]
    monkeypatch.setattr(converter, '_reference_samples', lambda *a, **k: samples)
    vehicle_interfaces = ('position.x', 'position.y', 'position.z', 'roll', 'pitch', 'yaw',
                          'velocity.x', 'velocity.y', 'velocity.z',
                          'angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z')

    def state(stamp, value):
        return SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=9, nanosec=stamp)),
            joint_names=['robot_IOs'] + ['robot__axis_' + a for a in 'edcba'],
            interface_values=[SimpleNamespace(interface_names=vehicle_interfaces, values=[value]*12)] +
                             [SimpleNamespace(interface_names=['position','velocity'], values=[value,0.])]*5)

    class Reader:
        def __init__(self):
            # A future state must not be selected for reset.
            self.messages = iter([state(990_000_000, 1.), state(1_001_000_000, 2.)])
            self.remaining = 2

        def get_all_topics_and_types(self):
            return [SimpleNamespace(name='/states', type='control_msgs/msg/DynamicJointState')]

        def has_next(self):
            return self.remaining > 0

        def read_next(self):
            self.remaining -= 1
            return '/states', next(self.messages), 0

    monkeypatch.setattr(converter, '_open_reader', lambda *a: Reader())
    monkeypatch.setattr(converter, 'deserialize_message', lambda msg, _: msg)
    rows, reset = converter._extract_reference_rows(tmp_path, storage_id='mcap', topic='/states',
                                                    robot_prefix='robot_', max_gap_sec=.1)
    np.testing.assert_allclose([row['time_sec'] for row in rows], [0, .013, .031], atol=1e-14)
    assert reset['manipulator_position'] == [1.]*5
    assert set(rows[0]) == {'time_sec', *converter.TARGET_COLUMNS}
    assert rows[-1]['arm_ddref_axis_a'] == reference['arm_ddref'][-1]
    path = tmp_path / 'reference.csv'
    converter._write_commands_csv(path, rows)
    with path.open() as handle:
        restored = list(csv.DictReader(handle))
    assert float(restored[-1]['arm_ddref_axis_a']) == reference['arm_ddref'][-1]
    assert not set(converter.ARM_COLUMNS) & set(restored[0])
    samples.append((11., reference))
    with pytest.raises(ValueError, match='gap'):
        converter._extract_reference_rows(tmp_path, storage_id='mcap', topic='/states',
                                          robot_prefix='robot_', max_gap_sec=.1)
    samples.pop()
    with pytest.raises(ValueError, match='no complete'):
        converter._extract_reference_rows(tmp_path, storage_id='mcap', topic='/states',
                                          robot_prefix='missing_', max_gap_sec=.1)


class _Logger:
    def __init__(self):
        self.messages = []

    def warn(self, *_args, **_kwargs):
        self.messages.append(("warn", _args))

    def error(self, *_args, **_kwargs):
        self.messages.append(("error", _args))

    def info(self, *_args, **_kwargs):
        self.messages.append(("info", _args))


class _Node:
    def __init__(self):
        self.logger = _Logger()

    def get_logger(self):
        return self.logger


def _controller(vehicle_mode="replay_command", manipulator_mode="replay_command"):
    controller = object.__new__(CmdReplayController)
    controller.node = _Node()
    controller.arm_dof = 4
    controller.time_column = CmdReplayController.DEFAULT_TIME_COLUMN
    controller.subsystem_mode = {
        "vehicle": vehicle_mode,
        "manipulator": manipulator_mode,
        "feedback_controller": "PID",
    }
    controller.repeats = 1
    controller.enabled = False
    controller.vehicle_columns = controller._parse_columns(
        ["ref_x", "ref_y", "ref_z", "ref_roll", "ref_pitch", "ref_yaw"],
        expected_size=6,
    )
    controller.arm_columns = controller._parse_columns(
        ["arm_ref_e", "arm_ref_d", "arm_ref_c", "arm_ref_b", "arm_ref_a"],
        expected_size=5,
    )
    controller.vehicle_reference_velocity_columns = controller._parse_columns(
        CmdReplayController.DEFAULT_VEHICLE_REFERENCE_VEL_COLUMNS,
        expected_size=6,
    )
    controller.vehicle_reference_acceleration_columns = controller._parse_columns(
        CmdReplayController.DEFAULT_VEHICLE_REFERENCE_ACC_COLUMNS,
        expected_size=6,
    )
    controller.arm_reference_velocity_columns = controller._parse_columns(
        CmdReplayController.DEFAULT_ARM_REFERENCE_VELOCITY_COLUMNS,
        expected_size=5,
    )
    controller.arm_reference_acceleration_columns = controller._parse_columns(
        CmdReplayController.DEFAULT_ARM_REFERENCE_ACCELERATION_COLUMNS,
        expected_size=5,
    )
    controller.times_sec = np.array([], dtype=float)
    controller.profile_name = ""
    controller.enabled = False
    controller._auto_start_pending = False
    controller._repeat_reset_requested = False
    controller._current_pass = 0
    controller._run_start_sim_time = None
    controller._last_sim_time = None
    controller._sample_time_sec = 0.0
    controller._reported_done = False
    controller._warned_time_jump = False
    controller.lifecycle_state = None
    return controller


def _write_csv(path, row):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def test_vehicle_columns_are_reference_pose_in_track_reference_mode(tmp_path):
    path = tmp_path / "commands.csv"
    _write_csv(
        path,
        {
            "time_sec": 0.0,
            "vehicle_fx": 10.0,
            "vehicle_fy": 11.0,
            "vehicle_fz": 12.0,
            "vehicle_tx": 13.0,
            "vehicle_ty": 14.0,
            "vehicle_tz": 15.0,
            "ref_x": 1.0,
            "ref_y": 2.0,
            "ref_z": 3.0,
            "ref_roll": 4.0,
            "ref_pitch": 5.0,
            "ref_yaw": 6.0,
        },
    )

    controller = _controller(vehicle_mode="track_reference")
    controller._load_csv(str(path))

    np.testing.assert_allclose(controller.vehicle_reference_pose[0], [1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(controller.vehicle_command_at(0), [0, 0, 0, 0, 0, 0])


def test_vehicle_columns_are_direct_commands_in_replay_command_mode(tmp_path):
    path = tmp_path / "commands.csv"
    _write_csv(
        path,
        {
            "time_sec": 0.0,
            "ref_x": 10.0,
            "ref_y": 11.0,
            "ref_z": 12.0,
            "ref_roll": 13.0,
            "ref_pitch": 14.0,
            "ref_yaw": 15.0,
            "target_ned_x": 1.0,
            "target_ned_y": 2.0,
            "target_ned_z": 3.0,
            "target_ned_roll": 4.0,
            "target_ned_pitch": 5.0,
            "target_ned_yaw": 6.0,
        },
    )

    controller = _controller(vehicle_mode="replay_command")
    controller._load_csv(str(path))

    np.testing.assert_allclose(controller.vehicle_command_at(0), [10, 11, 12, 13, 14, 15])
    np.testing.assert_allclose(controller.vehicle_reference_pose[0], [0, 0, 0, 0, 0, 0])


def test_manipulator_columns_are_reference_position_in_track_reference_mode(tmp_path):
    path = tmp_path / "commands.csv"
    _write_csv(
        path,
        {
            "time_sec": 0.0,
            "arm_ref_e": 3.1,
            "arm_ref_d": 0.7,
            "arm_ref_c": 0.4,
            "arm_ref_b": 2.1,
            "arm_ref_a": 0.2,
        },
    )

    controller = _controller(manipulator_mode="track_reference")
    controller._load_csv(str(path))

    np.testing.assert_allclose(controller.arm_reference_position[0], [3.1, 0.7, 0.4, 2.1, 0.2])
    np.testing.assert_allclose(controller.arm_command_at(0), [0, 0, 0, 0, 0])


def test_generated_manifest_columns_follow_selected_modes():
    args = SimpleNamespace(
        repeats=1,
        vehicle_mode="track_reference",
        manipulator_mode="hold_initial",
        feedback_controller="PID",
        dynamics_profile="dory_alpha",
    )
    reset = {
        "manipulator_position": [3.1, 0.7, 0.4, 2.1, 0.0],
        "manipulator_velocity": [0.0] * 5,
        "vehicle_pose": [0.0] * 6,
        "vehicle_twist": [0.0] * 6,
        "vehicle_wrench": [0.0] * 6,
    }

    manifest = _replay_manifest(args, reset)
    assert manifest["columns"]["vehicle"] == [
        "target_ned_x",
        "target_ned_y",
        "target_ned_z",
        "target_ned_roll",
        "target_ned_pitch",
        "target_ned_yaw",
    ]
    assert "vehicle_velocity" in manifest["columns"]
    assert "vehicle_acceleration" in manifest["columns"]
    assert "manipulator" not in manifest["columns"]

    args.vehicle_mode = "replay_command"
    args.manipulator_mode = "track_reference"
    manifest = _replay_manifest(args, reset)
    assert manifest["columns"]["vehicle"] == [
        "vehicle_fx",
        "vehicle_fy",
        "vehicle_fz",
        "vehicle_tx",
        "vehicle_ty",
        "vehicle_tz",
    ]
    assert manifest["columns"]["manipulator"] == [
        "arm_ref_axis_e",
        "arm_ref_axis_d",
        "arm_ref_axis_c",
        "arm_ref_axis_b",
        "arm_ref_axis_a",
    ]
    assert "manipulator_velocity" in manifest["columns"]
    assert "manipulator_acceleration" in manifest["columns"]


def test_replay_profile_requires_explicit_dynamics_profile(tmp_path):
    profile_dir = tmp_path / "missing_dynamics_profile"
    profile_dir.mkdir()
    (profile_dir / "replay.json").write_text(
        """
{
  "playback": {"repeats": 1},
  "subsystem_mode": {"vehicle": "replay_command", "manipulator": "replay_command"},
  "reset": {},
  "columns": {}
}
""",
        encoding="utf-8",
    )

    controller = _controller()
    controller.profiles_root = tmp_path

    assert controller.load_profile("missing_dynamics_profile") is False
    assert controller.lifecycle_state.value == "error"
    errors = [args[0] for level, args in controller.node.logger.messages if level == "error"]
    assert any("reset.robot_dynamics_profile" in message for message in errors)
