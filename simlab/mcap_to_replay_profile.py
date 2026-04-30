from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import rosbag2_py
from control_msgs.msg import DynamicJointState
from simlab_msgs.msg import ReferenceTargets
from rclpy.serialization import deserialize_message


ARM_AXES = ("e", "d", "c", "b", "a")
VEHICLE_INTERFACES = (
    "force.x",
    "force.y",
    "force.z",
    "torque.x",
    "torque.y",
    "torque.z",
)
VEHICLE_COLUMNS = (
    "vehicle_fx",
    "vehicle_fy",
    "vehicle_fz",
    "vehicle_tx",
    "vehicle_ty",
    "vehicle_tz",
)
ARM_COLUMNS = (
    "tau_axis_e",
    "tau_axis_d",
    "tau_axis_c",
    "tau_axis_b",
    "tau_axis_a",
)
TARGET_COLUMNS = (
    "target_ned_x",
    "target_ned_y",
    "target_ned_z",
    "target_ned_roll",
    "target_ned_pitch",
    "target_ned_yaw",
    "target_body_u",
    "target_body_v",
    "target_body_w",
    "target_body_p",
    "target_body_q",
    "target_body_r",
    "target_body_du",
    "target_body_dv",
    "target_body_dw",
    "target_body_dp",
    "target_body_dq",
    "target_body_dr",
    "arm_ref_axis_e",
    "arm_ref_axis_d",
    "arm_ref_axis_c",
    "arm_ref_axis_b",
    "arm_ref_axis_a",
    "arm_dref_axis_e",
    "arm_dref_axis_d",
    "arm_dref_axis_c",
    "arm_dref_axis_b",
    "arm_dref_axis_a",
    "arm_ddref_axis_e",
    "arm_ddref_axis_d",
    "arm_ddref_axis_c",
    "arm_ddref_axis_b",
    "arm_ddref_axis_a",
)
REFERENCE_TOPIC_TYPE = "simlab_msgs/msg/ReferenceTargets"


def _clean(value: float, eps: float = 1e-12) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return 0.0 if abs(value) < eps else value


def _interface_value(msg: DynamicJointState, name: str, interface: str, default: float = 0.0) -> float:
    try:
        joint_index = msg.joint_names.index(name)
        values = msg.interface_values[joint_index]
        interface_index = values.interface_names.index(interface)
        return _clean(values.values[interface_index])
    except (ValueError, IndexError):
        return _clean(default)


def _has_name(msg: DynamicJointState, name: str) -> bool:
    return name in msg.joint_names


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(v) for v in values)


def _open_reader(bag_path: Path, storage_id: str) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_options, converter_options)
    return reader


def _topic_type_map(reader: rosbag2_py.SequentialReader) -> dict[str, str]:
    return {topic.name: topic.type for topic in reader.get_all_topics_and_types()}


def _message_time_sec(msg, fallback_stamp_ns: int) -> float:
    stamp = getattr(msg, "header", None)
    if stamp is not None:
        return float(stamp.stamp.sec) + float(stamp.stamp.nanosec) * 1e-9
    return float(fallback_stamp_ns) * 1e-9


def _reference_topic(prefix: str) -> str:
    return f"/{prefix}/reference/targets"


def _fixed_vector(values, size: int) -> list[float]:
    out = [_clean(value) for value in list(values)[:size]]
    if len(out) < size:
        out.extend([0.0] * (size - len(out)))
    return out


def _reference_samples(
    bag_path: Path,
    *,
    storage_id: str,
    robot_prefix: str,
) -> list[tuple[float, dict[str, list[float]]]]:
    reader = _open_reader(bag_path, storage_id)
    topic_types = _topic_type_map(reader)
    reference_topic = _reference_topic(robot_prefix)
    if topic_types.get(reference_topic) != REFERENCE_TOPIC_TYPE:
        return []

    samples = []

    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic != reference_topic:
            continue
        msg = deserialize_message(data, ReferenceTargets)
        samples.append(
            (
                _message_time_sec(msg, stamp),
                {
                    "target_ned_pose": _fixed_vector(msg.ned_pose, 6),
                    "target_body_vel": _fixed_vector(msg.body_velocity, 6),
                    "target_body_acc": _fixed_vector(msg.body_acceleration, 6),
                    "arm_ref": _fixed_vector(msg.arm_position, 5),
                    "arm_dref": _fixed_vector(msg.arm_velocity, 5),
                    "arm_ddref": _fixed_vector(msg.arm_acceleration, 5),
                },
            )
        )

    return samples


def _nearest_reference_sample(
    samples: list[tuple[float, dict[str, list[float]]]],
    stamp_sec: float,
    max_age_sec: float,
) -> dict[str, list[float]] | None:
    if not samples:
        return None
    best_time, best_value = min(samples, key=lambda item: abs(item[0] - stamp_sec))
    if max_age_sec >= 0.0 and abs(best_time - stamp_sec) > max_age_sec:
        return None
    return best_value


def _target_row(
    reference_samples: list[tuple[float, dict[str, list[float]]]],
    *,
    time_sec: float,
    stamp_sec: float,
    max_age_sec: float,
) -> dict[str, float]:
    reference = _nearest_reference_sample(reference_samples, stamp_sec, max_age_sec) or {}
    target_ned_pose = reference.get("target_ned_pose", [0.0] * 6)
    target_body_vel = reference.get("target_body_vel", [0.0] * 6)
    target_body_acc = reference.get("target_body_acc", [0.0] * 6)
    arm_ref = reference.get("arm_ref", [0.0] * 5)
    arm_dref = reference.get("arm_dref", [0.0] * 5)
    arm_ddref = reference.get("arm_ddref", [0.0] * 5)
    values = [*target_ned_pose, *target_body_vel, *target_body_acc, *arm_ref, *arm_dref, *arm_ddref]
    row = {"time_sec": time_sec}
    row.update(dict(zip(TARGET_COLUMNS, values)))
    return row


def _extract_rows(
    bag_path: Path,
    *,
    storage_id: str,
    topic: str,
    robot_prefix: str,
    arm_interface: str,
    sample_period: float,
    target_max_age_sec: float,
) -> tuple[list[dict[str, float]], dict[str, list[float]]]:
    reader = _open_reader(bag_path, storage_id)
    topic_types = _topic_type_map(reader)
    if topic not in topic_types:
        raise RuntimeError(f"topic '{topic}' not found in bag")
    if topic_types[topic] != "control_msgs/msg/DynamicJointState":
        raise RuntimeError(f"topic '{topic}' has type '{topic_types[topic]}', expected control_msgs/msg/DynamicJointState")

    vehicle_name = f"{robot_prefix}IOs"
    arm_names = [f"{robot_prefix}_axis_{axis}" for axis in ARM_AXES]

    rows: list[dict[str, float]] = []
    reset: dict[str, list[float]] = {}
    first_sim_time: float | None = None
    last_written_time = -math.inf
    target_samples = _reference_samples(
        bag_path,
        storage_id=storage_id,
        robot_prefix=robot_prefix,
    )

    while reader.has_next():
        bag_topic, data, stamp = reader.read_next()
        if bag_topic != topic:
            continue

        msg = deserialize_message(data, DynamicJointState)
        if not _has_name(msg, vehicle_name) and not any(_has_name(msg, name) for name in arm_names):
            continue

        sim_time = _interface_value(msg, vehicle_name, "sim_time", default=math.nan)
        if not math.isfinite(sim_time):
            stamp = msg.header.stamp
            sim_time = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if first_sim_time is None:
            first_sim_time = sim_time
        time_sec = sim_time - first_sim_time
        if sample_period > 0.0 and rows and (time_sec - last_written_time) < sample_period:
            continue

        vehicle = [_interface_value(msg, vehicle_name, interface) for interface in VEHICLE_INTERFACES]
        arm = [_interface_value(msg, name, arm_interface) for name in arm_names]
        if not _finite(vehicle + arm):
            continue

        if not reset:
            reset = {
                "manipulator_position": [
                    _interface_value(msg, name, "position") for name in arm_names
                ],
                "manipulator_velocity": [
                    _interface_value(msg, name, "velocity") for name in arm_names
                ],
                "vehicle_pose": [
                    _interface_value(msg, vehicle_name, interface)
                    for interface in ("position.x", "position.y", "position.z", "roll", "pitch", "yaw")
                ],
                "vehicle_twist": [
                    _interface_value(msg, vehicle_name, interface)
                    for interface in (
                        "velocity.x",
                        "velocity.y",
                        "velocity.z",
                        "angular_velocity.x",
                        "angular_velocity.y",
                        "angular_velocity.z",
                    )
                ],
                "vehicle_wrench": [0.0] * 6,
            }

        row = {"time_sec": time_sec}
        row.update(dict(zip(VEHICLE_COLUMNS, vehicle)))
        row.update(dict(zip(ARM_COLUMNS, arm)))
        row.update(
            {
                key: value
                for key, value in _target_row(
                    target_samples,
                    time_sec=time_sec,
                    stamp_sec=_message_time_sec(msg, stamp),
                    max_age_sec=target_max_age_sec,
                ).items()
                if key != "time_sec"
            }
        )
        rows.append(row)
        last_written_time = time_sec

    if not rows:
        raise RuntimeError(f"no samples found for robot prefix '{robot_prefix}' on topic '{topic}'")
    return rows, reset


def _write_commands_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = ("time_sec", *VEHICLE_COLUMNS, *ARM_COLUMNS, *TARGET_COLUMNS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{float(row[key]):.9f}" for key in fieldnames})


def _replay_manifest(args, reset: dict[str, list[float]]) -> dict:
    return {
        "csv": "commands.csv",
        "time_column": "time_sec",
        "columns": {
            "vehicle": list(VEHICLE_COLUMNS),
            "manipulator": list(ARM_COLUMNS),
        },
        "playback": {
            "repeats": int(args.repeats),
        },
        "subsystem_mode": {
            "vehicle": str(args.vehicle_mode),
            "manipulator": str(args.manipulator_mode),
            "feedback_controller": str(args.feedback_controller),
        },
        "reset": {
            "hardware_settle": {
                "controller": "PID",
                "position_tolerance": 0.18,
                "velocity_tolerance": 0.03,
                "vehicle_position_tolerance": 0.08,
                "vehicle_velocity_tolerance": 0.05,
                "timeout_sec": 30.0,
            },
            "reset_manipulator": True,
            "reset_vehicle": True,
            "require_release_after_reset": False,
            "manipulator": {
                "enabled": True,
                "position": reset["manipulator_position"],
                "velocity": reset["manipulator_velocity"],
            },
            "vehicle": {
                "enabled": True,
                "pose": reset["vehicle_pose"],
                "twist": reset["vehicle_twist"],
                "wrench": reset["vehicle_wrench"],
            },
            "robot_dynamics_profile": str(args.dynamics_profile),
        },
        "recording": {
            "enabled": False,
        },
    }


def _default_output_root() -> Path:
    cwd_source_root = Path.cwd() / "src" / "uvms-simlab" / "resource" / "playback_profile"
    if cwd_source_root.exists():
        return cwd_source_root

    local_source_root = Path(__file__).resolve().parents[1] / "resource" / "playback_profile"
    if local_source_root.exists():
        return local_source_root

    return Path.cwd() / "playback_profile"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert one robot stream from a rosbag2 MCAP recording into a CmdReplay profile."
    )
    parser.add_argument("bag", help="rosbag2 bag directory, not the .mcap file inside it")
    parser.add_argument("profile", help="new replay profile name")
    parser.add_argument("--robot-prefix", required=True, help="robot prefix to extract, for example robot_1_")
    parser.add_argument("--output-root", default=str(_default_output_root()), help="directory that contains replay profiles")
    parser.add_argument("--topic", default="dynamic_joint_states", help="DynamicJointState topic recorded in the bag")
    parser.add_argument("--storage-id", default="mcap", help="rosbag2 storage id")
    parser.add_argument("--dynamics-profile", default="dory_alpha", help="robot dynamics profile referenced by replay.json")
    parser.add_argument("--arm-interface", default="effort", help="joint interface used as replay arm effort")
    parser.add_argument("--sample-period", type=float, default=0.0, help="minimum output sample period in seconds; 0 keeps every sample")
    parser.add_argument("--target-max-age", type=float, default=0.25, help="maximum allowed time offset when aligning desired target topics")
    parser.add_argument("--repeats", type=int, default=1, help="number of replay passes")
    subsystem_modes = ("replay_command", "track_reference", "hold_initial", "zero_command")
    parser.add_argument("--vehicle-mode", choices=subsystem_modes, default="replay_command")
    parser.add_argument("--manipulator-mode", choices=subsystem_modes, default="replay_command")
    parser.add_argument("--feedback-controller", default="PID")
    parser.add_argument("--force", action="store_true", help="overwrite an existing profile directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.exists():
        raise SystemExit(f"bag path does not exist: {bag_path}")
    if bag_path.is_file():
        raise SystemExit("bag path must be the rosbag2 directory, not the .mcap file")

    output_root = Path(args.output_root).expanduser().resolve()
    profile_dir = output_root / args.profile
    if profile_dir.exists() and not args.force:
        raise SystemExit(f"profile already exists: {profile_dir}; pass --force to overwrite")
    profile_dir.mkdir(parents=True, exist_ok=True)

    rows, reset = _extract_rows(
        bag_path,
        storage_id=args.storage_id,
        topic=args.topic,
        robot_prefix=args.robot_prefix,
        arm_interface=args.arm_interface,
        sample_period=max(0.0, float(args.sample_period)),
        target_max_age_sec=float(args.target_max_age),
    )
    _write_commands_csv(profile_dir / "commands.csv", rows)
    (profile_dir / "replay.json").write_text(
        json.dumps(_replay_manifest(args, reset), indent=2) + "\n",
        encoding="utf-8",
    )

    duration = rows[-1]["time_sec"] - rows[0]["time_sec"]
    print(f"wrote replay profile: {profile_dir}")
    print(f"samples: {len(rows)}, duration: {duration:.3f}s, robot_prefix: {args.robot_prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
