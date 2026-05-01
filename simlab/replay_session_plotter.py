#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ARM_AXES = ("e", "d", "c", "b")
ARM_ALIAS = {
    "axis_e": "e",
    "axis_d": "d",
    "axis_c": "c",
    "axis_b": "b",
    "e": "e",
    "d": "d",
    "c": "c",
    "b": "b",
}
VEHICLE_POSE = (
    ("x", "x [m]"),
    ("y", "y [m]"),
    ("z", "z [m]"),
    ("yaw", "yaw [rad]"),
)
VEHICLE_WRENCH = (
    ("fx", "Fx [N]"),
    ("fy", "Fy [N]"),
    ("fz", "Fz [N]"),
    ("tx", "Tx [Nm]"),
    ("ty", "Ty [Nm]"),
    ("tz", "Tz [Nm]"),
)
VEHICLE_POSE_NAMES = tuple(name for name, _ in VEHICLE_POSE)
VEHICLE_WRENCH_NAMES = tuple(name for name, _ in VEHICLE_WRENCH)


def latest_csv(directory: Path) -> Path:
    csvs = sorted(directory.glob("*.csv"), key=lambda path: path.stat().st_mtime)
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    return csvs[-1]


def has_columns(frame: pd.DataFrame, columns: list[str]) -> bool:
    return all(column in frame.columns for column in columns)


def any_nonzero(frame: pd.DataFrame, columns: list[str]) -> bool:
    present = [column for column in columns if column in frame.columns]
    if not present:
        return False
    return bool((frame[present].abs().sum(axis=1) > 1.0e-12).any())


def plot_columns(ax, frame: pd.DataFrame, time_col: str, items: list[tuple[str, str, str]], ylabel: str) -> bool:
    plotted = False
    for column, label, line_format in items:
        if column in frame.columns:
            ax.plot(frame[time_col], frame[column], line_format, label=label)
            plotted = True
    if plotted:
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", ncols=2)
    return plotted


def parse_csv_filter(value: str | None, allowed: tuple[str, ...], aliases: dict[str, str] | None = None) -> tuple[str, ...]:
    if value is None or value.strip().lower() in {"", "all"}:
        return allowed
    aliases = aliases or {}
    selected = []
    for raw_item in value.split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        item = aliases.get(item, item)
        if item not in allowed:
            raise ValueError(f"Invalid selection '{raw_item}'. Allowed: {', '.join(allowed)}")
        if item not in selected:
            selected.append(item)
    return tuple(selected)


def arm_items(
    measured_prefix: str,
    ref_prefix: str | None = None,
    axes: tuple[str, ...] = ARM_AXES,
) -> list[tuple[str, str, str]]:
    items = []
    for axis in axes:
        items.append((f"{measured_prefix}_{axis}", f"{axis} measured", "-"))
        if ref_prefix is not None:
            items.append((f"{ref_prefix}_{axis}", f"{axis} reference", "--"))
    return items


def vehicle_pose_items(dofs: tuple[str, ...] = VEHICLE_POSE_NAMES) -> list[tuple[str, str, str]]:
    items = []
    for name, _ in VEHICLE_POSE:
        if name not in dofs:
            continue
        items.append((f"vehicle_{name}", f"{name} measured", "-"))
        items.append((f"target_vehicle_{name}", f"{name} reference", "--"))
    return items


def vehicle_wrench_items(dofs: tuple[str, ...] = VEHICLE_WRENCH_NAMES) -> list[tuple[str, str, str]]:
    items = []
    for name, _ in VEHICLE_WRENCH:
        if name not in dofs:
            continue
        items.append((f"wrench_vehicle_{name}", f"{name} reported", "-"))
        items.append((f"cmd_vehicle_{name}", f"{name} command", "--"))
    return items


def replay_kind(frame: pd.DataFrame) -> str:
    arm_ref_cols = [f"ref_alpha_axis_{axis}" for axis in ARM_AXES]
    vehicle_ref_cols = [f"target_vehicle_{name}" for name, _ in VEHICLE_POSE]
    if any_nonzero(frame, arm_ref_cols + vehicle_ref_cols):
        return "reference replay"

    raw_arm_cols = [f"cmd_tau_axis_{axis}" for axis in ARM_AXES]
    raw_vehicle_cols = [f"cmd_vehicle_{name}" for name, _ in VEHICLE_WRENCH]
    if any_nonzero(frame, raw_arm_cols + raw_vehicle_cols):
        return "raw command replay"
    return "replay"


def resolve_csv_path(csv: str | Path | None, script_dir: Path | None = None) -> Path:
    script_dir = Path(__file__).resolve().parent if script_dir is None else script_dir
    if csv:
        csv_path = Path(csv)
    else:
        candidates = [
            Path.cwd() / "recordings/replay_sessions",
            Path("~/ros_ws/recordings/replay_sessions").expanduser(),
            script_dir / "replay_sessions",
            script_dir,
        ]
        csv_path = latest_csv(next((path for path in candidates if path.is_dir()), script_dir))
    return csv_path.expanduser().resolve()


def build_plot(
    csv_path: Path,
    time_col: str = "replay_time_sec",
    arm: str = "all",
    vehicle_pose: str = "all",
    vehicle_wrench: str = "all",
    only: str = "all",
):
    frame = pd.read_csv(csv_path)
    if time_col not in frame.columns:
        raise KeyError(f"Missing time column: {time_col}")

    kind = replay_kind(frame)
    arm_axes = parse_csv_filter(arm, ARM_AXES, ARM_ALIAS)
    vehicle_pose_dofs = parse_csv_filter(vehicle_pose, VEHICLE_POSE_NAMES)
    vehicle_wrench_dofs = parse_csv_filter(vehicle_wrench, VEHICLE_WRENCH_NAMES)

    rows = []
    include_q = only in {"all", "arm", "q"}
    include_dq = only in {"all", "arm", "dq"}
    include_ddq = only in {"all", "arm", "ddq"}
    include_effort = only in {"all", "arm", "effort"}
    include_pose = only in {"all", "vehicle", "pose"}
    include_wrench = only in {"all", "vehicle", "wrench"}

    if include_q and has_columns(frame, [f"q_alpha_axis_{axis}" for axis in arm_axes]):
        rows.append(("Arm position", arm_items("q_alpha_axis", "ref_alpha_axis", arm_axes), "q [rad]"))
    if include_dq and has_columns(frame, [f"dq_alpha_axis_{axis}" for axis in arm_axes]):
        rows.append(("Arm velocity", arm_items("dq_alpha_axis", "dref_alpha_axis", arm_axes), "dq [rad/s]"))
    if include_ddq and has_columns(frame, [f"ddq_alpha_axis_{axis}" for axis in arm_axes]):
        rows.append(("Arm acceleration", arm_items("ddq_alpha_axis", "ddref_alpha_axis", arm_axes), "ddq [rad/s^2]"))
    if include_effort and has_columns(frame, [f"effort_alpha_axis_{axis}" for axis in arm_axes]):
        rows.append(("Arm effort", arm_items("effort_alpha_axis", "cmd_tau_axis", arm_axes), "effort / command [Nm]"))

    pose_items = vehicle_pose_items(vehicle_pose_dofs)
    wrench_items = vehicle_wrench_items(vehicle_wrench_dofs)
    if include_pose and any(column in frame.columns for column, _, _ in pose_items):
        rows.append(("Vehicle pose", pose_items, "pose"))
    if include_wrench and any(column in frame.columns for column, _, _ in wrench_items):
        rows.append(("Vehicle wrench", wrench_items, "wrench"))

    if not rows:
        raise KeyError("No known replay-session columns found to plot.")

    fig, axes = plt.subplots(len(rows), 1, sharex=True, figsize=(14, max(8, 2.6 * len(rows))))
    if len(rows) == 1:
        axes = [axes]

    fig.suptitle(f"{csv_path.name} ({kind})")

    for ax, (title, items, ylabel) in zip(axes, rows):
        ax.set_title(title)
        plot_columns(ax, frame, time_col, items, ylabel)

    axes[-1].set_xlabel(time_col)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a CmdReplay session CSV.")
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        help="Replay session CSV. Defaults to newest CSV in recordings/replay_sessions.",
    )
    parser.add_argument(
        "--time",
        choices=("replay_time_sec", "sim_time_sec", "wall_time_sec"),
        default="replay_time_sec",
        help="Time column for the x-axis.",
    )
    parser.add_argument(
        "--arm",
        default="all",
        help="Comma-separated arm axes to plot: e,d,c,b or all.",
    )
    parser.add_argument(
        "--vehicle-pose",
        default="all",
        help="Comma-separated vehicle pose DOFs to plot: x,y,z,yaw or all.",
    )
    parser.add_argument(
        "--vehicle-wrench",
        default="all",
        help="Comma-separated vehicle wrench DOFs to plot: fx,fy,fz,tx,ty,tz or all.",
    )
    parser.add_argument(
        "--only",
        choices=("all", "arm", "vehicle", "q", "dq", "ddq", "effort", "wrench", "pose"),
        default="all",
        help="Limit plot groups.",
    )
    args = parser.parse_args()

    csv_path = resolve_csv_path(args.csv)
    build_plot(
        csv_path=csv_path,
        time_col=args.time,
        arm=args.arm,
        vehicle_pose=args.vehicle_pose,
        vehicle_wrench=args.vehicle_wrench,
        only=args.only,
    )
    plt.show()


if __name__ == "__main__":
    main()
