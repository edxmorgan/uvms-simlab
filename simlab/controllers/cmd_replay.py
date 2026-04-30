import json
import os
from enum import Enum
from pathlib import Path

import ament_index_python
import numpy as np
from rclpy.node import Node
from ros2_control_blue_reach_5.srv import ResetSimUvms

from simlab.controllers.base import ControllerTemplate
from simlab.dynamics_profiles import (
    is_valid_robot_dynamics_profile,
    load_robot_dynamics_profile,
    set_dynamics_request_from_profile,
)


class CmdReplayState(str, Enum):
    STOPPED = "stopped"
    RESETTING = "resetting"
    ARMED = "armed"
    PLAYING = "playing"
    COMPLETE = "complete"
    ERROR = "error"


class CmdReplayController(ControllerTemplate):
    """Replay CSV vehicle wrench and arm effort samples through normal command paths."""

    name = "CmdReplay"
    registry_name = "CmdReplay"

    DEFAULT_PROFILE = ""
    DEFAULT_TIME_COLUMN = "time_sec"
    DEFAULT_VEHICLE_COLUMNS = "vehicle_fx,vehicle_fy,vehicle_fz,vehicle_tx,vehicle_ty,vehicle_tz"
    DEFAULT_ARM_COLUMNS = "tau_axis_e,tau_axis_d,tau_axis_c,tau_axis_b,tau_axis_a"
    DEFAULT_VEHICLE_REFERENCE_POSE_COLUMNS = (
        "target_ned_x,target_ned_y,target_ned_z,target_ned_roll,target_ned_pitch,target_ned_yaw"
    )
    DEFAULT_VEHICLE_REFERENCE_VEL_COLUMNS = (
        "target_body_u,target_body_v,target_body_w,target_body_p,target_body_q,target_body_r"
    )
    DEFAULT_VEHICLE_REFERENCE_ACC_COLUMNS = (
        "target_body_du,target_body_dv,target_body_dw,target_body_dp,target_body_dq,target_body_dr"
    )
    DEFAULT_ARM_REFERENCE_POSITION_COLUMNS = (
        "arm_ref_axis_e,arm_ref_axis_d,arm_ref_axis_c,arm_ref_axis_b,arm_ref_axis_a"
    )
    DEFAULT_ARM_REFERENCE_VELOCITY_COLUMNS = (
        "arm_dref_axis_e,arm_dref_axis_d,arm_dref_axis_c,arm_dref_axis_b,arm_dref_axis_a"
    )
    DEFAULT_ARM_REFERENCE_ACCELERATION_COLUMNS = (
        "arm_ddref_axis_e,arm_ddref_axis_d,arm_ddref_axis_c,arm_ddref_axis_b,arm_ddref_axis_a"
    )
    REAL_SETTLE_POSITION_TOLERANCE = 0.40
    REAL_SETTLE_VELOCITY_TOLERANCE = 0.05
    REAL_SETTLE_TIMEOUT_SEC = 60.0

    def __init__(self, node: Node, arm_dof: int = 4, robot_prefix: str = ""):
        super().__init__(node, arm_dof, robot_prefix)
        self.profiles_root = Path(ament_index_python.get_package_share_directory("simlab")) / "playback_profile"
        self.profile_name = str(self._get_or_declare_parameter("cmd_replay_profile", self.DEFAULT_PROFILE)).strip()
        self.csv_path = ""
        self.config_path = ""
        self.time_column = self.DEFAULT_TIME_COLUMN
        self.vehicle_columns = self._parse_columns(self.DEFAULT_VEHICLE_COLUMNS, expected_size=6)
        self.arm_columns = self._parse_columns(self.DEFAULT_ARM_COLUMNS, expected_size=self.arm_dof + 1)
        self.repeats = 1
        self.enabled = bool(self._get_or_declare_parameter("cmd_replay_enabled", False, "csv_torque_playback_enabled"))
        self.max_sim_time_step_sec = float(
            self._get_or_declare_parameter("cmd_replay_max_sim_time_step_sec", 1.0)
        )

        self._run_start_sim_time = None
        self._last_sim_time = None
        self._sample_time_sec = 0.0
        self._warned_missing = False
        self._reported_done = False
        self._auto_start_pending = False
        self._warned_time_jump = False
        self.lifecycle_state = CmdReplayState.PLAYING if self.enabled else CmdReplayState.STOPPED
        self._reset_requires_release_after_reset = True
        self._current_pass = 0
        self._repeat_reset_requested = False
        self.times_sec = np.array([], dtype=float)
        self.vehicle_commands = np.zeros((0, 6), dtype=float)
        self.arm_commands = np.zeros((0, self.arm_dof + 1), dtype=float)
        self.vehicle_reference_pose = np.zeros((0, 6), dtype=float)
        self.vehicle_reference_vel = np.zeros((0, 6), dtype=float)
        self.vehicle_reference_acc = np.zeros((0, 6), dtype=float)
        self.arm_reference_position = np.zeros((0, self.arm_dof + 1), dtype=float)
        self.arm_reference_velocity = np.zeros((0, self.arm_dof + 1), dtype=float)
        self.arm_reference_acceleration = np.zeros((0, self.arm_dof + 1), dtype=float)
        self.duration_sec = 0.0
        self.reset_config = self._default_reset_config()
        self.subsystem_mode = self._default_subsystem_mode()
        self.recording_config = self._default_recording_config()

        if self.profile_name:
            self.load_profile(self.profile_name)
        else:
            self.node.get_logger().info("CmdReplay has no selected profile; choose a Cmd Replay profile before playback.")

    def _get_or_declare_parameter(self, name: str, default_value, legacy_name: str = None):
        if legacy_name and self.node.has_parameter(legacy_name):
            return self.node.get_parameter(legacy_name).value
        if not self.node.has_parameter(name):
            self.node.declare_parameter(name, default_value)
        return self.node.get_parameter(name).value

    def _parse_columns(self, value, expected_size: int) -> list:
        if isinstance(value, str):
            columns = [column.strip() or None for column in value.split(",")]
        else:
            columns = [str(column).strip() or None for column in value]
        if len(columns) < expected_size:
            columns.extend([None] * (expected_size - len(columns)))
        return columns[:expected_size]

    def list_profiles(self) -> list[str]:
        if not self.profiles_root.exists():
            return []
        return sorted(
            path.name
            for path in self.profiles_root.iterdir()
            if path.is_dir() and (path / "replay.json").exists()
        )

    def load_profile(self, profile_name: str) -> bool:
        profile_name = str(profile_name).strip()
        profile_dir = self.profiles_root / profile_name
        manifest_path = profile_dir / "replay.json"
        if not manifest_path.exists():
            self.node.get_logger().error(
                f"CmdReplay profile '{profile_name}' not found: {manifest_path}."
            )
            self.stop_playback()
            self.lifecycle_state = CmdReplayState.ERROR
            return False

        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as exc:
            self.node.get_logger().error(
                f"CmdReplay failed to load profile '{profile_name}' manifest: {exc}."
            )
            self.stop_playback()
            self.lifecycle_state = CmdReplayState.ERROR
            return False

        if not isinstance(manifest, dict):
            self.node.get_logger().error(f"CmdReplay profile '{profile_name}' replay.json must be an object.")
            self.stop_playback()
            self.lifecycle_state = CmdReplayState.ERROR
            return False

        playback = manifest.get("playback", {})
        columns = manifest.get("columns", {})
        subsystem_mode = manifest.get("subsystem_mode", {})
        if not isinstance(subsystem_mode, dict):
            self.node.get_logger().error(
                f"CmdReplay profile '{profile_name}' replay.json must contain a subsystem_mode object."
            )
            self.stop_playback()
            self.lifecycle_state = CmdReplayState.ERROR
            return False
        recording = manifest.get("recording", {})
        csv_name = str(manifest.get("csv", "commands.csv"))
        reset_section = manifest.get("reset", {})
        if not isinstance(reset_section, dict):
            reset_section = {}
        dynamics_profile_name = str(reset_section.get("robot_dynamics_profile", "dory_alpha"))
        dynamics_profile = load_robot_dynamics_profile(dynamics_profile_name, self.node)
        if not is_valid_robot_dynamics_profile(dynamics_profile):
            self.node.get_logger().error(
                f"CmdReplay profile '{profile_name}' references invalid robot dynamics profile "
                f"'{dynamics_profile_name}'."
            )
            self.stop_playback()
            self.lifecycle_state = CmdReplayState.ERROR
            return False

        self.stop_playback()
        self.profile_name = profile_name
        self.config_path = str(manifest_path)
        self.csv_path = str(profile_dir / csv_name)
        self.time_column = str(manifest.get("time_column", self.DEFAULT_TIME_COLUMN))
        self.vehicle_columns = self._parse_columns(
            columns.get("vehicle", self.DEFAULT_VEHICLE_COLUMNS),
            expected_size=6,
        )
        self.arm_columns = self._parse_columns(
            columns.get("manipulator", self.DEFAULT_ARM_COLUMNS),
            expected_size=self.arm_dof + 1,
        )
        self.repeats = max(1, int(playback.get("repeats", 1)))
        self.subsystem_mode = self._merge_subsystem_mode(self._default_subsystem_mode(), subsystem_mode)
        self.recording_config = self._merge_recording_config(self._default_recording_config(), recording)
        self._warned_invalid_feedback_controller = False
        self.reset_config = self._merge_reset_config(
            self._default_reset_config(),
            manifest.get("reset", {}),
        )
        self.times_sec = np.array([], dtype=float)
        self.vehicle_commands = np.zeros((0, 6), dtype=float)
        self.arm_commands = np.zeros((0, self.arm_dof + 1), dtype=float)
        self.vehicle_reference_pose = np.zeros((0, 6), dtype=float)
        self.vehicle_reference_vel = np.zeros((0, 6), dtype=float)
        self.vehicle_reference_acc = np.zeros((0, 6), dtype=float)
        self.arm_reference_position = np.zeros((0, self.arm_dof + 1), dtype=float)
        self.arm_reference_velocity = np.zeros((0, self.arm_dof + 1), dtype=float)
        self.arm_reference_acceleration = np.zeros((0, self.arm_dof + 1), dtype=float)
        self.duration_sec = 0.0

        self._load_csv(self.csv_path)
        self.node.get_logger().info(f"CmdReplay selected profile '{profile_name}'.")
        return self.times_sec.size > 0

    def _default_reset_config(self) -> dict:
        return {
            "reset_manipulator": True,
            "reset_vehicle": True,
            "require_release_after_reset": True,
            "manipulator": {
                "enabled": True,
                "position": [0.0] * (self.arm_dof + 1),
                "velocity": [0.0] * (self.arm_dof + 1),
            },
            "vehicle": {
                "enabled": True,
                "pose": [0.0] * 6,
                "twist": [0.0] * 6,
                "wrench": [0.0] * 6,
            },
            "robot_dynamics_profile": "dory_alpha",
        }

    def _default_subsystem_mode(self) -> dict:
        return {
            "vehicle": "replay_command",
            "manipulator": "replay_command",
            "feedback_controller": "PID",
        }

    def _default_recording_config(self) -> dict:
        return {
            "enabled": False,
        }

    def _merge_recording_config(self, base: dict, override: dict) -> dict:
        if not isinstance(override, dict):
            override = {}
        config = dict(base)
        config["enabled"] = bool(override.get("enabled", config["enabled"]))
        return config

    def recording_enabled(self) -> bool:
        return bool(self.recording_config.get("enabled", False))

    def _merge_subsystem_mode(self, base: dict, override: dict) -> dict:
        if not isinstance(override, dict):
            override = {}
        mode = dict(base)
        for key in ("vehicle", "manipulator"):
            value = str(override.get(key, mode[key])).strip().lower()
            if value not in ("replay_command", "track_reference", "hold_initial", "zero_command"):
                self.node.get_logger().warn(
                    f"CmdReplay invalid subsystem_mode.{key}='{value}'; using '{mode[key]}'."
                )
                continue
            mode[key] = value
        feedback_controller = str(override.get("feedback_controller", mode["feedback_controller"])).strip()
        if not feedback_controller:
            self.node.get_logger().warn(
                f"CmdReplay invalid subsystem_mode.feedback_controller='{feedback_controller}'; "
                f"using '{mode['feedback_controller']}'."
            )
        else:
            mode["feedback_controller"] = feedback_controller
        return mode

    def vehicle_subsystem_mode(self) -> str:
        return str(self.subsystem_mode.get("vehicle", "replay_command"))

    def manipulator_subsystem_mode(self) -> str:
        return str(self.subsystem_mode.get("manipulator", "replay_command"))

    def feedback_controller_name(self) -> str:
        return str(self.subsystem_mode.get("feedback_controller", "PID"))

    def _load_reset_config(self, config_path: str) -> None:
        path = Path(os.path.expanduser(config_path))
        if not path.exists():
            self.node.get_logger().warn(
                f"CmdReplay reset config not found: {path}. Using zero-state reset defaults."
            )
            return

        try:
            if path.suffix.lower() in (".yaml", ".yml"):
                import yaml

                loaded = yaml.safe_load(path.read_text()) or {}
            else:
                loaded = json.loads(path.read_text())
        except Exception as exc:
            self.node.get_logger().error(
                f"CmdReplay failed to load reset config {path}: {exc}. Using zero-state reset defaults."
            )
            return

        if not isinstance(loaded, dict):
            self.node.get_logger().error(
                f"CmdReplay reset config must be a JSON/YAML object: {path}. Using zero-state reset defaults."
            )
            return

        reset_section = loaded.get("reset", loaded)
        if not isinstance(reset_section, dict):
            self.node.get_logger().error(
                f"CmdReplay reset config 'reset' section must be an object: {path}. Using zero-state reset defaults."
            )
            return

        self.reset_config = self._merge_reset_config(self._default_reset_config(), reset_section)
        self.node.get_logger().info(f"CmdReplay loaded reset config from {path}.")

    def _merge_reset_config(self, base: dict, override: dict) -> dict:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_reset_config(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _vector(self, value, size: int, default: float = 0.0) -> list[float]:
        if value is None:
            values = []
        else:
            values = list(value)
        values = [float(v) for v in values[:size]]
        if len(values) < size:
            values.extend([float(default)] * (size - len(values)))
        return values

    def build_reset_request(self) -> ResetSimUvms.Request:
        config = self.reset_config
        manipulator = config.get("manipulator", {})
        vehicle = config.get("vehicle", {})
        dynamics = load_robot_dynamics_profile(
            config.get("robot_dynamics_profile", "dory_alpha"),
            self.node,
        )

        request = ResetSimUvms.Request()
        request.reset_manipulator = bool(config.get("reset_manipulator", True))
        request.reset_vehicle = bool(config.get("reset_vehicle", True))
        request.hold_commands = bool(config.get("require_release_after_reset", True))
        request.use_manipulator_state = bool(manipulator.get("enabled", True))
        request.manipulator_position = self._vector(manipulator.get("position"), 5)
        request.manipulator_velocity = self._vector(manipulator.get("velocity"), 5)
        request.use_vehicle_state = bool(vehicle.get("enabled", True))
        request.vehicle_pose = self._vector(vehicle.get("pose"), 6)
        request.vehicle_twist = self._vector(vehicle.get("twist"), 6)
        request.vehicle_wrench = self._vector(vehicle.get("wrench"), 6)
        dynamics_request = set_dynamics_request_from_profile(dynamics)
        request.use_coupled_dynamics = dynamics_request.use_coupled_dynamics
        request.set_manipulator_dynamics = dynamics_request.set_manipulator_dynamics
        request.manipulator_dynamics = dynamics_request.manipulator
        request.set_vehicle_dynamics = dynamics_request.set_vehicle_dynamics
        request.vehicle_dynamics = dynamics_request.vehicle
        return request

    def reset_mode(self) -> str:
        return "controller_settle" if "real" in self.robot_prefix else "sim_state"

    def controller_settle_config(self) -> dict:
        settle = self.reset_config.get("hardware_settle", {})
        if not isinstance(settle, dict):
            settle = {}
        config = {
            "controller": str(settle.get("controller", "PID")),
            "position_tolerance": float(settle.get("position_tolerance", 0.03)),
            "velocity_tolerance": float(settle.get("velocity_tolerance", 0.03)),
            "vehicle_position_tolerance": float(settle.get("vehicle_position_tolerance", 0.08)),
            "vehicle_velocity_tolerance": float(settle.get("vehicle_velocity_tolerance", 0.05)),
            "timeout_sec": float(settle.get("timeout_sec", 20.0)),
        }
        if "real" in self.robot_prefix:
            config["position_tolerance"] = max(config["position_tolerance"], self.REAL_SETTLE_POSITION_TOLERANCE)
            config["velocity_tolerance"] = max(config["velocity_tolerance"], self.REAL_SETTLE_VELOCITY_TOLERANCE)
            config["timeout_sec"] = max(config["timeout_sec"], self.REAL_SETTLE_TIMEOUT_SEC)
        return config

    def initial_manipulator_position(self) -> list[float]:
        manipulator = self.reset_config.get("manipulator", {})
        return self._vector(manipulator.get("position"), 5)

    def initial_vehicle_pose(self) -> list[float]:
        vehicle = self.reset_config.get("vehicle", {})
        return self._vector(vehicle.get("pose"), 6)

    def _command_matrix_from_columns(
        self,
        data,
        names: tuple,
        columns: list,
        label: str,
        warn_missing: bool = True,
    ) -> np.ndarray:
        rows = int(self.times_sec.size)
        command = np.zeros((rows, len(columns)), dtype=float)
        missing = []

        for index, column in enumerate(columns):
            if column is None:
                continue
            if column not in names:
                missing.append(column)
                continue
            command[:, index] = np.asarray(data[column], dtype=float).reshape(-1)

        if missing and warn_missing:
            self.node.get_logger().warn(
                f"CmdReplay missing {label} column(s) {missing}; using zero for those command slots."
            )
        return command

    def _load_csv(self, csv_path: str) -> None:
        path = Path(os.path.expanduser(csv_path))
        if not path.exists():
            self.node.get_logger().error(
                f"CmdReplay CSV not found: {path}. Controller will publish zero commands."
            )
            return

        data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
        if data.size == 0:
            self.node.get_logger().error(
                f"CmdReplay CSV is empty: {path}. Controller will publish zero commands."
            )
            return

        names = data.dtype.names or ()
        if self.time_column not in names:
            self.node.get_logger().error(
                f"CmdReplay CSV missing time column '{self.time_column}': {path}. "
                "Controller will publish zero commands."
            )
            return

        self.times_sec = np.asarray(data[self.time_column], dtype=float).reshape(-1)
        self.vehicle_commands = self._command_matrix_from_columns(
            data,
            names,
            self.vehicle_columns,
            "vehicle",
        )
        self.arm_commands = self._command_matrix_from_columns(
            data,
            names,
            self.arm_columns,
            "arm",
        )
        vehicle_tracks_reference = self.vehicle_subsystem_mode() == "track_reference"
        manipulator_tracks_reference = self.manipulator_subsystem_mode() == "track_reference"
        self.vehicle_reference_pose = self._command_matrix_from_columns(
            data,
            names,
            self._parse_columns(self.DEFAULT_VEHICLE_REFERENCE_POSE_COLUMNS, expected_size=6),
            "vehicle reference pose",
            warn_missing=vehicle_tracks_reference,
        )
        self.vehicle_reference_vel = self._command_matrix_from_columns(
            data,
            names,
            self._parse_columns(self.DEFAULT_VEHICLE_REFERENCE_VEL_COLUMNS, expected_size=6),
            "vehicle reference velocity",
            warn_missing=vehicle_tracks_reference,
        )
        self.vehicle_reference_acc = self._command_matrix_from_columns(
            data,
            names,
            self._parse_columns(self.DEFAULT_VEHICLE_REFERENCE_ACC_COLUMNS, expected_size=6),
            "vehicle reference acceleration",
            warn_missing=vehicle_tracks_reference,
        )
        self.arm_reference_position = self._command_matrix_from_columns(
            data,
            names,
            self._parse_columns(self.DEFAULT_ARM_REFERENCE_POSITION_COLUMNS, expected_size=self.arm_dof + 1),
            "arm reference position",
            warn_missing=manipulator_tracks_reference,
        )
        self.arm_reference_velocity = self._command_matrix_from_columns(
            data,
            names,
            self._parse_columns(self.DEFAULT_ARM_REFERENCE_VELOCITY_COLUMNS, expected_size=self.arm_dof + 1),
            "arm reference velocity",
            warn_missing=manipulator_tracks_reference,
        )
        self.arm_reference_acceleration = self._command_matrix_from_columns(
            data,
            names,
            self._parse_columns(self.DEFAULT_ARM_REFERENCE_ACCELERATION_COLUMNS, expected_size=self.arm_dof + 1),
            "arm reference acceleration",
            warn_missing=manipulator_tracks_reference,
        )

        order = np.argsort(self.times_sec)
        self.times_sec = self.times_sec[order]
        self.vehicle_commands = self.vehicle_commands[order]
        self.arm_commands = self.arm_commands[order]
        self.vehicle_reference_pose = self.vehicle_reference_pose[order]
        self.vehicle_reference_vel = self.vehicle_reference_vel[order]
        self.vehicle_reference_acc = self.vehicle_reference_acc[order]
        self.arm_reference_position = self.arm_reference_position[order]
        self.arm_reference_velocity = self.arm_reference_velocity[order]
        self.arm_reference_acceleration = self.arm_reference_acceleration[order]

        finite = (
            np.isfinite(self.times_sec)
            & np.all(np.isfinite(self.vehicle_commands), axis=1)
            & np.all(np.isfinite(self.arm_commands), axis=1)
            & np.all(np.isfinite(self.vehicle_reference_pose), axis=1)
            & np.all(np.isfinite(self.vehicle_reference_vel), axis=1)
            & np.all(np.isfinite(self.vehicle_reference_acc), axis=1)
            & np.all(np.isfinite(self.arm_reference_position), axis=1)
            & np.all(np.isfinite(self.arm_reference_velocity), axis=1)
            & np.all(np.isfinite(self.arm_reference_acceleration), axis=1)
        )
        self.times_sec = self.times_sec[finite]
        self.vehicle_commands = self.vehicle_commands[finite]
        self.arm_commands = self.arm_commands[finite]
        self.vehicle_reference_pose = self.vehicle_reference_pose[finite]
        self.vehicle_reference_vel = self.vehicle_reference_vel[finite]
        self.vehicle_reference_acc = self.vehicle_reference_acc[finite]
        self.arm_reference_position = self.arm_reference_position[finite]
        self.arm_reference_velocity = self.arm_reference_velocity[finite]
        self.arm_reference_acceleration = self.arm_reference_acceleration[finite]

        if self.times_sec.size == 0:
            self.node.get_logger().error(
                f"CmdReplay CSV has no finite samples: {path}. "
                "Controller will publish zero commands."
            )
            return

        if self.times_sec.size > 1:
            sample_period = float(np.median(np.diff(self.times_sec)))
            sample_period = max(sample_period, 0.0)
        else:
            sample_period = 0.0
        self.duration_sec = float(self.times_sec[-1]) + sample_period

        self.node.get_logger().info(
            f"CmdReplay loaded {self.times_sec.size} samples from {path}, "
            f"duration={self.duration_sec:.3f}s, repeats={self.repeats}, "
            f"enabled={self.enabled}."
        )

    def reset_playback(self) -> None:
        self._run_start_sim_time = None
        self._last_sim_time = None
        self._sample_time_sec = 0.0
        self._reported_done = False
        self._warned_time_jump = False

    def has_valid_playback(self) -> bool:
        return (
            bool(self.profile_name)
            and
            self.times_sec.size > 0
            and self.duration_sec > 0.0
            and self.vehicle_commands.shape[0] == self.times_sec.size
            and self.arm_commands.shape[0] == self.times_sec.size
            and self.vehicle_reference_pose.shape[0] == self.times_sec.size
            and self.vehicle_reference_vel.shape[0] == self.times_sec.size
            and self.vehicle_reference_acc.shape[0] == self.times_sec.size
            and self.arm_reference_position.shape[0] == self.times_sec.size
            and self.arm_reference_velocity.shape[0] == self.times_sec.size
            and self.arm_reference_acceleration.shape[0] == self.times_sec.size
        )

    def has_selected_profile(self) -> bool:
        return bool(self.profile_name)

    def request_reset(self, require_release_after_reset: bool) -> bool:
        if not self.has_valid_playback():
            self.enabled = False
            self._auto_start_pending = False
            self._repeat_reset_requested = False
            self.lifecycle_state = CmdReplayState.ERROR
            if not self.has_selected_profile():
                self.node.get_logger().error("CmdReplay has no selected profile; reset/playback rejected.")
            else:
                self.node.get_logger().error(
                    f"CmdReplay profile '{self.profile_name}' has no valid command samples; reset/playback rejected."
                )
            return False
        self.enabled = False
        self._auto_start_pending = False
        self._repeat_reset_requested = False
        self._reset_requires_release_after_reset = bool(require_release_after_reset)
        self.reset_playback()
        self.lifecycle_state = CmdReplayState.RESETTING
        self.node.get_logger().info("CmdReplay reset requested.")
        return True

    def mark_reset_succeeded(self) -> None:
        if self.lifecycle_state != CmdReplayState.RESETTING:
            self.node.get_logger().warn(
                f"CmdReplay reset success ignored from state {self.lifecycle_state.value}."
            )
            return
        self.enabled = False
        self._auto_start_pending = True
        self.lifecycle_state = CmdReplayState.ARMED
        self.node.get_logger().info("CmdReplay armed after reset.")

    def mark_reset_failed(self) -> None:
        self.enabled = False
        self._auto_start_pending = False
        self._repeat_reset_requested = False
        self.lifecycle_state = CmdReplayState.ERROR
        self.reset_playback()
        self.node.get_logger().warn("CmdReplay reset failed.")

    def request_auto_start(self) -> None:
        self._auto_start_pending = True
        self.lifecycle_state = CmdReplayState.ARMED
        self.node.get_logger().info("CmdReplay auto-start armed.")

    def has_pending_auto_start(self) -> bool:
        return self.lifecycle_state == CmdReplayState.ARMED and self._auto_start_pending

    def start_playback(self, sim_time_sec: float | None = None) -> bool:
        if not self.has_valid_playback():
            self.enabled = False
            self._auto_start_pending = False
            self.lifecycle_state = CmdReplayState.ERROR
            if not self.has_selected_profile():
                self.node.get_logger().error("CmdReplay has no selected profile; start rejected.")
            else:
                self.node.get_logger().error(
                    f"CmdReplay profile '{self.profile_name}' has no valid command samples; start rejected."
                )
            return False
        self.reset_playback()
        if sim_time_sec is not None:
            sim_time_sec = float(sim_time_sec)
            if np.isfinite(sim_time_sec):
                self._run_start_sim_time = sim_time_sec
                self._last_sim_time = sim_time_sec
                self._sample_time_sec = 0.0
        self._auto_start_pending = False
        self._repeat_reset_requested = False
        self.enabled = True
        self.lifecycle_state = CmdReplayState.PLAYING
        self.node.get_logger().info(f"CmdReplay started pass {self._current_pass + 1}/{self.repeats}.")
        return True

    def stop_playback(self) -> None:
        self.enabled = False
        self._auto_start_pending = False
        self._repeat_reset_requested = False
        self._current_pass = 0
        self.reset_playback()
        self.lifecycle_state = CmdReplayState.STOPPED
        self.node.get_logger().info("CmdReplay stopped.")

    def begin_sequence(self, require_release_after_reset: bool) -> bool:
        self._current_pass = 0
        return self.request_reset(require_release_after_reset)

    def needs_repeat_reset(self) -> bool:
        return self._repeat_reset_requested

    def consume_repeat_reset_request(self) -> bool:
        if not self._repeat_reset_requested:
            return False
        self._repeat_reset_requested = False
        self.request_reset(self._reset_requires_release_after_reset)
        return True

    def playback_status(self) -> str:
        if self.lifecycle_state in (
            CmdReplayState.RESETTING,
            CmdReplayState.ARMED,
            CmdReplayState.COMPLETE,
            CmdReplayState.ERROR,
        ):
            return self.lifecycle_state.value
        if not self.enabled:
            return "stopped"
        if self.times_sec.size == 0 or self.duration_sec <= 0.0:
            return "no_csv"
        if self._sample_time_sec >= self.duration_sec:
            return "complete"
        return "running"

    def set_sim_time(self, sim_time_sec: float) -> None:
        sim_time_sec = float(sim_time_sec)
        if not np.isfinite(sim_time_sec):
            return

        if self._run_start_sim_time is None or self._last_sim_time is None:
            self._run_start_sim_time = sim_time_sec
            self._reported_done = False
            self._last_sim_time = sim_time_sec
            return

        delta_sec = sim_time_sec - self._last_sim_time
        if delta_sec < 0.0 or delta_sec > self.max_sim_time_step_sec:
            if self.enabled and not self._warned_time_jump:
                self.node.get_logger().warn(
                    f"CmdReplay detected simulator time jump ({delta_sec:.3f}s); "
                    "re-anchoring playback timer without consuming CSV time."
                )
                self._warned_time_jump = True
            self._run_start_sim_time = sim_time_sec - self._sample_time_sec
            self._last_sim_time = sim_time_sec
            self._reported_done = False
            return

        self._last_sim_time = sim_time_sec
        if self.enabled:
            self._sample_time_sec = max(0.0, self._sample_time_sec + delta_sec)

    def vehicle_controller(
        self,
        state: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_acc: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        index = self._current_sample_index()
        if index is None:
            return np.zeros(6, dtype=float)
        return self.vehicle_command_at(index)

    def arm_controller(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        q_ref: np.ndarray,
        dq_ref: np.ndarray,
        ddq_ref: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        index = self._current_sample_index()
        if index is None:
            if not self._warned_missing:
                self.node.get_logger().warn(
                    "CmdReplay has no active CSV sample; publishing zero commands."
                )
                self._warned_missing = True
            return np.zeros(self.arm_dof + 1, dtype=float)

        return self.arm_command_at(index)

    def current_sample_index(self):
        return self._current_sample_index()

    def vehicle_command_at(self, index) -> np.ndarray:
        if index is None:
            return np.zeros(6, dtype=float)
        return self.vehicle_commands[int(index)].copy()

    def arm_command_at(self, index) -> np.ndarray:
        if index is None:
            return np.zeros(self.arm_dof + 1, dtype=float)
        return self.arm_commands[int(index)].copy()

    def vehicle_reference_at(self, index) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if index is None:
            return (
                np.zeros(6, dtype=float),
                np.zeros(6, dtype=float),
                np.zeros(6, dtype=float),
            )
        index = int(index)
        return (
            self.vehicle_reference_pose[index].copy(),
            self.vehicle_reference_vel[index].copy(),
            self.vehicle_reference_acc[index].copy(),
        )

    def arm_reference_at(self, index) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if index is None:
            return (
                np.zeros(self.arm_dof + 1, dtype=float),
                np.zeros(self.arm_dof + 1, dtype=float),
                np.zeros(self.arm_dof + 1, dtype=float),
            )
        index = int(index)
        return (
            self.arm_reference_position[index].copy(),
            self.arm_reference_velocity[index].copy(),
            self.arm_reference_acceleration[index].copy(),
        )

    def _current_sample_index(self):
        if not self.enabled:
            return None
        if self.times_sec.size == 0 or self.duration_sec <= 0.0:
            return None

        t = self._sample_time_sec

        if t < self.duration_sec:
            sample_t = t
        else:
            if self._current_pass + 1 < self.repeats:
                if not self._reported_done:
                    self.node.get_logger().info(
                        f"CmdReplay completed pass {self._current_pass + 1}/{self.repeats}; requesting reset."
                    )
                    self._reported_done = True
                self._current_pass += 1
                self.enabled = False
                self.lifecycle_state = CmdReplayState.RESETTING
                self._repeat_reset_requested = True
            else:
                if not self._reported_done:
                    self.node.get_logger().info("CmdReplay completed; publishing zero commands.")
                    self._reported_done = True
                self._current_pass = 0
                self.enabled = False
                self.lifecycle_state = CmdReplayState.COMPLETE
            return None

        index = int(np.searchsorted(self.times_sec, sample_t, side="right") - 1)
        return max(0, min(index, self.times_sec.size - 1))
