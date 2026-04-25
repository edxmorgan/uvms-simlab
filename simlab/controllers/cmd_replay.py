import json
import os
from enum import Enum
from pathlib import Path

import ament_index_python
import numpy as np
from rclpy.node import Node
from ros2_control_blue_reach_5.srv import ResetSimUvms

from simlab.controllers.base import ControllerTemplate


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

    DEFAULT_PROFILE = "phase1_payload_0p0kg"
    DEFAULT_TIME_COLUMN = "time_sec"
    DEFAULT_VEHICLE_COLUMNS = "vehicle_fx,vehicle_fy,vehicle_fz,vehicle_tx,vehicle_ty,vehicle_tz"
    DEFAULT_ARM_COLUMNS = "tau_axis_e,tau_axis_d,tau_axis_c,tau_axis_b,tau_axis_a"

    def __init__(self, node: Node, arm_dof: int = 4):
        super().__init__(node, arm_dof)
        self.profiles_root = Path(ament_index_python.get_package_share_directory("simlab")) / "csv_playback"
        self.profile_name = str(self._get_or_declare_parameter("cmd_replay_profile", self.DEFAULT_PROFILE))
        self.csv_path = ""
        self.config_path = ""
        self.time_column = self.DEFAULT_TIME_COLUMN
        self.vehicle_columns = self._parse_columns(self.DEFAULT_VEHICLE_COLUMNS, expected_size=6)
        self.arm_columns = self._parse_columns(self.DEFAULT_ARM_COLUMNS, expected_size=self.arm_dof + 1)
        self.repeats = 1
        self.loop = False
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
        self._reset_hold_commands = True
        self._current_pass = 0
        self._repeat_reset_requested = False
        self.times_sec = np.array([], dtype=float)
        self.vehicle_commands = np.zeros((0, 6), dtype=float)
        self.arm_commands = np.zeros((0, self.arm_dof + 1), dtype=float)
        self.duration_sec = 0.0
        self.reset_config = self._default_reset_config()

        self.load_profile(self.profile_name)

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
        csv_name = str(manifest.get("csv", "commands.csv"))

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
        self.loop = bool(playback.get("loop", False))
        self.reset_config = self._merge_reset_config(
            self._default_reset_config(),
            manifest.get("reset", {}),
        )
        self.times_sec = np.array([], dtype=float)
        self.vehicle_commands = np.zeros((0, 6), dtype=float)
        self.arm_commands = np.zeros((0, self.arm_dof + 1), dtype=float)
        self.duration_sec = 0.0

        self._load_csv(self.csv_path)
        self.node.get_logger().info(f"CmdReplay selected profile '{profile_name}'.")
        return self.times_sec.size > 0

    def _default_reset_config(self) -> dict:
        return {
            "reset_manipulator": True,
            "reset_vehicle": True,
            "hold_commands": True,
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
            "dynamics": {
                "gravity": 0.0,
                "payload_mass": 0.0,
                "payload_inertia": [0.0, 0.0, 0.0],
            },
        }

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
        dynamics = config.get("dynamics", {})

        request = ResetSimUvms.Request()
        request.reset_manipulator = bool(config.get("reset_manipulator", True))
        request.reset_vehicle = bool(config.get("reset_vehicle", True))
        request.hold_commands = bool(config.get("hold_commands", True))
        request.use_manipulator_state = bool(manipulator.get("enabled", True))
        request.manipulator_position = self._vector(manipulator.get("position"), 5)
        request.manipulator_velocity = self._vector(manipulator.get("velocity"), 5)
        request.use_vehicle_state = bool(vehicle.get("enabled", True))
        request.vehicle_pose = self._vector(vehicle.get("pose"), 6)
        request.vehicle_twist = self._vector(vehicle.get("twist"), 6)
        request.vehicle_wrench = self._vector(vehicle.get("wrench"), 6)
        request.gravity = float(dynamics.get("gravity", 0.0))
        request.payload_mass = float(dynamics.get("payload_mass", dynamics.get("mass", 0.0)))

        inertia = dynamics.get("payload_inertia", dynamics.get("inertia", [0.0, 0.0, 0.0]))
        inertia = self._vector(inertia, 3)
        request.payload_ixx = inertia[0]
        request.payload_iyy = inertia[1]
        request.payload_izz = inertia[2]
        return request

    def reset_mode(self) -> str:
        return str(self.reset_config.get("mode", "sim_state"))

    def controller_settle_config(self) -> dict:
        settle = self.reset_config.get("settle", {})
        if not isinstance(settle, dict):
            settle = {}
        return {
            "controller": str(settle.get("controller", "PID")),
            "position_tolerance": float(settle.get("position_tolerance", 0.03)),
            "velocity_tolerance": float(settle.get("velocity_tolerance", 0.03)),
            "vehicle_position_tolerance": float(settle.get("vehicle_position_tolerance", 0.08)),
            "vehicle_velocity_tolerance": float(settle.get("vehicle_velocity_tolerance", 0.05)),
            "timeout_sec": float(settle.get("timeout_sec", 20.0)),
        }

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

        if missing:
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

        order = np.argsort(self.times_sec)
        self.times_sec = self.times_sec[order]
        self.vehicle_commands = self.vehicle_commands[order]
        self.arm_commands = self.arm_commands[order]

        finite = (
            np.isfinite(self.times_sec)
            & np.all(np.isfinite(self.vehicle_commands), axis=1)
            & np.all(np.isfinite(self.arm_commands), axis=1)
        )
        self.times_sec = self.times_sec[finite]
        self.vehicle_commands = self.vehicle_commands[finite]
        self.arm_commands = self.arm_commands[finite]

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
            f"loop={self.loop}, enabled={self.enabled}."
        )

    def reset_playback(self) -> None:
        self._run_start_sim_time = None
        self._last_sim_time = None
        self._sample_time_sec = 0.0
        self._reported_done = False
        self._warned_time_jump = False

    def request_reset(self, hold_commands: bool) -> None:
        self.enabled = False
        self._auto_start_pending = False
        self._repeat_reset_requested = False
        self._reset_hold_commands = bool(hold_commands)
        self.reset_playback()
        self.lifecycle_state = CmdReplayState.RESETTING
        self.node.get_logger().info("CmdReplay reset requested.")

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

    def start_playback(self, sim_time_sec: float | None = None) -> None:
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

    def stop_playback(self) -> None:
        self.enabled = False
        self._auto_start_pending = False
        self._repeat_reset_requested = False
        self._current_pass = 0
        self.reset_playback()
        self.lifecycle_state = CmdReplayState.STOPPED
        self.node.get_logger().info("CmdReplay stopped.")

    def begin_sequence(self, hold_commands: bool) -> None:
        self._current_pass = 0
        self.request_reset(hold_commands)

    def needs_repeat_reset(self) -> bool:
        return self._repeat_reset_requested

    def consume_repeat_reset_request(self) -> bool:
        if not self._repeat_reset_requested:
            return False
        self._repeat_reset_requested = False
        self.request_reset(self._reset_hold_commands)
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
        if not self.loop and self._sample_time_sec >= self.duration_sec:
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
        return self.vehicle_commands[index].copy()

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

        return self.arm_commands[index].copy()

    def _current_sample_index(self):
        if not self.enabled:
            return None
        if self.times_sec.size == 0 or self.duration_sec <= 0.0:
            return None

        t = self._sample_time_sec

        if self.loop:
            if self.repeats != 1:
                self.node.get_logger().warn(
                    "CmdReplay loop=true ignores repeats because loop playback does not reset between cycles."
                )
            self.repeats = 1
            sample_t = t % self.duration_sec
        elif t < self.duration_sec:
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
