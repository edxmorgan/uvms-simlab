import csv
import json
import math
import os
import time
from typing import Any, Dict, List

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node

from simlab.alpha_reach import Params as alpha
from simlab.frame_utils import PoseX
from simlab.uvms_backend import UVMSBackendCore


class ExecutionBenchmarkRunner(Node):
    def __init__(self) -> None:
        super().__init__("execution_benchmark_runner")

        self.declare_parameter("robot_description", "")
        self.declare_parameter("robots_prefix", ["robot_1_"])
        self.declare_parameter("controller_name", "pid")
        self.declare_parameter("planner_name", "Bitstar")
        self.declare_parameter("goal_count", 3)
        self.declare_parameter("goal_seed", 11)
        self.declare_parameter("goal_file", "")
        self.declare_parameter("time_limit", 1.0)
        self.declare_parameter("robot_collision_radius", 0.574)
        self.declare_parameter("planning_timeout_sec", 15.0)
        self.declare_parameter("goal_timeout_sec", 75.0)
        self.declare_parameter("goal_tolerance_m", 0.40)
        self.declare_parameter("settle_time_sec", 1.0)
        self.declare_parameter("stop_speed_threshold", 0.05)
        self.declare_parameter("sample_min_clearance", 0.10)
        self.declare_parameter("sample_z_min", -4.0)
        self.declare_parameter("sample_z_max", -1.0)
        self.declare_parameter("min_goal_distance", 4.0)
        self.declare_parameter("max_goal_distance", 8.0)
        self.declare_parameter("max_sampling_attempts", 4000)
        self.declare_parameter("startup_delay_sec", 12.0)
        self.declare_parameter("wait_for_server_timeout_sec", 60.0)
        self.declare_parameter("inter_goal_pause_sec", 2.0)
        self.declare_parameter("abort_on_failure", False)
        self.declare_parameter("output_path", "/tmp/simlab_execution_benchmark_results.json")

        self.robot_description = str(self.get_parameter("robot_description").value)
        if not self.robot_description:
            raise RuntimeError("execution_benchmark_runner requires the robot_description parameter.")

        self.robots_prefix = list(self.get_parameter("robots_prefix").value)
        if len(self.robots_prefix) != 1:
            raise RuntimeError(
                f"execution_benchmark_runner currently supports exactly one robot, got {self.robots_prefix}."
            )

        self.controller_name = str(self.get_parameter("controller_name").value)
        self.planner_name = str(self.get_parameter("planner_name").value)
        self.goal_count = int(self.get_parameter("goal_count").value)
        self.goal_seed = int(self.get_parameter("goal_seed").value)
        self.goal_file = str(self.get_parameter("goal_file").value)
        self.time_limit = float(self.get_parameter("time_limit").value)
        self.robot_collision_radius = float(self.get_parameter("robot_collision_radius").value)
        self.planning_timeout_sec = float(self.get_parameter("planning_timeout_sec").value)
        self.goal_timeout_sec = float(self.get_parameter("goal_timeout_sec").value)
        self.goal_tolerance_m = float(self.get_parameter("goal_tolerance_m").value)
        self.settle_time_sec = float(self.get_parameter("settle_time_sec").value)
        self.stop_speed_threshold = float(self.get_parameter("stop_speed_threshold").value)
        self.sample_min_clearance = float(self.get_parameter("sample_min_clearance").value)
        self.sample_z_min = float(self.get_parameter("sample_z_min").value)
        self.sample_z_max = float(self.get_parameter("sample_z_max").value)
        self.min_goal_distance = float(self.get_parameter("min_goal_distance").value)
        self.max_goal_distance = float(self.get_parameter("max_goal_distance").value)
        self.max_sampling_attempts = int(self.get_parameter("max_sampling_attempts").value)
        self.startup_delay_sec = float(self.get_parameter("startup_delay_sec").value)
        self.wait_for_server_timeout_sec = float(self.get_parameter("wait_for_server_timeout_sec").value)
        self.inter_goal_pause_sec = float(self.get_parameter("inter_goal_pause_sec").value)
        self.abort_on_failure = bool(self.get_parameter("abort_on_failure").value)
        self.output_path = str(self.get_parameter("output_path").value)

        self.world_frame = "world"
        self.vehicle_target_frame = "vehicle_marker_frame"
        self.arm_base_target_frame = "arm_base_marker_frame"
        self.world_endeffector_target_frame = "world_endeffector_marker_frame"
        arm_base_pose = PoseX.from_pose(
            xyz=alpha.base_T0_new[0:3],
            rot=alpha.base_T0_new[3:6],
            rot_rep="euler_xyz",
            frame="NWU",
        ).get_pose_as_Pose_msg()

        self.uvms_backend = UVMSBackendCore(
            self,
            self.robot_description,
            arm_base_pose,
            self.vehicle_target_frame,
            self.arm_base_target_frame,
            self.world_frame,
            self.world_endeffector_target_frame,
            alpha,
        )
        self.uvms_backend.fcl_world.set_robot_collision_radius(self.robot_collision_radius)
        self.robot = self.uvms_backend.robot_selected
        self.robot.set_controller(self.controller_name)
        self.robot.set_planner(self.planner_name)

        self._loaded_goals = self._load_goals()
        self._goal_cursor = 0
        self._rng = np.random.default_rng(self.goal_seed)
        self.results: List[Dict[str, Any]] = []

        self._phase = "startup"
        self._startup_started_at = time.perf_counter()
        self._cooldown_until = 0.0
        self._server_ready_checked = False
        self._current_run: Dict[str, Any] | None = None
        self._plan_sent_at = 0.0
        self._plan_completed_at = 0.0
        self._goal_reached_since: float | None = None
        self._error_history: List[float] = []
        self._actual_positions: List[np.ndarray] = []
        self._min_actual_clearance_m: float | None = None
        self._poll_timer = self.create_timer(0.10, self._tick)

        self.get_logger().info(
            f"Prepared execution benchmark controller={self.controller_name} planner={self.planner_name} "
            f"goals={self.goal_count if not self._loaded_goals else len(self._loaded_goals)}"
        )

    def _safe_float(self, value: Any) -> float | None:
        try:
            value = float(value)
        except Exception:
            return None
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    def _vector_to_list(self, value: np.ndarray | List[float]) -> List[float]:
        return [float(v) for v in np.asarray(value, dtype=float).reshape(-1).tolist()]

    def _pose_to_xyz_quat(self, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
        xyz = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
        quat = np.array(
            [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z],
            dtype=float,
        )
        return xyz, quat

    def _goal_pose_msg(self, xyz: np.ndarray, quat_wxyz: np.ndarray) -> Pose:
        pose = Pose()
        pose.position.x = float(xyz[0])
        pose.position.y = float(xyz[1])
        pose.position.z = float(xyz[2])
        pose.orientation.w = float(quat_wxyz[0])
        pose.orientation.x = float(quat_wxyz[1])
        pose.orientation.y = float(quat_wxyz[2])
        pose.orientation.z = float(quat_wxyz[3])
        return pose

    def _get_current_world_pose(self) -> Pose | None:
        return self.robot._pose_from_state_in_frame(self.world_frame)

    def _get_current_world_xyz(self) -> np.ndarray | None:
        pose = self._get_current_world_pose()
        if pose is None:
            return None
        xyz, _ = self._pose_to_xyz_quat(pose)
        return xyz

    def _load_goals(self) -> List[Dict[str, np.ndarray]]:
        if not self.goal_file:
            return []
        with open(self.goal_file, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, list) or not loaded:
            raise RuntimeError("goal_file must contain a non-empty JSON list.")

        goals: List[Dict[str, np.ndarray]] = []
        for idx, goal in enumerate(loaded):
            goals.append(
                {
                    "goal_id": str(goal.get("goal_id", f"goal_{idx:03d}")),
                    "goal_xyz": np.asarray(goal["goal_xyz"], dtype=float).reshape(3),
                    "goal_quat_wxyz": np.asarray(
                        goal.get("goal_quat_wxyz", [1.0, 0.0, 0.0, 0.0]),
                        dtype=float,
                    ).reshape(4),
                }
            )
        return goals

    def _sample_goal(self, start_xyz: np.ndarray) -> Dict[str, np.ndarray]:
        x_min, x_max, y_min, y_max, env_z_min, env_z_max = self.uvms_backend.fcl_world.env_xyz_bounds

        xy_margin = self.robot_collision_radius + self.sample_min_clearance
        x_low = x_min + xy_margin
        x_high = x_max - xy_margin
        y_low = y_min + xy_margin
        y_high = y_max - xy_margin
        z_low = max(env_z_min + self.robot_collision_radius + self.sample_min_clearance, self.sample_z_min)
        z_high = min(env_z_max - 1e-3, self.sample_z_max)
        if not (x_low < x_high and y_low < y_high and z_low < z_high):
            raise RuntimeError("Sampling bounds are invalid; adjust sample_z_* or clearance parameters.")

        for attempt in range(1, self.max_sampling_attempts + 1):
            goal_xyz = np.array(
                [
                    self._rng.uniform(x_low, x_high),
                    self._rng.uniform(y_low, y_high),
                    self._rng.uniform(z_low, z_high),
                ],
                dtype=float,
            )
            if np.linalg.norm(goal_xyz - start_xyz) < self.min_goal_distance:
                continue
            if self.max_goal_distance > 0.0 and np.linalg.norm(goal_xyz - start_xyz) > self.max_goal_distance:
                continue
            if self.uvms_backend.fcl_world.min_distance_xyz(goal_xyz) < self.sample_min_clearance:
                continue
            return {
                "goal_id": f"goal_{len(self.results):03d}",
                "goal_xyz": goal_xyz,
                "goal_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            }

        raise RuntimeError(
            f"Failed to sample a valid goal after {self.max_sampling_attempts} attempts "
            f"from start={start_xyz.tolist()}."
        )

    def _next_goal(self, start_xyz: np.ndarray) -> Dict[str, np.ndarray] | None:
        if self._loaded_goals:
            if self._goal_cursor >= len(self._loaded_goals):
                return None
            goal = self._loaded_goals[self._goal_cursor]
            self._goal_cursor += 1
            return goal

        if self._goal_cursor >= self.goal_count:
            return None
        self._goal_cursor += 1
        return self._sample_goal(start_xyz)

    def _start_next_run(self) -> None:
        pose_now = self._get_current_world_pose()
        if pose_now is None:
            self.get_logger().warn("Current world pose unavailable; delaying next execution run.")
            return

        start_xyz, start_quat_wxyz = self._pose_to_xyz_quat(pose_now)
        goal = self._next_goal(start_xyz)
        if goal is None:
            self._finish()
            return

        goal_xyz = np.asarray(goal["goal_xyz"], dtype=float)
        goal_quat_wxyz = np.asarray(goal["goal_quat_wxyz"], dtype=float)
        goal_pose = self._goal_pose_msg(goal_xyz, goal_quat_wxyz)
        self.uvms_backend.target_vehicle_pose = goal_pose

        self.robot.set_controller(self.controller_name)
        self.robot.set_planner(self.planner_name)
        self.robot.abrupt_planner_stop()

        self._current_run = {
            "goal_id": goal["goal_id"],
            "controller_name": self.controller_name,
            "planner_name": self.planner_name,
            "start_xyz": self._vector_to_list(start_xyz),
            "start_quat_wxyz": self._vector_to_list(start_quat_wxyz),
            "goal_xyz": self._vector_to_list(goal_xyz),
            "goal_quat_wxyz": self._vector_to_list(goal_quat_wxyz),
        }
        self._error_history = []
        self._actual_positions = [start_xyz.copy()]
        self._min_actual_clearance_m = self._safe_float(self.uvms_backend.fcl_world.min_distance_xyz(start_xyz))
        self._goal_reached_since = None
        self._plan_completed_at = 0.0
        self._plan_sent_at = time.perf_counter()

        sent = self.robot.plan_vehicle_trajectory_action(
            goal_pose=goal_pose,
            time_limit=self.time_limit,
            robot_collision_radius=self.robot_collision_radius,
        )
        if not sent:
            self._finalize_current_run(success=False, message="Planner action request was not sent.")
            return

        self._phase = "waiting_for_plan"
        self.get_logger().info(
            f"Started execution run {self._current_run['goal_id']} "
            f"start={self._current_run['start_xyz']} goal={self._current_run['goal_xyz']}"
        )

    def _update_execution_samples(self, now: float) -> None:
        if self._current_run is None:
            return

        current_xyz = self._get_current_world_xyz()
        if current_xyz is None:
            return

        goal_xyz = np.asarray(self._current_run["goal_xyz"], dtype=float)
        goal_error = float(np.linalg.norm(current_xyz - goal_xyz))
        self._error_history.append(goal_error)

        if not self._actual_positions or np.linalg.norm(current_xyz - self._actual_positions[-1]) > 1e-6:
            self._actual_positions.append(current_xyz.copy())

        clearance = self._safe_float(self.uvms_backend.fcl_world.min_distance_xyz(current_xyz))
        if clearance is not None:
            if self._min_actual_clearance_m is None:
                self._min_actual_clearance_m = clearance
            else:
                self._min_actual_clearance_m = min(self._min_actual_clearance_m, clearance)

        body_vel = np.asarray(self.robot.get_state()["body_vel"][0:3], dtype=float)
        speed = float(np.linalg.norm(body_vel))
        if goal_error <= self.goal_tolerance_m and speed <= self.stop_speed_threshold:
            if self._goal_reached_since is None:
                self._goal_reached_since = now
        else:
            self._goal_reached_since = None

    def _actual_path_length(self) -> float | None:
        if len(self._actual_positions) < 2:
            return 0.0
        diffs = np.diff(np.asarray(self._actual_positions, dtype=float), axis=0)
        return float(np.linalg.norm(diffs, axis=1).sum())

    def _finalize_current_run(self, *, success: bool, message: str) -> None:
        if self._current_run is None:
            return

        finished_at = time.perf_counter()
        self._update_execution_samples(finished_at)
        final_goal_error_m = self._error_history[-1] if self._error_history else None
        rms_goal_error_m = None
        max_goal_error_m = None
        if self._error_history:
            error_arr = np.asarray(self._error_history, dtype=float)
            rms_goal_error_m = float(np.sqrt(np.mean(np.square(error_arr))))
            max_goal_error_m = float(np.max(error_arr))

        planned_result = None if self.robot.planner is None else self.robot.planner.planned_result
        record = dict(self._current_run)
        record.update(
            {
                "success": bool(success),
                "message": str(message),
                "plan_wall_time_sec": None if self._plan_completed_at == 0.0 else self._plan_completed_at - self._plan_sent_at,
                "execution_time_sec": None if self._plan_completed_at == 0.0 else finished_at - self._plan_completed_at,
                "end_to_end_time_sec": finished_at - self._plan_sent_at,
                "planned_waypoint_count": 0 if not planned_result else int(planned_result.get("count", 0)),
                "planned_geom_length": None if not planned_result else self._safe_float(planned_result.get("geom_length")),
                "planned_path_length_cost": None if not planned_result else self._safe_float(planned_result.get("path_length_cost")),
                "actual_path_length_m": self._actual_path_length(),
                "final_goal_error_m": self._safe_float(final_goal_error_m),
                "rms_goal_error_m": self._safe_float(rms_goal_error_m),
                "max_goal_error_m": self._safe_float(max_goal_error_m),
                "min_actual_clearance_m": self._safe_float(self._min_actual_clearance_m),
            }
        )
        self.results.append(record)

        self.get_logger().info(
            f"Finished {record['goal_id']}: success={record['success']} "
            f"plan={record['plan_wall_time_sec']}s end_to_end={record['end_to_end_time_sec']:.3f}s "
            f"final_err={record['final_goal_error_m']} message='{record['message']}'"
        )

        self.robot.abrupt_planner_stop()
        self._current_run = None
        if self.abort_on_failure and not success:
            self.get_logger().warn("Aborting benchmark after first failed goal to keep chained-goal results comparable.")
            self._finish()
            return
        self._phase = "cooldown"
        self._cooldown_until = finished_at + self.inter_goal_pause_sec

    def _summarize_results(self) -> Dict[str, Any]:
        def _mean_or_none(field: str, rows: List[Dict[str, Any]]) -> float | None:
            values = [float(row[field]) for row in rows if row.get(field) is not None]
            if not values:
                return None
            return float(np.mean(values))

        success_rows = [row for row in self.results if row["success"]]
        return {
            "controller_name": self.controller_name,
            "planner_name": self.planner_name,
            "runs": len(self.results),
            "successes": len(success_rows),
            "success_rate": 0.0 if not self.results else len(success_rows) / len(self.results),
            "mean_plan_wall_time_sec": _mean_or_none("plan_wall_time_sec", self.results),
            "mean_execution_time_sec": _mean_or_none("execution_time_sec", success_rows),
            "mean_end_to_end_time_sec": _mean_or_none("end_to_end_time_sec", self.results),
            "mean_final_goal_error_m": _mean_or_none("final_goal_error_m", self.results),
            "mean_rms_goal_error_m": _mean_or_none("rms_goal_error_m", success_rows),
            "mean_actual_path_length_m": _mean_or_none("actual_path_length_m", success_rows),
            "mean_min_actual_clearance_m": _mean_or_none("min_actual_clearance_m", success_rows),
        }

    def _write_results(self) -> None:
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        summary = self._summarize_results()
        payload = {
            "parameters": {
                "controller_name": self.controller_name,
                "planner_name": self.planner_name,
                "goal_count": self.goal_count if not self._loaded_goals else len(self._loaded_goals),
                "goal_seed": self.goal_seed,
                "time_limit": self.time_limit,
                "robot_collision_radius": self.robot_collision_radius,
                "planning_timeout_sec": self.planning_timeout_sec,
                "goal_timeout_sec": self.goal_timeout_sec,
                "goal_tolerance_m": self.goal_tolerance_m,
                "settle_time_sec": self.settle_time_sec,
                "sample_min_clearance": self.sample_min_clearance,
                "sample_z_min": self.sample_z_min,
                "sample_z_max": self.sample_z_max,
                "min_goal_distance": self.min_goal_distance,
                "max_goal_distance": self.max_goal_distance,
                "abort_on_failure": self.abort_on_failure,
            },
            "summary": summary,
            "results": self.results,
        }
        with open(self.output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        csv_path = os.path.splitext(self.output_path)[0] + ".csv"
        fieldnames = [
            "goal_id",
            "controller_name",
            "planner_name",
            "success",
            "message",
            "plan_wall_time_sec",
            "execution_time_sec",
            "end_to_end_time_sec",
            "planned_waypoint_count",
            "planned_geom_length",
            "planned_path_length_cost",
            "actual_path_length_m",
            "final_goal_error_m",
            "rms_goal_error_m",
            "max_goal_error_m",
            "min_actual_clearance_m",
            "start_xyz",
            "start_quat_wxyz",
            "goal_xyz",
            "goal_quat_wxyz",
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in self.results:
                flat_row = dict(row)
                flat_row["start_xyz"] = json.dumps(flat_row["start_xyz"])
                flat_row["start_quat_wxyz"] = json.dumps(flat_row["start_quat_wxyz"])
                flat_row["goal_xyz"] = json.dumps(flat_row["goal_xyz"])
                flat_row["goal_quat_wxyz"] = json.dumps(flat_row["goal_quat_wxyz"])
                writer.writerow(flat_row)

        self.get_logger().info(f"Wrote execution benchmark JSON to {self.output_path}")
        self.get_logger().info(f"Wrote execution benchmark CSV to {csv_path}")
        self.get_logger().info(
            f"Execution summary: success_rate={summary['success_rate']:.3f}, "
            f"mean_plan_wall_time_sec={summary['mean_plan_wall_time_sec']}, "
            f"mean_execution_time_sec={summary['mean_execution_time_sec']}, "
            f"mean_final_goal_error_m={summary['mean_final_goal_error_m']}"
        )

    def _finish(self) -> None:
        self._write_results()
        rclpy.shutdown()

    def _tick(self) -> None:
        now = time.perf_counter()

        if self._phase == "startup":
            if now - self._startup_started_at < self.startup_delay_sec:
                return
            if self.robot.get_state()["status"] != "active":
                self.get_logger().info("Waiting for active robot state before starting execution benchmark.")
                return
            if self._get_current_world_pose() is None:
                self.get_logger().info("Waiting for world-frame pose before starting execution benchmark.")
                return
            if not self._server_ready_checked:
                self.get_logger().info(
                    f"Waiting for planner action server for up to {self.wait_for_server_timeout_sec:.1f}s."
                )
                if not self.robot.planner_action_client.wait_for_server(self.wait_for_server_timeout_sec):
                    self.get_logger().error("Planner action server did not become ready in time.")
                    self._finish()
                    return
                self._server_ready_checked = True
            self._phase = "cooldown"
            self._cooldown_until = now
            return

        if self._phase == "cooldown":
            if now < self._cooldown_until:
                return
            self._start_next_run()
            return

        if self._phase == "waiting_for_plan":
            if self._current_run is None:
                self._phase = "cooldown"
                self._cooldown_until = now
                return
            if now - self._plan_sent_at > self.planning_timeout_sec:
                self._finalize_current_run(success=False, message="Planner result timeout.")
                return

            planned_result = None if self.robot.planner is None else self.robot.planner.planned_result
            if planned_result is None:
                return

            self._plan_completed_at = now
            if not planned_result.get("is_success", False):
                self._finalize_current_run(
                    success=False,
                    message=str(planned_result.get("message", "Planner failed.")),
                )
                return

            self._phase = "executing"
            return

        if self._phase == "executing":
            if self._current_run is None:
                self._phase = "cooldown"
                self._cooldown_until = now
                return

            self._update_execution_samples(now)
            if self._goal_reached_since is not None and (now - self._goal_reached_since) >= self.settle_time_sec:
                self._finalize_current_run(success=True, message="Reached goal tolerance.")
                return
            if now - self._plan_sent_at > self.goal_timeout_sec:
                self._finalize_current_run(success=False, message="Goal timeout.")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExecutionBenchmarkRunner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()