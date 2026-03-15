import csv
import json
import math
import os
import time
from typing import Any, Dict, List

import numpy as np
import rclpy
from rclpy.node import Node

from simlab.fcl_checker import FCLWorld
from simlab.planner_action_client import PlannerActionClient


class PlannerBenchmarkRunner(Node):
    def __init__(self) -> None:
        super().__init__("planner_benchmark_runner")

        self.declare_parameter("robot_description", "")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("planner_names", "Bitstar,RRTstar")
        self.declare_parameter("scenario_count", 6)
        self.declare_parameter("scenario_seed", 7)
        self.declare_parameter("time_limit", 1.0)
        self.declare_parameter("robot_collision_radius", 0.574)
        self.declare_parameter("sample_min_clearance", 0.10)
        self.declare_parameter("sample_z_min", -4.0)
        self.declare_parameter("sample_z_max", -1.0)
        self.declare_parameter("min_goal_distance", 4.0)
        self.declare_parameter("max_sampling_attempts", 4000)
        self.declare_parameter("startup_delay_sec", 12.0)
        self.declare_parameter("wait_for_server_timeout_sec", 60.0)
        self.declare_parameter("output_path", "/tmp/simlab_planner_benchmark_results.json")
        self.declare_parameter("scenario_file", "")

        self.robot_description = str(self.get_parameter("robot_description").value)
        if not self.robot_description:
            raise RuntimeError("planner_benchmark_runner requires the robot_description parameter.")

        self.world_frame = str(self.get_parameter("world_frame").value)
        self.planner_names = self._parse_csv_list(str(self.get_parameter("planner_names").value))
        self.scenario_count = int(self.get_parameter("scenario_count").value)
        self.scenario_seed = int(self.get_parameter("scenario_seed").value)
        self.time_limit = float(self.get_parameter("time_limit").value)
        self.robot_collision_radius = float(self.get_parameter("robot_collision_radius").value)
        self.sample_min_clearance = float(self.get_parameter("sample_min_clearance").value)
        self.sample_z_min = float(self.get_parameter("sample_z_min").value)
        self.sample_z_max = float(self.get_parameter("sample_z_max").value)
        self.min_goal_distance = float(self.get_parameter("min_goal_distance").value)
        self.max_sampling_attempts = int(self.get_parameter("max_sampling_attempts").value)
        self.startup_delay_sec = float(self.get_parameter("startup_delay_sec").value)
        self.wait_for_server_timeout_sec = float(self.get_parameter("wait_for_server_timeout_sec").value)
        self.output_path = str(self.get_parameter("output_path").value)
        self.scenario_file = str(self.get_parameter("scenario_file").value)

        self.fcl_world = FCLWorld(
            urdf_string=self.robot_description,
            world_frame=self.world_frame,
            vehicle_radius=self.robot_collision_radius,
        )
        self.fcl_world.set_robot_collision_radius(self.robot_collision_radius)

        self.scenarios = self._load_or_generate_scenarios()
        self.run_queue = self._build_run_queue()
        self.results: List[Dict[str, Any]] = []
        self._current_run: Dict[str, Any] | None = None
        self._current_started_at = 0.0

        self.client = PlannerActionClient(self, action_name="planner", on_result=self._on_result)
        self._startup_timer = self.create_timer(self.startup_delay_sec, self._start_benchmark_once)

        self.get_logger().info(
            f"Prepared {len(self.scenarios)} scenarios and {len(self.run_queue)} planner runs. "
            f"Planners={self.planner_names}"
        )

    def _parse_csv_list(self, raw_value: str) -> List[str]:
        items = [item.strip() for item in raw_value.split(",") if item.strip()]
        if not items:
            raise RuntimeError("planner_names must contain at least one planner.")
        return items

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

    def _load_or_generate_scenarios(self) -> List[Dict[str, Any]]:
        if self.scenario_file:
            with open(self.scenario_file, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, list) or not loaded:
                raise RuntimeError("scenario_file must contain a non-empty JSON list.")
            scenarios = []
            for idx, scenario in enumerate(loaded):
                start_xyz = np.asarray(scenario["start_xyz"], dtype=float).reshape(3)
                goal_xyz = np.asarray(scenario["goal_xyz"], dtype=float).reshape(3)
                scenarios.append(
                    {
                        "scenario_id": str(scenario.get("scenario_id", f"scenario_{idx:03d}")),
                        "start_xyz": start_xyz,
                        "goal_xyz": goal_xyz,
                        "start_quat_wxyz": np.asarray(scenario.get("start_quat_wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=float).reshape(4),
                        "goal_quat_wxyz": np.asarray(scenario.get("goal_quat_wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=float).reshape(4),
                    }
                )
            return scenarios

        rng = np.random.default_rng(self.scenario_seed)
        x_min, x_max, y_min, y_max, env_z_min, env_z_max = self.fcl_world.env_xyz_bounds

        xy_margin = self.robot_collision_radius + self.sample_min_clearance
        x_low = x_min + xy_margin
        x_high = x_max - xy_margin
        y_low = y_min + xy_margin
        y_high = y_max - xy_margin
        z_low = max(env_z_min + self.robot_collision_radius + self.sample_min_clearance, self.sample_z_min)
        z_high = min(env_z_max - 1e-3, self.sample_z_max)
        if not (x_low < x_high and y_low < y_high and z_low < z_high):
            raise RuntimeError("Sampling bounds are invalid; adjust sample_z_* or clearance parameters.")

        scenarios: List[Dict[str, Any]] = []
        attempts = 0
        while len(scenarios) < self.scenario_count and attempts < self.max_sampling_attempts:
            attempts += 1
            start_xyz = np.array(
                [
                    rng.uniform(x_low, x_high),
                    rng.uniform(y_low, y_high),
                    rng.uniform(z_low, z_high),
                ],
                dtype=float,
            )
            goal_xyz = np.array(
                [
                    rng.uniform(x_low, x_high),
                    rng.uniform(y_low, y_high),
                    rng.uniform(z_low, z_high),
                ],
                dtype=float,
            )
            if np.linalg.norm(goal_xyz - start_xyz) < self.min_goal_distance:
                continue
            if self.fcl_world.min_distance_xyz(start_xyz) < self.sample_min_clearance:
                continue
            if self.fcl_world.min_distance_xyz(goal_xyz) < self.sample_min_clearance:
                continue

            scenarios.append(
                {
                    "scenario_id": f"scenario_{len(scenarios):03d}",
                    "start_xyz": start_xyz,
                    "goal_xyz": goal_xyz,
                    "start_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
                    "goal_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
                }
            )

        if len(scenarios) != self.scenario_count:
            raise RuntimeError(
                f"Only generated {len(scenarios)} valid scenarios after {attempts} attempts "
                f"(requested {self.scenario_count})."
            )

        self.get_logger().info(
            f"Generated {len(scenarios)} collision-free scenarios from bounds "
            f"x[{x_low:.2f}, {x_high:.2f}] y[{y_low:.2f}, {y_high:.2f}] z[{z_low:.2f}, {z_high:.2f}]"
        )
        return scenarios

    def _build_run_queue(self) -> List[Dict[str, Any]]:
        queue: List[Dict[str, Any]] = []
        for scenario in self.scenarios:
            straight_line = float(
                np.linalg.norm(np.asarray(scenario["goal_xyz"], dtype=float) - np.asarray(scenario["start_xyz"], dtype=float))
            )
            for planner_name in self.planner_names:
                queue.append(
                    {
                        "planner_name": planner_name,
                        "scenario_id": scenario["scenario_id"],
                        "start_xyz": np.asarray(scenario["start_xyz"], dtype=float),
                        "goal_xyz": np.asarray(scenario["goal_xyz"], dtype=float),
                        "start_quat_wxyz": np.asarray(scenario["start_quat_wxyz"], dtype=float),
                        "goal_quat_wxyz": np.asarray(scenario["goal_quat_wxyz"], dtype=float),
                        "straight_line_distance": straight_line,
                    }
                )
        return queue

    def _start_benchmark_once(self) -> None:
        self._startup_timer.cancel()
        self.get_logger().info(
            f"Waiting for planner action server for up to {self.wait_for_server_timeout_sec:.1f}s."
        )
        if not self.client.wait_for_server(self.wait_for_server_timeout_sec):
            self.get_logger().error("Planner action server did not become ready in time.")
            self._finish()
            return
        self._run_next()

    def _run_next(self) -> None:
        if not self.run_queue:
            self._finish()
            return

        self._current_run = self.run_queue.pop(0)
        self._current_started_at = time.perf_counter()
        sent = self.client.send_goal(
            start_xyz=self._current_run["start_xyz"],
            start_quat_wxyz=self._current_run["start_quat_wxyz"],
            goal_xyz=self._current_run["goal_xyz"],
            goal_quat_wxyz=self._current_run["goal_quat_wxyz"],
            planner_name=self._current_run["planner_name"],
            time_limit=self.time_limit,
            robot_collision_radius=self.robot_collision_radius,
        )
        if not sent:
            self.get_logger().error(
                f"Failed to submit run {self._current_run['scenario_id']} with planner "
                f"{self._current_run['planner_name']}."
            )
            self.results.append(
                {
                    "planner_name": self._current_run["planner_name"],
                    "scenario_id": self._current_run["scenario_id"],
                    "start_xyz": self._vector_to_list(self._current_run["start_xyz"]),
                    "goal_xyz": self._vector_to_list(self._current_run["goal_xyz"]),
                    "success": False,
                    "wall_time_sec": None,
                    "waypoint_count": 0,
                    "path_length_cost": None,
                    "geom_length": None,
                    "min_path_clearance_m": None,
                    "straight_line_distance": self._current_run["straight_line_distance"],
                    "message": "Goal submission failed.",
                }
            )
            self._current_run = None
            self._run_next()

    def _on_result(self, plan_result: Dict[str, Any]) -> None:
        if self._current_run is None:
            self.get_logger().warn("Received planner result with no active benchmark run.")
            return

        wall_time_sec = time.perf_counter() - self._current_started_at
        success = bool(plan_result.get("is_success", False))
        waypoints = np.asarray(plan_result.get("xyz", []), dtype=float).reshape(-1, 3)
        min_path_clearance = None
        if success and waypoints.size > 0:
            clearances = [self.fcl_world.min_distance_xyz(point) for point in waypoints]
            if clearances:
                min_path_clearance = float(min(clearances))

        record = {
            "planner_name": self._current_run["planner_name"],
            "scenario_id": self._current_run["scenario_id"],
            "start_xyz": self._vector_to_list(self._current_run["start_xyz"]),
            "goal_xyz": self._vector_to_list(self._current_run["goal_xyz"]),
            "success": success,
            "wall_time_sec": wall_time_sec,
            "waypoint_count": int(plan_result.get("count", len(waypoints))),
            "path_length_cost": self._safe_float(plan_result.get("path_length_cost")),
            "geom_length": self._safe_float(plan_result.get("geom_length")),
            "min_path_clearance_m": self._safe_float(min_path_clearance),
            "straight_line_distance": self._current_run["straight_line_distance"],
            "message": str(plan_result.get("message", "")),
        }
        self.results.append(record)

        self.get_logger().info(
            f"Completed {record['scenario_id']} with {record['planner_name']}: "
            f"success={record['success']} wall_time={record['wall_time_sec']:.3f}s "
            f"geom_length={record['geom_length']}"
        )

        self._current_run = None
        self._run_next()

    def _summarize_results(self) -> Dict[str, Dict[str, Any]]:
        def _mean_or_none(values: List[float | None]) -> float | None:
            filtered = [float(v) for v in values if v is not None]
            if not filtered:
                return None
            return float(np.mean(filtered))

        summary: Dict[str, Dict[str, Any]] = {}
        for planner_name in self.planner_names:
            rows = [row for row in self.results if row["planner_name"] == planner_name]
            if not rows:
                continue
            success_rows = [row for row in rows if row["success"]]
            summary[planner_name] = {
                "runs": len(rows),
                "successes": len(success_rows),
                "success_rate": len(success_rows) / len(rows),
                "mean_wall_time_sec": _mean_or_none([row["wall_time_sec"] for row in rows]),
                "mean_geom_length": _mean_or_none([row["geom_length"] for row in success_rows]),
                "mean_min_path_clearance_m": _mean_or_none([row["min_path_clearance_m"] for row in success_rows]),
            }
        return summary

    def _write_results(self) -> Dict[str, Dict[str, Any]]:
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        summary = self._summarize_results()
        payload = {
            "parameters": {
                "planner_names": self.planner_names,
                "scenario_count": len(self.scenarios),
                "scenario_seed": self.scenario_seed,
                "time_limit": self.time_limit,
                "robot_collision_radius": self.robot_collision_radius,
                "sample_min_clearance": self.sample_min_clearance,
                "sample_z_min": self.sample_z_min,
                "sample_z_max": self.sample_z_max,
                "min_goal_distance": self.min_goal_distance,
            },
            "summary": summary,
            "results": self.results,
        }
        with open(self.output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        csv_path = os.path.splitext(self.output_path)[0] + ".csv"
        fieldnames = [
            "planner_name",
            "scenario_id",
            "success",
            "wall_time_sec",
            "waypoint_count",
            "path_length_cost",
            "geom_length",
            "min_path_clearance_m",
            "straight_line_distance",
            "message",
            "start_xyz",
            "goal_xyz",
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.results:
                flat_row = dict(row)
                flat_row["start_xyz"] = json.dumps(flat_row["start_xyz"])
                flat_row["goal_xyz"] = json.dumps(flat_row["goal_xyz"])
                writer.writerow(flat_row)

        self.get_logger().info(f"Wrote benchmark JSON to {self.output_path}")
        self.get_logger().info(f"Wrote benchmark CSV to {csv_path}")
        for planner_name, planner_summary in summary.items():
            self.get_logger().info(
                f"{planner_name}: success_rate={planner_summary['success_rate']:.3f}, "
                f"mean_wall_time_sec={planner_summary['mean_wall_time_sec']}, "
                f"mean_geom_length={planner_summary['mean_geom_length']}, "
                f"mean_min_path_clearance_m={planner_summary['mean_min_path_clearance_m']}"
            )
        return summary

    def _finish(self) -> None:
        self._write_results()
        rclpy.shutdown()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlannerBenchmarkRunner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
