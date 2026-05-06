from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from scipy.spatial.transform import Rotation

from ros2_control_blue_reach_5.msg import DynamicObstacle, DynamicObstacleArray


@dataclass(frozen=True)
class DynamicObstacleState:
    obstacle_id: str
    center_world: np.ndarray
    obstacle_to_world_rotation: np.ndarray
    world_to_obstacle_rotation: np.ndarray
    linear_velocity_world: np.ndarray
    angular_velocity_world: np.ndarray
    collision_type: int
    collision_dimensions: tuple[float, ...]


@dataclass(frozen=True)
class DynamicClearance:
    obstacle_id: str
    distance_m: float


class DynamicWorldModel:
    """Shared dynamic-obstacle snapshot for planning, visualization, and safety checks."""

    def __init__(
        self,
        node: Node,
        *,
        world_frame: str,
        robot_radius_provider: Callable[[], float],
        topic: str = "/dynamic_obstacles",
    ):
        self.node = node
        self.world_frame = world_frame
        self.robot_radius_provider = robot_radius_provider
        self.obstacles: dict[str, DynamicObstacleState] = {}
        self._warned_frame_mismatch = False

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.subscription = self.node.create_subscription(
            DynamicObstacleArray,
            topic,
            self._obstacles_callback,
            qos,
        )

    def close(self) -> None:
        self.obstacles.clear()
        if self.subscription is not None:
            self.node.destroy_subscription(self.subscription)
            self.subscription = None

    def _obstacles_callback(self, msg: DynamicObstacleArray) -> None:
        frame_id = msg.header.frame_id or self.world_frame
        if frame_id != self.world_frame:
            if not self._warned_frame_mismatch:
                self.node.get_logger().warn(
                    f"Ignoring dynamic obstacles in frame '{frame_id}'; expected '{self.world_frame}'."
                )
                self._warned_frame_mismatch = True
            return

        next_obstacles: dict[str, DynamicObstacleState] = {}
        for index, obstacle in enumerate(msg.obstacles):
            obstacle_id = obstacle.id.strip() or f"obstacle_{index}"
            if obstacle.collision_type == DynamicObstacle.GEOMETRY_NONE:
                continue
            center_world = np.array(
                [
                    obstacle.pose.position.x,
                    obstacle.pose.position.y,
                    obstacle.pose.position.z,
                ],
                dtype=float,
            )
            q = obstacle.pose.orientation
            quat_xyzw = np.array([q.x, q.y, q.z, q.w], dtype=float)
            if np.linalg.norm(quat_xyzw) < 1e-9:
                obstacle_to_world_rotation = np.eye(3)
                world_to_obstacle_rotation = np.eye(3)
            else:
                obstacle_to_world_rotation = Rotation.from_quat(quat_xyzw).as_matrix()
                world_to_obstacle_rotation = obstacle_to_world_rotation.T
            linear_velocity_world = np.array(
                [
                    obstacle.twist.linear.x,
                    obstacle.twist.linear.y,
                    obstacle.twist.linear.z,
                ],
                dtype=float,
            )
            angular_velocity_world = np.array(
                [
                    obstacle.twist.angular.x,
                    obstacle.twist.angular.y,
                    obstacle.twist.angular.z,
                ],
                dtype=float,
            )
            next_obstacles[obstacle_id] = DynamicObstacleState(
                obstacle_id=obstacle_id,
                center_world=center_world,
                obstacle_to_world_rotation=obstacle_to_world_rotation,
                world_to_obstacle_rotation=world_to_obstacle_rotation,
                linear_velocity_world=linear_velocity_world,
                angular_velocity_world=angular_velocity_world,
                collision_type=int(obstacle.collision_type),
                collision_dimensions=tuple(float(v) for v in obstacle.collision_dimensions),
            )
        self.obstacles = next_obstacles

    def in_collision_at_xyz(self, xyz: np.ndarray, *, t_offset: float = 0.0) -> bool:
        clearance = self.min_clearance_xyz(xyz, t_offset=t_offset)
        return clearance is not None and clearance.distance_m <= 0.0

    def min_clearance_xyz(self, xyz: np.ndarray, *, t_offset: float = 0.0) -> DynamicClearance | None:
        if not self.obstacles:
            return None

        point = np.asarray(xyz, dtype=float).reshape(3)
        robot_radius = max(0.0, float(self.robot_radius_provider()))
        best: DynamicClearance | None = None
        for obstacle in self.obstacles.values():
            distance_m = self._distance_to_obstacle(point, obstacle, robot_radius, float(t_offset))
            if best is None or distance_m < best.distance_m:
                best = DynamicClearance(obstacle.obstacle_id, distance_m)
        return best

    def _distance_to_obstacle(
        self,
        point_world: np.ndarray,
        obstacle: DynamicObstacleState,
        robot_radius: float,
        t_offset: float,
    ) -> float:
        center_world, world_to_obstacle_rotation = self._predicted_obstacle_transform(obstacle, t_offset)
        if obstacle.collision_type == DynamicObstacle.GEOMETRY_SPHERE:
            return self._distance_to_sphere(point_world, obstacle, robot_radius, center_world)
        if obstacle.collision_type == DynamicObstacle.GEOMETRY_BOX:
            return self._distance_to_box(point_world, obstacle, robot_radius, center_world, world_to_obstacle_rotation)
        if obstacle.collision_type == DynamicObstacle.GEOMETRY_CYLINDER:
            return self._distance_to_cylinder(point_world, obstacle, robot_radius, center_world, world_to_obstacle_rotation)
        if obstacle.collision_type == DynamicObstacle.GEOMETRY_MESH:
            return self._distance_to_mesh_proxy(point_world, obstacle, robot_radius, center_world, world_to_obstacle_rotation)
        return float("inf")

    def _distance_to_sphere(
        self,
        point_world: np.ndarray,
        obstacle: DynamicObstacleState,
        robot_radius: float,
        center_world: np.ndarray,
    ) -> float:
        if len(obstacle.collision_dimensions) < 1:
            return float("inf")
        radius = max(0.0, float(obstacle.collision_dimensions[0]))
        return float(np.linalg.norm(point_world - center_world) - radius - robot_radius)

    def _distance_to_box(
        self,
        point_world: np.ndarray,
        obstacle: DynamicObstacleState,
        robot_radius: float,
        center_world: np.ndarray,
        world_to_obstacle_rotation: np.ndarray,
    ) -> float:
        if len(obstacle.collision_dimensions) < 3:
            return float("inf")
        point_local = self._point_in_obstacle_frame(point_world, center_world, world_to_obstacle_rotation)
        half_extents = 0.5 * np.asarray(obstacle.collision_dimensions[:3], dtype=float)
        return self._signed_distance_to_box(point_local, half_extents) - robot_radius

    def _distance_to_cylinder(
        self,
        point_world: np.ndarray,
        obstacle: DynamicObstacleState,
        robot_radius: float,
        center_world: np.ndarray,
        world_to_obstacle_rotation: np.ndarray,
    ) -> float:
        if len(obstacle.collision_dimensions) < 2:
            return float("inf")
        point_local = self._point_in_obstacle_frame(point_world, center_world, world_to_obstacle_rotation)
        radius = max(0.0, float(obstacle.collision_dimensions[0]))
        half_height = 0.5 * max(0.0, float(obstacle.collision_dimensions[1]))
        q = np.array([np.linalg.norm(point_local[:2]) - radius, abs(point_local[2]) - half_height])
        outside = float(np.linalg.norm(np.maximum(q, 0.0)))
        inside = float(min(max(q[0], q[1]), 0.0))
        return outside + inside - robot_radius

    def _distance_to_mesh_proxy(
        self,
        point_world: np.ndarray,
        obstacle: DynamicObstacleState,
        robot_radius: float,
        center_world: np.ndarray,
        world_to_obstacle_rotation: np.ndarray,
    ) -> float:
        dimensions = obstacle.collision_dimensions
        if len(dimensions) == 1:
            return self._distance_to_sphere(point_world, obstacle, robot_radius, center_world)
        if len(dimensions) >= 3:
            return self._distance_to_box(
                point_world,
                obstacle,
                robot_radius,
                center_world,
                world_to_obstacle_rotation,
            )
        return float("inf")

    @staticmethod
    def _point_in_obstacle_frame(
        point_world: np.ndarray,
        center_world: np.ndarray,
        world_to_obstacle_rotation: np.ndarray,
    ) -> np.ndarray:
        return world_to_obstacle_rotation @ (point_world - center_world)

    @staticmethod
    def _predicted_obstacle_transform(
        obstacle: DynamicObstacleState,
        t_offset: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if abs(t_offset) < 1e-9:
            return obstacle.center_world, obstacle.world_to_obstacle_rotation

        center_world = obstacle.center_world + obstacle.linear_velocity_world * t_offset
        angular_delta = obstacle.angular_velocity_world * t_offset
        if np.linalg.norm(angular_delta) < 1e-9:
            return center_world, obstacle.world_to_obstacle_rotation

        delta_rotation = Rotation.from_rotvec(angular_delta).as_matrix()
        obstacle_to_world_rotation = delta_rotation @ obstacle.obstacle_to_world_rotation
        return center_world, obstacle_to_world_rotation.T

    @staticmethod
    def _signed_distance_to_box(point_local: np.ndarray, half_extents: np.ndarray) -> float:
        q = np.abs(point_local) - np.maximum(half_extents, 0.0)
        outside = float(np.linalg.norm(np.maximum(q, 0.0)))
        inside = float(min(max(q[0], q[1], q[2]), 0.0))
        return outside + inside
