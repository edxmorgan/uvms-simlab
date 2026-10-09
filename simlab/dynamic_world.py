from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import fcl
from bringup.collision_geometry import collision_geometry, geometry_from_obstacle
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
    collision_mesh_resource: str = ""
    collision_mesh_scale: tuple[float, ...] = ()


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
        self.error: str | None = "waiting for initial dynamic obstacle snapshot"
        self.bodies: list[dict] = []

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
        self.bodies.clear()
        if self.subscription is not None:
            self.node.destroy_subscription(self.subscription)
            self.subscription = None

    def update_from_msg(self, msg: DynamicObstacleArray) -> None:
        self._obstacles_callback(msg)

    def _obstacles_callback(self, msg: DynamicObstacleArray) -> None:
        try:
            self._accept_snapshot(msg)
            self.error = None
        except Exception as exc:
            error = str(exc)
            if error != self.error:
                self.node.get_logger().error(f"Dynamic collision geometry unavailable: {error}")
            self.error = error

    def _accept_snapshot(self, msg: DynamicObstacleArray) -> None:
        frame_id = msg.header.frame_id or self.world_frame
        if frame_id != self.world_frame:
            raise ValueError(f"obstacle frame '{frame_id}' does not match '{self.world_frame}'")

        next_obstacles: dict[str, DynamicObstacleState] = {}
        next_bodies = []
        old_bodies = {body['name']: body for body in self.bodies}
        seen = set()
        for index, obstacle in enumerate(msg.obstacles):
            obstacle_id = obstacle.id.strip() or f"obstacle_{index}"
            if obstacle_id in seen:
                raise ValueError(f"duplicate obstacle id: {obstacle_id}")
            seen.add(obstacle_id)
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
            if not all(np.all(np.isfinite(v)) for v in (center_world, quat_xyzw, linear_velocity_world, angular_velocity_world)):
                raise ValueError(f"non-finite obstacle state: {obstacle_id}")
            geometry, _ = geometry_from_obstacle(obstacle)
            name = f"dynamic/{obstacle_id}"
            body = old_bodies.get(name)
            if body is None or body['geom'] is not geometry:
                body = {'name': name, 'geom': geometry, 'fcl_obj': fcl.CollisionObject(geometry)}
            body['fcl_obj'].setTransform(fcl.Transform(obstacle_to_world_rotation, center_world))
            next_bodies.append(body)
            next_obstacles[obstacle_id] = DynamicObstacleState(
                obstacle_id=obstacle_id,
                center_world=center_world,
                obstacle_to_world_rotation=obstacle_to_world_rotation,
                world_to_obstacle_rotation=world_to_obstacle_rotation,
                linear_velocity_world=linear_velocity_world,
                angular_velocity_world=angular_velocity_world,
                collision_type=int(obstacle.collision_type),
                collision_dimensions=tuple(float(v) for v in obstacle.collision_dimensions),
                collision_mesh_resource=obstacle.collision_mesh_resource,
                collision_mesh_scale=tuple(obstacle.collision_mesh_scale),
            )
        self.obstacles = next_obstacles
        self.bodies = next_bodies

    def in_collision_at_xyz(self, xyz: np.ndarray, *, t_offset: float = 0.0) -> bool:
        clearance = self.min_clearance_xyz(xyz, t_offset=t_offset)
        return clearance is not None and clearance.distance_m <= 0.0

    def min_clearance_xyz(self, xyz: np.ndarray, *, t_offset: float = 0.0) -> DynamicClearance | None:
        if self.error:
            # Unknown geometry must not be interpreted as free space by planners.
            return DynamicClearance("unavailable", float("-inf"))
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
            return self._distance_to_mesh(point_world, obstacle, robot_radius, center_world, world_to_obstacle_rotation)
        raise ValueError(f"unsupported collision type: {obstacle.collision_type}")

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

    def _distance_to_mesh(
        self,
        point_world: np.ndarray,
        obstacle: DynamicObstacleState,
        robot_radius: float,
        center_world: np.ndarray,
        world_to_obstacle_rotation: np.ndarray,
    ) -> float:
        geometry, _ = collision_geometry(
            obstacle.collision_type, obstacle.collision_dimensions,
            obstacle.collision_mesh_resource, obstacle.collision_mesh_scale,
        )
        # Prediction uses separate objects; never mutate the live visualization pose.
        mesh = fcl.CollisionObject(geometry, fcl.Transform(world_to_obstacle_rotation.T, center_world))
        sphere = fcl.CollisionObject(fcl.Sphere(max(robot_radius, 1e-9)), fcl.Transform(point_world))
        result = fcl.CollisionResult()
        if fcl.collide(sphere, mesh, fcl.CollisionRequest(), result):
            # BVH contact does not provide a trustworthy signed penetration depth.
            return -1e-3
        distance = float(fcl.distance(sphere, mesh, fcl.DistanceRequest(), fcl.DistanceResult()))
        return distance

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
