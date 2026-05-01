# Copyright (C) 2026 Edward Morgan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.

#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
import rclpy
import tf2_ros
import trimesh
from geometry_msgs.msg import TransformStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import CameraInfo, Image
from simlab.mesh_utils import collect_env_meshes, conc_env_trimesh, rpy_xyz_to_mat
from simlab.shutdown import install_signal_shutdown_handler, shutdown_node, spin_until_shutdown


def _topic_prefix(robot_prefix: str) -> str:
    return f"/{robot_prefix.rstrip('_')}/camera"


def _trimesh_to_polydata(mesh: trimesh.Trimesh) -> pv.PolyData:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.size == 0:
        return pv.PolyData(vertices)
    vtk_faces = np.column_stack(
        [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
    ).reshape(-1)
    return pv.PolyData(vertices, vtk_faces)


def _transform_to_matrix(transform: TransformStamped) -> np.ndarray:
    t = transform.transform.translation
    q = transform.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        r = np.eye(3)
    else:
        s = 2.0 / n
        xx, yy, zz = x * x * s, y * y * s, z * z * s
        xy, xz, yz = x * y * s, x * z * s, y * z * s
        wx, wy, wz = w * x * s, w * y * s, w * z * s
        r = np.array(
            [
                [1.0 - (yy + zz), xy - wz, xz + wy],
                [xy + wz, 1.0 - (xx + zz), yz - wx],
                [xz - wy, yz + wx, 1.0 - (xx + yy)],
            ],
            dtype=float,
        )
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = r
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix


def _load_mesh(path: str, scale: List[float], rpy: List[float], xyz: List[float]) -> Optional[pv.PolyData]:
    try:
        scene_or_mesh = trimesh.load(path, force="scene")
        mesh = (
            scene_or_mesh.dump(concatenate=True)
            if isinstance(scene_or_mesh, trimesh.Scene)
            else scene_or_mesh
        )
    except Exception:
        return None
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return None
    mesh.apply_scale(np.asarray(scale, dtype=float))
    mesh.apply_transform(rpy_xyz_to_mat(rpy, xyz))
    return _trimesh_to_polydata(mesh)


class SimCameraRendererNode(Node):
    def __init__(self):
        super().__init__("sim_camera_renderer_node")
        self.declare_parameter("robot_description", "")
        self.declare_parameter("robots_prefix", ["robot_1_"])
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("render_rate", 10.0)
        self.declare_parameter("horizontal_fov_deg", 75.0)
        self.declare_parameter("selected_prefix", "")
        self.declare_parameter("publish_selected_output", True)

        urdf_string = self.get_parameter("robot_description").value
        if not urdf_string:
            raise RuntimeError("sim_camera_renderer_node requires robot_description")

        self.robot_prefixes = [str(p) for p in self.get_parameter("robots_prefix").value]
        self.world_frame = str(self.get_parameter("world_frame").value)
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.render_rate = float(self.get_parameter("render_rate").value)
        self.horizontal_fov_deg = float(self.get_parameter("horizontal_fov_deg").value)
        self.selected_prefix = str(self.get_parameter("selected_prefix").value)
        self.publish_selected_output = bool(self.get_parameter("publish_selected_output").value)
        if not self.selected_prefix:
            self.selected_prefix = next((p for p in self.robot_prefixes if p != "robot_real_"), "")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        image_qos = QoSProfile(depth=1)
        info_qos = QoSProfile(depth=1)
        self.image_publishers: Dict[str, object] = {}
        self.info_publishers: Dict[str, object] = {}
        for prefix in self.robot_prefixes:
            if prefix == "robot_real_":
                continue
            topic_base = _topic_prefix(prefix)
            self.image_publishers[prefix] = self.create_publisher(Image, f"{topic_base}/image_raw", image_qos)
            self.info_publishers[prefix] = self.create_publisher(CameraInfo, f"{topic_base}/camera_info", info_qos)
        self.selected_image_publisher = self.create_publisher(Image, "/alpha/image_raw", image_qos)
        self.selected_info_publisher = self.create_publisher(CameraInfo, "/alpha/camera_info", info_qos)
        self.parameter_callback_handle = self.add_on_set_parameters_callback(self._set_parameters_callback)

        self.plotter = pv.Plotter(off_screen=True, window_size=(self.width, self.height))
        self.plotter.set_background("#022032", top="#001018")
        self.plotter.enable_lightkit()
        self.robot_actors: List[Tuple[str, object]] = []
        self._build_scene(urdf_string)
        self._configure_camera_intrinsics()

        self.timer = self.create_timer(1.0 / max(self.render_rate, 0.1), self.render_all)
        self.get_logger().info(
            f"sim_camera_renderer_node publishing rendered images for "
            f"{[p for p in self.robot_prefixes if p != 'robot_real_']}"
        )

    def _set_parameters_callback(self, parameters):
        result = SetParametersResult()
        result.successful = True
        for parameter in parameters:
            if parameter.name == "selected_prefix":
                selected_prefix = str(parameter.value)
                if selected_prefix not in self.image_publishers:
                    result.successful = False
                    result.reason = f"selected_prefix must be one of {sorted(self.image_publishers)}"
                    return result
                self.selected_prefix = selected_prefix
                self.get_logger().info(f"Selected simulated camera feed set to {self.selected_prefix}")
            elif parameter.name == "publish_selected_output":
                self.publish_selected_output = bool(parameter.value)
        return result

    def destroy_node(self):
        try:
            self.plotter.close()
        except Exception:
            pass
        super().destroy_node()

    def _configure_camera_intrinsics(self) -> None:
        horizontal_fov = math.radians(self.horizontal_fov_deg)
        self.fx = 0.5 * self.width / math.tan(0.5 * horizontal_fov)
        self.fy = self.fx
        self.cx = 0.5 * (self.width - 1)
        self.cy = 0.5 * (self.height - 1)

    def _build_scene(self, urdf_string: str) -> None:
        robot_mesh_infos, env_mesh_infos, floor_depth = collect_env_meshes(urdf_string)
        env_mesh = conc_env_trimesh(env_mesh_infos)
        if env_mesh is not None:
            env_mesh = env_mesh.copy()
            env_mesh.apply_translation([0.0, 0.0, float(floor_depth)])
            self.plotter.add_mesh(
                _trimesh_to_polydata(env_mesh),
                color="#7d7055",
                smooth_shading=True,
                ambient=0.55,
                diffuse=0.65,
                specular=0.05,
            )

        water = pv.Plane(
            center=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, 1.0),
            i_size=60.0,
            j_size=40.0,
        )
        self.plotter.add_mesh(
            water,
            color="#2f9bbd",
            opacity=0.18,
            ambient=0.6,
            diffuse=0.3,
        )

        for info in robot_mesh_infos:
            link = info["link"]
            poly = _load_mesh(info["uri"], info["scale"], info["rpy"], info["xyz"])
            if poly is None:
                continue
            actor = self.plotter.add_mesh(
                poly,
                color="#f0c84b",
                smooth_shading=True,
                ambient=0.35,
                diffuse=0.75,
                specular=0.15,
            )
            self.robot_actors.append((link, actor))

    def _lookup_world_transform(self, frame_id: str) -> Optional[np.ndarray]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame,
                frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.02),
            )
        except Exception:
            return None
        return _transform_to_matrix(transform)

    def _update_robot_actors(self) -> None:
        for link, actor in self.robot_actors:
            transform = self._lookup_world_transform(link)
            if transform is None:
                continue
            actor.user_matrix = transform

    def _set_camera_from_transform(self, transform: np.ndarray) -> None:
        position = transform[:3, 3]
        rotation = transform[:3, :3]
        forward = rotation[:, 2]
        up = -rotation[:, 1]
        focal_point = position + forward
        self.plotter.camera.position = tuple(position.tolist())
        self.plotter.camera.focal_point = tuple(focal_point.tolist())
        self.plotter.camera.up = tuple(up.tolist())
        self.plotter.camera.view_angle = self.horizontal_fov_deg
        self.plotter.camera.clipping_range = (0.03, 100.0)

    def _camera_info(self, prefix: str, stamp) -> CameraInfo:
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = f"{prefix}camera_link"
        info.height = self.height
        info.width = self.width
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def _image_msg(self, prefix: str, stamp, rgb: np.ndarray) -> Image:
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = f"{prefix}camera_link"
        msg.height = self.height
        msg.width = self.width
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = self.width * 3
        msg.data = bgr.tobytes()
        return msg

    def _has_subscribers(self, publisher) -> bool:
        return publisher.get_subscription_count() > 0

    def _should_render_prefix(self, prefix: str) -> bool:
        if self._has_subscribers(self.image_publishers[prefix]):
            return True
        if (
            self.publish_selected_output
            and prefix == self.selected_prefix
            and self._has_subscribers(self.selected_image_publisher)
        ):
            return True
        return False

    def render_all(self) -> None:
        active_prefixes = [
            prefix for prefix in self.image_publishers
            if self._should_render_prefix(prefix)
        ]
        if not active_prefixes:
            return

        self._update_robot_actors()
        stamp = self.get_clock().now().to_msg()
        for prefix in active_prefixes:
            camera_tf = self._lookup_world_transform(f"{prefix}camera_link")
            if camera_tf is None:
                continue
            self._set_camera_from_transform(camera_tf)
            self.plotter.camera.Modified()
            self.plotter.renderer.ResetCameraClippingRange()
            self.plotter.render()
            rgb = self.plotter.screenshot(return_img=True)
            if rgb is None:
                continue
            image_msg = self._image_msg(prefix, stamp, np.asarray(rgb))
            info_msg = self._camera_info(prefix, stamp)
            if self._has_subscribers(self.image_publishers[prefix]):
                self.image_publishers[prefix].publish(image_msg)
            if self._has_subscribers(self.info_publishers[prefix]):
                self.info_publishers[prefix].publish(info_msg)
            if self.publish_selected_output and prefix == self.selected_prefix:
                if self._has_subscribers(self.selected_image_publisher):
                    self.selected_image_publisher.publish(image_msg)
                if self._has_subscribers(self.selected_info_publisher):
                    self.selected_info_publisher.publish(info_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimCameraRendererNode()
    install_signal_shutdown_handler()
    try:
        spin_until_shutdown(node)
    finally:
        shutdown_node(node)


if __name__ == "__main__":
    main()
