# Copyright (C) 2025 Edward Morgan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

#!/usr/bin/env python3
import os
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
import trimesh
from ament_index_python.packages import get_package_share_directory
from mesh_utils import collect_env_meshes, conc_env_trimesh, points_to_cloud2
from datetime import datetime

class VoxelVizNode(Node):
    def __init__(self):
        super().__init__("voxel_viz_node")
        self.declare_parameter('robot_description', '')

        urdf_string = self.get_parameter(
            'robot_description'
        ).get_parameter_value().string_value

        if not urdf_string:
            self.get_logger().error(
                'robot_description param is empty. Did you load it into the param server in launch'
            )
            raise RuntimeError('no robot_description')

        # collect meshes
        robot_mesh_infos, env_mesh_infos, floor_depth = collect_env_meshes(urdf_string)
        if len(env_mesh_infos) == 0:
            self.get_logger().warn(
                "No env meshes with prefix bathymetry_ found"
            )
        else:
            self.get_logger().info(
                f"env links {[x['link'] for x in env_mesh_infos]}"
            )

        # merge meshes into one Trimesh in world frame
        env_mesh = conc_env_trimesh(env_mesh_infos)
        if env_mesh is None:
            self.get_logger().error("No environment mesh could be built")
            raise RuntimeError("empty env mesh")

        self.get_logger().info(
            f"concatenated env meshes, {len(env_mesh.faces)} faces total"
        )

        # choose voxel resolution
        self.voxel_size = 0.10  # meters per cell

        # load cached voxel centers from ros2_control_blue_reach_5 share
        # or build them and save there
        self.centers = self.load_or_build_voxels(env_mesh, self.voxel_size)

        self.get_logger().info(
            f"env voxel grid ready, {self.centers.shape[0]} occupied voxels at "
            f"{self.voxel_size} m"
        )

        self.cloud_pub = self.create_publisher(
            PointCloud2,
            '/env_voxels_cloud',
            10
        )

        self.overlay_text_publisher = self.create_publisher(
            String,
            "chatter",
            10
        )

        self.overlay_text_timer = self.create_timer(1.0 / 30, self.publish_overlay_text_callback)
        # timer
        self.timer = self.create_timer(0.1, self.tick)

    def publish_overlay_text_callback(self) -> None:
        str_msg = String()
        str_msg.data = f"© {datetime.now().year} Louisiana State University. Research use."
        self.overlay_text_publisher.publish(str_msg)


    def tick(self):
        """
        Periodic publish.
        1. Publish point cloud of voxel centers to RViz for geometric sanity check.
        2. Placeholder for OccupancyGrid projection from octree later.
        """
        # Publish voxel centers as PointCloud2
        if self.centers is not None and self.centers.shape[0] > 0:
            cloud_msg = points_to_cloud2(
                self.centers,
                frame_id="world_bottom",
                stamp=self.get_clock().now().to_msg()
            )
            self.cloud_pub.publish(cloud_msg)


    def get_cache_path(self, voxel_size: float):
        pkg_share = get_package_share_directory('ros2_control_blue_reach_5')

        # Put voxel cache in Bathymetry/voxels under that share directory
        voxels_dir = os.path.join(pkg_share, 'Bathymetry', 'voxels')
        os.makedirs(voxels_dir, exist_ok=True)

        fname = f"env_voxels_{voxel_size:.3f}m.npy"
        cache_path = os.path.join(voxels_dir, fname)

        return cache_path

    def load_or_build_voxels(self,
                             mesh: trimesh.Trimesh,
                             voxel_size: float):
        """
        Try to load cached centers from ros2_control_blue_reach_5 share.
        If not present, voxelize, save there, then return.
        """
        cache_path = self.get_cache_path(voxel_size)

        if os.path.exists(cache_path):
            self.get_logger().info(
                f"loading cached voxel centers from {cache_path}"
            )
            centers = np.load(cache_path)
            return centers

        # cache miss case
        self.get_logger().info(
            f"no cache found, voxelizing at {voxel_size} m and saving to {cache_path}"
        )

        centers, _ = self.voxelize_mesh(
            mesh,
            voxel_size,
            solid=False
        )

        # write cache file
        np.save(cache_path, centers)
        self.get_logger().info(
            f"saved {centers.shape[0]} voxel centers to {cache_path}"
        )

        return centers

    def voxelize_mesh(self,
                      mesh: trimesh.Trimesh,
                      voxel_size: float,
                      solid: bool = False):
        """
        Voxelize with trimesh.
        solid False gives surface shell voxels.
        solid True fills interior.
        """
        v = mesh.voxelized(pitch=voxel_size, method="subdivide")
        if solid:
            v = v.fill()

        centers = v.points.copy()
        return centers, voxel_size


def main():
    rclpy.init()
    node = VoxelVizNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
