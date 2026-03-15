#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch

class MidasRGBToCloudFast(Node):
    def __init__(self):
        super().__init__('midas_rgb_to_cloud_fast')

        # Parameters
        self.declare_parameter('image_topic', '/alpha/image_raw')
        self.declare_parameter('cloud_topic', '/alpha/points_midas')
        self.declare_parameter('hfov_deg', 80.0)
        self.declare_parameter('vfov_deg', 64.0)
        self.declare_parameter('stride', 2)
        self.declare_parameter('model_type', 'DPT_Large')  # MiDaS_small, DPT_Hybrid, DPT_Large
        self.declare_parameter('median_depth_m', 2.0)
        self.declare_parameter('min_depth_m', 0.3)
        self.declare_parameter('max_depth_m', 10.0)
        self.declare_parameter('frame_id', 'camera_optical_frame')
        self.declare_parameter('frame_skip', 2)       # process one of every N frames
        self.declare_parameter('net_height', 256)     # input height to MiDaS
        self.declare_parameter('net_width',  448)     # input width to MiDaS

        p = self.get_parameter
        self.image_topic = p('image_topic').get_parameter_value().string_value
        self.cloud_topic = p('cloud_topic').get_parameter_value().string_value
        self.hfov = math.radians(float(p('hfov_deg').value))
        self.vfov = math.radians(float(p('vfov_deg').value))
        self.stride = int(p('stride').value)
        self.median_depth_m = float(p('median_depth_m').value)
        self.min_depth_m = float(p('min_depth_m').value)
        self.max_depth_m = float(p('max_depth_m').value)
        self.default_frame = p('frame_id').get_parameter_value().string_value
        self.frame_skip = int(p('frame_skip').value)
        self.net_h = int(p('net_height').value)
        self.net_w = int(p('net_width').value)

        # MiDaS
        self.model_type = p('model_type').get_parameter_value().string_value
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'\033[35mTorch device : {self.device}\033[0m')

        torch.set_grad_enabled(False)
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        self.midas = torch.hub.load('intel-isl/MiDaS', self.model_type, trust_repo=True).to(self.device)
        self.midas.eval()
        self.transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        if self.model_type in ('DPT_Large', 'DPT_Hybrid'):
            self.transform = self.transforms.dpt_transform
        else:
            self.transform = self.transforms.small_transform

        # Half precision on GPU
        self.use_fp16 = (self.device.type == 'cuda' and self.model_type != 'MiDaS_small') or (self.device.type == 'cuda')
        if self.use_fp16:
            self.midas.half()

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, self.image_topic, self.cb, 10)
        self.pub = self.create_publisher(PointCloud2, self.cloud_topic, 10)

        self.frame_count = 0
        self.cached_size = None
        self.fx = self.fy = self.cx = self.cy = None
        self.grid_u = self.grid_v = None
        self.sample_idx = None

        self.get_logger().info(
            f'MiDaS fast node, topic={self.image_topic}, model={self.model_type}, stride={self.stride}, skip={self.frame_skip}, net={self.net_w}x{self.net_h}'
        )

    def _to_rgb8(self, msg: Image) -> np.ndarray:
        enc = msg.encoding.lower()
        try:
            if enc == 'rgb8':
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            elif enc == 'bgr8':
                bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            elif enc in ('mono8', '8uc1'):
                gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            elif enc in ('yuyv', 'yuv422', 'yuv422_yuy2', 'yuv422_yuyv', 'yuy2'):
                yuyv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                img = cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUYV)
            elif enc in ('uyvy', 'yuv422_uyvy'):
                uyvy = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
                img = cv2.cvtColor(uyvy, cv2.COLOR_YUV2RGB_UYVY)
            else:
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge conversion failed, {e}, falling back to rgb8')
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        return img

    def _prepare_projection_cache(self, h: int, w: int):
        if self.cached_size == (h, w):
            return
        # Intrinsics from FOV
        self.fx = w / (2.0 * math.tan(self.hfov * 0.5))
        self.fy = h / (2.0 * math.tan(self.vfov * 0.5))
        self.cx, self.cy = w * 0.5, h * 0.5

        ys = np.arange(0, h, self.stride, dtype=np.int32)
        xs = np.arange(0, w, self.stride, dtype=np.int32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        self.grid_u = grid_x.reshape(-1).astype(np.float32)
        self.grid_v = grid_y.reshape(-1).astype(np.float32)
        # index array for fast sampling
        self.sample_idx = (grid_y * w + grid_x).reshape(-1)

        self.cached_size = (h, w)
        self.get_logger().info(f'Cached projection for {w}x{h}, stride={self.stride}, points={self.grid_u.size}')

    def _predict_depth(self, img_rgb: np.ndarray) -> np.ndarray:
        # Resize for the network to reduce compute, use standard MiDaS preprocessing
        ih, iw = img_rgb.shape[:2]
        inp = cv2.resize(img_rgb, (self.net_w, self.net_h), interpolation=cv2.INTER_AREA)
        t_in = self.transform(inp).to(self.device)
        if self.use_fp16:
            t_in = t_in.half()

        with torch.no_grad():
            pred = self.midas(t_in)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=(ih, iw), mode='bicubic', align_corners=False
            ).squeeze(1).squeeze(0)

        inv_np = pred.float().cpu().numpy()
        inv_med = float(np.median(inv_np[inv_np > 0]))
        if inv_med <= 0:
            return None
        k = self.median_depth_m * inv_med
        depth = k / np.maximum(inv_np, 1e-6)
        depth = np.clip(depth, self.min_depth_m, self.max_depth_m).astype(np.float32)
        return depth

    def _make_cloud(self, stamp, frame_id, points_xyz, colors_0to1):
        c = np.clip(colors_0to1 * 255.0, 0, 255).astype(np.uint32)
        rgb_uint32 = (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
        rgb_float32 = rgb_uint32.view(np.float32)
        cloud = np.column_stack([points_xyz, rgb_float32]).astype(np.float32)

        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        return PointCloud2(
            header=header,
            height=1,
            width=cloud.shape[0],
            fields=fields,
            is_bigendian=False,
            point_step=16,
            row_step=16 * cloud.shape[0],
            is_dense=True,
            data=cloud.tobytes()
        )

    def cb(self, msg: Image):
        self.frame_count += 1
        if self.frame_skip > 1 and (self.frame_count % self.frame_skip) != 0:
            return

        img = self._to_rgb8(msg)
        h, w, _ = img.shape
        self._prepare_projection_cache(h, w)

        depth_m = self._predict_depth(img)
        if depth_m is None:
            self.get_logger().warn('Invalid MiDaS output, skipping frame')
            return

        # Sample depth and colors using cached indices
        z = depth_m.reshape(-1)[self.sample_idx]
        x = (self.grid_u - self.cx) * z / self.fx
        y = (self.grid_v - self.cy) * z / self.fy
        points_xyz = np.column_stack((x, y, z)).astype(np.float32)

        sampled = img.reshape(-1, 3)[self.sample_idx]
        colors_0to1 = sampled.astype(np.float32) / 255.0

        cloud_msg = self._make_cloud(
            stamp=msg.header.stamp,
            frame_id=msg.header.frame_id or self.default_frame,
            points_xyz=points_xyz,
            colors_0to1=colors_0to1
        )
        self.pub.publish(cloud_msg)

def main():
    rclpy.init()
    node = MidasRGBToCloudFast()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
