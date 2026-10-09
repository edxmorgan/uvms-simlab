"""Default behavior: stationary unless the caller specifies a twist."""
import copy

import numpy as np
from scipy.spatial.transform import Rotation

from simlab.dynamic_obstacle_sources.behavior import ObstacleBehaviorSource


class ScriptedMotionSource(ObstacleBehaviorSource):
    registry_name = "scripted_motion"

    def initialize(self, scene, config, seed):
        if config:
            raise ValueError("scripted_motion uses per-obstacle twists; no source parameters are supported")

    def step(self, scene, context):
        result = copy.deepcopy(scene)
        for obstacle in result.obstacles:
            p, q = obstacle.pose.position, obstacle.pose.orientation
            v, w = obstacle.twist.linear, obstacle.twist.angular
            p.x += context.dt * v.x
            p.y += context.dt * v.y
            p.z += context.dt * v.z
            # Both linear and angular velocities are expressed in the world frame.
            delta = Rotation.from_rotvec(context.dt * np.array([w.x, w.y, w.z]))
            rotation = delta * Rotation.from_quat([q.x, q.y, q.z, q.w])
            q.x, q.y, q.z, q.w = rotation.as_quat().tolist()
        return result
