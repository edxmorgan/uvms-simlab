"""Vehicle hover task for the batched mock UVMS environment."""

from __future__ import annotations

from typing import Any

import numpy as np

from uvms_rl.tasks.base import TaskBase


def _range_pair(config: dict[str, Any], key: str, default: tuple[float, float]) -> tuple[float, float]:
    value = config.get(key, default)
    if isinstance(value, dict):
        raise ValueError(f"{key} must be a two-element range, not a mapping")
    if len(value) != 2:
        raise ValueError(f"{key} must contain [low, high]")
    return float(value[0]), float(value[1])


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class HoverVehicleTask(TaskBase):
    """Reach and hold a target xyz/yaw with the vehicle base."""

    name = "hover_vehicle"

    @property
    def policy_observation_dim(self) -> int:
        # xyz error, yaw error, uvw, pqr, arm q, arm qd, previous action
        return 3 + 1 + 3 + 3 + 5 + 5 + 13

    def reset(self, env) -> np.ndarray:
        cfg = self.config
        n = self.robot_count
        obs = np.zeros((n, env.observation_dim), dtype=np.float32)

        x_low, x_high = _range_pair(cfg, "target_x", (-2.0, 2.0))
        y_low, y_high = _range_pair(cfg, "target_y", (-2.0, 2.0))
        z_low, z_high = _range_pair(cfg, "target_z", (-1.5, -0.2))
        yaw_low, yaw_high = _range_pair(cfg, "target_yaw", (-np.pi, np.pi))

        self.target_xyz = np.column_stack(
            [
                self.rng.uniform(x_low, x_high, size=n),
                self.rng.uniform(y_low, y_high, size=n),
                self.rng.uniform(z_low, z_high, size=n),
            ]
        ).astype(np.float32)
        self.target_yaw = self.rng.uniform(yaw_low, yaw_high, size=n).astype(np.float32)
        self.success_streak = np.zeros(n, dtype=np.int32)

        xyz_noise = float(cfg.get("initial_xyz_noise", 2.0))
        yaw_noise = float(cfg.get("initial_yaw_noise", np.pi))
        velocity_noise = float(cfg.get("initial_velocity_noise", 0.05))

        obs[:, 0:3] = self.target_xyz + self.rng.uniform(-xyz_noise, xyz_noise, size=(n, 3)).astype(np.float32)
        obs[:, 5] = self.target_yaw + self.rng.uniform(-yaw_noise, yaw_noise, size=n).astype(np.float32)
        obs[:, 6:12] = self.rng.uniform(-velocity_noise, velocity_noise, size=(n, 6)).astype(np.float32)
        return obs

    def policy_observation(self, env, sim_obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        xyz = sim_obs[:, 0:3]
        yaw = sim_obs[:, 5]
        uvw = sim_obs[:, 6:9]
        pqr = sim_obs[:, 9:12]
        arm_q = sim_obs[:, 12:17]
        arm_qd = sim_obs[:, 17:22]

        xyz_error = self.target_xyz - xyz
        yaw_error = _wrap_to_pi(self.target_yaw - yaw)[:, None]
        return np.concatenate([xyz_error, yaw_error, uvw, pqr, arm_q, arm_qd, actions], axis=1).astype(np.float32)

    def reward_done(self, env, sim_obs: np.ndarray, actions: np.ndarray):
        cfg = self.config
        xyz = sim_obs[:, 0:3]
        yaw = sim_obs[:, 5]
        uvw = sim_obs[:, 6:9]
        pqr = sim_obs[:, 9:12]

        pos_err = np.linalg.norm(xyz - self.target_xyz, axis=1)
        yaw_err = np.abs(_wrap_to_pi(yaw - self.target_yaw))
        lin_speed = np.linalg.norm(uvw, axis=1)
        ang_speed = np.linalg.norm(pqr, axis=1)
        effort = np.linalg.norm(actions, axis=1)

        reward = (
            -float(cfg.get("position_weight", 2.0)) * pos_err
            -float(cfg.get("yaw_weight", 0.5)) * yaw_err
            -float(cfg.get("linear_velocity_weight", 0.1)) * lin_speed
            -float(cfg.get("angular_velocity_weight", 0.05)) * ang_speed
            -float(cfg.get("action_weight", 0.01)) * effort
        ).astype(np.float32)

        pos_tol = float(cfg.get("success_position_tolerance", 0.15))
        yaw_tol = float(cfg.get("success_yaw_tolerance", 0.15))
        required_streak = int(cfg.get("success_streak_steps", 10))
        success_now = (pos_err < pos_tol) & (yaw_err < yaw_tol)
        self.success_streak = np.where(success_now, self.success_streak + 1, 0)
        success = self.success_streak >= required_streak

        timeout = env.episode_steps >= env.max_episode_steps
        out_of_bounds = np.linalg.norm(xyz, axis=1) > float(cfg.get("out_of_bounds_radius", 10.0))

        reward += success.astype(np.float32) * float(cfg.get("success_bonus", 10.0))
        reward -= out_of_bounds.astype(np.float32) * float(cfg.get("out_of_bounds_penalty", 10.0))
        done = success | timeout | out_of_bounds

        info = {
            "mean_position_error": float(np.mean(pos_err)),
            "mean_yaw_error": float(np.mean(yaw_err)),
            "success_rate": float(np.mean(success)),
            "timeout_rate": float(np.mean(timeout)),
            "out_of_bounds_rate": float(np.mean(out_of_bounds)),
        }
        return reward, done, info
