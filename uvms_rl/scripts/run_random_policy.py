"""Run a random-policy smoke test for a uvms_rl experiment config."""

from __future__ import annotations

import argparse

import numpy as np

from uvms_rl import UvmsBatchEnv
from uvms_rl.config import load_experiment_config
from uvms_rl.tensor import as_numpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="hover_vehicle", help="Packaged config name or YAML path")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--action-scale", type=float, default=0.2)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    env_cfg = config.get("env", {})
    task_cfg = config.get("task", {})
    task_name = task_cfg.get("name", "hover_vehicle")

    env = UvmsBatchEnv(
        robot_count=int(env_cfg.get("robot_count", 1024)),
        control_dt=float(env_cfg.get("control_dt", env_cfg.get("dt", 0.01))),
        sim_dt=float(env_cfg.get("sim_dt", env_cfg.get("control_dt", env_cfg.get("dt", 0.01)))),
        max_episode_steps=int(env_cfg.get("max_episode_steps", 500)),
        seed=env_cfg.get("seed"),
        task=task_name,
        task_config=task_cfg,
        backend=str(env_cfg.get("backend", "cpu")),
    )
    rng = np.random.default_rng(env_cfg.get("seed"))
    obs = env.reset()

    for _ in range(args.steps):
        actions = rng.uniform(-args.action_scale, args.action_scale, size=(env.robot_count, env.action_dim)).astype(np.float32)
        obs, rewards, dones, info = env.step(actions)

    print("task", task_name)
    print("backend", env.backend)
    print("control_dt", env.control_dt, "control_hz", 1.0 / env.control_dt)
    print("sim_dt", env.sim_dt, "sim_hz", 1.0 / env.sim_dt, "substeps", env.substeps)
    print("policy_obs", obs.shape)
    rewards_np = as_numpy(rewards)
    dones_np = as_numpy(dones)
    print("reward", rewards_np.shape, "mean", float(np.mean(rewards_np)), "first5envs", rewards_np[:5])
    print("done", dones_np.shape, "rate", float(np.mean(dones_np)), "first5envs", dones_np[:5])
    print("tick", info.tick_id, "sim_time", info.sim_time)
    print("task_info", info.task)


if __name__ == "__main__":
    main()
