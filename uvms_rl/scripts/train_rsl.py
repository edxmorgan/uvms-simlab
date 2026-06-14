"""Train a uvms_rl experiment with RSL-RL PPO."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner
from uvms_rl.rsl_adapter import RslRlUvmsEnv


def _pd_hover_teacher_actions(env: RslRlUvmsEnv, obs, trainer: dict[str, Any]) -> torch.Tensor:
    return env.teacher_actions(obs, dict(trainer.get("teacher", {})))


def _pretrain_with_teacher(runner: OnPolicyRunner, env: RslRlUvmsEnv, trainer: dict[str, Any]) -> None:
    teacher = dict(trainer.get("teacher", {}))
    if str(teacher.get("name", "")).strip().lower() != "pd_hover":
        return
    steps = int(teacher.get("pretrain_steps", 0))
    if steps <= 0:
        return

    actor_critic = getattr(runner.alg, "policy", None)
    if actor_critic is None:
        actor_critic = runner.alg.actor_critic
    actor_critic.train()
    optimizer = torch.optim.Adam(
        actor_critic.actor.parameters(),
        lr=float(teacher.get("pretrain_learning_rate", 1.0e-3)),
    )
    obs = env.reset().to(runner.device)
    print("teacher_pretrain", f"name=pd_hover", f"steps={steps}")
    for step in range(steps):
        with torch.no_grad():
            actor_critic.update_normalization(obs)
            if env.residual_teacher:
                target = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=runner.device)
            else:
                target = _pd_hover_teacher_actions(env, obs.to(env.device), trainer).to(runner.device)
        pred = actor_critic.act_inference(obs)
        loss = torch.nn.functional.mse_loss(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.inference_mode():
            obs, _, _, _ = env.step(target.to(env.device))
            obs = obs.to(runner.device)

        interval = max(1, steps // 5)
        if (step + 1) % interval == 0 or step == 0:
            print("teacher_pretrain", f"step={step + 1}", f"loss={float(loss.detach().cpu().item()):.6f}")


def _make_log_dir(root: str | Path, experiment_name: str, run_name: str) -> Path:
    root_path = Path(root).expanduser()
    if not root_path.is_absolute():
        root_path = Path.cwd() / root_path
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return root_path / experiment_name / f"{stamp}_{run_name}"


def _trainer_cfg(experiment_cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    trainer = dict(experiment_cfg.get("trainer", {}))
    rsl_cfg = copy.deepcopy(trainer.get("rsl_rl", {}))
    if not rsl_cfg:
        raise ValueError("experiment config is missing trainer.rsl_rl")
    return trainer, rsl_cfg


def _evaluate(runner: OnPolicyRunner, env: RslRlUvmsEnv, steps: int) -> dict[str, float]:
    obs = env.reset().to(runner.device)
    policy = runner.get_inference_policy(device=runner.device)
    reward_sum = torch.zeros(env.num_envs, dtype=torch.float32, device=runner.device)
    done_count = torch.zeros(env.num_envs, dtype=torch.float32, device=runner.device)
    log_sums: dict[str, float] = {}
    last_log: dict[str, float] = {}
    log_count = 0
    with torch.inference_mode():
        for _ in range(int(steps)):
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions.to(env.device))
            obs = obs.to(runner.device)
            reward_sum += rewards.to(runner.device)
            done_count += dones.to(runner.device, dtype=torch.float32)
            for key, value in dict(extras.get("log", {})).items():
                if key.startswith("/uvms_rl/") and isinstance(value, (int, float)):
                    metric = key.removeprefix("/uvms_rl/")
                    log_sums[metric] = log_sums.get(metric, 0.0) + float(value)
                    last_log[metric] = float(value)
            log_count += 1
    result = {
        "mean_step_reward": float((reward_sum / float(steps)).mean().item()),
        "mean_episode_completions": float(done_count.mean().item()),
    }
    if log_count > 0:
        for key, value in log_sums.items():
            result[key] = value / float(log_count)
        for key in ("mean_position_error", "mean_yaw_error", "success_rate", "out_of_bounds_rate"):
            if key in last_log:
                result[f"final_{key}"] = last_log[key]
    return result


def _print_eval(label: str, metrics: dict[str, float]) -> None:
    fields = [
        f"mean_step_reward={metrics['mean_step_reward']:.6f}",
        f"mean_episode_completions={metrics['mean_episode_completions']:.3f}",
    ]
    for key in (
        "mean_position_error",
        "mean_yaw_error",
        "success_rate",
        "timeout_rate",
        "out_of_bounds_rate",
        "final_mean_position_error",
        "final_mean_yaw_error",
    ):
        if key in metrics:
            fields.append(f"{key}={metrics[key]:.4f}")
    print(label, *fields)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="hover_vehicle", help="Packaged experiment name or experiment folder")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--backend", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--device", default=None, help="Torch training device, e.g. cpu or cuda")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--clip-actions", type=float, default=None)
    parser.add_argument("--init-random-episode-length", action="store_true")
    args = parser.parse_args()

    env = RslRlUvmsEnv.from_experiment(
        args.config,
        backend=args.backend,
        robot_count=args.num_envs,
        clip_actions=args.clip_actions,
    )
    trainer, train_cfg = _trainer_cfg(env.cfg)
    if args.iterations is not None:
        train_cfg["num_learning_iterations"] = int(args.iterations)
    if args.clip_actions is None and "clip_actions" in trainer:
        env.clip_actions = float(trainer["clip_actions"])
    if "action_scale" in trainer:
        env.action_scale = env._make_action_scale(trainer["action_scale"])
    residual_teacher = dict(trainer.get("residual_teacher", {}))
    if residual_teacher:
        residual_scale = residual_teacher.pop("residual_scale", None)
        env.configure_residual_teacher(residual_teacher, residual_scale)

    seed = int(env.cfg.get("env", {}).get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = args.device or ("cuda" if env.unwrapped.backend == "gpu" else "cpu")
    eval_steps = int(args.eval_steps if args.eval_steps is not None else trainer.get("eval_steps", 200))
    iterations = int(train_cfg.pop("num_learning_iterations"))
    run_name = str(train_cfg.get("run_name", Path(str(args.config)).stem or "uvms_rl"))
    log_dir = Path(args.log_dir) if args.log_dir else _make_log_dir(trainer.get("log_root", "recordings/rl_runs"), env.cfg.get("name", args.config), run_name)
    log_dir.mkdir(parents=True, exist_ok=True)

    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir=str(log_dir), device=device)

    before = _evaluate(runner, env, eval_steps)
    _print_eval("before_training", before)

    _pretrain_with_teacher(runner, env, trainer)
    after_teacher = _evaluate(runner, env, eval_steps)
    _print_eval("after_teacher", after_teacher)

    if iterations > 0:
        runner.learn(
            num_learning_iterations=iterations,
            init_at_random_ep_len=bool(args.init_random_episode_length),
        )
    else:
        print("ppo_finetune skipped iterations=0")

    after = _evaluate(runner, env, eval_steps)
    final_model = log_dir / "model_final.pt"
    runner.save(str(final_model), infos={"before": before, "after_teacher": after_teacher, "after": after})
    _print_eval("after_training", after)
    print("log_dir", log_dir)
    print("final_model", final_model)


if __name__ == "__main__":
    main()
