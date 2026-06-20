"""Train a uvms_rl experiment with RSL-RL PPO."""

from __future__ import annotations

import argparse
import copy
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner
from uvms_rl.rsl_adapter import RslRlUvmsEnv


def _pd_hover_teacher_actions(env: RslRlUvmsEnv, obs, trainer: dict[str, Any]) -> torch.Tensor:
    return env.teacher_actions(obs, dict(trainer.get("teacher", {})))


def _pretrain_with_teacher(runner: OnPolicyRunner, env: RslRlUvmsEnv, trainer: dict[str, Any]) -> bool:
    teacher = dict(trainer.get("teacher", {}))
    if str(teacher.get("name", "")).strip().lower() != "pd_hover":
        return False
    steps = int(teacher.get("pretrain_steps", 0))
    if steps <= 0:
        return False

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
    return True


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


def _configure_adapter(env: RslRlUvmsEnv, trainer: dict[str, Any], *, clip_actions: float | None) -> None:
    if clip_actions is None and "clip_actions" in trainer:
        env.clip_actions = float(trainer["clip_actions"])
    if "action_scale" in trainer:
        env.action_scale = env._make_action_scale(trainer["action_scale"])
    residual_teacher = dict(trainer.get("residual_teacher", {}))
    if residual_teacher:
        residual_scale = residual_teacher.pop("residual_scale", None)
        env.configure_residual_teacher(residual_teacher, residual_scale)


def _zero_actor_output(runner: OnPolicyRunner) -> None:
    actor_critic = getattr(runner.alg, "policy", None)
    if actor_critic is None:
        actor_critic = runner.alg.actor_critic
    actor = getattr(actor_critic, "actor", None)
    if actor is None:
        raise RuntimeError("cannot zero actor output: actor network not found")
    last_linear = None
    for module in actor.modules():
        if isinstance(module, torch.nn.Linear):
            last_linear = module
    if last_linear is None:
        raise RuntimeError("cannot zero actor output: no Linear layer found in actor")
    with torch.no_grad():
        last_linear.weight.zero_()
        if last_linear.bias is not None:
            last_linear.bias.zero_()
    print("actor_output_init zero")


def _make_eval_env(args: argparse.Namespace, trainer: dict[str, Any], train_env: RslRlUvmsEnv) -> RslRlUvmsEnv:
    eval_envs = int(getattr(args, "eval_envs", 0) or trainer.get("eval_envs", train_env.num_envs))
    eval_env = RslRlUvmsEnv.from_experiment(
        args.config,
        backend=args.backend,
        robot_count=eval_envs,
        clip_actions=args.clip_actions,
    )
    _configure_adapter(eval_env, trainer, clip_actions=args.clip_actions)
    return eval_env


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
        for key in (
            "mean_position_error",
            "mean_yaw_error",
            "success_now_rate",
            "success_rate",
            "out_of_bounds_rate",
            "surface_violation_rate",
        ):
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
        "success_now_rate",
        "success_rate",
        "timeout_rate",
        "out_of_bounds_rate",
        "surface_violation_rate",
        "final_mean_position_error",
        "final_mean_yaw_error",
        "final_success_now_rate",
        "final_success_rate",
    ):
        if key in metrics:
            fields.append(f"{key}={metrics[key]:.4f}")
    print(label, *fields)


def _tracking_score(metrics: dict[str, float]) -> float:
    mean_pos_err = metrics.get("mean_position_error", float("inf"))
    mean_yaw_err = metrics.get("mean_yaw_error", float("inf"))
    final_pos_err = metrics.get("final_mean_position_error", mean_pos_err)
    final_yaw_err = metrics.get("final_mean_yaw_error", mean_yaw_err)
    success = metrics.get("final_success_rate", metrics.get("success_rate", 0.0))
    return (
        float(mean_pos_err)
        + 0.5 * float(final_pos_err)
        + 0.25 * float(mean_yaw_err)
        + 0.1 * float(final_yaw_err)
        - 0.02 * float(success)
    )


def _log_eval_metrics(runner: OnPolicyRunner, label: str, metrics: dict[str, float], step: int) -> None:
    writer = getattr(getattr(runner, "logger", None), "writer", None)
    if writer is None:
        return
    safe_label = label.replace(" ", "_").replace("=", "_")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"Eval/{safe_label}/{key}", float(value), int(step))
    writer.add_scalar(f"Eval/{safe_label}/tracking_score", _tracking_score(metrics), int(step))
    flush = getattr(writer, "flush", None)
    if callable(flush):
        flush()


def _save_best_if_needed(
    runner: OnPolicyRunner,
    metrics: dict[str, float],
    *,
    candidate_model: Path,
    best_model: Path,
    best_score: float,
    label: str,
) -> tuple[float, bool]:
    score = _tracking_score(metrics)
    if score < best_score:
        runner.save(str(candidate_model), infos={label: metrics})
        shutil.copyfile(candidate_model, best_model)
        print("best_model_update", f"source={candidate_model.name}", f"score={score:.6f}", f"label={label}")
        return score, True
    return best_score, False


def _learn_with_periodic_best(
    runner: OnPolicyRunner,
    env: RslRlUvmsEnv,
    *,
    iterations: int,
    init_at_random_ep_len: bool,
    eval_steps: int,
    eval_interval: int,
    early_stop_patience: int,
    best_model: Path,
    best_score: float,
) -> float:
    if iterations <= 0:
        print("ppo_finetune skipped iterations=0")
        return best_score
    if eval_interval <= 0:
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=init_at_random_ep_len)
        return best_score

    remaining = int(iterations)
    first_chunk = True
    since_best = 0
    while remaining > 0:
        chunk = min(int(eval_interval), remaining)
        runner.learn(
            num_learning_iterations=chunk,
            init_at_random_ep_len=bool(init_at_random_ep_len and first_chunk),
        )
        first_chunk = False
        remaining -= chunk

        candidate_model = best_model.with_name(f"model_eval_{runner.current_learning_iteration}.pt")
        metrics = _evaluate(runner, env, eval_steps)
        _print_eval(f"periodic_eval iteration={runner.current_learning_iteration}", metrics)
        _log_eval_metrics(runner, "periodic_eval", metrics, runner.current_learning_iteration)
        best_score, improved = _save_best_if_needed(
            runner,
            metrics,
            candidate_model=candidate_model,
            best_model=best_model,
            best_score=best_score,
            label=f"iteration_{runner.current_learning_iteration}",
        )
        since_best = 0 if improved else since_best + chunk
        if early_stop_patience > 0 and since_best >= early_stop_patience:
            print(
                "early_stop",
                f"iteration={runner.current_learning_iteration}",
                f"since_best={since_best}",
                f"patience={early_stop_patience}",
            )
            break
        # RSL-RL stores the last completed iteration. Advance it before chunked continuation.
        runner.current_learning_iteration += 1
    return best_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="hover_vehicle", help="Packaged experiment name or experiment folder")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--backend", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--device", default=None, help="Torch training device, e.g. cpu or cuda")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--eval-envs", type=int, default=None, help="Number of environments used by the separate eval env")
    parser.add_argument("--clip-actions", type=float, default=None)
    parser.add_argument("--init-random-episode-length", action="store_true")
    parser.add_argument("--best-eval-interval", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--logger", choices=["tensorboard", "wandb", "neptune"], default=None)
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
    _configure_adapter(env, trainer, clip_actions=args.clip_actions)
    eval_env = _make_eval_env(args, trainer, env)

    seed = int(env.cfg.get("env", {}).get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = args.device or ("cuda" if env.unwrapped.backend == "gpu" else "cpu")
    eval_steps = int(args.eval_steps if args.eval_steps is not None else trainer.get("eval_steps", 200))
    best_eval_interval = int(
        args.best_eval_interval if args.best_eval_interval is not None else trainer.get("best_eval_interval", 0)
    )
    early_stop_patience = int(
        args.early_stop_patience if args.early_stop_patience is not None else trainer.get("early_stop_patience", 0)
    )
    iterations = int(train_cfg.pop("num_learning_iterations"))
    train_cfg["logger"] = args.logger or str(train_cfg.get("logger", "tensorboard"))
    run_name = str(train_cfg.get("run_name", Path(str(args.config)).stem or "uvms_rl"))
    log_dir = Path(args.log_dir) if args.log_dir else _make_log_dir(trainer.get("log_root", "recordings/rl_runs"), env.cfg.get("name", args.config), run_name)
    log_dir.mkdir(parents=True, exist_ok=True)

    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir=str(log_dir), device=device)
    if bool(trainer.get("zero_actor_output", False)):
        _zero_actor_output(runner)

    before = _evaluate(runner, eval_env, eval_steps)
    _print_eval("before_training", before)
    _log_eval_metrics(runner, "before_training", before, runner.current_learning_iteration)

    used_teacher = _pretrain_with_teacher(runner, env, trainer)
    pretrain_label = "after_teacher" if used_teacher else "initial_policy"
    after_teacher = _evaluate(runner, eval_env, eval_steps)
    teacher_model = log_dir / ("model_after_teacher.pt" if used_teacher else "model_initial.pt")
    runner.save(str(teacher_model), infos={"before": before, pretrain_label: after_teacher})
    best_model = log_dir / "model_best.pt"
    shutil.copyfile(teacher_model, best_model)
    best_score = _tracking_score(after_teacher)
    _print_eval(pretrain_label, after_teacher)
    _log_eval_metrics(runner, pretrain_label, after_teacher, runner.current_learning_iteration)
    print("best_model_update", f"source={teacher_model.name}", f"score={best_score:.6f}", f"label={pretrain_label}")

    best_score = _learn_with_periodic_best(
        runner,
        eval_env,
        iterations=iterations,
        init_at_random_ep_len=bool(args.init_random_episode_length),
        eval_steps=eval_steps,
        eval_interval=best_eval_interval,
        early_stop_patience=early_stop_patience,
        best_model=best_model,
        best_score=best_score,
    )

    after = _evaluate(runner, eval_env, eval_steps)
    final_model = log_dir / "model_final.pt"
    runner.save(str(final_model), infos={"before": before, pretrain_label: after_teacher, "after": after})
    final_score = _tracking_score(after)
    if final_score < best_score:
        shutil.copyfile(final_model, best_model)
        best_source = final_model
        best_score = final_score
    else:
        best_source = best_model
    _log_eval_metrics(runner, "after_training", after, runner.current_learning_iteration)
    _print_eval("after_training", after)
    print("log_dir", log_dir)
    if str(train_cfg.get("logger", "tensorboard")).lower() == "tensorboard":
        print("tensorboard", f"tensorboard --logdir {log_dir}")
    print("initial_model", teacher_model)
    print("final_model", final_model)
    print("best_model", best_model, "source", best_source.name, f"score={best_score:.6f}")


if __name__ == "__main__":
    main()
