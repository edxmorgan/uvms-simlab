import importlib

import numpy as np
import pytest

from uvms_rl import UvmsBatchEnv
from uvms_rl.config import load_experiment
from uvms_rl.tensor import as_numpy, torch_available


def _initial_observations(robot_count, seed=123):
    rng = np.random.default_rng(seed)
    obs = rng.normal(0.0, 0.05, size=(robot_count, UvmsBatchEnv.observation_dim)).astype(np.float32)
    obs[:, 3:6] *= 0.1
    obs[:, 12:] *= 0.1
    return obs


def _actions(step_count, robot_count, seed=456):
    rng = np.random.default_rng(seed)
    return rng.uniform(
        -0.2,
        0.2,
        size=(step_count, robot_count, UvmsBatchEnv.action_dim),
    ).astype(np.float32)


def _gpu_available():
    if not torch_available():
        return False
    return importlib.util.find_spec("ros2_control_blue_reach_5._batch_uvms_gpu_core") is not None


def _rsl_rl_available():
    return (
        importlib.util.find_spec("rsl_rl") is not None
        and importlib.util.find_spec("tensordict") is not None
        and importlib.util.find_spec("torch") is not None
    )


def test_cpu_uvms_rl_hover_vehicle_shapes_and_types():
    experiment = load_experiment("hover_vehicle")
    env_cfg = experiment.config["env"]
    env = UvmsBatchEnv(
        robot_count=16,
        control_dt=1.0 / 150.0,
        sim_dt=1.0 / 150.0,
        seed=7,
        task=experiment.task_cls,
        task_config=experiment.config["task"],
        backend="cpu",
        dynamics_profile=env_cfg["dynamics_profile"],
    )

    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (16, 35)

    actions = np.zeros((16, env.action_dim), dtype=np.float32)
    obs, rewards, dones, info = env.step(actions)

    assert obs.shape == (16, 35)
    assert rewards.shape == (16,)
    assert rewards.dtype == np.float32
    assert dones.shape == (16,)
    assert dones.dtype == bool
    assert info.backend == "cpu"
    assert info.tick_id == 2

@pytest.mark.skipif(not _rsl_rl_available(), reason="RSL-RL training dependencies are not installed")
def test_rsl_adapter_returns_tensordict_and_same_step_reset():
    import torch

    from uvms_rl.rsl_adapter import RslRlUvmsEnv

    experiment = load_experiment("hover_vehicle")
    env_cfg = experiment.config["env"]
    env = UvmsBatchEnv(
        robot_count=8,
        control_dt=1.0 / 150.0,
        sim_dt=1.0 / 150.0,
        max_episode_steps=3,
        seed=7,
        task=experiment.task_cls,
        task_config=experiment.config["task"],
        backend="cpu",
        dynamics_profile=env_cfg["dynamics_profile"],
    )
    rsl_env = RslRlUvmsEnv(env)

    obs = rsl_env.get_observations()
    assert set(obs.keys()) == {"policy"}
    assert tuple(obs["policy"].shape) == (8, 35)
    assert rsl_env.num_envs == 8
    assert rsl_env.num_actions == env.action_dim
    assert rsl_env.device == torch.device("cpu")
    rsl_env.episode_length_buf.fill_(2)

    actions = torch.zeros((8, env.action_dim), dtype=torch.float32)
    next_obs, rewards, dones, extras = rsl_env.step(actions)

    assert set(next_obs.keys()) == {"policy"}
    assert tuple(next_obs["policy"].shape) == (8, 35)
    assert tuple(rewards.shape) == (8,)
    assert rewards.dtype == torch.float32
    assert tuple(dones.shape) == (8,)
    assert dones.dtype == torch.bool
    assert torch.all(dones)
    assert torch.all(extras["time_outs"])
    assert torch.all(rsl_env.episode_length_buf == 0)
    assert "/uvms_rl/sim_time" in extras["log"]


@pytest.mark.skipif(not _gpu_available(), reason="UVMS GPU backend is not available")
def test_gpu_uvms_rl_hover_vehicle_returns_cuda_tensors():
    import torch

    experiment = load_experiment("hover_vehicle")
    env_cfg = experiment.config["env"]
    env = UvmsBatchEnv(
        robot_count=16,
        control_dt=1.0 / 150.0,
        sim_dt=1.0 / 150.0,
        seed=7,
        task=experiment.task_cls,
        task_config=experiment.config["task"],
        backend="gpu",
        dynamics_profile=env_cfg["dynamics_profile"],
    )

    obs = env.reset()
    assert torch.is_tensor(obs)
    assert obs.is_cuda
    assert tuple(obs.shape) == (16, 35)

    actions = torch.zeros((16, env.action_dim), dtype=torch.float32, device="cuda")
    obs, rewards, dones, info = env.step(actions)

    assert obs.is_cuda
    assert rewards.is_cuda
    assert dones.is_cuda
    assert tuple(obs.shape) == (16, 35)
    assert tuple(rewards.shape) == (16,)
    assert tuple(dones.shape) == (16,)
    assert info.backend == "gpu"
    assert info.tick_id == 2


@pytest.mark.skipif(not _gpu_available(), reason="UVMS GPU backend is not available")
def test_cpu_gpu_uvms_rl_hover_vehicle_parity():
    import torch

    robot_count = 32
    step_count = 12
    dt = 1.0 / 150.0
    init_obs = _initial_observations(robot_count)
    actions = _actions(step_count, robot_count)
    experiment = load_experiment("hover_vehicle")
    env_cfg = experiment.config["env"]

    cpu = UvmsBatchEnv(
        robot_count=robot_count,
        control_dt=dt,
        sim_dt=dt,
        seed=9,
        task=experiment.task_cls,
        task_config=experiment.config["task"],
        backend="cpu",
        dynamics_profile=env_cfg["dynamics_profile"],
    )
    gpu = UvmsBatchEnv(
        robot_count=robot_count,
        control_dt=dt,
        sim_dt=dt,
        seed=9,
        task=experiment.task_cls,
        task_config=experiment.config["task"],
        backend="gpu",
        dynamics_profile=env_cfg["dynamics_profile"],
    )

    cpu.reset()
    gpu.reset()
    cpu.reset(observations=init_obs)
    gpu.reset(observations=init_obs)

    max_sim_error = 0.0
    max_policy_error = 0.0
    max_reward_error = 0.0
    for i in range(step_count):
        cpu_obs, cpu_rewards, cpu_dones, _ = cpu.step(actions[i])
        gpu_obs, gpu_rewards, gpu_dones, _ = gpu.step(torch.as_tensor(actions[i], device="cuda"))

        max_sim_error = max(
            max_sim_error,
            float(np.max(np.abs(cpu.sim_observations() - as_numpy(gpu.sim_observations())))),
        )
        max_policy_error = max(
            max_policy_error,
            float(np.max(np.abs(cpu_obs - as_numpy(gpu_obs)))),
        )
        max_reward_error = max(
            max_reward_error,
            float(np.max(np.abs(cpu_rewards - as_numpy(gpu_rewards)))),
        )
        np.testing.assert_array_equal(cpu_dones, as_numpy(gpu_dones))

    assert max_sim_error < 5e-3
    assert max_policy_error < 5e-3
    assert max_reward_error < 5e-3


@pytest.mark.skipif(not _rsl_rl_available(), reason="RSL-RL training dependencies are not installed")
def test_pd_hover_teacher_matches_diffuv_hold_restoring_wrench():
    import torch

    from uvms_rl.rsl_adapter import RslRlUvmsEnv

    experiment = load_experiment("hover_vehicle")
    env_cfg = experiment.config["env"]
    env = UvmsBatchEnv(
        robot_count=3,
        control_dt=1.0 / 150.0,
        sim_dt=1.0 / 150.0,
        seed=7,
        task=experiment.task_cls,
        task_config=experiment.config["task"],
        backend="cpu",
        dynamics_profile=env_cfg["dynamics_profile"],
    )
    state = np.zeros((3, env.observation_dim), dtype=np.float32)
    state[:, 2] = 0.5
    env.reset(observations=state)
    env.task.target_xyz[:] = state[:, 0:3]
    env.task.target_yaw[:] = state[:, 5]
    env.task.sync_reset_observations(env, state)
    rsl_env = RslRlUvmsEnv(env, reset_on_init=False)

    actions = rsl_env.teacher_actions(rsl_env.get_observations(), {"name": "pd_hover"})
    expected = rsl_env._restoring_wrench(torch.as_tensor(state, dtype=torch.float32))

    torch.testing.assert_close(actions[:, 0:6], expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actions[:, 6:], torch.zeros_like(actions[:, 6:]), rtol=0.0, atol=0.0)

@pytest.mark.skipif(not _rsl_rl_available(), reason="RSL-RL training dependencies are not installed")
def test_pd_hover_teacher_maps_ned_error_through_body_jacobian_transpose():
    import torch

    from uvms_rl.rsl_adapter import RslRlUvmsEnv

    experiment = load_experiment("hover_vehicle")
    env_cfg = experiment.config["env"]
    env = UvmsBatchEnv(
        robot_count=1,
        control_dt=1.0 / 150.0,
        sim_dt=1.0 / 150.0,
        seed=7,
        task=experiment.task_cls,
        task_config=experiment.config["task"],
        backend="cpu",
        dynamics_profile=env_cfg["dynamics_profile"],
    )
    state = np.zeros((1, env.observation_dim), dtype=np.float32)
    state[0, 2] = 0.5
    state[0, 5] = np.pi / 2.0
    env.reset(observations=state)
    env.task.target_xyz[:] = state[:, 0:3]
    env.task.target_xyz[:, 0] += 1.0
    env.task.target_yaw[:] = state[:, 5]
    env.task.sync_reset_observations(env, state)
    rsl_env = RslRlUvmsEnv(env, reset_on_init=False)

    teacher = {"name": "pd_hover", "force_kp": 10.0, "force_kd": 0.0, "yaw_kp": 0.0, "angular_kd": 0.0}
    actions = rsl_env.teacher_actions(rsl_env.get_observations(), teacher)
    restoring = rsl_env._restoring_wrench(torch.as_tensor(state, dtype=torch.float32))
    pid_only = actions[:, 0:6] - restoring

    expected = torch.tensor([[0.0, -10.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(pid_only, expected, rtol=1e-6, atol=1e-6)

def test_uvms_batch_env_requires_explicit_dynamics_profile():
    with pytest.raises(ValueError, match="dynamics_profile must be provided explicitly"):
        UvmsBatchEnv(robot_count=1)


def test_uvms_batch_env_rejects_unknown_backend():
    experiment = load_experiment("hover_vehicle")
    env_cfg = experiment.config["env"]

    for backend in ("", "cuda", "gpuu", "anything_else"):
        with pytest.raises(ValueError, match="Expected 'cpu' or 'gpu'"):
            UvmsBatchEnv(
                robot_count=1,
                backend=backend,
                dynamics_profile=env_cfg["dynamics_profile"],
            )


def test_uvms_batch_env_normalizes_cpu_backend_case():
    experiment = load_experiment("hover_vehicle")
    env_cfg = experiment.config["env"]
    env = UvmsBatchEnv(
        robot_count=1,
        backend=" CPU ",
        dynamics_profile=env_cfg["dynamics_profile"],
    )

    assert env.backend == "cpu"


def test_hover_success_can_avoid_episode_termination():
    experiment = load_experiment("hover_vehicle_stage1_fixed")
    env_cfg = experiment.config["env"]
    task_cfg = dict(experiment.config["task"])
    task_cfg.update(
        {
            "initial_xyz_noise": 0.0,
            "initial_yaw_noise": 0.0,
            "initial_velocity_noise": 0.0,
            "success_position_tolerance": 1.0,
            "success_yaw_tolerance": 1.0,
            "success_streak_steps": 1,
            "success_terminates": False,
        }
    )
    env = UvmsBatchEnv(
        robot_count=2,
        control_dt=env_cfg["control_dt"],
        sim_dt=env_cfg["sim_dt"],
        task=experiment.task_cls,
        task_config=task_cfg,
        backend="cpu",
        dynamics_profile=env_cfg["dynamics_profile"],
    )
    env.reset()

    _, _, dones, info = env.step(np.zeros((2, env.action_dim), dtype=np.float32))

    assert not np.any(dones)
    assert info.task["success_rate"] == pytest.approx(1.0)
    assert info.task["success_now_rate"] == pytest.approx(1.0)
