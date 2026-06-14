import importlib

import numpy as np
import pytest

from uvms_rl import UvmsBatchEnv
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


def test_cpu_uvms_rl_hover_vehicle_shapes_and_types():
    env = UvmsBatchEnv(
        robot_count=16,
        control_dt=1.0 / 150.0,
        sim_dt=1.0 / 150.0,
        seed=7,
        task="hover_vehicle",
        backend="cpu",
    )

    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (16, 31)

    actions = np.zeros((16, env.action_dim), dtype=np.float32)
    obs, rewards, dones, info = env.step(actions)

    assert obs.shape == (16, 31)
    assert rewards.shape == (16,)
    assert rewards.dtype == np.float32
    assert dones.shape == (16,)
    assert dones.dtype == bool
    assert info.backend == "cpu"
    assert info.tick_id == 2


@pytest.mark.skipif(not _gpu_available(), reason="UVMS GPU backend is not available")
def test_gpu_uvms_rl_hover_vehicle_returns_cuda_tensors():
    import torch

    env = UvmsBatchEnv(
        robot_count=16,
        control_dt=1.0 / 150.0,
        sim_dt=1.0 / 150.0,
        seed=7,
        task="hover_vehicle",
        backend="gpu",
    )

    obs = env.reset()
    assert torch.is_tensor(obs)
    assert obs.is_cuda
    assert tuple(obs.shape) == (16, 31)

    actions = torch.zeros((16, env.action_dim), dtype=torch.float32, device="cuda")
    obs, rewards, dones, info = env.step(actions)

    assert obs.is_cuda
    assert rewards.is_cuda
    assert dones.is_cuda
    assert tuple(obs.shape) == (16, 31)
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

    cpu = UvmsBatchEnv(
        robot_count=robot_count,
        control_dt=dt,
        sim_dt=dt,
        seed=9,
        task="hover_vehicle",
        backend="cpu",
    )
    gpu = UvmsBatchEnv(
        robot_count=robot_count,
        control_dt=dt,
        sim_dt=dt,
        seed=9,
        task="hover_vehicle",
        backend="gpu",
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
