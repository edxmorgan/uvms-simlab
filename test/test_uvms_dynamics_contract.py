import re
from pathlib import Path

import numpy as np
import pytest

from simlab.dynamics_profiles import (
    load_robot_dynamics_profile,
    manipulator_dynamics_from_config,
    vehicle_dynamics_from_config,
)
from uvms_rl import UvmsBatchEnv
from uvms_rl.config import load_experiment
from uvms_rl.rl_env import arm_environment, pack_arm_params, pack_vehicle_params


SRC_ROOT = Path(__file__).resolve().parents[2]
SIMULATOR_ROOT = SRC_ROOT / "uvms-simulator"


def _numbers_from_cpp_array(source: str, name: str) -> list[float]:
    match = re.search(rf"{re.escape(name)}\[[^\]]+\]\s*=\s*\{{(?P<body>.*?)\}};", source, re.S)
    assert match, f"missing C++ array {name}"
    return [float(item.replace("F", "")) for item in re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?F?", match.group("body"))]


def _xacro_joint_params() -> dict[str, dict[str, float]]:
    path = SIMULATOR_ROOT / "description/ros2_control/robot_system_multi_interface.ros2_control.xacro"
    text = path.read_text(encoding="utf-8")
    result = {}
    for axis in ("e", "d", "c", "b", "a"):
        match = re.search(rf'<joint name="\${{prefix}}_axis_{axis}">(?P<body>.*?)</joint>', text, re.S)
        assert match, f"missing axis_{axis} joint in {path}"
        body = match.group("body")
        params = {}
        for name in (
            "max_effort",
            "soft_min_position",
            "soft_max_position",
            "kt",
            "forward_I_static",
            "backward_I_static",
        ):
            param = re.search(rf'<param name="{name}">(?P<value>[-+]?\d*\.?\d+)</param>', body)
            assert param, f"missing {name} for axis_{axis}"
            params[name] = float(param.group("value"))
        result[axis] = params
    return result


def test_batch_arm_actuator_constants_match_hardware_xacro():
    cpp = (SIMULATOR_ROOT / "hardware/batch_uvms_core.cpp").read_text(encoding="utf-8")
    xacro = _xacro_joint_params()
    axes = ("e", "d", "c", "b", "a")

    expected = {
        "kArmPositionMin": [xacro[axis]["soft_min_position"] for axis in axes],
        "kArmPositionMax": [xacro[axis]["soft_max_position"] for axis in axes],
        "kArmCurrentMax": [xacro[axis]["max_effort"] for axis in axes],
        "kArmMotorKt": [xacro[axis]["kt"] for axis in axes],
        "kArmForwardStaticCurrent": [xacro[axis]["forward_I_static"] for axis in axes],
        "kArmBackwardStaticCurrent": [xacro[axis]["backward_I_static"] for axis in axes],
    }

    for name, values in expected.items():
        np.testing.assert_allclose(_numbers_from_cpp_array(cpp, name), values, rtol=0.0, atol=1e-7)


def test_rl_dynamics_pack_order_matches_profile_message_helpers():
    profile = load_robot_dynamics_profile("dory_alpha")
    vehicle = vehicle_dynamics_from_config(profile["vehicle"])
    manipulator = manipulator_dynamics_from_config(profile["manipulator"])

    expected_vehicle = np.asarray(
        [
            vehicle.m_x_du,
            vehicle.m_y_dv,
            vehicle.m_z_dw,
            vehicle.mz_g_x_dq,
            vehicle.mz_g_y_dp,
            vehicle.mz_g_k_dv,
            vehicle.mz_g_m_du,
            vehicle.i_x_k_dp,
            vehicle.i_y_m_dq,
            vehicle.i_z_n_dr,
            vehicle.weight,
            vehicle.buoyancy,
            vehicle.x_g_weight_minus_x_b_buoyancy,
            vehicle.y_g_weight_minus_y_b_buoyancy,
            vehicle.z_g_weight_minus_z_b_buoyancy,
            vehicle.x_u,
            vehicle.y_v,
            vehicle.z_w,
            vehicle.k_p,
            vehicle.m_q,
            vehicle.n_r,
            vehicle.x_uu,
            vehicle.y_vv,
            vehicle.z_ww,
            vehicle.k_pp,
            vehicle.m_qq,
            vehicle.n_rr,
            *vehicle.current_velocity,
        ],
        dtype=np.float32,
    )
    expected_arm = np.asarray(
        [
            *manipulator.link_masses,
            *manipulator.link_first_moments,
            *manipulator.link_inertias,
            *manipulator.viscous_friction,
            *manipulator.coulomb_friction,
            *manipulator.static_friction,
            *manipulator.stribeck_velocity,
            *manipulator.gravity_vector,
            *manipulator.payload_com,
            manipulator.payload_mass,
            *manipulator.base_pose,
            *manipulator.world_pose,
            *manipulator.tip_offset_pose,
        ],
        dtype=np.float32,
    )

    vehicle_params = pack_vehicle_params(profile)
    arm_params = pack_arm_params(profile)
    assert vehicle_params.shape == (33,)
    assert arm_params.shape == (81,)
    np.testing.assert_array_equal(vehicle_params, expected_vehicle)
    np.testing.assert_array_equal(arm_params, expected_arm)
    assert arm_environment(profile) == (
        pytest.approx(manipulator.endeffector_mass),
        pytest.approx(manipulator.endeffector_damping),
        pytest.approx(manipulator.endeffector_stiffness),
        pytest.approx(manipulator.baumgarte_alpha),
    )


def test_hover_vehicle_rejects_negative_depth_targets_by_default():
    experiment = load_experiment("hover_vehicle")
    env_cfg = experiment.config["env"]
    task_cfg = dict(experiment.config["task"])
    task_cfg["target_z"] = [-0.8, -0.4]

    with pytest.raises(ValueError, match="target_z is below the minimum valid depth"):
        UvmsBatchEnv(
            robot_count=2,
            control_dt=env_cfg["control_dt"],
            sim_dt=env_cfg["sim_dt"],
            task=experiment.task_cls,
            task_config=task_cfg,
            backend="cpu",
            dynamics_profile=env_cfg["dynamics_profile"],
        )


def test_hover_vehicle_can_allow_negative_depth_for_explicit_surface_tests():
    experiment = load_experiment("hover_vehicle")
    env_cfg = experiment.config["env"]
    task_cfg = dict(experiment.config["task"])
    task_cfg["target_z"] = [-0.8, -0.4]
    task_cfg["allow_negative_z"] = True

    env = UvmsBatchEnv(
        robot_count=2,
        control_dt=env_cfg["control_dt"],
        sim_dt=env_cfg["sim_dt"],
        task=experiment.task_cls,
        task_config=task_cfg,
        backend="cpu",
        dynamics_profile=env_cfg["dynamics_profile"],
    )
    obs = env.reset()
    assert obs.shape == (2, env.policy_observation_dim)
