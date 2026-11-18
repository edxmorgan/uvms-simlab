from dataclasses import dataclass
import hashlib
import json
import numpy as np
import os
from pathlib import Path

import ament_index_python
from rclpy.node import Node
from ros2_control_blue_reach_5.msg import SimCameraDynamics, SimManipulatorDynamics, SimVehicleDynamics
from ros2_control_blue_reach_5.srv import SetSimDynamics


def dynamics_profiles_root() -> Path:
    return Path(ament_index_python.get_package_share_directory("simlab")) / "dynamics_profiles"


def list_robot_dynamics_profiles() -> list[str]:
    root = dynamics_profiles_root()
    if not root.exists():
        return []
    return sorted(path.stem for path in root.glob("*.json") if path.is_file())


def resolve_robot_dynamics_profile(profile_name: str) -> Path:
    profile = str(profile_name or "").strip()
    if not profile:
        raise ValueError("robot dynamics profile name is required")
    path = Path(os.path.expanduser(profile))
    if path.is_absolute():
        return path
    if path.suffix:
        return dynamics_profiles_root() / path
    return dynamics_profiles_root() / f"{profile}.json"


def load_robot_dynamics_profile(profile_name: str, node: Node | None = None) -> dict:
    try:
        profile_path = resolve_robot_dynamics_profile(profile_name)
        loaded = json.loads(profile_path.read_text())
    except Exception as exc:
        if node is not None:
            node.get_logger().error(f"Failed to load robot dynamics profile '{profile_name}': {exc}")
        return {}
    if not isinstance(loaded, dict):
        if node is not None:
            node.get_logger().error(f"Robot dynamics profile must be a JSON object: {profile_path}")
        return {}
    return loaded


def is_valid_robot_dynamics_profile(profile: dict) -> bool:
    if not isinstance(profile, dict):
        return False
    return (
        isinstance(profile.get("manipulator"), dict)
        or isinstance(profile.get("vehicle"), dict)
        or isinstance(profile.get("camera"), dict)
    )


def required_float(config: dict, key: str) -> float:
    if key not in config:
        raise KeyError(f"missing required dynamics parameter '{key}'")
    return float(config[key])


def required_vector(config: dict, key: str, size: int) -> list[float]:
    if key not in config:
        raise KeyError(f"missing required dynamics vector '{key}'")
    values = [float(v) for v in list(config[key])]
    if len(values) != size:
        raise ValueError(f"dynamics vector '{key}' must contain exactly {size} values")
    return values


def required_matrix(config: dict, key: str, rows: int, cols: int) -> list[float]:
    if key not in config:
        raise KeyError(f"missing required dynamics matrix '{key}'")
    source = config[key]
    if isinstance(source, list) and all(isinstance(row, list) for row in source):
        if len(source) != rows or any(len(row) != cols for row in source):
            raise ValueError(f"dynamics matrix '{key}' must be {rows}x{cols}")
        values = [float(item) for row in source for item in row]
    else:
        values = [float(item) for item in list(source)]
    size = rows * cols
    if len(values) != size:
        raise ValueError(f"dynamics matrix '{key}' must contain exactly {size} values")
    return values


def optional_bool(config: dict, key: str) -> bool | None:
    if key not in config:
        return None
    value = config[key]
    if value is None:
        raise ValueError(f"boolean profile parameter '{key}' must be true or false")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"boolean profile parameter '{key}' must be true or false")


def vehicle_dynamics_from_config(dynamics: dict) -> SimVehicleDynamics:
    msg = SimVehicleDynamics()
    msg.m_x_du = required_float(dynamics, "m_x_du")
    msg.m_y_dv = required_float(dynamics, "m_y_dv")
    msg.m_z_dw = required_float(dynamics, "m_z_dw")
    msg.mz_g_x_dq = required_float(dynamics, "mz_g_x_dq")
    msg.mz_g_y_dp = required_float(dynamics, "mz_g_y_dp")
    msg.mz_g_k_dv = required_float(dynamics, "mz_g_k_dv")
    msg.mz_g_m_du = required_float(dynamics, "mz_g_m_du")
    msg.i_x_k_dp = required_float(dynamics, "i_x_k_dp")
    msg.i_y_m_dq = required_float(dynamics, "i_y_m_dq")
    msg.i_z_n_dr = required_float(dynamics, "i_z_n_dr")
    msg.weight = required_float(dynamics, "weight")
    msg.buoyancy = required_float(dynamics, "buoyancy")
    msg.x_g_weight_minus_x_b_buoyancy = required_float(dynamics, "x_g_weight_minus_x_b_buoyancy")
    msg.y_g_weight_minus_y_b_buoyancy = required_float(dynamics, "y_g_weight_minus_y_b_buoyancy")
    msg.z_g_weight_minus_z_b_buoyancy = required_float(dynamics, "z_g_weight_minus_z_b_buoyancy")
    msg.x_u = required_float(dynamics, "x_u")
    msg.y_v = required_float(dynamics, "y_v")
    msg.z_w = required_float(dynamics, "z_w")
    msg.k_p = required_float(dynamics, "k_p")
    msg.m_q = required_float(dynamics, "m_q")
    msg.n_r = required_float(dynamics, "n_r")
    msg.x_uu = required_float(dynamics, "x_uu")
    msg.y_vv = required_float(dynamics, "y_vv")
    msg.z_ww = required_float(dynamics, "z_ww")
    msg.k_pp = required_float(dynamics, "k_pp")
    msg.m_qq = required_float(dynamics, "m_qq")
    msg.n_rr = required_float(dynamics, "n_rr")
    msg.current_velocity = required_vector(dynamics, "current_velocity", 6)
    msg.thrust_configuration_matrix = required_matrix(dynamics, "thrust_configuration_matrix", 6, 8)
    return msg


def manipulator_dynamics_from_config(dynamics: dict) -> SimManipulatorDynamics:
    msg = SimManipulatorDynamics()
    msg.link_masses = required_vector(dynamics, "link_masses", 4)
    msg.link_first_moments = required_vector(dynamics, "link_first_moments", 12)
    msg.link_inertias = required_vector(dynamics, "link_inertias", 24)
    msg.viscous_friction = required_vector(dynamics, "viscous_friction", 4)
    msg.coulomb_friction = required_vector(dynamics, "coulomb_friction", 4)
    msg.static_friction = required_vector(dynamics, "static_friction", 4)
    msg.stribeck_velocity = required_vector(dynamics, "stribeck_velocity", 4)
    msg.gravity_vector = required_vector(dynamics, "gravity_vector", 3)
    msg.payload_com = required_vector(dynamics, "payload_com", 3)
    msg.payload_mass = required_float(dynamics, "payload_mass")
    msg.payload_inertia = required_vector(dynamics, "payload_inertia", 3)
    msg.base_pose = required_vector(dynamics, "base_pose", 6)
    msg.world_pose = required_vector(dynamics, "world_pose", 6)
    msg.tip_offset_pose = required_vector(dynamics, "tip_offset_pose", 6)
    msg.joint_lock_on_deadband = required_float(dynamics, "joint_lock_on_deadband")
    msg.joint_lock_off_deadband = required_float(dynamics, "joint_lock_off_deadband")
    msg.baumgarte_alpha = required_float(dynamics, "baumgarte_alpha")
    msg.endeffector_mass = required_float(dynamics, "endeffector_mass")
    msg.endeffector_damping = required_float(dynamics, "endeffector_damping")
    msg.endeffector_stiffness = required_float(dynamics, "endeffector_stiffness")
    return msg


def set_dynamics_request_from_profile(profile: dict, include_vehicle: bool = True) -> SetSimDynamics.Request:
    request = SetSimDynamics.Request()
    request.use_coupled_dynamics = bool(profile.get("use_coupled_dynamics", False))
    manipulator = profile.get("manipulator")
    if isinstance(manipulator, dict):
        request.set_manipulator_dynamics = True
        request.manipulator = manipulator_dynamics_from_config(manipulator)
    vehicle = profile.get("vehicle")
    if include_vehicle and isinstance(vehicle, dict):
        request.set_vehicle_dynamics = True
        request.vehicle = vehicle_dynamics_from_config(vehicle)
    return request


def camera_dynamics_from_profile(profile: dict) -> SimCameraDynamics | None:
    camera = profile.get("camera")
    if not isinstance(camera, dict):
        return None

    msg = SimCameraDynamics()
    if "underwater_effect" in camera:
        msg.set_underwater_effect = True
        msg.underwater_effect = optional_bool(camera, "underwater_effect")
    if "underwater_haze" in camera:
        msg.set_underwater_haze = True
        msg.underwater_haze = float(camera["underwater_haze"])
    if "underwater_tint" in camera:
        msg.set_underwater_tint = True
        msg.underwater_tint = float(camera["underwater_tint"])
    if "underwater_blur" in camera:
        msg.set_underwater_blur = True
        msg.underwater_blur = float(camera["underwater_blur"])
    if "underwater_noise" in camera:
        msg.set_underwater_noise = True
        msg.underwater_noise = float(camera["underwater_noise"])
    if "underwater_vignette" in camera:
        msg.set_underwater_vignette = True
        msg.underwater_vignette = float(camera["underwater_vignette"])
    return msg


# Nominal controller layouts for the generated dynamics functions.
VEHICLE_FIELDS = (
    'm_x_du', 'm_y_dv', 'm_z_dw', 'mz_g_x_dq', 'mz_g_y_dp',
    'mz_g_k_dv', 'mz_g_m_du', 'i_x_k_dp', 'i_y_m_dq', 'i_z_n_dr',
    'weight', 'buoyancy', 'x_g_weight_minus_x_b_buoyancy',
    'y_g_weight_minus_y_b_buoyancy', 'z_g_weight_minus_z_b_buoyancy',
    'x_u', 'y_v', 'z_w', 'k_p', 'm_q', 'n_r',
    'x_uu', 'y_vv', 'z_ww', 'k_pp', 'm_qq', 'n_rr',
)
ARM_FIELDS = (
    ('link_masses', 4), ('link_first_moments', 12), ('link_inertias', 24),
    ('viscous_friction', 4), ('coulomb_friction', 4), ('static_friction', 4),
    ('stribeck_velocity', 4), ('gravity_vector', 3), ('payload_com', 3),
    ('payload_mass', 1), ('base_pose', 6), ('world_pose', 6),
)


def _values(config, key, size):
    values = np.asarray(config[key], dtype=float).reshape(-1)
    if values.size != size or not np.all(np.isfinite(values)):
        raise ValueError(f'nominal {key} must contain {size} finite values')
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class NominalModel:
    sha256: str
    arm_baseline: tuple
    arm_oges: tuple
    vehicle_baseline: tuple
    vehicle_oges: tuple
    current_velocity: tuple
    gripper_parameters: tuple = (300., 400., 0.)

    @classmethod
    def from_profile(cls, profile):
        if profile.get('use_coupled_dynamics', False):
            raise ValueError('controller model profiles currently require uncoupled parameter blocks')
        arm, vehicle = profile['manipulator'], profile['vehicle']
        # These generated model signatures have no payload-inertia inputs.
        if any(_values(arm, 'payload_inertia', 3)):
            raise ValueError('nominal controller models do not represent payload inertia')
        arm75 = sum((_values(arm, key, size) for key, size in ARM_FIELDS), ())
        vehicle27 = tuple(_values(vehicle, key, 1)[0] for key in VEHICLE_FIELDS)
        current = _values(vehicle, 'current_velocity', 6)
        digest = hashlib.sha256(json.dumps(
            profile, sort_keys=True, separators=(',', ':'), allow_nan=False,
        ).encode()).hexdigest()
        return cls(digest, arm75, arm75 + _values(arm, 'tip_offset_pose', 6),
                   vehicle27, vehicle27 + current, current,
                   tuple(_values(arm, key, 1)[0] for key in
                         ('endeffector_mass', 'endeffector_damping', 'endeffector_stiffness')))
