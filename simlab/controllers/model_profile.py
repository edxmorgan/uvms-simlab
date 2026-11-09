"""Fixed simulation-controller model; never selected by the plant profile menu.

The 75/81 and 27/33 element layouts belong to different generated CasADi
functions. Pack by physical field, not by copying an arbitrary plant vector.
"""
from dataclasses import dataclass
import hashlib
import json

import numpy as np


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
