"""Public controller configuration boundary; independent of plant configuration."""
import json
import numpy as np

from simlab.controllers.model_profile import NominalModel
from simlab.dynamics_profiles import load_robot_dynamics_profile


def controller_configuration(robot, request):
    controller = robot.controller_instance(request.controller_name)
    if controller is None or robot._controller_is_replay(controller):
        raise ValueError('requested feedback controller is unavailable')
    if request.operation not in ('get', 'set_model', 'set_gains'):
        raise ValueError('operation must be get, set_model or set_gains')
    if request.operation != 'get':
        if 'real' in robot.prefix:
            raise ValueError('runtime model configuration is simulation-only')
        if robot.status != 'inactive' and not robot.sim_reset_hold:
            raise ValueError('controller configuration requires inactive or held robot')
        if getattr(robot, '_replay_reset_pending', False):
            raise ValueError('cannot configure while a replay reset response is pending')
        replay = robot.controller_instance('CmdReplay')
        if replay is not None and replay.playback_status() not in ('stopped', 'complete', 'error', 'no_csv'):
            raise ValueError('cannot configure during replay preparation or execution')
        if request.operation == 'set_model':
            model = NominalModel.from_profile(load_robot_dynamics_profile(request.model_profile))
            controller.set_model_profile(model)
        else:
            allowed = {
                'PID': ('kp','ki','kd','arm_kp','arm_ki','arm_kd'),
                'InvDyn': ('kp','ki','kd','arm_kp','arm_ki','arm_kd'),
                'SMC': tuple(f'{part}_{field}' for part in ('vehicle','arm')
                             for field in ('surface','damping','switching','boundary')),
            }.get(controller.registry_name, ())
            gains = json.loads(request.gains_json)
            if not isinstance(gains, dict) or not gains or not set(gains).issubset(allowed):
                raise ValueError('set_gains requires supported gain arrays only')
            validated = {}
            for name, values in gains.items():
                original = np.asarray(getattr(controller,name), dtype=float)
                value = np.asarray(values, dtype=float)
                if value.shape != original.shape or not np.isfinite(value).all() or np.any(value < 0):
                    raise ValueError(f'invalid gain array {name}')
                if name.endswith(('_surface','_boundary')) and np.any(value <= 0):
                    raise ValueError(f'{name} must be positive')
                validated[name] = value.copy()
            # Atomic validation: no partial write if any supplied field is invalid.
            for name, value in validated.items():
                setattr(controller,name,value)
            controller.reset_controller_state()
    # allow_nan=False fails closed on non-finite effective settings.
    return json.dumps(controller.configuration_snapshot(), sort_keys=True, allow_nan=False)
