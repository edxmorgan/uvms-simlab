"""Public controller configuration boundary; independent of plant configuration."""
import json

from simlab.controllers.model_profile import NominalModel
from simlab.dynamics_profiles import load_robot_dynamics_profile


def controller_configuration(robot, request):
    controller = robot.controller_instance(request.controller_name)
    if controller is None or robot._controller_is_replay(controller):
        raise ValueError('requested feedback controller is unavailable')
    if request.operation not in ('get', 'set_model'):
        raise ValueError('operation must be get or set_model')
    if request.operation == 'set_model':
        if 'real' in robot.prefix:
            raise ValueError('runtime model configuration is simulation-only')
        if robot.status != 'inactive' and not robot.sim_reset_hold:
            raise ValueError('controller configuration requires inactive or held robot')
        if getattr(robot, '_replay_reset_pending', False):
            raise ValueError('cannot configure while a replay reset response is pending')
        replay = robot.controller_instance('CmdReplay')
        if replay is not None and replay.playback_status() not in ('stopped', 'complete', 'error', 'no_csv'):
            raise ValueError('cannot configure during replay preparation or execution')
        model = NominalModel.from_profile(load_robot_dynamics_profile(request.model_profile))
        controller.set_model_profile(model)
    # allow_nan=False fails closed on non-finite effective settings.
    return json.dumps(controller.configuration_snapshot(), sort_keys=True, allow_nan=False)
