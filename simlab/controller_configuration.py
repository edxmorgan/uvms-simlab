"""Public controller configuration boundary; independent of plant configuration."""
import hashlib
import math
import time
import uuid
from enum import Enum
from std_msgs.msg import String
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import json
import numpy as np

from simlab.dynamics_profiles import NominalModel, load_robot_dynamics_profile


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
    if request.operation != 'get' and hasattr(robot, '_record_controller_context'):
        robot._record_controller_context()
    # allow_nan=False fails closed on non-finite effective settings.
    return json.dumps(controller.configuration_snapshot(), sort_keys=True, allow_nan=False)


def json_value(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def encode(value):
    return json.dumps(json_value(value), sort_keys=True, allow_nan=False)


class ControllerTrace:
    def __init__(self, node, prefix):
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.context_pub = node.create_publisher(String, f'/{prefix}/controller/context', qos)
        self.kinematics_pub = node.create_publisher(String, f'/{prefix}/performance/kinematics', 10)
        self.session = str(uuid.uuid4())
        self.revision = 0
        self._signature = None
        self._identity = None
        self._configs = {}
        self._checked = -float('inf')

    def context(self, robot, stamp, force=False):
        components = {part: getattr(getattr(robot, part+'_controller_fn', None), '__self__', None)
                      for part in ('vehicle', 'arm')}
        identity = tuple(id(c) for c in components.values())
        now = time.monotonic()
        if force or identity != self._identity or now-self._checked >= 1.:
            self._configs = {}
            for part, controller in components.items():
                if controller is None:
                    self._configs[part] = {'available': False}
                    continue
                try:
                    self._configs[part] = controller.configuration_snapshot()
                except Exception as exc:
                    self._configs[part] = {'available': False, 'error': str(exc),
                        'controller': getattr(controller, 'registry_name', type(controller).__name__)}
            self._identity, self._checked = identity, now
        context = dict(schema_version=1, robot_prefix=robot.prefix,
            selected_controller=getattr(robot, 'controller_name', None),
            control_mode=getattr(robot, 'control_mode', None),
            control_space=getattr(robot, 'control_space', None),
            component_semantics='bound controller implementations; control_mode determines routing',
            components=self._configs)
        signature = encode(context)
        if signature != self._signature:
            self.revision += 1
            self._signature = signature
            context.update(session_id=self.session, revision=self.revision,
                stamp_ns=stamp.sec*1000000000+stamp.nanosec,
                configuration_sha256=hashlib.sha256(signature.encode()).hexdigest())
            self.context_pub.publish(String(data=encode(context)))
        return dict(session_id=self.session, revision=self.revision)

    def kinematics(self, robot, stamp, state, reference, metrics):
        context = self.context(robot, stamp)
        fields = ('pose','body_vel','body_acc','q','dq','ddq','tool_pose',
                  'trajectory_direction','sim_time','feedback_age_sec','feedback_valid','tool_geometry')
        payload = dict(schema_version=1, stamp_ns=stamp.sec*1000000000+stamp.nanosec,
            robot_prefix=robot.prefix, context=context,
            controller_interval_sec=getattr(robot, '_control_interval', None),
            frames={'vehicle_pose':'map NED: xyz metres, xyz Euler radians',
                    'body_velocity':'body FRD: m/s, rad/s',
                    'body_acceleration':'body-coordinate derivatives: m/s^2, rad/s^2',
                    'tool_pose':'map NWU: xyz metres, xyz Euler radians',
                    'joint_order':['axis_e','axis_d','axis_c','axis_b']},
            current={k:state[k] for k in fields if k in state},
            reference={k:reference[k] for k in fields if k in reference},
            published_command=getattr(robot, '_metric_recorded_command', None),
            metrics=metrics)
        self.kinematics_pub.publish(String(data=encode(payload)))
