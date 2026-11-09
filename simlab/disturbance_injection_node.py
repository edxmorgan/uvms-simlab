"""Persistent simulation-only waveform action server; idle until given a goal."""
import json
import time
import uuid
from collections import deque
import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation
from ament_index_python.packages import get_package_share_directory
from control_msgs.msg import DynamicJointState
from geometry_msgs.msg import WrenchStamped
from rclpy.qos import qos_profile_sensor_data
from simlab.motion_planning.kinematics.forward.tool_wrench import ToolWrenchMapper
from simlab.uvms_parameters_sim import ReachParams

import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.executors import SingleThreadedExecutor
from rclpy.clock import Clock, ClockType
from rclpy.task import Future
from types import SimpleNamespace
from rclpy.node import Node
from std_srvs.srv import Trigger
from ros2_control_blue_reach_5.msg import DisturbanceCommand, DisturbanceState
from simlab.action import RunDisturbance
from simlab.msg import DisturbanceRunStatus
from simlab.disturbance_waveforms import validate_options, load_vector
from simlab.disturbances import waveform_identity
from simlab.task_wrench_estimation import TaskWrenchObserver


def goal_options(goal):
    return validate_options(SimpleNamespace(
        space=goal.space, wrench_frame=goal.wrench_frame, dof=list(goal.dofs), amplitude=list(goal.amplitudes),
        waveform=list(goal.waveforms), phase=list(goal.phases_deg), delay=list(goal.delays_s),
        frequency=goal.frequency_hz, period=None, cycles=goal.cycles, seed=goal.seed,
        end_frequency=goal.end_frequency_hz or None,
        correlation_time=goal.correlation_time_s))


class DisturbanceInjectionNode(Node):
    def __init__(self):
        super().__init__('disturbance_injection_node')
        self.prefix = self.declare_parameter('robot_prefix', 'robot_1_').value
        self.base = '/' + self.prefix + '/disturbance/'
        self.task_observer = TaskWrenchObserver(self, self.prefix)
        self.busy = False
        self.stop_requested = False
        self.run = None
        self.states = {}
        self.joint_states = deque(maxlen=200)
        self.task_mapper = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.joint_subscription = self.create_subscription(DynamicJointState,
            '/dynamic_joint_states', lambda msg: self.joint_states.append((msg, time.monotonic())),
            qos_profile_sensor_data)
        self.task_wrench_pub = self.create_publisher(WrenchStamped, self.base+'task_wrench', 10)
        self.status = DisturbanceRunStatus(state='waiting_for_simulator')
        self.publisher = self.create_publisher(DisturbanceCommand, self.base+'command', 1)
        self.status_pub = self.create_publisher(DisturbanceRunStatus, self.base+'status', 10)
        self.subscriptions_ = [self.create_subscription(
            DisturbanceState, self.base+component+'/state',
            lambda msg, c=component: self.receive_state(c, msg), 1)
            for component in ('vehicle', 'arm')]
        self.server = ActionServer(self, RunDisturbance, self.base+'run', self.execute,
            goal_callback=self.accept, cancel_callback=lambda _: CancelResponse.ACCEPT)
        self.stop_service = self.create_service(Trigger, self.base+'stop', self.stop)
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.timer = self.create_timer(.05, self.tick, clock=self.steady_clock)
        self.get_logger().info(f'Idle: action {self.base}run; stop service {self.base}stop')

    def receive_state(self, component, msg):
        self.states[component] = (msg, time.monotonic())

    def task_state(self, frame):
        if not self.joint_states or time.monotonic()-self.joint_states[-1][1] >= .5:
            raise ValueError('Task disturbance needs fresh joint-state telemetry')
        msg = self.joint_states[-1][0]
        stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        age = (self.get_clock().now()-stamp).nanoseconds/1e9
        if not 0. <= age < .5:
            raise ValueError('Task disturbance joint-state timestamp is stale or in the future')
        rows = {name: dict(zip(values.interface_names, values.values))
                for name, values in zip(msg.joint_names, msg.interface_values)}
        joints = [rows.get(self.prefix+'_axis_'+axis, {}).get('position', np.nan) for axis in 'edcb']
        if not np.all(np.isfinite(joints)):
            raise ValueError('Task disturbance joint positions are missing/nonfinite')
        rotation = None
        if frame == 'world':
            try:
                # Injection belongs to the simulated plant: use its physical
                # attitude, not the EKF pose published as body TF. Only the
                # static placement of the robot's map comes from TF.
                vehicle = rows.get(self.prefix+'IOs', {})
                attitude = np.array([vehicle.get(k, np.nan) for k in ('roll', 'pitch', 'yaw')])
                if not np.all(np.isfinite(attitude)):
                    raise ValueError('Physical vehicle attitude unavailable for wrench injection')
                transform = self.tf_buffer.lookup_transform('world', self.prefix+'map', stamp)
                q = transform.transform.rotation
                world_from_map = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                convention = np.diag([1., -1., -1.])
                rotation = world_from_map @ convention @ Rotation.from_euler('xyz', attitude).as_matrix() @ convention
            except (tf2_ros.TransformException, ValueError) as error:
                raise ValueError('Task disturbance world/body transform unavailable: '+str(error)) from error
        if self.task_mapper is None:
            self.task_mapper = ToolWrenchMapper(get_package_share_directory('simlab'),
                ReachParams.base_T0_new, ReachParams.tipOffset)
        return joints, rotation, msg.header.stamp

    def publish_task_wrench(self, values, stamp=None):
        message = WrenchStamped()
        message.header.frame_id = self.prefix+'joint_4'
        message.header.stamp = stamp or self.get_clock().now().to_msg()
        for i, vector in enumerate((message.wrench.force, message.wrench.torque)):
            for j, axis in enumerate('xyz'):
                setattr(vector, axis, float(values[3*i+j]))
        self.task_wrench_pub.publish(message)

    def receivers_ready(self):
        now = time.monotonic()
        return len(self.states)==2 and all(now-t < .5 and not m.lease_active
                                           for m,t in self.states.values())

    def accept(self, goal):
        try:
            options = goal_options(goal)
            if options.space == 'task':
                self.task_state(options.wrench_frame)
        except (ValueError, OverflowError, RuntimeError) as error:
            self.get_logger().warning(f'Rejected disturbance configuration: {error}')
            return GoalResponse.REJECT
        if self.busy or not self.receivers_ready():
            self.get_logger().warning('Rejected run: busy, owned or simulation receivers unavailable')
            return GoalResponse.REJECT
        self.busy = True
        self.stop_requested = False
        return GoalResponse.ACCEPT

    def stop(self, request, response):
        self.stop_requested = True
        response.success = True
        response.message = 'Stop requested; receiver release is confirmed in action/status (watchdog: 1 s)'
        return response

    def publish_status(self):
        self.status.header.stamp = self.get_clock().now().to_msg()
        self.status_pub.publish(self.status)

    def command(self, session, sequence, values, acquire=False, release=False):
        msg = DisturbanceCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.prefix+'base_link'
        msg.session_id, msg.sequence = session, sequence
        msg.acquire, msg.release = acquire, release
        msg.joint_names = [self.prefix+'_axis_'+axis for axis in 'edcb']
        msg.joint_torques = list(values[6:])
        for i, vector in enumerate((msg.body_wrench.force, msg.body_wrench.torque)):
            for j, axis in enumerate('xyz'):
                setattr(vector, axis, float(values[3*i+j]))
        self.publisher.publish(msg)

    async def execute(self, handle):
        options = goal_options(handle.request)
        duration = 2.+options.max_delay+options.period*options.cycles+3.
        session = str(uuid.uuid4())
        self.status = DisturbanceRunStatus(session_id=session, state='acquiring', duration_s=duration,
            configuration_json=json.dumps(dict(channels=[vars(c) for c in options.channels],
                frequency_hz=options.frequency, end_frequency_hz=options.end_frequency,
                cycles=options.cycles, seed=options.seed, correlation_time_s=options.correlation_time,
                waveform_implementations=[waveform_identity(name) for name in dict.fromkeys(options.waveform)],
                space=options.space, wrench_frame=options.wrench_frame,
                application_frame=self.prefix+'joint_4' if options.space == 'task' else None,
                mounting=ReachParams.base_T0_new if options.space == 'task' else None,
                tip=ReachParams.tipOffset if options.space == 'task' else None,
                robot_prefix=self.prefix, clock='vehicle accumulated dynamics-step time')))
        self.run = SimpleNamespace(handle=handle, options=options, session=session, sequence=0,
            elapsed=0., duration=duration, baseline=None, generations=None,
            deadline=time.monotonic()+3., finishing=None, finish_time=None, future=Future())
        # Yield to the executor; only the steady-clock timer advances this run.
        future = self.run.future
        return await future

    def send(self, values, acquire=False, release=False):
        self.run.sequence += 1
        self.command(self.run.session, self.run.sequence, values, acquire=acquire, release=release)

    def begin_finish(self, reason):
        if self.run.finishing is None:
            self.run.finishing = reason
            self.run.finish_time = time.monotonic()
            self.status.state = 'stopping'
        self.send([0.]*10, release=True)
        if self.run.options.space == 'task':
            self.publish_task_wrench([0.]*6)

    def finish(self, confirmed):
        run = self.run
        reason = run.finishing
        if not confirmed:
            reason += '; receiver release not confirmed (local 1 s watchdog still applies)'
        if run.handle.is_cancel_requested:
            reason = 'Canceled' if confirmed else reason
        success = reason == 'Completed'
        self.status.state = ('completed' if success else 'stopped' if reason in
                             ('Canceled','Stopped by client') else 'failed')
        self.status.message = reason
        self.status.elapsed_s = run.elapsed
        if run.handle.is_cancel_requested:
            run.handle.canceled()
        elif success:
            run.handle.succeed()
        else:
            run.handle.abort()
        run.future.set_result(RunDisturbance.Result(success=success, message=reason,
            session_id=run.session, elapsed_s=run.elapsed))
        self.run = None
        self.busy = False

    def tick(self):
        run = self.run
        if run is None and self.status.state in ('idle','waiting_for_simulator'):
            self.status.state = 'idle' if self.receivers_ready() else 'waiting_for_simulator'
        try:
            if run is not None:
                now = time.monotonic()
                if run.finishing is not None:
                    # Both receivers must acknowledge loss of ownership. On an
                    # unpaused plant, the completed-step load must also be zero.
                    confirmed = len(self.states)==2 and all(
                        now-t < .5 and m.lease_owner!=run.session and
                        (m.paused or (all(v==0. for v in m.applied_joint_torques) and
                         all(getattr(v,axis)==0. for v in (m.applied_body_wrench.force,
                             m.applied_body_wrench.torque) for axis in 'xyz')))
                        for m,t in self.states.values())
                    if confirmed or now-run.finish_time>1.5:
                        self.finish(confirmed)
                    else:
                        self.send([0.]*10, release=True)
                elif run.handle.is_cancel_requested or self.stop_requested:
                    self.begin_finish('Canceled' if run.handle.is_cancel_requested else 'Stopped by client')
                elif len(self.states)!=2 or any(now-t > .5 for m,t in self.states.values()):
                    self.begin_finish('Simulator receiver telemetry stale/unavailable')
                elif run.baseline is None:
                    self.send([0.]*10, acquire=True)
                    if all(m.lease_active and m.lease_owner==run.session for m,t in self.states.values()):
                        run.baseline = self.states['vehicle'][0].dynamics_time
                        run.generations = {c:m.generation for c,(m,t) in self.states.items()}
                        self.status.state = 'running'
                    elif now > run.deadline:
                        self.begin_finish('Could not acquire both simulator receivers')
                elif any(m.generation!=run.generations[c] for c,(m,t) in self.states.items()):
                    self.begin_finish('Simulator reset during injection')
                elif any(not m.lease_active or m.lease_owner!=run.session for m,t in self.states.values()):
                    self.begin_finish('Injection lease lost')
                else:
                    elapsed = self.states['vehicle'][0].dynamics_time-run.baseline
                    paused = [m.paused for m,t in self.states.values()]
                    if elapsed < run.elapsed:
                        self.begin_finish('Dynamics clock moved backwards')
                    elif any(paused) and not all(paused):
                        self.begin_finish('Vehicle and arm pause states differ')
                    else:
                        run.elapsed = elapsed
                        if elapsed >= run.duration:
                            self.begin_finish('Completed')
                        else:
                            a = run.options
                            values = load_vector(a.channels, elapsed-2., a.frequency,
                                a.period*a.cycles, a.end_frequency, seed=a.seed,
                                correlation_time=a.correlation_time)
                            if a.space == 'task':
                                joints, rotation, stamp = self.task_state(a.wrench_frame)
                                values, tool_wrench = self.task_mapper.resolve(
                                    joints, values[:6], a.wrench_frame, rotation)
                                self.publish_task_wrench(tool_wrench, stamp)
                            self.send(values)
                            self.status.state = 'paused' if any(paused) else 'running'
                if self.run is not None:
                    self.status.elapsed_s = run.elapsed
                    run.handle.publish_feedback(RunDisturbance.Feedback(session_id=run.session,
                        state=self.status.state, elapsed_s=run.elapsed, duration_s=run.duration))
        except Exception as error:
            if self.run is not None:
                self.begin_finish(str(error))
            self.get_logger().error(str(error))
        self.publish_status()


def main(args=None):
    rclpy.init(args=args)
    node = DisturbanceInjectionNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        # If ROS has already shut down, the receiver watchdog clears the lease.
        if node.run is not None and rclpy.ok():
            node.send([0.]*10, release=True)
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
