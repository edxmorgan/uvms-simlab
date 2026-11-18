"""Explicit-action gate shared by interactive simulation and hardware.

This never arms hardware. Only the simulator has a one-shot startup release.
"""
import time
from controller_manager_msgs.srv import ListControllers
from std_msgs.msg import String
from rclpy.qos import QoSProfile, DurabilityPolicy


class ExecutionGate:
    def __init__(self, robot, startup_paused=False, simulated_components=("vehicle", "manipulator")):
        self.robot = robot
        self.simulated_components = frozenset(simulated_components)
        self.active_components = set()
        self.startup_paused = bool(startup_paused)
        self.authorized = False
        self.releasing = False
        self.controllers_ready = False
        self.checked_at = 0.
        self.query = None
        self.last_status = None
        self.client = robot.node.create_client(ListControllers, '/controller_manager/list_controllers')
        self.publisher = robot.node.create_publisher(String, f'/{robot.prefix}/control_status',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.timer = robot.node.create_timer(.5, self.poll)

    @property
    def startup_paused(self):
        return bool(self.paused_components)

    @startup_paused.setter
    def startup_paused(self, paused):
        self.paused_components = set(getattr(self, 'simulated_components',
                                            ('vehicle', 'manipulator'))) if paused else set()

    def ready(self):
        r = self.robot
        received = getattr(r, '_last_state_received', None)
        return (self.controllers_ready and time.monotonic()-self.checked_at < 2.
                and received is not None and time.monotonic()-received < .5
                and not getattr(r, '_feedback_fault', False)
                and r.get_state()['status'] == 'active'
                and getattr(r, '_hardware_estimator_valid', True)
                and not r.sim_reset_hold
                and not getattr(r, '_simulation_pause_pending', False))

    def poll(self):
        if (self.query is not None and not self.query.done()
                and time.monotonic() - self.query_started > 2.):
            # A busy executor can delay both this timer and the response. A
            # stale snapshot withholds output, but is not a deactivation event.
            self.query.cancel()
        if self.query is None and self.client.service_is_ready():
            self.query_started = time.monotonic()
            try:
                self.query = self.client.call_async(ListControllers.Request())
            except Exception as error:
                self.controllers_ready = False
                self.robot.node.get_logger().error(f'Controller readiness request failed: {error}')
                if self.authorized:
                    self.robot.stop_on_control_fault('Controller readiness request failed')
                return
            requested_at = self.query_started
            def done(future):
                if future.cancelled():
                    self.query = None
                    return
                if time.monotonic() - requested_at >= 2.:
                    self.query = None
                    self.controllers_ready = False
                    if self.authorized:
                        self.robot.stop_on_control_fault('Controller liveness confirmation timed out')
                    return
                try:
                    active = {c.name for c in future.result().controller if c.state == 'active'}
                    self.controllers_ready = {
                        f'vehicle_effort_controller_{self.robot.prefix}',
                        f'manipulation_effort_controller_{self.robot.prefix}'} <= active
                    self.checked_at = time.monotonic()
                except Exception:
                    self.controllers_ready = False
                self.query = None
                if not self.controllers_ready and self.authorized and not self.releasing:
                    self.robot.stop_on_control_fault('Command controllers became inactive')
            self.query.add_done_callback(done)
        if self.authorized and time.monotonic() - self.checked_at >= 2.:
            self.robot.stop_on_control_fault('Controller liveness confirmation timed out')
        state = ('Waiting for feedback/controllers' if not self.ready() else
                 'Starting execution' if self.releasing else
                 'Control active' if self.authorized else
                 'Ready — simulation paused' if self.startup_paused else 'Ready — idle')
        if state != self.last_status:
            self.publisher.publish(String(data=state))
            self.robot.node.get_logger().info(f'[{self.robot.prefix}] {state}')
            self.last_status = state

    def authorize(self, components=('vehicle', 'manipulator')):
        if getattr(self.robot, 'scene_switch_hold', False):
            return False
        requested = set(components)
        if not requested or not requested <= {'vehicle', 'manipulator'}:
            return False
        if not self.ready():
            self.robot.node.get_logger().warning('Motion rejected: fresh feedback and active command controllers required')
            return False
        # Do not widen an in-flight release. The caller can retry after its ack.
        if self.releasing:
            return requested <= self.active_components
        self.authorized = True
        self.active_components = getattr(self, 'active_components', set()) | requested
        pending = requested & self.paused_components
        if pending:
            self.releasing = True
            epoch = getattr(self, '_authorization_epoch', 0)
            def released():
                if epoch != getattr(self, '_authorization_epoch', 0):
                    return
                self.paused_components.difference_update(pending)
                self.releasing = False
                # Do not reauthorize here: Stop may have arrived during release.
            def failed():
                if epoch != getattr(self, '_authorization_epoch', 0):
                    return
                self.releasing = False
                self.authorized = False
                self.active_components.clear()
            self.robot.release_simulation(components=pending, on_success=released, on_failure=failed)
        return True

    def stop(self):
        self._authorization_epoch = getattr(self, '_authorization_epoch', 0) + 1
        self.authorized = False
        self.active_components = set()
        self.releasing = False
        cancel = getattr(self.robot, 'cancel_simulation_release', None)
        if cancel is not None:
            cancel()

    def allows_output(self, component=None):
        active = getattr(self, 'active_components', {'vehicle', 'manipulator'})
        permitted = active - self.paused_components
        return (self.authorized and not self.releasing and self.ready()
                and (bool(permitted) if component is None else component in permitted))
