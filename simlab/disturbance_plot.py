"""Read-only live disturbance viewer. Recording belongs to the standard MCAP pipeline."""
import math
import json
import textwrap
import os
import time
import sys
import numpy as np
from collections import deque
from types import SimpleNamespace
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from control_msgs.msg import DynamicJointState
from ros2_control_blue_reach_5.msg import DisturbanceState
from simlab.msg import DisturbanceRunStatus, TaskWrenchEstimate
from geometry_msgs.msg import WrenchStamped
from rcl_interfaces.srv import SetParametersAtomically
from rcl_interfaces.msg import SetParametersResult
from simlab.disturbance_waveforms import parse_args, display_channels
from simlab.disturbances import waveform_class

def estimate_status(data, index, key, age):
    if not math.isfinite(age) or age < 0. or age >= .5:
        return math.nan, 'estimate stale/unavailable'
    if data.get('estimator_valid') != 1.:
        return math.nan, 'estimate invalid'
    if index >= 6:
        axis = key[-1]
        if data.get('estimator_joint_locked.'+axis) == 1.:
            return math.nan, 'joint locked / unobservable'
        if data.get('estimator_joint_at_limit.'+axis) == 1.:
            return math.nan, 'joint at stop / unobservable'
        if data.get('estimator_disturbance_observable.'+axis) != 1.:
            return math.nan, 'disturbance unobservable'
    value = data.get('estimated_disturbance.'+key, math.nan)
    return value, 'estimate valid' if math.isfinite(value) else 'estimate unavailable'


def estimate_sigma(data, key, estimate):
    """Published marginal EKF uncertainty; never invent missing/invalid variance."""
    variance = data.get('estimated_disturbance.variance.'+key, math.nan)
    if not math.isfinite(estimate) or not math.isfinite(variance) or variance < 0.:
        return math.nan
    return math.sqrt(variance)


def uncertainty_polygons(times, estimates, sigmas):
    """Separate finite runs so unavailable covariance leaves gaps in the band."""
    t, d, sigma = map(np.asarray, (times, estimates, sigmas))
    valid = np.isfinite(t) & np.isfinite(d) & np.isfinite(sigma) & (sigma >= 0.)
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    polygons = []
    for start, end in zip(edges[::2], edges[1::2]):
        x, y, width = t[start:end], d[start:end], 2.*sigma[start:end]
        polygons.append(np.column_stack((np.r_[x, x[::-1]],
                                         np.r_[y-width, (y+width)[::-1]])))
    return polygons


class LiveDisturbancePlot:
    """GUI-only blitted display; never sends commands or processes ROS callbacks."""
    def __init__(self, a):
        import matplotlib
        if matplotlib.get_backend().lower() == 'agg':
            matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.ion()
        self.fig = plt.figure()
        self.fig.canvas.manager.set_window_title('Live disturbance / EKF')
        self.blit = self.fig.canvas.supports_blit
        self.fig.canvas.mpl_connect('draw_event', self._capture_background)
        self.configure(a)
        plt.show(block=False)
        self.fig.canvas.flush_events()

    def configure(self, a):
        """Replace traces and layout in the existing GUI window for a new run."""
        from matplotlib.collections import PolyCollection
        self.fig.clear()
        channels = display_channels(a)
        columns = min(3, len(channels))
        groups = math.ceil(len(channels)/columns)
        self.fig.set_size_inches(max(9, 4.5*columns), min(10., 3.2*groups+.6), forward=True)
        grid = self.fig.add_gridspec(groups+1, columns,
            height_ratios=[.3]+[3.]*groups, hspace=.65, wspace=.30)
        header = self.fig.add_subplot(grid[0, :])
        header.set_axis_off()
        self.panels, self.artists = {}, []
        self.blit = self.fig.canvas.supports_blit
        for j, c in enumerate(channels):
            row, column = j//columns, j%columns
            panel_grid = grid[row+1, column].subgridspec(2, 1, height_ratios=[1., .2], hspace=.65)
            ax = self.fig.add_subplot(panel_grid[0])
            status_ax = self.fig.add_subplot(panel_grid[1])
            status_ax.set_axis_off()
            waveform = waveform_class(c.waveform)
            title = (waveform.plot_title(c, a) if a.space == 'generalized'
                     else f'{c.dof}: equivalent wrench\nTool axes, about tool origin')
            ax.set_title('\n'.join(textwrap.fill(line, width=44) for line in title.splitlines()),
                         fontsize=10, pad=8)
            applied, = ax.plot([], [], color='black', lw=2, label='Applied disturbance' if a.space == 'generalized' else 'Requested tool wrench')
            if waveform.plot_style is not None:
                applied.set_drawstyle(waveform.plot_style)
            estimate, = ax.plot([], [], color='#1874d1', lw=2, label='EKF estimate' if a.space == 'generalized' else 'Equivalent wrench estimate')
            band = PolyCollection([], facecolors='#1874d1', alpha=.18, edgecolors='none',
                                  label='EKF ±2σ' if a.space == 'generalized' else 'Approx. ±2σ', zorder=1)
            ax.add_collection(band)
            ax.set_ylabel(f'Disturbance ({c.unit})')
            if j == 0:
                header.legend(handles=[applied, estimate, band], loc='center',
                              ncol=3, frameon=False, fontsize=10)
            ax.grid(alpha=.2)
            ax.axhline(0, color='gray', lw=.6)
            ax.set_xlim(0, 5.+a.period*a.cycles+a.max_delay)
            ax.set_ylim(-max(.01,abs(c.amplitude)*1.4),max(.01,abs(c.amplitude)*1.4))
            status = status_ax.text(0., 1., 'Waiting for telemetry',
                                    transform=status_ax.transAxes, va='top', fontsize=9)
            self.panels[c.dof] = SimpleNamespace(ax=ax,applied=applied,
                estimate=estimate,status=status,band=band,sigma=deque(maxlen=6000),
                t=deque(maxlen=6000),u=deque(maxlen=6000),d=deque(maxlen=6000))
            self.artists.extend([band,applied,estimate,status])
        for artist in self.artists:
            artist.set_animated(self.blit)
        self.fig.supxlabel('Elapsed time (s)', y=.01, fontsize=10)
        self.fig.subplots_adjust(left=.08, right=.98, bottom=.07, top=.97)
        self.background = None
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _capture_background(self, event):
        if self.blit:
            self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
            # Full draws omit animated artists. Restore them on resize/expose
            # and after completion too, even if no new telemetry arrives.
            for artist in self.artists:
                if artist.get_animated():
                    artist.axes.draw_artist(artist)

    def is_open(self):
        return self.plt.fignum_exists(self.fig.number)

    def update(self, rows):
        changed = set()
        for dof, t, u, d, sigma, status in rows:
            p = self.panels[dof]
            p.t.append(t); p.u.append(u); p.d.append(d); p.sigma.append(sigma)
            p.status.set_text(textwrap.fill(status.replace(' | ', ' · '), width=44))
            changed.add(dof)
        if rows:
            redraw = False
            for dof in changed:
                p = self.panels[dof]
                p.applied.set_data(p.t,p.u)
                p.estimate.set_data(p.t,p.d)
                p.band.set_verts(uncertainty_polygons(p.t, p.d, p.sigma))
                values = [p.u[-1],p.d[-1],p.d[-1]-2.*p.sigma[-1],
                          p.d[-1]+2.*p.sigma[-1]]
                peak = max([abs(v) for v in values if math.isfinite(v)] or [0.])
                if peak > p.ax.get_ylim()[1]:
                    p.ax.set_ylim(-peak*1.3,peak*1.3)
                    redraw = True
            if redraw:
                self.fig.canvas.draw()
            if self.blit and self.background is not None:
                self.fig.canvas.restore_region(self.background)
                for artist in self.artists:
                    artist.axes.draw_artist(artist)
                self.fig.canvas.blit(self.fig.bbox)
            else:
                self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def finish(self, status, close):
        if self.is_open():
            for panel in self.panels.values():
                # Keep validity/observability visible after the run ends.
                if not panel.t:
                    panel.status.set_text(status)
            for artist in self.artists:
                artist.set_animated(False)
            self.plt.ioff()
            if close:
                self.plt.close(self.fig)
            else:
                print('Live view finished. Close plot to exit.', flush=True)
                self.plt.show(block=True)


def main(argv=None):
    import threading
    from rclpy.signals import SignalHandlerOptions
    ros_args = sys.argv if argv is None else [sys.argv[0], *argv]
    a = parse_args(remove_ros_args(args=ros_args)[1:])
    live = None if a.no_plot else LiveDisturbancePlot(a)
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = Node('disturbance_plot')
    base = '/'+a.robot_prefix+'/disturbance/'
    latest, pending = {}, deque(maxlen=6000)
    mutex, stop = threading.Lock(), threading.Event()
    subscriptions = []
    terminal_at = None
    seen_session = None
    reconfigure = None
    ignored_session = None

    def configure(request, response):
        nonlocal a, reconfigure, seen_session, ignored_session, terminal_at
        try:
            if len(request.parameters) != 1 or request.parameters[0].name != 'argv':
                raise ValueError('Expected argv parameter')
            args = json.loads(request.parameters[0].value.string_value)
            if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
                raise ValueError('argv must be a list of strings')
            options = parse_args(args)
            if options.robot_prefix != a.robot_prefix:
                raise ValueError('Cannot change the robot of an existing viewer')
            if stop.is_set():
                raise ValueError('Viewer is closing; retry the command')
            current = latest.get('status')
            ignored_session = current[0].session_id if current else seen_session
            seen_session = terminal_at = None
            a = options
            with mutex:
                pending.clear()
                reconfigure = options
            response.result = SetParametersResult(successful=True)
        except (ValueError, TypeError, SystemExit) as error:
            response.result = SetParametersResult(successful=False, reason=str(error))
        return response

    node.create_service(SetParametersAtomically, base+'viewer/configure', configure)

    def receive(key, msg):
        nonlocal terminal_at, seen_session
        latest[key] = (msg, time.monotonic())
        if key == 'status' and msg.session_id:
            if msg.session_id == ignored_session:
                return
            if seen_session is None and msg.state not in ('acquiring','running','paused'):
                return
            if seen_session is None:
                seen_session = msg.session_id
            if msg.session_id==seen_session and msg.state in ('completed','failed','stopped'):
                terminal_at = terminal_at or time.monotonic()

    for key, typ, topic in [('vehicle',DisturbanceState,base+'vehicle/state'),
                          ('arm',DisturbanceState,base+'arm/state'),
                          ('status',DisturbanceRunStatus,base+'status'),
                          ('task',TaskWrenchEstimate,base+'task_estimate'),
                          ('requested',WrenchStamped,base+'task_wrench'),
                          ('joints',DynamicJointState,'/dynamic_joint_states')]:
        subscriptions.append(node.create_subscription(typ, topic, lambda m,k=key: receive(k,m), qos_profile_sensor_data))

    def tick():
        now = time.monotonic()
        if terminal_at is not None and now-terminal_at > 1.:
            if a.close_on_finish:
                stop.set()
            return  # Retain the completed traces while waiting for another run.
        if seen_session is None or 'status' not in latest or not latest['status'][0].session_id:
            return
        status = latest['status'][0]
        if status.session_id!=seen_session:
            stop.set(); return
        elapsed = status.elapsed_s
        status_fresh = now-latest['status'][1] < .5
        joint_msg, received = latest.get('joints', (None, -math.inf))
        fields = {} if joint_msg is None else {
            n:dict(zip(v.interface_names,v.values)) for n,v in zip(joint_msg.joint_names,joint_msg.interface_values)}
        rows = []
        for c in display_channels(a):
            if a.space == 'task':
                estimate = applied = sigma = math.nan
                label = 'wrench estimate stale/unavailable'
                message, receipt = latest.get('task', (None, -math.inf))
                if message is not None and now-receipt < .5:
                    label = message.status
                    if message.valid:
                        vector = message.wrench.force if c.index < 3 else message.wrench.torque
                        estimate = getattr(vector, 'xyz'[c.index % 3])
                        variance = message.covariance[7*c.index]
                        sigma = math.sqrt(variance) if variance >= 0. else math.nan
                request, receipt = latest.get('requested', (None, -math.inf))
                if request is not None and now-receipt < .5:
                    vector = request.wrench.force if c.index < 3 else request.wrench.torque
                    applied = getattr(vector, 'xyz'[c.index % 3])
                if not status_fresh:
                    applied = estimate = sigma = math.nan
                    label = 'run status stale/unavailable'
                # Detailed assumptions and residuals are on task_estimate/MCAP.
                if label.startswith('equivalent wrench;'):
                    label = 'valid; approx. uncertainty'
                    if message.observations == 6:
                        label += '; no redundancy'
                rows.append((c.dof, elapsed, applied, estimate, sigma, label))
                continue
            data = fields.get(a.robot_prefix+('IOs' if c.index<6 else '_arm_IOs'),{})
            estimate, label = estimate_status(data, c.index, c.key, now-received)
            state, receipt = latest.get('vehicle' if c.index<6 else 'arm', (None,-math.inf))
            applied = math.nan
            if state is not None and now-receipt < .5:
                if c.index<6:
                    v = state.applied_body_wrench.force if c.index<3 else state.applied_body_wrench.torque
                    applied = getattr(v,'xyz'[c.index%3])
                else:
                    applied = state.applied_joint_torques[c.index-6]
            if not status_fresh:
                applied = estimate = math.nan
                label = 'run status stale/unavailable'
            sigma = estimate_sigma(data, c.key, estimate)
            if math.isfinite(estimate) and not math.isfinite(sigma):
                label += ' | uncertainty unavailable'
            rows.append((c.dof,elapsed,applied,estimate,sigma,status.state+' | '+label))
        with mutex:
            pending.extend(rows)

    node.create_timer(.05,tick)
    errors = []
    def worker():
        ready_fd = a.ready_fd
        try:
            while rclpy.ok() and not stop.is_set():
                rclpy.spin_once(node,timeout_sec=.05)
                if (ready_fd is not None and all(k in latest for k in ('vehicle','arm','status','joints'))
                        and latest['status'][0].state != 'waiting_for_simulator'):
                    os.write(ready_fd, b'1'); os.close(ready_fd); ready_fd = None
        except Exception as error:
            errors.append(str(error)); stop.set()
        finally:
            if ready_fd is not None: os.close(ready_fd)
    thread = threading.Thread(target=worker)
    thread.start()
    try:
        while not stop.is_set():
            if live is not None:
                if not live.is_open():
                    break
                with mutex:
                    options = reconfigure; reconfigure = None
                    rows = list(pending); pending.clear()
                if options is not None:
                    live.configure(options)
                live.update(rows)
            time.sleep(1./30.)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set(); thread.join()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
        if live is not None:
            live.finish('Viewer stopped; injection controlled separately', a.close_on_finish)
    if errors: raise RuntimeError('; '.join(errors))


if __name__ == '__main__':
    main()
