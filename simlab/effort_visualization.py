"""Read-only, SI-unit WrenchStamped adapters for the standard RViz Wrench display.

Vehicle dynamics use FRD; URDF vehicle frames use FLU. Joint generalized
forces are projected onto the URDF joint axis in its child-link frame.
Unavailable values are NaN, never fictitious zero measurements. RViz's
Accept NaN Values option hides these vectors; text explains their status.
"""
import math
import time
import xml.etree.ElementTree as ET
from itertools import zip_longest

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from control_msgs.msg import DynamicJointState
from geometry_msgs.msg import WrenchStamped, Point
from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker, MarkerArray
from rviz_2d_overlay_msgs.msg import OverlayText


AXES = ('e', 'd', 'c', 'b', 'a')
COMPONENTS = ('force.x', 'force.y', 'force.z', 'torque.x', 'torque.y', 'torque.z')


def joint_geometry(description, prefix):
    root = ET.fromstring(description)
    result = {}
    for axis in AXES:
        name = f'{prefix}_axis_{axis}'
        joint = root.find(f"joint[@name='{name}']")
        if joint is None:
            raise ValueError(f'Missing URDF joint {name}')
        kind = joint.get('type')
        if kind not in ('revolute', 'continuous', 'prismatic'):
            raise ValueError(f'Unsupported effort joint {name}: {kind}')
        element = joint.find('axis')
        vector = tuple(float(v) for v in (element.get('xyz') if element is not None else '1 0 0').split())
        norm = math.sqrt(sum(v*v for v in vector))
        if len(vector) != 3 or not math.isfinite(norm) or norm <= 0:
            raise ValueError(f'Invalid joint axis: {name}')
        result[axis] = (joint.find('child').get('link'), kind, tuple(v/norm for v in vector))
    return result


def joint_wrench(value, kind, axis):
    vector = [value*v for v in axis]
    return vector + [0.]*3 if kind == 'prismatic' else [0.]*3 + vector


def vehicle_wrench(values):
    # Proper 180-degree rotation about X, applied to both force and torque.
    return [v*s for v, s in zip(values, (1., -1., -1., 1., -1., -1.))]


def finite_values(row, keys):
    values = [row.get(k, math.nan) for k in keys]
    return values if all(math.isfinite(v) for v in values) else None


def arm_estimate(row, axis):
    if axis == 'a':
        return None, 'not estimated'
    if row.get('estimator_valid') != 1.:
        return None, 'unavailable'
    if row.get(f'estimator_joint_locked.{axis}') == 1.:
        return None, 'locked / unobservable'
    if row.get(f'estimator_joint_at_limit.{axis}') == 1.:
        return None, 'joint stop / unobservable'
    if row.get(f'estimator_disturbance_observable.{axis}') != 1.:
        return None, 'unobservable'
    values = finite_values(row, [f'estimated_disturbance.torque.{axis}'])
    return (values[0], '') if values else (None, 'unavailable')


def arm_effort_markers(frame, stamp, kind, axis, value, source):
    """Linear display scaling; SI wrench topics and numeric values stay exact.

    Arc radius: 0.25 m/Nm. Force length: 0.1 m/N. Stroke: 12 mm.
    Opposite axial offsets separate command/estimate rings without changing sign.
    """
    markers = []
    for ident in range(2):
        m = Marker()
        # These are joint-attached live indicators, not a historical trajectory.
        # Latest TF + frame locking avoids exact-stamp TF queue gaps and keeps
        # the graphic attached between 20 Hz magnitude updates. The raw wrench
        # message still preserves the source measurement timestamp.
        m.header.frame_id, m.header.stamp = frame, Time()
        m.frame_locked = True
        m.ns, m.id = source, ident
        m.pose.orientation.w = 1.
        m.lifetime.sec = 1
        m.color.r, m.color.g, m.color.b = ((230/255, 140/255, 30/255)
                                         if source == 'command' else (40/255, 160/255, 1.))
        m.color.a = 1.
        m.action = Marker.DELETE
        markers.append(m)
    if value is None or not math.isfinite(value) or value == 0.:
        return markers
    sign = 1. if value > 0. else -1.
    if kind == 'prismatic':
        m = markers[0]
        m.action, m.type = Marker.ADD, Marker.ARROW
        length = 0.1 * abs(value)
        m.points = [Point(), Point(x=axis[0]*length*sign,
                                  y=axis[1]*length*sign, z=axis[2]*length*sign)]
        m.scale.x, m.scale.y, m.scale.z = .012, .2*length, .25*length
        return markers

    def cross(a, b):
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

    reference = (1., 0., 0.) if abs(axis[0]) < .8 else (0., 1., 0.)
    u = cross(axis, reference)
    norm = math.sqrt(sum(x*x for x in u))
    u = tuple(x/norm for x in u)
    v = cross(axis, u)
    radius = .25 * abs(value)
    offset = -.025 if source == 'command' else .025

    def point(angle):
        xyz = [offset*axis[j] + radius*(math.cos(angle)*u[j] + math.sin(angle)*v[j])
               for j in range(3)]
        return Point(x=xyz[0], y=xyz[1], z=xyz[2])

    arc, head = markers
    arc.action, arc.type, arc.scale.x = Marker.ADD, Marker.LINE_STRIP, .012
    arc.points = [point(sign*1.5*math.pi*i/40) for i in range(41)]
    angle = sign*1.5*math.pi
    end = arc.points[-1]
    tangent = [sign*(-math.sin(angle)*u[j] + math.cos(angle)*v[j]) for j in range(3)]
    head.action, head.type = Marker.ADD, Marker.ARROW
    head_length = .4 * radius
    head.points = [Point(x=end.x-head_length*tangent[0], y=end.y-head_length*tangent[1], z=end.z-head_length*tangent[2]), end]
    head.scale.x, head.scale.y, head.scale.z = .012, .35*radius, head_length
    return markers


class EffortVisualization(Node):
    def __init__(self):
        super().__init__('effort_visualization')
        self.declare_parameter('robot_prefixes', ['robot_1_'])
        self.declare_parameter('robot_description', '')
        self.declare_parameter('stale_after_seconds', 0.5)
        self.prefixes = list(self.get_parameter('robot_prefixes').value)
        self.geometry = {p: joint_geometry(self.get_parameter('robot_description').value, p)
                         for p in self.prefixes}
        self.publishers_by_key = {}
        self.marker_publishers = {}
        for p in self.prefixes:
            for part in ('vehicle', *AXES):
                for source in ('command', 'estimated_disturbance'):
                    if part == 'a' and source == 'estimated_disturbance':
                        continue
                    topic = f'/{p}/visualization/effort/{part}/{source}'
                    self.publishers_by_key[p, part, source] = self.create_publisher(WrenchStamped, topic, 1)
                    if part != "vehicle":
                        self.marker_publishers[p, part, source] = self.create_publisher(MarkerArray, topic + "/markers", 1)
        self.overlay = self.create_publisher(OverlayText, '/effort_visualization/status', 1)
        self.labels = self.create_publisher(MarkerArray, '/effort_visualization/labels', 1)
        self.rows = {}
        self.received = -math.inf
        self.stamp = None
        self.create_subscription(DynamicJointState, '/dynamic_joint_states', self.receive,
                                 qos_profile_sensor_data)
        self.create_timer(0.05, self.publish)

    def receive(self, msg):
        self.rows = {name: dict(zip(values.interface_names, values.values))
                     for name, values in zip(msg.joint_names, msg.interface_values)}
        self.stamp = msg.header.stamp
        self.received = time.monotonic()

    def emit(self, prefix, part, source, frame, vector, stamp):
        msg = WrenchStamped()
        msg.header.frame_id = frame
        msg.header.stamp = stamp
        values = vector if vector is not None else [math.nan]*6
        msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = values[:3]
        msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = values[3:]
        self.publishers_by_key[prefix, part, source].publish(msg)
        if part != 'vehicle':
            _, kind, axis = self.geometry[prefix][part]
            components = values[:3] if kind == 'prismatic' else values[3:]
            value = sum(a*b for a, b in zip(axis, components))
            self.marker_publishers[prefix, part, source].publish(MarkerArray(markers=
                arm_effort_markers(frame, stamp, kind, axis, value, source)))

    def publish(self):
        stale = time.monotonic() - self.received > self.get_parameter('stale_after_seconds').value
        rows = {} if stale else self.rows
        stamp = self.get_clock().now().to_msg() if stale else self.stamp
        labels = MarkerArray()
        table = []
        for p in self.prefixes:
            vehicle = rows.get(f'{p}IOs', {})
            vehicle_values = {}
            for source in ('command', 'estimated_disturbance'):
                keys = COMPONENTS if source == 'command' else tuple('estimated_disturbance.' + k for k in COMPONENTS)
                values = finite_values(vehicle, keys)
                if source != 'command' and vehicle.get('estimator_valid') != 1.:
                    values = None
                vehicle_values[source] = values
                self.emit(p, 'vehicle', source, f'{p}base_link',
                          vehicle_wrench(values) if values is not None else None, stamp)
            vehicle_table = [f'{"DOF":<7}{"Unit":<6}{"Command":>13}  {"External disturbance (EKF)":>27}']
            for i, dof in enumerate(('Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw')):
                command = vehicle_values['command']
                estimate = vehicle_values['estimated_disturbance']
                cmd_text = f'{command[i]:+.3f}' if command is not None else 'unavailable'
                est_text = f'{estimate[i]:+.3f}' if estimate is not None else 'Unavailable'
                unit = 'N' if i < 3 else 'Nm'
                vehicle_table.append(f'{dof:<7}{unit:<6}{cmd_text:>13}  {est_text:>27}')
            arm_table = [f'{"Joint":<7}{"Unit":<6}{"Motor effort":>13}  {"External disturbance (EKF)":>27}']
            arm = rows.get(f'{p}_arm_IOs', {})
            for axis, (frame, kind, vector) in self.geometry[p].items():
                command = finite_values(rows.get(f'{p}_axis_{axis}', {}), ['effort'])
                estimate, status = arm_estimate(arm, axis)
                self.emit(p, axis, 'command', frame,
                          joint_wrench(command[0], kind, vector) if command else None, stamp)
                if axis != 'a':
                    self.emit(p, axis, 'estimated_disturbance', frame,
                              joint_wrench(estimate, kind, vector) if estimate is not None else None, stamp)
                unit = 'N' if kind == 'prismatic' else 'Nm'
                cmd_text = f'{command[0]:+.3f}' if command else 'unavailable'
                reason = {'locked': 'joint locked', 'at limit': 'joint at limit'}.get(status, status)
                est_text = f'{estimate:+.3f}' if estimate is not None else f'Unavailable: {reason}'
                row = f'{axis:<7}{unit:<6}{cmd_text:>13}'
                if axis != 'a':
                    row += f'  {est_text:>27}'
                arm_table.append(row)
                text = f'{axis} [{unit}] Motor effort: {cmd_text}'
                if axis != 'a':
                    text += f'\nExternal disturbance (EKF): {est_text}'
                if stale:
                    text += '\nSTALE / no data'
                marker = Marker()
                marker.header.frame_id, marker.header.stamp = frame, stamp
                marker.ns, marker.id = p, AXES.index(axis)
                marker.type = Marker.TEXT_VIEW_FACING
                marker.pose.orientation.w = 1.
                marker.pose.position.z = 0.10
                marker.scale.z = 0.035
                marker.color.r = marker.color.g = marker.color.b = 1.
                marker.color.a = 1.
                marker.text = text
                marker.lifetime.sec = 1
                labels.markers.append(marker)
            column_width = max(len(line) for line in vehicle_table)
            table.extend(f'{left:<{column_width}}    |    {right}'
                         for left, right in zip_longest(vehicle_table, arm_table, fillvalue=''))
            marker = Marker()
            marker.header.frame_id, marker.header.stamp = f'{p}base_link', stamp
            marker.ns, marker.id, marker.type = p, 5, Marker.TEXT_VIEW_FACING
            marker.pose.orientation.w = 1.
            marker.pose.position.z = 0.5
            marker.scale.z = 0.045
            marker.color.r = marker.color.g = marker.color.b = marker.color.a = 1.
            marker.text = ('Vehicle command: orange / External disturbance (EKF): blue\n'
                           'Force: N / Torque: Nm')
            if stale:
                marker.text += '\nSTALE / no data'
            elif vehicle.get('estimator_valid') != 1.:
                marker.text += '\nVehicle external disturbance (EKF) unavailable'
            marker.lifetime.sec = 1
            labels.markers.append(marker)
        self.labels.publish(labels)
        if stale:
            table.append('STALE / no data - vectors hidden')
        elif any(rows.get(f'{p}IOs', {}).get('estimator_valid') != 1. for p in self.prefixes):
            table.append('Vehicle external disturbance (EKF) unavailable')
        overlay = OverlayText()
        overlay.action = OverlayText.ADD
        overlay.width, overlay.height = 1800, 40 + 40 * len(table)
        overlay.horizontal_alignment, overlay.vertical_alignment = OverlayText.RIGHT, OverlayText.BOTTOM
        overlay.horizontal_distance = overlay.vertical_distance = 10
        overlay.text_size, overlay.font = 12., 'DejaVu Sans Mono'
        overlay.fg_color.r = overlay.fg_color.g = overlay.fg_color.b = overlay.fg_color.a = 1.
        overlay.bg_color.a = 0.7
        overlay.text = '\n'.join(table)
        self.overlay.publish(overlay)


def main(args=None):
    rclpy.init(args=args)
    node = EffortVisualization()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
