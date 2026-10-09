"""RViz obstacle authoring, independent of robot targets and active paths."""
import copy
import json
import math
from pathlib import Path
import uuid

from ament_index_python.packages import get_package_share_directory
from interactive_markers.menu_handler import MenuHandler
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback, Marker
from bringup.obstacle_description import obstacle_from_config, visual_geometry
from bringup.collision_geometry import collision_geometry
from simlab.msg import ObstacleScene
from simlab.srv import EditDynamicObstacles, BackendWorldCommand
from simlab.world_profiles import list_world_profiles


def load_mesh_catalog(path):
    """Load named mesh drafts using the same geometry contract as scene edits."""
    if not path:
        return {}
    entries = json.loads(Path(path).expanduser().read_text())
    if not isinstance(entries, dict):
        raise ValueError('mesh catalog must be an object mapping names to mesh configurations')
    catalog = {}
    for name, config in entries.items():
        if not name.strip() or not isinstance(config, dict):
            raise ValueError('mesh catalog entries require a name and configuration object')
        catalog[name] = obstacle_from_config({**config, 'id': 'preview', 'type': 'mesh'}, 0)
    return catalog


class ObstacleEditor:
    def __init__(self, node, server, world_frame):
        self.node, self.server, self.frame = node, server, world_frame
        self.name = 'obstacle_editor'
        default_catalog = str(Path(get_package_share_directory('simlab')) / 'obstacle_meshes' / 'catalog.json')
        for name, value in (('obstacle_mesh_resource', ''), ('obstacle_mesh_catalog', default_catalog),
                            ('obstacle_preview_size', 0.8)):
            if not node.has_parameter(name):
                node.declare_parameter(name, value)
        self.selected = None
        self.items = {}
        self.scene = None
        self.draft = obstacle_from_config({'id': 'preview', 'type': 'sphere',
            'dimensions': [float(node.get_parameter('obstacle_preview_size').value)],
            'position': [2., 0., -1.]}, 0)
        self.client = node.create_client(EditDynamicObstacles,
            '/dynamic_obstacle_sim_node/edit_dynamic_obstacles')
        self.world_client = node.create_client(BackendWorldCommand, '/backend/world_command')
        self.subscription = node.create_subscription(ObstacleScene, '/obstacle_scene', self._snapshot,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._render()

    def _snapshot(self, message):
        self.scene = copy.deepcopy(message)
        old_ids = set(self.items)
        self.items = {o.id: copy.deepcopy(o) for o in message.obstacles.obstacles}
        if self.selected is not None and self.selected not in self.items:
            self.selected = None
        if old_ids != set(self.items):
            self._render()

    def _menus(self):
        self.menu = MenuHandler()
        selection = self.menu.insert('Select obstacle')
        self._choice('New obstacle (preview)', selection, self.selected is None, lambda _: self._new())
        self._selection_handles = {}
        for name in sorted(self.items):
            self._selection_handles[name] = self._choice(
                name, selection, name == self.selected, lambda _, n=name: self._select(n))

        settings = self.menu.insert('Obstacle settings (preview)')
        shape = self.menu.insert('Shape', parent=settings)
        for number, kind in enumerate(('sphere', 'box', 'cylinder', 'mesh'), 1):
            self._choice(kind.title(), shape, self.draft.collision_type == number,
                         lambda _, k=kind: self._shape(k))
        meshes = self.menu.insert('Mesh catalog', parent=shape)
        self.menu.insert('Reload catalog', parent=meshes, callback=lambda _: self._reload_catalog())
        try:
            catalog = load_mesh_catalog(self.node.get_parameter('obstacle_mesh_catalog').value)
        except (OSError, ValueError) as exc:
            catalog = {}
            self.node.get_logger().warn(f'Mesh catalog rejected: {exc}')
        groups = {}
        for name, draft in sorted(catalog.items()):
            group, separator, label = name.partition(' / ')
            parent = meshes
            if separator:
                if group not in groups:
                    groups[group] = self.menu.insert(group, parent=meshes)
                parent = groups[group]
            else:
                label = name
            fields = ('collision_type', 'collision_mesh_resource', 'collision_mesh_scale',
                      'visual_type', 'visual_mesh_resource', 'visual_dimensions')
            self._choice(label, parent, all(getattr(self.draft, f) == getattr(draft, f) for f in fields),
                         lambda _, d=draft: self._mesh(d))
        kind = self.draft.collision_type
        orientation = self.menu.insert('Orientation', parent=settings)
        q = self.draft.pose.orientation
        self._choice('Reset to asset axes (+X forward, Z up)', orientation,
                     all(abs(v) < 1e-9 for v in (q.x, q.y, q.z)) and abs(abs(q.w)-1.) < 1e-9,
                     lambda _: self._reset_orientation())
        size_label = {1: 'Radius (m)', 2: 'Cube side length (m)',
                      3: 'Radius / height (m)', 4: 'Mesh scale (largest component)'}
        sizes = self.menu.insert(size_label.get(kind, 'Size'), parent=settings)
        for size in (0.25, 0.5, 0.8, 1., 2.):
            expected = {1: [size], 2: [size]*3, 3: [size, 2*size], 4: [size]}.get(kind, [])
            actual = ([max(self.draft.collision_mesh_scale)] if kind == 4
                      else list(self.draft.collision_dimensions))
            checked = len(actual) == len(expected) and all(
                math.isclose(a, b) for a, b in zip(actual, expected))
            title = f'{size:g} / {2*size:g}' if kind == 3 else f'{size:g}'
            self._choice(title, sizes, checked, lambda _, s=size: self._size(s))
        motion = self.menu.insert('Initial velocity (world frame)', parent=settings)
        linear, angular = self.draft.twist.linear, self.draft.twist.angular
        for title, speed in (('Stationary', 0.), ('+X 0.25 m/s', .25), ('-X 0.25 m/s', -.25)):
            checked = math.isclose(linear.x, speed, abs_tol=1e-9) and all(
                abs(v) < 1e-9 for v in (linear.y, linear.z, angular.x, angular.y, angular.z))
            self._choice(title, motion, checked, lambda _, s=speed: self._velocity(s))

        if self.selected is None:
            self.menu.insert('Add obstacle', callback=lambda _: self._send('add'))
        else:
            self.menu.insert('Update selected obstacle', callback=lambda _: self._send('update'))
            self.menu.insert('Discard edits / reload selected', callback=lambda _: self._select(self.selected))
            deletion = self.menu.insert('Delete selected obstacle', callback=lambda _: None)
            self.menu.insert('Confirm delete', parent=deletion, callback=lambda _: self._send('remove'))
        scene = self.menu.insert('Scene / source')
        self.menu.insert('Source status', parent=scene, callback=lambda _: self._source_status())
        for title, operation in (('Clear all obstacles', 'clear'), ('Reset source and scene', 'reset')):
            parent = self.menu.insert(title, parent=scene, callback=lambda _: None)
            self.menu.insert('Confirm', parent=parent, callback=lambda _, op=operation: self._send(op))
        profiles = self.menu.insert('Load world profile (replaces scene)', parent=scene)
        for name in list_world_profiles():
            self.menu.insert(name, parent=profiles,
                callback=lambda _, n=name: self._world_command('set_world_profile', n))
        replanning = self.menu.insert('Replanning')
        for title, command in (('Enable', 'enable_dynamic_replanning'),
                               ('Disable', 'disable_dynamic_replanning'),
                               ('Status', 'dynamic_replanning_status')):
            self.menu.insert(title, parent=replanning,
                callback=lambda _, c=command: self._world_command(c))

    def _choice(self, title, parent, checked, callback):
        handle = self.menu.insert(title, parent=parent, callback=callback)
        self.menu.setCheckState(handle, MenuHandler.CHECKED if checked else MenuHandler.UNCHECKED)
        return handle

    def _new(self):
        self.selected = None
        self.draft.id = 'preview'
        self._render()

    def _select(self, name):
        if name not in self.items:
            self.node.get_logger().warn(f'Obstacle {name!r} is no longer in the scene')
            return
        self.selected = name
        self.draft = copy.deepcopy(self.items[name])
        self._render()

    def _reload_catalog(self):
        self._render()

    def _mesh(self, draft):
        draft = copy.deepcopy(draft)
        draft.pose = copy.deepcopy(self.draft.pose)
        draft.twist = copy.deepcopy(self.draft.twist)
        self.draft = draft
        self._render()

    def _source_status(self):
        if self.scene is None:
            self.node.get_logger().warn('Waiting for obstacle source snapshot')
            return
        self.node.get_logger().info(
            f'Obstacle source: {self.scene.source}; revision={self.scene.revision}; '
            f'obstacles={len(self.items)}; fault={self.scene.fault or "none"}')

    def _world_command(self, command, name=''):
        # Menu ownership does not couple behavior sources to planner/controller code.
        if not self.world_client.service_is_ready():
            self.node.get_logger().warn('World command service is not ready')
            return
        request = BackendWorldCommand.Request(command=command, name=name)
        self.world_client.call_async(request).add_done_callback(self._done)

    def _shape(self, kind):
        config = {'id': 'preview', 'type': kind}
        if kind == 'mesh':
            config.update(collision_mesh_resource=self.node.get_parameter('obstacle_mesh_resource').value,
                          collision_mesh_scale=[1., 1., 1.])
        else:
            config['dimensions'] = {'sphere': [0.8], 'box': [0.8]*3, 'cylinder': [0.8, 1.6]}[kind]
        try:
            draft = obstacle_from_config(config, 0)
            self._mesh(draft)
        except ValueError as exc:
            self.node.get_logger().warn(f'Obstacle draft rejected: {exc}')

    def _size(self, size):
        kind = self.draft.collision_type
        values = {1: [size], 2: [size]*3, 3: [size, 2*size], 4: [size]*3}[kind]
        if kind == 4:
            # Scale the complete authored object, retaining any collision/visual
            # scale ratio and nonuniform proportions from the mesh catalog.
            factor = size / max(self.draft.collision_mesh_scale)
            self.draft.collision_mesh_scale = [v * factor for v in self.draft.collision_mesh_scale]
            self.draft.visual_dimensions = [v * factor for v in self.draft.visual_dimensions]
        else:
            self.draft.collision_dimensions = values
            self.draft.visual_dimensions = values
        self._render()

    def _feedback(self, feedback):
        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            self.draft.pose = copy.deepcopy(feedback.pose)

    def _velocity(self, speed):
        from geometry_msgs.msg import Twist
        self.draft.twist = Twist()
        self.draft.twist.linear.x = speed
        self._render()

    def _reset_orientation(self):
        from geometry_msgs.msg import Quaternion
        self.draft.pose.orientation = Quaternion(w=1.)
        self._render()

    def _render(self):
        self._menus()
        marker = InteractiveMarker()
        marker.header.frame_id = self.frame
        marker.name = self.name
        marker.description = ('Obstacle: ' + self.selected + '\nPreview changes → Update selected obstacle'
                              if self.selected else 'New obstacle preview\nPosition / settings → Add obstacle')
        marker.pose = copy.deepcopy(self.draft.pose)
        kind, dimensions, resource = visual_geometry(self.draft)
        radius = (collision_geometry(kind, dimensions if kind != 4 else (), resource,
                                     dimensions if kind == 4 else ())[1] if kind else 0.)
        # Default axis handles are sized relative to the interactive marker.
        # Enclose the displayed asset, not just its collision proxy or mesh scale.
        marker.scale = max(1.5, 2.5 * radius + 0.5)
        # Primitive/STL previews are translucent. Material-bearing assets keep
        # their original colours. Authoritative geometry changes only on Update.
        from bringup.dynamic_obstacle_sim import DynamicObstacleTransport
        visual = Marker()
        visual.type = DynamicObstacleTransport._marker_type(self.draft)
        DynamicObstacleTransport._set_marker_scale(visual, self.draft)
        visual.pose.orientation.w = 1.
        visual.color.r, visual.color.g, visual.color.b, visual.color.a = (0.2, 0.8, 1., 0.35)
        visual.mesh_resource = DynamicObstacleTransport._marker_mesh_resource(self.draft)
        if visual.type == Marker.MESH_RESOURCE and not visual.mesh_resource.lower().endswith('.stl'):
            visual.mesh_use_embedded_materials = True
            visual.color.r = visual.color.g = visual.color.b = visual.color.a = 0.
        control = InteractiveMarkerControl()
        control.name = 'move_preview'
        control.description = 'Drag to move; right-click for menu'
        control.always_visible = True
        control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
        control.orientation.w = 1.
        control.orientation_mode = InteractiveMarkerControl.VIEW_FACING
        # The drag plane follows the camera, but the obstacle itself must not.
        control.independent_marker_orientation = True
        control.markers.append(visual)
        marker.controls.append(control)
        for axis in ('x', 'y', 'z'):
            for mode in (InteractiveMarkerControl.MOVE_AXIS, InteractiveMarkerControl.ROTATE_AXIS):
                control = InteractiveMarkerControl()
                control.name = f'{axis}_{mode}'
                control.orientation.w = 0.7071067811865476
                setattr(control.orientation, {'x': 'x', 'y': 'z', 'z': 'y'}[axis], 0.7071067811865476)
                control.interaction_mode = mode
                marker.controls.append(control)
        self.server.insert(marker, feedback_callback=self._feedback)
        self.menu.apply(self.server, self.name)
        self.server.applyChanges()

    def _send(self, operation):
        if operation in ('update', 'remove') and self.selected is None:
            self.node.get_logger().warn('Select an existing obstacle first')
            return
        if not self.client.service_is_ready():
            self.node.get_logger().warn('Obstacle service is not ready')
            return
        request = EditDynamicObstacles.Request(operation=operation)
        request.obstacles.header.frame_id = self.frame
        if operation in ('add', 'update'):
            item = copy.deepcopy(self.draft)
            item.id = f'obstacle_{uuid.uuid4().hex[:12]}' if operation == 'add' else self.selected
            request.obstacles.obstacles = [item]
        elif operation == 'remove':
            request.ids = [self.selected]
        self.client.call_async(request).add_done_callback(self._done)

    def _done(self, future):
        try:
            result = future.result()
            log = self.node.get_logger().info if result.success else self.node.get_logger().warn
            log(f'Obstacle menu: {result.message}')
        except Exception as exc:
            self.node.get_logger().warn(f'Obstacle menu request failed: {exc}')
