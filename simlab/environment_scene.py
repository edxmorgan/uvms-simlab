"""Adapters for the shared, held static-scene transition protocol."""
import numpy as np
import rclpy
from bringup.scene_runtime import SceneParticipant, seed_environment_tf
from simlab.fcl_checker import FCLWorld
from simlab.utils.meshes import collect_env_meshes


def prepare_world(node, owner, description, tf_buffer):
    current = getattr(owner, 'fcl_world', getattr(owner, 'world', None))
    world = FCLWorld(description, world_frame=owner.world_frame, vehicle_radius=current.vehicle_radius)
    seed_environment_tf(tf_buffer, description, node)
    return world, description


def attach_backend(backend):
    node = backend.node
    from rclpy.qos import QoSProfile, DurabilityPolicy
    from visualization_msgs.msg import MarkerArray
    from simlab.utils.geometry import environment_grid_markers
    backend.environment_grid_publisher = node.create_publisher(
        MarkerArray, '/environment_grid', QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    def publish_grid():
        backend.environment_grid_publisher.publish(environment_grid_markers(
            backend.fcl_world.env_xyz_bounds, backend.world_frame))
    publish_grid()
    def hold():
        from simlab.robot import ControlMode
        if any(robot.control_mode == ControlMode.TELEOP for robot in backend.robots):
            raise ValueError('Switch from joystick to interactive feedback control before changing scenes')
        if any(robot._active_controller_is_replay() for robot in backend.robots):
            raise ValueError('Select a feedback controller before changing scenes')
        backend.scene_transition = True
        for robot in backend.robots:
            robot.scene_switch_hold = True
            # Cancel references tied to the old world without revoking control
            # authorization or sending reset/pause commands to any component.
            robot._cancel_motion_calculation()
            robot._cancel_planning_settle()
            robot._accept_planner_results = False
            robot.planner_action_client.cancel_active_goal()
            robot._zero_teleop_commands()
            backend.vehicle_waypoint_missions[robot.k_robot].stop()
            if robot.vehicle_cart_traj is not None:
                robot.vehicle_cart_traj.active = False
            robot.final_goal_map_ned_6 = None
            robot.last_vehicle_goal_pose_world = None
            if robot.planner is not None:
                robot.planner.planned_result = None
                stamp = node.get_clock().now().to_msg()
                robot.planner.clear_path(stamp, robot.world_frame)
                robot.planner.clear_target(stamp, robot.world_frame)
            robot.hold_current_state_with_feedback()
            robot.pose_command = np.asarray(robot.get_state()['pose'], dtype=float).tolist()
    def prepare(description):
        # Only build detached geometry on the worker. Live TF and controller
        # state are changed by commit/verify in the normal callback context.
        return FCLWorld(description, world_frame=backend.world_frame,
                        vehicle_radius=backend.fcl_world.vehicle_radius), description
    def commit(staged):
        backend.fcl_world, backend.urdf_string = staged
        seed_environment_tf(backend.tf_buffer, backend.urdf_string, node)
        publish_grid()
    def verify():
        if not backend.fcl_world.update_from_tf(backend.tf_buffer, rclpy.time.Time()):
            raise ValueError('Robot/environment TF unavailable during scene verification')
    def release():
        for robot in backend.robots:
            robot.scene_switch_hold = False
        backend.scene_transition = False
    backend.scene_transition = False
    backend.scene_participant = SceneParticipant(node, prepare, commit, hold=hold, release=release, verify=verify, prepare_async=True)


def attach_planner(node):
    def prepare(description):
        # An executing action can use a reentrant callback group. Do not replace
        # its world while the solver still holds references to it.
        with node._scene_execution_lock:
            return prepare_world(node, node, description, node.tf_buffer)
    def commit(staged):
        with node._scene_execution_lock:
            node.fcl_world = staged[0]
            node.planner_world.fcl_world = node.fcl_world
            node.env_bounds = tuple(float(v) for v in node.fcl_world.env_xyz_bounds)
            node._fcl_ok = False
            from simlab.motion_planning.planners import DEFAULT_PLANNER_CLASSES
            node._planners = {cls.registry_name:cls(node) for cls in DEFAULT_PLANNER_CLASSES}
    node.scene_participant = SceneParticipant(node, prepare, commit)


def attach_contact(node):
    def commit(staged):
        node.world = staged[0]
    node.scene_participant = SceneParticipant(node,
        lambda d:prepare_world(node, node, d, node.tf_buf), commit)


def attach_voxels(node):
    def prepare(description):
        _, meshes, _ = collect_env_meshes(description)
        # Geometry is staged by the same collision-surface loader used by FCL;
        # voxel generation remains asynchronous and is only a visualization.
        from bringup.collision_geometry import collision_surface_mesh
        for info in meshes:
            spec = node._static_spec(info)
            collision_surface_mesh(*spec)
            node.caches['static'].request(spec)
        return description, meshes
    def commit(staged):
        description, node.static_meshes = staged
        seed_environment_tf(node.tf_buffer, description, node)
        node._signatures.clear()
        from simlab.utils.meshes import points_to_cloud2
        # Never retain an old-scene cloud while the replacement builds.
        node.cloud_pub.publish(points_to_cloud2(np.empty((0, 3)), frame_id=node.world_frame))
        node.caches['static'].retain({node._static_spec(info) for info in node.static_meshes})
        # The old cloud is already cleared. The normal timer publishes only
        # when all new-scene voxels are ready; display work cannot block physics.
    node.scene_participant = SceneParticipant(node, prepare, commit)
