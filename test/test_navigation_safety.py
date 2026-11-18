from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from geometry_msgs.msg import Pose

from simlab.dynamic_world import DynamicClearance
from simlab.motion_planning.dynamic_replanners.base import ReplanDecision
from simlab.motion_planning.dynamic_replanners.clearance_hysteresis import ClearanceHysteresisReplanner
from simlab.motion_planning.trajectory_generators.ruckig import RuckigVehicleTrajectoryGenerator
from simlab.vehicle_waypoint_mission import VehicleWaypointMission
from simlab.robot import ControlMode
from simlab.planner_action_client import PlannerActionClient
from simlab.robot import Robot as ActualRobot


@pytest.mark.parametrize('phase', ['unstarted', 'start', 'updated', 'reset', 'closed'])
def test_actual_trajectory_visualization_uses_read_only_reference(phase):
    generator = RuckigVehicleTrajectoryGenerator(Mock(), 3, .02, 100)
    if phase != 'unstarted':
        generator.start_from_path([0., 0., 0.], [[0., 0., 0.], [1., 0., 0.]],
                                  [.5]*3, [.2]*3, [1.]*3)
    if phase == 'updated':
        generator.update(0.)
    if phase == 'closed':
        generator.close()
    robot = SimpleNamespace(vehicle_cart_traj=generator, planner=Mock(), node=Mock(),
                            world_frame='world', control_mode=ControlMode.PLANNER,
                            sim_reset_hold=phase == 'reset', task_based_controller=False)
    elapsed = None if generator.out is None else generator.out.time
    reference = generator.current_reference()
    ActualRobot.trajectory_viz_callback(robot)
    assert (None if generator.out is None else generator.out.time) == elapsed
    if phase in ('start', 'updated'):
        robot.planner.update_target_viz.assert_called_once()
        np.testing.assert_allclose(robot.planner.update_target_viz.call_args.kwargs['xyz'], reference[0])
        np.testing.assert_allclose(robot.planner.quat_wxyz_from_x_to_vec_scipy.call_args.args[0], reference[1])
        robot.planner.clear_target.assert_not_called()
        # Consumers cannot corrupt the generator by mutating the snapshot.
        reference[0][:] = 99.
        reference[1][:] = 99.
        reference[2][:] = 99.
        assert all(np.all(np.abs(value) < 99.) for value in generator.current_reference())
    else:
        robot.planner.update_target_viz.assert_not_called()
        robot.planner.clear_target.assert_called_once()


@pytest.mark.parametrize('count', [2, 3, 4, 5, 7])
def test_path_markers_have_distance_based_density(count):
    from builtin_interfaces.msg import Time
    from simlab.planner_markers import PathPlanner
    pub = Mock()
    planner = PathPlanner(pub)
    points = np.column_stack([np.arange(count), np.zeros(count), np.zeros(count)])
    planner.update_path_viz(Time(), 'world', points, step=3)
    waypoint_marker = pub.publish.call_args_list[0].args[0]
    xs = [p.x for p in waypoint_marker.points]
    assert len(xs) > count-1
    assert set(range(count-1)).issubset(xs)
    assert max(np.diff(xs + [float(count-1)])) <= .6
    np.testing.assert_array_equal(points[:, 0], np.arange(count))


def test_display_densifies_long_edges_without_cutting_corners_and_refreshes():
    from builtin_interfaces.msg import Time
    from simlab.planner_markers import PathPlanner
    from visualization_msgs.msg import Marker
    pub = Mock()
    planner = PathPlanner(pub)
    path = np.array([[0., 0., 0.], [8., 0., 0.], [8., 4., 0.]])
    planner.update_path_viz(Time(), 'world', path)
    points = pub.publish.call_args_list[0].args[0].points
    assert len(points) >= 20
    assert all(p.y == 0. or p.x == 8. for p in points)
    assert any(p.x == 8. and p.y == 0. for p in points)
    pub.reset_mock()
    planner.update_path_viz(Time(sec=1), 'world', path)
    assert pub.publish.call_count == 2  # late subscribers receive the path
    pub.reset_mock()
    planner.clear_path(Time(sec=2), 'world')
    assert all(call.args[0].action == Marker.DELETE for call in pub.publish.call_args_list)


class World:
    def __init__(self):
        self.obstacles = {'obstacle': SimpleNamespace(owner_robot='', velocity_valid=True,
            angular_velocity_world=np.zeros(3), linear_velocity_world=np.zeros(3), collision_type=1)}
        self.unavailable_robot_owners = set()
        self.clearance = 2.0
    def snapshot(self):
        return self
    def availability_error(self, exclude_owner=""):
        return None
    def min_clearance_xyz(self, xyz, *, t_offset=0., exclude_owner=""):
        self.last_exclude_owner = exclude_owner
        return DynamicClearance('obstacle', self.clearance)


class Robot:
    prefix = 'test_robot'
    control_mode = ControlMode.PLANNER
    sim_reset_hold = False
    task_based_controller = False
    max_traj_acc = [0.1]*3
    max_traj_jerk = [1.]*3
    world_frame = 'world'
    max_traj_vel = [0.5, 0., 0.]
    def __init__(self):
        self._navigation_epoch = 0
        self.planner = SimpleNamespace(planned_result={'xyz': [[0.,0.,0.],[5.,0.,0.]]})
        self.vehicle_cart_traj = SimpleNamespace(active=True, current_reference=lambda: None)
        self.planner_action_client = SimpleNamespace(busy=False)
        self.last_vehicle_goal_pose_world = Pose()
        self.last_vehicle_goal_pose_world.position.x = 5.
        self.stops = 0
        self.plans = []
        self.speed = 0.05
    def _pose_from_state_in_frame(self, frame):
        return Pose()
    def _current_vehicle_velocity_world_nwu(self):
        return np.array([self.speed, 0., 0.])
    def abrupt_planner_stop(self, **kwargs):
        self._navigation_epoch += 1
        self.stops += 1
        self.vehicle_cart_traj.active = False
        self.last_vehicle_goal_pose_world = None
        self.planner_action_client.busy = False
    def hold_current_state_with_feedback(self):
        pass
    def plan_vehicle_trajectory_action(self, **kwargs):
        self._navigation_epoch += 1
        self.plans.append(kwargs)
        self.planner_action_client.busy = True
        return True


def setup(hardware=False):
    robot = Robot()
    world = World()
    backend = SimpleNamespace(dynamic_replanning_enabled=True, node=SimpleNamespace(get_logger=lambda: Mock()),
        world_frame='world', dynamic_world=world, use_vehicle_hardware=hardware,
        fcl_world=SimpleNamespace(vehicle_radius=0.574,
            min_distance_xyz=lambda xyz: float('inf'), planner_in_collision_at_xyz=lambda xyz: False))
    mission = VehicleWaypointMission('test_robot')
    mission.add_waypoint(robot.last_vehicle_goal_pose_world)
    mission.start()
    mission.mark_planning()
    replanner = ClearanceHysteresisReplanner(backend, robot, mission,
        cooldown_s=0., lookahead_time_s=4., safety_margin_m=0.25)
    replanner._remaining_path_decision = lambda _, **kwargs: ReplanDecision('replan', 'blocked', 'obstacle', 0.1, 'path', 4.)
    return replanner, robot, world, mission


def test_scene_or_path_change_invalidates_suppression():
    replanner, robot, world, mission = setup()
    replanner.tick()
    robot.planner_action_client.busy = False
    replanner.tick()
    assert len(robot.plans) == 1
    world.obstacles['new'] = SimpleNamespace(**vars(world.obstacles['obstacle']))
    replanner.tick()
    assert len(robot.plans) == 2
    robot.planner_action_client.busy = False
    replanner._remaining_path_decision = lambda _, **kwargs: ReplanDecision('replan', 'blocked', 'obstacle', 0.12, 'new_path', 4.)
    replanner.tick()
    assert len(robot.plans) == 3


def test_all_path_points_participate_in_signature():
    path = np.zeros((7, 3))
    first = ClearanceHysteresisReplanner._path_signature(path)
    path[1, 0] = 1.
    assert ClearanceHysteresisReplanner._path_signature(path) != first


def test_actual_ruckig_preview_does_not_advance_execution():
    generator = RuckigVehicleTrajectoryGenerator(SimpleNamespace(get_logger=lambda: Mock()), 3, 0.02, 0)
    generator.start_from_path([0., 0., 0.], np.array([[0., 0., 0.], [1., 0., 0.]]),
                              [0.5]*3, [0.2]*3, [1.]*3)
    assert generator.preview_samples() == []
    generator.update(0.)
    first = generator.preview_samples(sample_dt=0.05)
    second = generator.preview_samples(sample_dt=0.05)
    np.testing.assert_allclose([p for _, p in first], [p for _, p in second])
    np.testing.assert_allclose(first[0][1], generator.out.new_position)
    np.testing.assert_allclose(first[-1][1], [1., 0., 0.])
    initial_time = generator.out.time
    generator.update(0.)
    assert generator.out.time > initial_time
    generator.update(0.)
    assert generator.out.time == pytest.approx(0.06)
    generator.close()


def test_dynamic_monitor_reads_restored_ruckig_without_changing_execution():
    replanner, robot, world, _ = setup()
    generator = RuckigVehicleTrajectoryGenerator(Mock(), 3, .02, 100)
    robot.vehicle_cart_traj = generator
    generator.start_from_path([0., 0., 0.], [[0., 0., 0.], [1., 0., 0.]],
                              [.5]*3, [.2]*3, [1.]*3)
    robot.planner = SimpleNamespace(planned_result={'xyz': [[0., 0., 0.], [1., 0., 0.]]})
    decision = ClearanceHysteresisReplanner._remaining_path_decision(replanner, robot)
    assert decision.action == 'pending'
    generator.update(0.)
    before = generator.out.time
    world.clearance = .1
    decision = ClearanceHysteresisReplanner._remaining_path_decision(replanner, robot)
    assert decision.action == 'replan'
    assert generator.out.time == before
    world.clearance = 2.
    decision = ClearanceHysteresisReplanner._remaining_path_decision(replanner, robot)
    assert decision.action == 'keep'
    assert generator.out.time == before


def test_missing_dynamic_world_is_not_reported_as_valid():
    replanner, robot, _, _ = setup()
    robot.vehicle_cart_traj.preview_samples = lambda **kwargs: []
    replanner.backend.dynamic_world = None
    decision = ClearanceHysteresisReplanner._remaining_path_decision(replanner, robot)
    assert decision.should_replan
    assert 'dynamic world unavailable' in decision.reason
    assert decision.t_offset_s == 0.


def test_late_action_acceptance_is_cancelled_and_stale_result_ignored():
    client = PlannerActionClient.__new__(PlannerActionClient)
    client._generation = 2
    client._busy = True
    client._on_result = Mock()
    client._goal_handle = 'current'
    handle = Mock(accepted=True)
    future = Mock()
    future.result.return_value = handle
    client._goal_response_callback(future, generation=1)
    handle.cancel_goal_async.assert_called_once()
    client._result_callback(future, generation=1)
    assert client._goal_handle == 'current' and client._busy
    client._on_result.assert_not_called()


def test_actual_bitstar_mesh_path_to_incremental_ruckig_trajectory(tmp_path):
    import trimesh
    from bringup.obstacle_description import obstacle_from_config
    from ros2_control_blue_reach_5.msg import DynamicObstacleArray
    from simlab.dynamic_world import DynamicWorldModel
    from simlab.planner_world import PlannerWorld
    from simlab.motion_planning.planners.ompl import OmplPlanner
    mesh = tmp_path / 'blocking_box.stl'
    trimesh.creation.box(extents=[.6, .6, .6]).export(mesh)
    obstacle = obstacle_from_config(dict(id='blocking_mesh', type='mesh',
        collision_mesh_resource=str(mesh), collision_mesh_scale=[1., 1., 1.]), 0)
    clock_node = Mock()
    clock_node.get_clock.return_value.now.return_value.nanoseconds = 0
    dynamic = DynamicWorldModel(clock_node, world_frame='world', robot_radius_provider=lambda: .1)
    snapshot = DynamicObstacleArray(obstacles=[obstacle])
    snapshot.header.frame_id = 'world'
    dynamic.update_from_msg(snapshot)
    world = PlannerWorld(fcl_world=SimpleNamespace(min_distance_xyz=lambda xyz: float('inf'),
        planner_in_collision_at_xyz=lambda xyz: False), dynamic_world=dynamic)
    node = SimpleNamespace(planner_world=world, get_logger=lambda: Mock())
    planner = OmplPlanner(node, safety_margin=.05, env_bounds=(-2., 2., -2., 2., -2., 2.))
    start, goal = [-1., 0., 0.], [1., 0., 0.]
    result = planner.plan_se3_path(start, [1., 0., 0., 0.], goal, [1., 0., 0., 0.], time_limit=.5)
    assert result.is_success, result.message
    generator = RuckigVehicleTrajectoryGenerator(node, 3, .01, 100)
    generator.start_from_path(start, result.xyz, [.3]*3, [.2]*3, [.5]*3)
    generator.update(0.)
    assert len(generator.preview_samples(sample_dt=.01)) > 2
    assert world.in_collision_xyz([-.3, 0., 0.])


@pytest.mark.parametrize("settling", [False, True])
def test_waypoint_mission_waits_for_busy_planner(settling):
    from simlab.uvms_backend import UVMSBackendCore
    robot = SimpleNamespace(k_robot=0,
        control_mode=ControlMode.PLANNER, sim_reset_hold=False,
        planner_action_client=SimpleNamespace(busy=not settling),
        _planning_settle_timer=object() if settling else None,
        abrupt_planner_stop=Mock())
    mission = VehicleWaypointMission('test')
    goal = Pose()
    goal.position.x = 5.
    mission.add_waypoint(goal)
    mission.start()
    mission.mark_planning()
    backend = SimpleNamespace(robots=[robot], vehicle_waypoint_missions={0: mission},
        _has_robot_reached_vehicle_waypoint=lambda *args, **kwargs: (False, None))
    UVMSBackendCore.vehicle_waypoint_execution_callback(backend)
    assert mission.executing and mission.active_index == 0
    robot.abrupt_planner_stop.assert_not_called()


def test_planning_action_requires_explicit_scene_robot_identity():
    from simlab.action import PlanVehicle
    from simlab.planner_action_server import PlannerActionServer
    from rclpy.action import GoalResponse
    request = PlanVehicle.Goal()
    request.start_xyz = [0.,0.,-1.]
    request.goal_xyz = [1.,0.,-1.]
    request.start_quat_wxyz = request.goal_quat_wxyz = [1.,0.,0.,0.]
    request.robot_collision_radius = .5
    request.planner_name = 'Bitstar'
    server = SimpleNamespace(scene_participant=SimpleNamespace(held=False),
        fcl_world=SimpleNamespace(robot_prefixes={'robot_real_','robot_2_'}),
        _planners={'Bitstar':None}, _tag='test', get_logger=lambda:Mock())
    assert PlannerActionServer.goal_callback(server, request) == GoalResponse.REJECT
    request.robot_prefix = 'robot_real_'
    assert PlannerActionServer.goal_callback(server, request) == GoalResponse.ACCEPT
    request.robot_prefix = 'robot_missing_'
    assert PlannerActionServer.goal_callback(server, request) == GoalResponse.REJECT


def test_navigation_clearance_calls_exclude_own_robot():
    replanner, robot, world, mission = setup()
    robot.vehicle_cart_traj = SimpleNamespace(preview_samples=lambda **_: [(0.,np.zeros(3))], speed_bound=lambda: 0., preview_position=lambda t: np.zeros(3))
    ClearanceHysteresisReplanner._remaining_path_decision(replanner, robot)
    assert world.last_exclude_owner == robot.prefix


def test_navigation_signature_tracks_other_robots_and_missing_telemetry():
    replanner, robot, world, mission = setup()
    world.obstacles = {'self':SimpleNamespace(owner_robot=robot.prefix, pose=0.),
                       'other':SimpleNamespace(owner_robot='robot_2_', pose=1.)}
    original = replanner._world_signature()
    world.obstacles['self'].pose = 2.
    assert replanner._world_signature() == original
    world.obstacles['other'].pose = 2.
    assert replanner._world_signature() != original
    updated = replanner._world_signature()
    world.unavailable_robot_owners.add('robot_2_')
    assert replanner._world_signature() != updated


def test_disabled_auto_replanning_does_not_plan_for_distant_conflict():
    replanner, robot, world, mission = setup()
    replanner.backend.dynamic_replanning_enabled = False
    world.clearance = 10.
    replanner._remaining_path_decision = lambda _, **kwargs: ReplanDecision('replan', 'distant', 'other', .1, 'path', 30.)
    replanner.tick()
    assert robot.stops == 0
    assert not robot.plans


def test_proactive_clearance_requests_replan_without_emergency_hold():
    replanner, robot, world, _ = setup()
    robot.speed = 0.
    world.clearance = .1
    robot.vehicle_cart_traj.preview_samples = lambda **_: [(0., np.zeros(3)), (1., np.zeros(3))]
    robot.vehicle_cart_traj.speed_bound = lambda: 0.
    robot.vehicle_cart_traj.preview_position = lambda t: np.zeros(3)
    replanner._remaining_path_decision = lambda robot, **kw: ClearanceHysteresisReplanner._remaining_path_decision(replanner, robot, **kw)
    replanner.tick()
    assert robot.stops == 0
    assert len(robot.plans) == 1




@pytest.mark.parametrize('enabled,busy', [(False,False),(True,False),(True,True)])
def test_collision_replanning_never_interrupts_execution(enabled, busy):
    replanner, robot, world, mission = setup()
    replanner.backend.dynamic_replanning_enabled = enabled
    robot.planner_action_client.busy = busy
    replanner._remaining_path_decision = lambda *a, **kw: ReplanDecision(
        'replan', 'overlap', 'obstacle', -.2, 'path', 0.)
    replanner.tick()
    assert robot.stops == 0
    assert robot.vehicle_cart_traj.active
    assert len(robot.plans) == int(enabled and not busy)
    if robot.plans:
        assert robot.plans[0]['preempt_current'] is False
