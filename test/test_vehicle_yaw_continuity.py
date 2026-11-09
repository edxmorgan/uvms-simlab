import numpy as np

from scipy.spatial.transform import Rotation as R

from simlab.robot import Robot, yaw_only_quat_wxyz


def test_vehicle_command_yaw_unwraps_against_previous_command():
    robot = Robot.__new__(Robot)
    robot.ned_pose = [0.0, 0.0, 0.0, 0.0, 0.0, -2.10]
    robot._last_vehicle_cmd_yaw = None
    robot._last_vehicle_target_yaw = None
    robot._last_vehicle_cmd_yaw_step = 0.0

    max_step = 0.02
    first = robot.continuous_vehicle_command_yaw(0.852, fallback_yaw=robot.ned_pose[5], max_step=max_step)
    second = robot.continuous_vehicle_command_yaw(-5.046, fallback_yaw=robot.ned_pose[5], max_step=max_step)
    third = robot.continuous_vehicle_command_yaw(0.877, fallback_yaw=robot.ned_pose[5], max_step=max_step)

    assert abs(first - robot.ned_pose[5]) <= max_step + 1e-12
    assert abs(second - first) <= max_step + 1e-12
    assert abs(third - second) <= max_step + 1e-12
    assert first > robot.ned_pose[5]
    assert second > first
    assert third > second


def test_vehicle_reference_pose_unwraps_before_controller_dispatch():
    robot = Robot.__new__(Robot)
    robot.ned_pose = [0.0, 0.0, 0.0, 0.0, 0.0, -2.107]
    robot._last_vehicle_cmd_yaw = None
    robot._last_vehicle_target_yaw = None
    robot._last_vehicle_cmd_yaw_step = 0.0
    positive_branch = np.zeros(6)
    negative_branch = np.zeros(6)
    positive_branch[5] = 0.852
    negative_branch[5] = -5.046

    first = robot.continuous_vehicle_reference_pose(positive_branch, fallback_yaw=robot.ned_pose[5])
    second = robot.continuous_vehicle_reference_pose(negative_branch, fallback_yaw=robot.ned_pose[5])

    assert abs((first[5] - robot.ned_pose[5]) - 2.959) < 1e-6
    assert abs(second[5] - first[5]) < 0.5
    assert second[5] > 0.0
    assert negative_branch[5] == -5.046


def test_vehicle_command_yaw_keeps_shortest_turn_when_direction_changes():
    robot = Robot.__new__(Robot)
    robot.ned_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 2.0]
    robot._last_vehicle_cmd_yaw = 2.0
    robot._last_vehicle_target_yaw = 2.0
    robot._last_vehicle_cmd_yaw_step = 0.02

    command = robot.continuous_vehicle_command_yaw(0.0, fallback_yaw=robot.ned_pose[5], max_step=0.02)

    assert command < 2.0
    assert abs(command - 1.98) < 1e-12
    assert robot._last_vehicle_cmd_yaw_step < 0.0


def test_vehicle_path_heading_yaw_uses_ned_direction():
    robot = Robot.__new__(Robot)

    north_yaw = robot._yaw_from_ned_direction([1.0, 0.0, 0.0], fallback_yaw=1.0)
    east_yaw = robot._yaw_from_ned_direction([0.0, 1.0, 0.0], fallback_yaw=0.0)

    assert north_yaw == 0.0
    assert east_yaw == np.pi / 2.0


def test_vehicle_path_heading_yaw_holds_when_direction_is_too_small():
    robot = Robot.__new__(Robot)

    yaw = robot._yaw_from_ned_direction([0.001, 0.0, 0.0], fallback_yaw=1.25, speed_threshold=0.03)

    assert yaw == 1.25


def test_planner_yaw_only_quaternion_removes_transient_roll_pitch():
    quat_xyzw = R.from_euler("xyz", [0.2, -0.25, 1.1]).as_quat()
    quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

    projected = yaw_only_quat_wxyz(quat_wxyz)
    projected_xyzw = [projected[1], projected[2], projected[3], projected[0]]
    roll, pitch, yaw = R.from_quat(projected_xyzw).as_euler("xyz", degrees=False)

    assert abs(roll) < 1e-12
    assert abs(pitch) < 1e-12
    assert abs(yaw - 1.1) < 1e-12


def test_completed_translation_keeps_advancing_goal_heading_with_rate_limit():
    from unittest.mock import Mock
    from simlab.robot import ControlMode
    from simlab.motion_planning.trajectory_generators.ruckig import RuckigVehicleTrajectoryGenerator
    from ruckig import Result
    r=Robot.__new__(Robot)
    r.node=Mock();r.prefix='robot_1_';r.sim_reset_hold=False;r.task_based_controller=False
    r.control_mode=ControlMode.PLANNER;r.get_state=lambda:{'status':'active'}
    r.world_frame='world';r.ned_pose=[0.]*6;r.pose_command=[0.]*6
    r.final_goal_map_ned_6=np.array([.001,0,0,0,0,1.2])
    r.planner=Mock();r.planner.planned_result={'is_success':True,'xyz':[[0.,0.,0.],[.001,0.,0.]],'quat_wxyz':[[1.,0.,0.,0.]]*2}
    r.compute_errors=lambda:(None,None,None,np.zeros(3),None)
    r.pos_blend_threshold=.5;r.trajectory_sample_period=.02;r.max_yaw_command_rate=.3
    r.vehicle_path_yaw_mode='goal';r._last_vehicle_cmd_yaw=None;r._last_vehicle_target_yaw=None
    r.reference_pub=Mock();r.world_nwu_to_map_ned=lambda **kw:(np.array(kw['xyz_world_nwu']),np.array([0.,0.,1.2]))
    r.world_nwu_vect6_to_map_ned_vect6=lambda x,**kw:x
    r.vehicle_J=Mock();r.vehicle_J.return_value.full.return_value=np.eye(6)
    r.to_body_velocity=lambda *args:np.zeros(6);r.to_body_acceleration=lambda *args:np.zeros(6);r.v_c=np.zeros(6)
    r.vehicle_cart_traj=RuckigVehicleTrajectoryGenerator(Mock(),3,.02,100)
    r.vehicle_cart_traj.start_from_path([0.]*3,np.array([[0.,0.,0.],[.001,0.,0.]]),[.5]*3,[.2]*3,[1.]*3)
    commands=[];at_finish=None
    for _ in range(300):
        r.com_trajectory_sampler_callback();commands.append(r.pose_command[5])
        if at_finish is None and not r.vehicle_cart_traj.active:at_finish=r.pose_command[5]
    assert at_finish is not None and at_finish<1.2-.1
    assert abs(commands[-1]-1.2)<1e-9
    assert np.max(np.abs(np.diff(commands)))<=.3*.02+1e-10
    # An error/cancellation must not be interpreted as successful completion.
    r.vehicle_cart_traj.last_result=Result.Error;r.final_goal_map_ned_6[5]=2.
    r.com_trajectory_sampler_callback();assert r.pose_command[5]==commands[-1]


def test_waypoint_completion_uses_measured_yaw_and_speed_not_blend():
    from types import SimpleNamespace
    from simlab.uvms_backend import UVMSBackendCore as B
    from geometry_msgs.msg import Pose
    m={'distance_m':.01,'linear_speed_mps':0.,'yaw_error_rad':.4,'yaw_blend':1.,'yaw_blend_threshold':.98}
    backend=SimpleNamespace(_robot_waypoint_tracking_metrics=lambda *a:m)
    def reached():
        return B._has_robot_reached_vehicle_waypoint(backend,object(),Pose(),
            position_tolerance_m=.35,yaw_tolerance_rad=np.radians(5),speed_tolerance_mps=.02)[0]
    assert not reached()
    m['yaw_error_rad']=.01;m['linear_speed_mps']=.1
    assert not reached()
    m['linear_speed_mps']=.001
    assert reached()


def test_completed_translation_waits_for_waypoint_heading_instead_of_stopping():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from geometry_msgs.msg import Pose
    from ruckig import Result
    from simlab.uvms_backend import UVMSBackendCore as B
    from simlab.robot import ControlMode
    from simlab.vehicle_waypoint_mission import VehicleWaypointMission
    mission=VehicleWaypointMission('robot_1_');mission.add_waypoint(Pose());mission.start();mission.mark_planning()
    robot=SimpleNamespace(k_robot=0,prefix='robot_1_',navigation_blocked_goal=None,
        control_mode=ControlMode.PLANNER,sim_reset_hold=False,planner_action_client=SimpleNamespace(busy=False),
        vehicle_cart_traj=SimpleNamespace(active=False,last_result=Result.Finished),abrupt_planner_stop=Mock(),
        final_goal_map_ned_6=np.zeros(6),_accept_planner_results=True,
        planner=SimpleNamespace(planned_result={'is_success':True}))
    backend=SimpleNamespace(robots=[robot],vehicle_waypoint_missions={0:mission},node=Mock(),
        _has_robot_reached_vehicle_waypoint=lambda *a,**kw:(False,None))
    B.vehicle_waypoint_execution_callback(backend)
    assert mission.executing and mission.active_index==0
    robot.abrupt_planner_stop.assert_not_called()
    backend._has_robot_reached_vehicle_waypoint=lambda *a,**kw:(True,{})
    B.vehicle_waypoint_execution_callback(backend)
    assert not mission.executing


def test_waypoint_inside_tolerance_does_not_skip_unfinished_translation():
    from types import SimpleNamespace as NS
    from unittest.mock import Mock
    from geometry_msgs.msg import Pose
    from ruckig import Result
    from simlab.uvms_backend import UVMSBackendCore as B
    from simlab.robot import ControlMode
    from simlab.vehicle_waypoint_mission import VehicleWaypointMission
    mission=VehicleWaypointMission('robot_1_');mission.add_waypoint(Pose());mission.start();mission.mark_planning()
    robot=NS(k_robot=0,prefix='robot_1_',navigation_blocked_goal=None,
        control_mode=ControlMode.PLANNER,sim_reset_hold=False,planner_action_client=NS(busy=False),
        vehicle_cart_traj=NS(active=True,last_result=Result.Working),abrupt_planner_stop=Mock(),
        final_goal_map_ned_6=np.zeros(6),_accept_planner_results=True,
        planner=NS(planned_result={'is_success':True}),hold_current_state_with_feedback=Mock())
    backend=NS(robots=[robot],vehicle_waypoint_missions={0:mission},node=Mock(),
        _has_robot_reached_vehicle_waypoint=lambda *a,**kw:(True,None))
    B.vehicle_waypoint_execution_callback(backend)
    assert mission.executing and robot.vehicle_cart_traj.active
    assert mission.active_index==0
    # A stale Finished result after cancellation must not resurrect settling.
    robot.vehicle_cart_traj.active=False;robot.vehicle_cart_traj.last_result=Result.Finished
    robot.final_goal_map_ned_6=None;robot._accept_planner_results=False;robot.planner.planned_result=None
    backend._has_robot_reached_vehicle_waypoint=lambda *a,**kw:(False,None)
    B.vehicle_waypoint_execution_callback(backend)
    assert not mission.executing
    robot.abrupt_planner_stop.assert_called_once()


def test_waypoint_metrics_require_valid_velocity_and_orientation():
    from types import SimpleNamespace as NS
    from geometry_msgs.msg import Pose
    from simlab.uvms_backend import UVMSBackendCore as B
    actual=Pose();actual.orientation.w=1.;goal=Pose();goal.orientation.w=1.
    state={}
    robot=NS(_pose_from_state_in_frame=lambda _:actual,get_state=lambda:state,
             vehicle_cart_traj=None,vehicle_path_yaw_mode='goal')
    backend=NS(world_frame='world')
    assert B._robot_waypoint_tracking_metrics(backend,robot,goal) is None
    state['ned_vel']=[float('nan')]*6
    assert B._robot_waypoint_tracking_metrics(backend,robot,goal) is None
    state['ned_vel']=[0.]*6
    assert B._robot_waypoint_tracking_metrics(backend,robot,goal)['yaw_error_rad']==0.
    goal.orientation.w=0.
    assert B._robot_waypoint_tracking_metrics(backend,robot,goal) is None


def test_pid_task_endpoint_uses_nearest_equivalent_heading():
    """Actual PID must not unwind accumulated waypoint turns for an IK endpoint."""
    import casadi as ca
    from pathlib import Path
    from simlab.uvms_parameters import select_robot_params
    _, params = select_robot_params('robot_1_')
    pid = ca.Function.load(str(Path(__file__).parents[1] /
        'resource/model_functions/vehicle/uv_reg_pid_controller.casadi'))
    robot = Robot.__new__(Robot)
    for turns in (-3, -1, 1, 3):
        state = np.zeros(12)
        state[5] = turns * 2*np.pi + .2
        target = np.zeros(6)
        target[5] = robot.normalize_angle(.3, state[5])
        output = pid(params.model_params, np.ones(6), np.zeros(6), np.zeros(6),
                     np.zeros(6), state, target, np.zeros(6), .01)
        np.testing.assert_allclose(float(output[0][5]), .1, atol=1e-12)
