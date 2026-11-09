from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from geometry_msgs.msg import Pose
from visualization_msgs.msg import InteractiveMarkerFeedback
from simlab.robot import Robot
from simlab.uvms_backend import UVMSBackendCore
from simlab.interactive_control import InteractiveControlsNode

def fixture():
    robot = SimpleNamespace(arm=SimpleNamespace(), k_robot=0, prefix='robot_1_',
        sim_reset_hold=False, task_based_controller=False,
        get_state=lambda: {'pose':[0.]*6, 'q':[3.1,.7,.4,2.1]},
        enable_planner_output=Mock())
    backend = SimpleNamespace(robot_selected=robot, target_arm_base_endeffector_pose=Pose(),
        set_robot_selected=Mock(), is_valid_arm_base_task=Mock(return_value=True),
        solve_execute_inverse_kinematics_wrt_vehicle_frame=Mock(
            return_value={'is_success':True,'result':[3.14,.41,.18,0.]}))
    return robot,backend

def test_hold_survives_timer_until_explicit_valid_target():
    robot,backend=fixture()
    Robot.hold_pose_with_feedback(robot)
    for _ in range(20):
        UVMSBackendCore.plan_and_execute_task_trajectory_wrt_vehicle(backend)
    np.testing.assert_array_equal(robot.arm.q_command,[3.1,.7,.4,2.1])
    backend.solve_execute_inverse_kinematics_wrt_vehicle_frame.assert_not_called()
    backend.is_valid_arm_base_task.return_value=False
    assert not UVMSBackendCore.set_task_target_arm_base(backend,robot,Pose())[0]
    assert robot._arm_feedback_hold
    backend.is_valid_arm_base_task.return_value=True
    assert UVMSBackendCore.set_task_target_arm_base(backend,robot,Pose())[0]
    UVMSBackendCore.plan_and_execute_task_trajectory_wrt_vehicle(backend)
    np.testing.assert_array_equal(robot.arm.q_command,[3.14,.41,.18,0.])

@pytest.mark.parametrize('event',[InteractiveMarkerFeedback.KEEP_ALIVE,
    InteractiveMarkerFeedback.MOUSE_DOWN,InteractiveMarkerFeedback.MOUSE_UP,
    InteractiveMarkerFeedback.BUTTON_CLICK,InteractiveMarkerFeedback.MENU_SELECT])
def test_non_motion_feedback_does_not_release_hold(event):
    ui=SimpleNamespace(uvms_backend=SimpleNamespace(set_task_target_arm_base=Mock()))
    InteractiveControlsNode.arm_base_task_marker_processFeedback(ui,
        InteractiveMarkerFeedback(event_type=event))
    ui.uvms_backend.set_task_target_arm_base.assert_not_called()

def test_pose_feedback_still_submits_explicit_target():
    backend=SimpleNamespace(robot_selected=object(),target_arm_base_endeffector_pose=Pose(),
        arm_base_target_frame='arm_base',set_task_target_arm_base=Mock(return_value=(True,'accepted')))
    ui=SimpleNamespace(uvms_backend=backend,sync_endeffector_world_marker_pose=Mock(),
        get_logger=Mock(return_value=Mock()))
    event=InteractiveMarkerFeedback(event_type=InteractiveMarkerFeedback.POSE_UPDATE)
    InteractiveControlsNode.arm_base_task_marker_processFeedback(ui,event)
    backend.set_task_target_arm_base.assert_called_once_with(backend.robot_selected,event.pose)
    ui.sync_endeffector_world_marker_pose.assert_called_once()


def test_vehicle_only_hold_preserves_explicit_arm_goal_and_ownership():
    robot,backend=fixture()
    robot.arm.q_command=[2.,1.,1.2,3.]
    robot.arm.dq_command=[.1]*4;robot.arm.ddq_command=[.2]*4
    robot._arm_feedback_hold=False
    Robot.hold_pose_with_feedback(robot,hold_arm=False)
    assert robot.arm.q_command==[2.,1.,1.2,3.]
    assert robot.arm.dq_command==[.1]*4 and robot.arm.ddq_command==[.2]*4
    assert not robot._arm_feedback_hold
    UVMSBackendCore.plan_and_execute_task_trajectory_wrt_vehicle(backend)
    backend.solve_execute_inverse_kinematics_wrt_vehicle_frame.assert_called_once()


def test_starting_vehicle_plan_does_not_replace_new_arm_goal():
    robot,backend=fixture()
    robot.node=Mock();robot.planner_name='Bitstar'
    robot._replay_is_active=lambda:False
    robot._pose_from_state_in_frame=lambda frame:Pose()
    robot.world_frame='world';robot.abrupt_planner_stop=Mock()
    robot.hold_current_state_with_feedback=lambda **kw:Robot.hold_pose_with_feedback(robot,**kw)
    robot._pose_to_xyz_quat_wxyz=lambda pose:([0.,0.,0.],[1.,0.,0.,0.])
    robot._get_vehicle_goal_from_marker=lambda pose:([1.,0.,0.],[1.,0.,0.,0.])
    robot._log_plan_context=Mock();robot._save_vehicle_goal_from_target=Mock()
    robot.planner_action_client=Mock();robot.planner_action_client.send_goal.return_value=True
    robot.arm.q_command=[2.,1.,1.2,3.];robot.arm.dq_command=[0.]*4;robot.arm.ddq_command=[0.]*4
    robot._arm_feedback_hold=False
    assert Robot.plan_vehicle_trajectory_action(robot,Pose())
    assert robot.arm.q_command==[2.,1.,1.2,3.]
    assert not robot._arm_feedback_hold
