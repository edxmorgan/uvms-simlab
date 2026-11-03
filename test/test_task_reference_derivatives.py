import pytest
from types import SimpleNamespace
from simlab.robot import Robot, ControlSpace


def robot():
    r=SimpleNamespace(body_vel_command=[.15,0,.068,0,0,0],
        body_acc_command=[-.0016,.0093,0,0,0,0],
        arm=SimpleNamespace(q_command=[3.1,.7,.4,2.1],dq_command=[.1]*4,ddq_command=[.2]*4))
    r._clear_task_reference_derivatives=lambda:Robot._clear_task_reference_derivatives(r)
    return r


def test_switch_to_task_drops_old_trajectory_feedforward():
    r=robot()
    Robot.set_control_space(r,ControlSpace.TASK_SPACE)
    assert r.task_based_controller
    assert r.body_vel_command==[0.]*6 and r.body_acc_command==[0.]*6
    assert r.arm.dq_command==[0.]*4 and r.arm.ddq_command==[0.]*4
    assert r.arm.q_command==[3.1,.7,.4,2.1]


def test_joint_mode_preserves_trajectory_derivatives():
    r=robot();before=r.body_vel_command.copy()
    Robot.set_control_space(r,ControlSpace.JOINT_SPACE)
    assert r.body_vel_command==before


@pytest.mark.parametrize("axis", [[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
@pytest.mark.parametrize('conversion_ok', [True, False])
def test_accepted_ik_endpoint_clears_derivatives(axis, conversion_ok):
    # Exercise the production acceptance path with frame conversion supplied.
    import numpy as np
    from unittest.mock import Mock
    from geometry_msgs.msg import Pose
    from scipy.spatial.transform import Rotation as R
    from simlab.motion_planning.kinematics.inverse.whole_body import EndpointIK
    from simlab.uvms_parameters_sim import ReachParams as P
    from pathlib import Path
    r=robot();r.world_frame='world';r.prefix='robot_1_'
    r.node=Mock();r.get_state=lambda:{'q':r.arm.q_command}
    current=Pose();current.orientation.w=1.;current.position.z=-2.
    r._pose_from_state_in_frame=lambda _:current
    r.endpoint_ik=EndpointIK(Path(__file__).resolve().parents[1]/'resource')
    r.fk_eval=r.endpoint_ik.fk
    r.ik_joint_lower=np.array([.01]*4);r.ik_joint_upper=np.array([6.27,3.4,3.4,5.7])
    r.ik_tool_axis=np.array(axis);r.ik_base_align_w=1.;r.ik_position_tolerance=.01;r.ik_orientation_tolerance=.05
    pose=np.array(r.fk_eval(r.arm.q_command,P.base_T0_new,[0,0,-2,0,0,0],P.tipOffset)[-1]).ravel()
    target=Pose();target.position.x,target.position.y,target.position.z=pose[:3]
    target.orientation.w,target.orientation.x,target.orientation.y,target.orientation.z=pose[3:]
    r.world_nwu_to_map_ned=lambda **kwargs:(kwargs['xyz_world_nwu'],R.from_quat(np.array(kwargs['quat_world_wxyz'])[[1,2,3,0]]).as_euler('xyz'))
    r._vehicle_desired_pose_from_ik_ = 'previous marker'
    r.pose_command = [42.] * 6
    previous_joints = list(r.arm.q_command)
    if not conversion_ok:
        r.world_nwu_to_map_ned = lambda **kwargs: None
    r.endpoint_ik.solve=Mock(wraps=r.endpoint_ik.solve)
    Robot.solve_inverse_kinematics_wrt_world_frame(r,target)
    assert r.endpoint_ik.solve.call_args.kwargs["base_align_weight"]==1.
    # The selected marker axis must reach the solver as the requested tool +Z.
    marker_rotation=R.from_quat(pose[[4,5,6,3]]).as_matrix()
    np.testing.assert_allclose(r.endpoint_ik.solve.call_args.args[3][:,2],marker_rotation@axis,atol=1e-10)
    assert r.endpoint_ik.last_report['position_error_m']<1e-6
    assert r.endpoint_ik.last_report['orientation_error_rad']<1e-6
    if conversion_ok:
        assert r.body_vel_command==[0.]*6 and r.arm.dq_command==[0.]*4
    else:
        assert r._vehicle_desired_pose_from_ik_ == 'previous marker'
        assert r.pose_command == [42.] * 6
        assert r.arm.q_command == previous_joints
        assert r.arm.dq_command == [.1] * 4


def joint_switch_fixture():
    import numpy as np
    from pathlib import Path
    from unittest.mock import Mock
    from simlab.motion_planning.kinematics.inverse.whole_body import EndpointIK
    from simlab.uvms_backend import UVMSBackendCore as B
    from simlab.robot import ControlMode
    from geometry_msgs.msg import TransformStamped
    from tf2_geometry_msgs import do_transform_pose
    from scipy.spatial.transform import Rotation as R
    r=robot();r.prefix='robot_1_';r.k_robot=0;r.map_frame='robot_1_map'
    r.task_based_controller=True;r.control_space=ControlSpace.TASK_SPACE;r.control_mode=ControlMode.PLANNER
    r.pose_command=[99.]*6;r.arm.q_command=[1.,1.,1.,1.]
    r.sim_reset_hold=False
    r.state={'pose':[1.,2.,3.,.2,-.1,.4],'q':[3.1,.7,.4,2.1],'status':'active'}
    r.get_state=lambda:r.state
    r.list_control_spaces=lambda:[m.value for m in ControlSpace]
    r.set_control_space=lambda name:Robot.set_control_space(r,name)
    r.fk_eval=EndpointIK(Path(__file__).resolve().parents[1]/'resource').fk
    r.abrupt_planner_stop=Mock();r.enable_planner_output=Mock()
    tf=TransformStamped();tf.transform.translation.x=8.;tf.transform.translation.y=-3.
    q=R.from_euler('z',.6).as_quat();tf.transform.rotation.x,tf.transform.rotation.y,tf.transform.rotation.z,tf.transform.rotation.w=q
    r.try_transform_pose=Mock(side_effect=lambda pose,**kw:do_transform_pose(pose,tf))
    b=SimpleNamespace(world_frame='world',vehicle_waypoint_missions={0:Mock()},robot_selected=r,
        target_vehicle_pose=None,target_arm_base_endeffector_pose=None,target_world_endeffector_pose=None)
    b._hold_measured_joint_space=lambda rr:B._hold_measured_joint_space(b,rr)
    return r,b


def test_joint_switch_holds_measured_state_and_rebases_markers():
    import numpy as np
    from simlab.uvms_backend import UVMSBackendCore as B
    from simlab.uvms_parameters import ReachParams as P
    from simlab.utils.frames import PoseX
    from scipy.spatial.transform import Rotation as R
    r,b=joint_switch_fixture()
    assert B.set_robot_control_space(b,r,ControlSpace.JOINT_SPACE)[0]
    assert not r.task_based_controller
    np.testing.assert_array_equal(r.pose_command,r.state['pose'])
    np.testing.assert_array_equal(r.arm.q_command,r.state['q'])
    assert r.body_vel_command==[0.]*6 and r.body_acc_command==[0.]*6
    assert r.arm.dq_command==[0.]*4 and r.arm.ddq_command==[0.]*4
    assert r._arm_feedback_hold
    r.abrupt_planner_stop.assert_called_once_with(publish_zero=False)
    b.vehicle_waypoint_missions[0].stop.assert_called_once()
    # Nonidentity world/map transform and NED/NWU conversion must both apply.
    expected_map=PoseX.from_pose(xyz=r.state['pose'][:3],rot=r.state['pose'][3:],
        rot_rep='euler_xyz',frame='NED').get_pose_as_Pose_msg(frame='NWU')
    expected=r.try_transform_pose(expected_map)
    assert b.target_vehicle_pose==expected
    tool=b.target_arm_base_endeffector_pose
    local=np.array(r.fk_eval(r.state['q'],np.zeros(6),np.zeros(6),P.tipOffset)[-1]).ravel()
    np.testing.assert_allclose([tool.position.x,tool.position.y,tool.position.z],local[:3])
    actual_world=np.array([b.target_world_endeffector_pose.position.x,
        b.target_world_endeffector_pose.position.y,b.target_world_endeffector_pose.position.z])
    vehicle_rot=R.from_quat([expected.orientation.x,expected.orientation.y,expected.orientation.z,expected.orientation.w])
    arm_rot=R.from_euler('xyz',P.base_T0_new[3:])
    np.testing.assert_allclose(actual_world,np.array([expected.position.x,expected.position.y,expected.position.z])+
        vehicle_rot.apply(np.array(P.base_T0_new[:3])+arm_rot.apply(local[:3])),atol=1e-10)
    # The existing timer must not re-solve the held arm onto another IK branch.
    B.plan_and_execute_task_trajectory_wrt_vehicle(b)
    np.testing.assert_array_equal(r.arm.q_command,r.state['q'])


@pytest.mark.parametrize('failure',['tf','nonfinite','inactive'])
def test_joint_switch_rejection_preserves_mode_and_goals(failure):
    from simlab.uvms_backend import UVMSBackendCore as B
    r,b=joint_switch_fixture()
    if failure=='tf':r.try_transform_pose.side_effect=None;r.try_transform_pose.return_value=None
    elif failure=='nonfinite':r.state['q'][0]=float('nan')
    else:r.state['status']='inactive'
    assert not B.set_robot_control_space(b,r,ControlSpace.JOINT_SPACE)[0]
    assert r.task_based_controller and r.pose_command==[99.]*6
    assert r.arm.q_command==[1.]*4 and b.target_vehicle_pose is None
    r.abrupt_planner_stop.assert_not_called()


def test_joint_markers_use_captured_targets_without_resubmitting_old_goal():
    from unittest.mock import Mock
    from geometry_msgs.msg import Pose
    from simlab.interactive_control import InteractiveControlsNode as UI
    b=SimpleNamespace(robot_selected=object(),target_vehicle_pose=Pose(),
        target_arm_base_endeffector_pose=Pose(),_vehicle_desired_pose_from_ik_=Pose(),
        set_vehicle_target=Mock(),set_task_target_arm_base=Mock())
    b.target_vehicle_pose.position.x=4.;b._vehicle_desired_pose_from_ik_.position.x=99.
    u=SimpleNamespace(uvms_backend=b,uv_marker=SimpleNamespace(name='uv',pose=None),
        arm_base_task_marker=SimpleNamespace(name='arm',pose=None),server=Mock(),menu_handler=Mock(),
        _clear_markers=Mock(),_ensure_menu_control=Mock(),_configure_arm_marker_mode=Mock(),vehicle_marker_processFeedback=Mock(),
        arm_base_task_marker_processFeedback=Mock(),get_logger=Mock(return_value=Mock()))
    UI._apply_joint_control_mode(u,b.robot_selected,abort_motion=False)
    assert u.uv_marker.pose==b.target_vehicle_pose
    assert u.arm_base_task_marker.pose==b.target_arm_base_endeffector_pose
    b.set_vehicle_target.assert_not_called();b.set_task_target_arm_base.assert_not_called()


def test_reselecting_joint_space_does_not_discard_user_goal():
    from simlab.uvms_backend import UVMSBackendCore as B
    r,b=joint_switch_fixture();r.set_control_space(ControlSpace.JOINT_SPACE)
    assert B.set_robot_control_space(b,r,ControlSpace.JOINT_SPACE)[0]
    assert r.pose_command==[99.]*6
    r.abrupt_planner_stop.assert_not_called()
