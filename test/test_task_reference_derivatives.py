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


def test_accepted_ik_endpoint_clears_derivatives():
    # Exercise the production acceptance path with frame conversion supplied.
    import numpy as np
    from unittest.mock import Mock
    from geometry_msgs.msg import Pose
    from scipy.spatial.transform import Rotation as R
    from simlab.robot import EndpointIK
    from simlab.uvms_parameters_sim import ReachParams as P
    from pathlib import Path
    r=robot();r.world_frame='world';r.prefix='robot_1_'
    r.node=Mock();r.get_state=lambda:{'q':r.arm.q_command}
    current=Pose();current.orientation.w=1.;current.position.z=-2.
    r._pose_from_state_in_frame=lambda _:current
    r.endpoint_ik=EndpointIK(Path(__file__).resolve().parents[1]/'resource')
    r.fk_eval=r.endpoint_ik.fk
    r.ik_joint_lower=np.array([.01]*4);r.ik_joint_upper=np.array([6.27,3.4,3.4,5.7])
    r.ik_base_align_w=1.;r.ik_position_tolerance=.01;r.ik_orientation_tolerance=.05
    pose=np.array(r.fk_eval(r.arm.q_command,P.base_T0_new,[0,0,-2,0,0,0],P.tipOffset)[-1]).ravel()
    target=Pose();target.position.x,target.position.y,target.position.z=pose[:3]
    target.orientation.w,target.orientation.x,target.orientation.y,target.orientation.z=pose[3:]
    r.world_nwu_to_map_ned=lambda **kwargs:(kwargs['xyz_world_nwu'],R.from_quat(np.array(kwargs['quat_world_wxyz'])[[1,2,3,0]]).as_euler('xyz'))
    r.endpoint_ik.solve=Mock(wraps=r.endpoint_ik.solve)
    Robot.solve_inverse_kinematics_wrt_world_frame(r,target)
    assert r.endpoint_ik.solve.call_args.kwargs["base_align_weight"]==1.
    assert r.body_vel_command==[0.]*6 and r.arm.dq_command==[0.]*4
