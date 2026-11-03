"""Exercise menu callbacks without sending commands to ROS or a robot."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
import pytest
from interactive_markers.menu_handler import MenuHandler as MH
from simlab.interactive_control import InteractiveControlsNode as UI
from simlab.uvms_backend import UVMSBackendCore as Backend
from simlab.robot import ControlSpace, ControlMode


@pytest.fixture
def ui():
    robot=Mock(prefix='robot_1_',k_robot=0,ik_base_align_w=1.,
               ik_tool_axis=np.array([0.,0.,1.]),control_space=ControlSpace.TASK_SPACE,
               controller_name='PID',planner_name='RRT',active_dynamics_profile='default')
    robot.arm=NS(grasp_command=.0137,grasper_open_position=.0137,grasper_closed_position=.0004)
    backend=Mock(robot_selected=robot,robots=[robot])
    node=NS(uvms_backend=backend,menu_handler=MH(),server=Mock(),get_logger=Mock(return_value=Mock()))
    for name in ('_refresh_robot_menu_state','_set_mcap_recording_menu_state'):
        setattr(node,name,lambda *a,_name=name,**kw:getattr(UI,_name)(node,*a,**kw))
    for name in ('_refresh_vehicle_waypoint_delete_menu','_apply_task_control_mode',
                 '_apply_joint_control_mode','_set_selected_robot_menu_state','configure_selected_robot'):
        setattr(node,name,Mock())
    for field,values in [('axis_menu_map',[np.eye(3)[i] for i in range(3)]+[None]),
                         ('grasp_menu_map',['open','close']),('controller_menu_map',['PID','InvDyn']),
                         ('planner_menu_map',['RRT','Bitstar']),('control_space_menu_map',list(ControlSpace)),
                         ('csv_profile_menu_map',['example']),('dynamics_profile_menu_map',['default'])]:
        setattr(node,field,{node.menu_handler.insert(str(i)):v for i,v in enumerate(values)})
    for name in ('start_recording_handle','stop_recording_handle','obstacle_marker_handle'):
        setattr(node,name,node.menu_handler.insert(name))
    return node


def feedback(handle=0):
    return NS(menu_entry_id=handle,marker_name='arm_base_task_marker')


@pytest.mark.parametrize('index',[0,1,2])
def test_axis_menu_sets_real_backend_setting(ui,index):
    ui.uvms_backend.set_robot_ik_tool_axis.side_effect=lambda r,a:Backend.set_robot_ik_tool_axis(None,r,a)
    h=list(ui.axis_menu_map)[index]
    UI.toggle_endeffector_axis_align(ui,feedback(h))
    np.testing.assert_array_equal(ui.uvms_backend.robot_selected.ik_tool_axis,np.eye(3)[index])
    assert ui.menu_handler.getCheckState(h)==MH.CHECKED


def test_alignment_toggle_uses_robot_state_not_stale_checkbox(ui):
    ui.uvms_backend.set_robot_ik_base_align_weight.side_effect=lambda r,w:Backend.set_robot_ik_base_align_weight(None,r,w)
    h=list(ui.axis_menu_map)[-1];ui.menu_handler.setCheckState(h,MH.UNCHECKED)
    UI.toggle_align_with_base_weight(ui,feedback(h))
    assert ui.uvms_backend.robot_selected.ik_base_align_w==0
    UI.toggle_align_with_base_weight(ui,feedback(h))
    assert ui.uvms_backend.robot_selected.ik_base_align_w==1


def test_robot_refresh_restores_ik_and_grasper_checks(ui):
    robot=ui.uvms_backend.robot_selected;robot.ik_tool_axis=np.array([1.,0.,0.]);robot.ik_base_align_w=0
    ui._refresh_robot_menu_state(0)
    assert [ui.menu_handler.getCheckState(h) for h in ui.axis_menu_map]==[MH.CHECKED,MH.UNCHECKED,MH.UNCHECKED,MH.UNCHECKED]
    assert [ui.menu_handler.getCheckState(h) for h in ui.grasp_menu_map]==[MH.CHECKED,MH.UNCHECKED]


@pytest.mark.parametrize('method,field,backend',[
 ('switch_controller_type','controller_menu_map','set_robot_controller'),
 ('switch_planner_type','planner_menu_map','set_robot_planner'),
 ('switch_control_space_type','control_space_menu_map','set_robot_control_space'),
 ('switch_csv_playback_profile','csv_profile_menu_map','select_replay_profile'),
 ('grasper_callback','grasp_menu_map','command_grasper')])
@pytest.mark.parametrize('success',[False,True])
def test_selection_dispatch_and_rejection(ui,method,field,backend,success):
    mapping=getattr(ui,field);h=next(iter(mapping));ui.menu_handler.setCheckState(h,MH.UNCHECKED)
    call=getattr(ui.uvms_backend,backend);call.return_value=(success,'result')
    getattr(UI,method)(ui,feedback(h))
    call.assert_called_once_with(ui.uvms_backend.robot_selected,mapping[h])
    assert ui.menu_handler.getCheckState(h)==(MH.CHECKED if success else MH.UNCHECKED)


@pytest.mark.parametrize('method,backend',[
 ('plan_execute','plan_execute_selected'),('reset_simulation','reset_selected_simulation'),
 ('release_simulation','release_selected_simulation'),('clear_vehicle_waypoints','clear_selected_vehicle_waypoints'),
 ('stop_vehicle_waypoints','stop_selected_vehicle_waypoints')])
def test_action_dispatch(ui,method,backend):
    getattr(UI,method)(ui,feedback());getattr(ui.uvms_backend,backend).assert_called_once_with()


@pytest.mark.parametrize('method,backend',[('reset_csv_playback','start_replay'),('stop_csv_playback','stop_replay')])
def test_replay_dispatch(ui,method,backend):
    call=getattr(ui.uvms_backend,backend);call.return_value=(True,'ok')
    getattr(UI,method)(ui,feedback());call.assert_called_once_with(ui.uvms_backend.robot_selected)


def test_waypoint_add_delete_dispatch(ui):
    ui.uvms_backend.add_vehicle_waypoint_for_robot.return_value=(True,'ok')
    UI.add_vehicle_waypoint(ui,feedback())
    ui.uvms_backend.add_vehicle_waypoint_for_robot.assert_called_once_with(ui.uvms_backend.robot_selected)
    ui.delete_vehicle_waypoint_menu_map={42:(0,3)}
    UI.delete_vehicle_waypoint(ui,feedback(42))
    ui.uvms_backend.remove_vehicle_waypoint_for_robot.assert_called_once_with(0,3)


def test_dynamics_menu_waits_for_success(ui):
    h=next(iter(ui.dynamics_profile_menu_map));ui.menu_handler.setCheckState(h,MH.UNCHECKED)
    ui.uvms_backend.set_robot_dynamics_profile.return_value=(True,'pending')
    UI.switch_dynamics_profile(ui,feedback(h))
    assert ui.menu_handler.getCheckState(h)==MH.UNCHECKED
    ui.uvms_backend.set_robot_dynamics_profile.call_args.kwargs['on_success']()
    assert ui.menu_handler.getCheckState(h)==MH.CHECKED


def test_recording_state_waits_for_service_success(ui):
    UI.start_mcap_recording(ui,feedback())
    ui.uvms_backend.start_mcap_recording.call_args.kwargs['on_success']()
    assert ui.mcap_recording_active
    assert ui.menu_handler.getCheckState(ui.start_recording_handle)==MH.CHECKED
    assert ui.menu_handler.getCheckState(ui.stop_recording_handle)==MH.UNCHECKED
    UI.stop_mcap_recording(ui,feedback())
    ui.uvms_backend.stop_mcap_recording.call_args.kwargs['on_success']()
    assert not ui.mcap_recording_active
    assert ui.menu_handler.getCheckState(ui.stop_recording_handle)==MH.CHECKED


def test_robot_selection_dispatch(ui):
    robot=ui.uvms_backend.robot_selected;robot.user_id=42
    UI.switch_robot_in_use(ui,feedback(42));ui.configure_selected_robot.assert_called_once_with(robot)


@pytest.mark.parametrize('axis',[None,[1,1,0],[0,0,0],[float('nan'),0,1]])
def test_invalid_axis_does_not_change_setting(axis):
    robot=NS(ik_tool_axis=np.array([0,0,1]))
    assert not Backend.set_robot_ik_tool_axis(None,robot,axis)[0]
    np.testing.assert_array_equal(robot.ik_tool_axis,[0,0,1])


def test_arm_marker_offers_only_supported_dofs_and_restores_task_rotation():
    import copy
    from geometry_msgs.msg import Pose
    from visualization_msgs.msg import InteractiveMarkerControl as C
    from simlab.utils.interactive_markers import make_UVMS_Dof_Marker
    marker=make_UVMS_Dof_Marker('arm','arm','base','arm_task',False,
        C.MOVE_ROTATE_3D,Pose(),.2,show_6dof=True)
    node=NS(arm_base_task_marker=marker,_arm_full_controls=copy.deepcopy(marker.controls))
    UI._configure_arm_marker_mode(node,False)
    assert marker.controls[0].interaction_mode==C.MOVE_3D
    assert all(c.interaction_mode!=C.ROTATE_AXIS for c in marker.controls)
    UI._configure_arm_marker_mode(node,True)
    assert sum(c.interaction_mode==C.ROTATE_AXIS for c in marker.controls)==3
    assert marker.controls[0].interaction_mode==C.MOVE_ROTATE_3D


