from types import SimpleNamespace
from unittest.mock import Mock
from simlab.robot import Robot,ControlMode


def test_task_mode_clears_vehicle_target_even_with_active_old_trajectory():
    planner=Mock();trajectory=Mock(active=True)
    robot=SimpleNamespace(planner=planner,vehicle_cart_traj=trajectory,
        node=Mock(),world_frame='world',sim_reset_hold=False,
        task_based_controller=True,control_mode=ControlMode.PLANNER)
    Robot.trajectory_viz_callback(robot)
    planner.clear_target.assert_called_once()
    trajectory.current_reference.assert_not_called()
    planner.update_target_viz.assert_not_called()
    Robot.planner_viz_callback(robot)
    planner.clear_path.assert_called_once()
    planner.update_path_viz.assert_not_called()


def test_cached_workspace_clouds_clear_and_restore_on_mode_switch():
    from simlab.uvms_backend import UVMSBackendCore
    from sensor_msgs.msg import PointCloud2
    clouds = (PointCloud2(width=10), PointCloud2(width=20), PointCloud2(width=0))
    backend = SimpleNamespace(_target_cloud_messages=clouds,
        selected_robot=SimpleNamespace(task_based_controller=False),
        taskspace_pc_publisher_=Mock(), rov_pc_publisher_=Mock())
    for publisher in (backend.taskspace_pc_publisher_, backend.rov_pc_publisher_):
        publisher.get_subscription_count.return_value = 1
    UVMSBackendCore.vehicle_target_cloud_timer_callback(backend)
    assert backend.taskspace_pc_publisher_.publish.call_args.args[0] is clouds[0]
    assert backend.rov_pc_publisher_.publish.call_args.args[0] is clouds[1]
    backend.selected_robot.task_based_controller = True
    UVMSBackendCore.vehicle_target_cloud_timer_callback(backend)
    assert backend.taskspace_pc_publisher_.publish.call_args.args[0] is clouds[2]
    assert backend.rov_pc_publisher_.publish.call_args.args[0] is clouds[2]
    backend.selected_robot.task_based_controller = False
    UVMSBackendCore.vehicle_target_cloud_timer_callback(backend)
    assert backend.taskspace_pc_publisher_.publish.call_args.args[0] is clouds[0]


def test_scene_transition_clears_clouds_until_reset_target_is_committed():
    from simlab.uvms_backend import UVMSBackendCore
    from sensor_msgs.msg import PointCloud2
    clouds = (PointCloud2(width=10), PointCloud2(width=20), PointCloud2(width=0))
    backend = SimpleNamespace(_target_cloud_messages=clouds, scene_transition=True,
        selected_robot=SimpleNamespace(task_based_controller=False),
        taskspace_pc_publisher_=Mock(), rov_pc_publisher_=Mock())
    for publisher in (backend.taskspace_pc_publisher_, backend.rov_pc_publisher_):
        publisher.get_subscription_count.return_value = 1
    UVMSBackendCore.vehicle_target_cloud_timer_callback(backend)
    assert backend.taskspace_pc_publisher_.publish.call_args.args[0] is clouds[2]
    assert backend.rov_pc_publisher_.publish.call_args.args[0] is clouds[2]
    backend.scene_transition = False
    UVMSBackendCore.vehicle_target_cloud_timer_callback(backend)
    assert backend.taskspace_pc_publisher_.publish.call_args.args[0] is clouds[0]
    assert backend.rov_pc_publisher_.publish.call_args.args[0] is clouds[1]
