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
