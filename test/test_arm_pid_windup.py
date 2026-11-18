"""Exercise the exported controller, including saturation and reversal."""
from pathlib import Path
import casadi as ca
import numpy as np
from simlab.controllers.pid import LowLevelPidController
from simlab.uvms_parameters import select_robot_params


def controller():
    pid = object.__new__(LowLevelPidController)
    pid.arm_dof = 4
    p, _ = select_robot_params('robot_real_')
    root = Path(__file__).resolve().parents[1]/'resource/model_functions/arm/arm_pid.casadi'
    pid.arm_pid_controller = ca.Function.load(str(root))
    pid.arm_model_params = p.sim_p
    pid.arm_kp = np.array([4., 4., 2., 2., 1000.])
    pid.arm_ki = np.array([1., 1., .5, .25, 0.])
    pid.arm_kd = np.zeros(5)
    pid.arm_u_min = np.array(list(p.u_min)+list(p.grasper_u_min))
    pid.arm_u_max = np.array(list(p.u_max)+list(p.grasper_u_max))
    pid.arm_pid_i_buffer = np.zeros(5)
    return pid


def test_saturated_arm_reverses_without_integral_windup():
    pid = controller()
    q = np.array([2.834, .629, .416, .211, .0004])
    offset = np.array([2., 2., 2., 2., 0.])
    for _ in range(2000):
        u = pid.arm_controller(q, np.zeros(5), q+offset, np.zeros(5), np.zeros(5), .005)
    np.testing.assert_allclose(pid.arm_pid_i_buffer, 0.)
    assert np.all(u[:4] == pid.arm_u_max[:4])
    u = pid.arm_controller(q, np.zeros(5), q-offset, np.zeros(5), np.zeros(5), .005)
    assert np.all(u[:4] == pid.arm_u_min[:4])


def test_integral_accumulates_unsaturated_error_and_stops_at_limit():
    pid = controller()
    pid.arm_kp[:] = 0.
    q = np.array([2.834, .629, .416, .211, .0004])
    target = q.copy();target[0] += .1
    for _ in range(100):
        pid.arm_controller(q, np.zeros(5), target, np.zeros(5), np.zeros(5), .01)
    assert .09 < pid.arm_pid_i_buffer[0] < .11
    for _ in range(4000):
        pid.arm_controller(q, np.zeros(5), target, np.zeros(5), np.zeros(5), .01)
    assert pid.arm_pid_i_buffer[0] <= pid.arm_u_max[0]/pid.arm_ki[0]
    before = pid.arm_pid_i_buffer[0]
    target[0] = q[0]-.1
    pid.arm_controller(q, np.zeros(5), target, np.zeros(5), np.zeros(5), .01)
    assert pid.arm_pid_i_buffer[0] < before


def test_startup_gains_preserve_gripper_and_effort_limits():
    import rclpy
    from rclpy.parameter import Parameter
    rclpy.init()
    node = rclpy.create_node('pid_startup_test', parameter_overrides=[
        Parameter('robot_real_pid_arm_ki', value=[1., 1., .5, .25])])
    try:
        pid = LowLevelPidController(node, robot_prefix='robot_real_')
        np.testing.assert_allclose(pid.arm_ki, [1., 1., .5, .25, 0.])
        np.testing.assert_allclose(pid.arm_u_max, [3., 2., 2., 1., 10.])
        assert pid.arm_kp[-1] == 1000.
    finally:
        node.destroy_node();rclpy.shutdown()
