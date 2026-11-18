"""Equivalent-wrench inference, observability and invalid-data regression tests."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from simlab.motion_planning.kinematics.forward.tool_wrench import ToolWrenchMapper
from simlab.task_wrench_estimation import infer_wrench, TaskWrenchObserver
from simlab.uvms_parameters_sim import ReachParams


def mapper():
    return ToolWrenchMapper(Path(__file__).parents[1]/'resource',
                            ReachParams.base_T0_new, ReachParams.tipOffset)


def test_recovery_frames_and_covariance():
    model = mapper()
    rng = np.random.default_rng(730)
    for _ in range(30):
        q, wrench = rng.uniform(.2, 2.5, 4), rng.normal(size=6)
        tau = model.map(q, wrench)
        variance = rng.uniform(.01, .1, 10)
        result = infer_wrench(model, q, tau, variance)
        assert result['valid'] and result['rank'] == 6
        np.testing.assert_allclose(result['wrench'], wrench, atol=1e-12)
        np.testing.assert_allclose(result['residual'], 0., atol=1e-12)
        assert np.linalg.eigvalsh(result['covariance']).min() > 0.
        # Independent normal-equation check (production uses more stable SVD).
        a = np.column_stack([model.map(q, e) for e in np.eye(6)])
        np.testing.assert_allclose(result['covariance'],
            np.linalg.inv(a.T @ np.diag(1./variance) @ a), atol=1e-12)


def test_excluded_arm_rows_and_rank_failure():
    model, q = mapper(), [.8, 1., .9, 1.2]
    wrench = np.arange(6.)
    tau = model.map(q, wrench)
    tau[6:] = np.nan
    result = infer_wrench(model, q, tau, np.ones(10))
    assert result['valid'] and result['observations'] == 6
    np.testing.assert_allclose(result['wrench'], wrench, atol=1e-12)
    assert np.isnan(result['residual'][6:]).all()
    tau[0] = np.nan
    result = infer_wrench(model, q, tau, np.ones(10))
    assert not result['valid'] and result['rank'] == 5
    assert np.isnan(result['wrench']).all()
    result = infer_wrench(model, q, tau, np.zeros(10))
    assert not result['valid'] and result['rank'] == 0


def test_incompatible_generalized_load_has_residual():
    model, q = mapper(), [.8, 1., .9, 1.2]
    tau = model.map(q, np.ones(6))
    tau[6] += 2.
    result = infer_wrench(model, q, tau, np.ones(10))
    assert np.linalg.norm(result['residual']) > .1


def test_observer_locked_and_stale(monkeypatch):
    import rclpy
    from rclpy.node import Node
    from control_msgs.msg import DynamicJointState, InterfaceValue
    from simlab.disturbance_waveforms import DOFS
    monkeypatch.setenv('ROS_DOMAIN_ID', '178')
    rclpy.init()
    node = Node('test_wrench_observer')
    observer = TaskWrenchObserver(node, 'robot_1_')
    captured = []
    observer.publisher = SimpleNamespace(publish=captured.append)
    observer.mapper = mapper()
    try:
        q = [.8, 1., .9, 1.2]
        expected = np.arange(6.)
        tau = observer.mapper.map(q, expected)
        rows = {'robot_1_IOs': {'estimator_valid': 1.},
                'robot_1__arm_IOs': {'estimator_valid': 1.}}
        for index, _, key in DOFS.values():
            data = rows['robot_1_IOs' if index < 6 else 'robot_1__arm_IOs']
            data['estimated_disturbance.'+key] = float(tau[index]) if index < 6 else 1000.
            data['estimated_disturbance.variance.'+key] = .1
            if index >= 6:
                data['estimator_joint_locked.'+key[-1]] = 1.
                data['estimator_joint_at_limit.'+key[-1]] = 0.
                data['estimator_disturbance_observable.'+key[-1]] = 0.
        rows.update({'robot_1__axis_'+axis: {'filtered_position': value, 'position': value + 1.0} for axis, value in zip('edcb', q)})
        message = DynamicJointState()
        message.header.stamp = node.get_clock().now().to_msg()
        message.joint_names = list(rows)
        message.interface_values = [InterfaceValue(interface_names=list(v), values=list(v.values()))
                                    for v in rows.values()]
        observer.receive(message)
        observer.publish()
        output = captured[-1]
        assert output.valid and output.observations == 6
        np.testing.assert_allclose([getattr(v, a) for v in (output.wrench.force, output.wrench.torque)
                                    for a in 'xyz'], expected, atol=1e-12)
        assert output.header.stamp == message.header.stamp
        body_cov = np.eye(6)*.1
        body_cov[0,4]=body_cov[4,0]=.04
        rows['robot_1_IOs'].update({f'estimated_disturbance.covariance.{i}.{j}':float(body_cov[i,j])
                                  for i in range(6) for j in range(6)})
        message.interface_values = [InterfaceValue(interface_names=list(v), values=list(v.values()))
                                    for v in rows.values()]
        observer.receive(message);observer.publish()
        assert 'full vehicle wrench covariance' in captured[-1].status
        a=np.column_stack([observer.mapper.map(q,e) for e in np.eye(6)])[:6]
        inverse=np.linalg.inv(a)
        np.testing.assert_allclose(np.array(captured[-1].covariance).reshape(6,6),
                                   inverse@body_cov@inverse.T,atol=1e-12)
        observer.latest = (message, 0.)
        observer.publish()
        assert not captured[-1].valid and 'stale' in captured[-1].status
        assert np.isnan(captured[-1].covariance).all()
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_correlated_vehicle_covariance_and_force_moment_cancellation():
    model, q = mapper(), [3.15, 1.29, 2.58, 2.1]
    a = np.column_stack([model.map(q, e) for e in np.eye(6)])[:6]
    # Independent uncertainties at the TOOL become correlated at the vehicle.
    tool_cov = np.diag([.2, .1, .1, .003, .002, .003])
    body_cov = a @ tool_cov @ a.T
    covariance = np.eye(10)
    covariance[:6,:6] = body_cov
    values = np.r_[a @ np.arange(6.), [np.nan]*4]
    result = infer_wrench(model, q, values, np.diag(covariance), covariance)
    np.testing.assert_allclose(result['wrench'], np.arange(6.), atol=1e-12)
    np.testing.assert_allclose(result['covariance'], tool_cov, atol=1e-12)
    diagonal = infer_wrench(model, q, values, np.diag(covariance))
    assert abs(diagonal['covariance'][4,4]-tool_cov[4,4]) > .01
    # Covariance changes uncertainty; it cannot alter the square inverse mean.
    np.testing.assert_allclose(diagonal['wrench'], result['wrench'], atol=1e-12)


def test_full_covariance_weighted_redundant_observations():
    model, q = mapper(), [.8, 1., .9, 1.2]
    rng = np.random.default_rng(731)
    a = np.column_stack([model.map(q, e) for e in np.eye(6)])
    l = rng.normal(size=(10,10));covariance=l@l.T+np.eye(10)
    values = rng.normal(size=10)
    result = infer_wrench(model, q, values, np.diag(covariance), covariance)
    expected_cov = np.linalg.inv(a.T@np.linalg.solve(covariance,a))
    np.testing.assert_allclose(result['covariance'],expected_cov,atol=1e-10)
    np.testing.assert_allclose(result['wrench'],expected_cov@a.T@np.linalg.solve(covariance,values),atol=1e-10)


def test_observer_preserves_arm_covariance_and_masks_unavailable_arm(monkeypatch):
    import rclpy
    from rclpy.node import Node
    from control_msgs.msg import DynamicJointState, InterfaceValue
    from simlab.disturbance_waveforms import DOFS
    monkeypatch.setenv('ROS_DOMAIN_ID', '178')
    rclpy.init()
    node = Node('test_full_arm_covariance_observer')
    observer = TaskWrenchObserver(node, 'robot_1_')
    captured = []
    observer.publisher = SimpleNamespace(publish=captured.append)
    observer.mapper = mapper()
    try:
        q = [3.0, .9, 1.2, 1.9]
        mapping = np.column_stack([observer.mapper.map(q, e) for e in np.eye(6)])
        values = mapping @ np.arange(6.)*.1 + np.arange(10)*.02
        covariance = np.diag([.1]*6+[.04, .06, .08, .1])
        covariance[7,8] = covariance[8,7] = .025
        rows = {'robot_1_IOs': {'estimator_valid': 1.},
                'robot_1__arm_IOs': {'estimator_valid': 1.}}
        for index, _, key in DOFS.values():
            data = rows['robot_1_IOs' if index < 6 else 'robot_1__arm_IOs']
            data['estimated_disturbance.'+key] = float(values[index])
            data['estimated_disturbance.variance.'+key] = float(covariance[index,index])
            if index >= 6:
                data['estimator_joint_locked.'+key[-1]] = 0.
                data['estimator_joint_at_limit.'+key[-1]] = 0.
                data['estimator_disturbance_observable.'+key[-1]] = 1.
        for prefix, start, count in [('robot_1_IOs',0,6),('robot_1__arm_IOs',6,4)]:
            rows[prefix].update({f'estimated_disturbance.covariance.{i}.{j}':
                                float(covariance[start+i,start+j])
                                for i in range(count) for j in range(count)})
        rows.update({'robot_1__axis_'+axis: {'filtered_position': value, 'position': value + 1.0} for axis,value in zip('edcb',q)})
        def publish():
            message = DynamicJointState()
            message.header.stamp = node.get_clock().now().to_msg()
            message.joint_names = list(rows)
            message.interface_values = [InterfaceValue(interface_names=list(v),values=list(v.values()))
                                        for v in rows.values()]
            observer.receive(message); observer.publish()
            return captured[-1]
        output = publish()
        assert output.valid and output.observations == 10
        assert 'full arm wrench covariance' in output.status
        assert 'cross-system independence approximation' in output.status
        normal = mapping.T @ np.linalg.solve(covariance,mapping)
        expected_cov = np.linalg.solve(normal,np.eye(6))
        expected_mean = expected_cov @ mapping.T @ np.linalg.solve(covariance,values)
        actual = [getattr(v,a) for v in (output.wrench.force,output.wrench.torque) for a in 'xyz']
        np.testing.assert_allclose(actual,expected_mean,atol=1e-12)
        np.testing.assert_allclose(np.array(output.covariance).reshape(6,6),expected_cov,atol=1e-12)
        diagonal = infer_wrench(observer.mapper,q,values,np.diag(covariance))
        assert np.linalg.norm(expected_mean-diagonal['wrench']) > .001
        del rows['robot_1__arm_IOs']['estimated_disturbance.covariance.1.2']
        output = publish()
        assert not output.valid and 'incomplete arm wrench covariance' in output.status
        for axis in 'edcb': rows['robot_1__arm_IOs']['estimator_joint_locked.'+axis] = 1.
        output = publish()
        assert output.valid and output.observations == 6
        assert 'full arm wrench covariance' not in output.status
    finally:
        node.destroy_node()
        rclpy.shutdown()
