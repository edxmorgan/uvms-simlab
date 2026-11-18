"""Read-only equivalent tool-wrench inference; never uses injection as an input."""
import time

import numpy as np
from ament_index_python.packages import get_package_share_directory
from control_msgs.msg import DynamicJointState
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data
from simlab.msg import TaskWrenchEstimate
from simlab.disturbance_waveforms import DOFS
from simlab.motion_planning.kinematics.forward.tool_wrench import ToolWrenchMapper
from simlab.uvms_parameters_sim import ReachParams


def infer_wrench(mapper, joints, estimates, variances, covariance=None):
    """Weighted SVD with rank detection; missing rows are never replaced by zero."""
    jacobian, transform = mapper.geometry(joints)
    rotation = np.zeros((6, 6))
    rotation[:3, :3] = rotation[3:, 3:] = transform[:3, :3]
    mapping = jacobian.T @ rotation
    values, variance = np.asarray(estimates), np.asarray(variances)
    mask = np.isfinite(values) & np.isfinite(variance) & (variance > 0.)
    # Whiten with the complete covariance: force/torque correlations matter
    # when translating wrench origins. Never discard them after publishing.
    matrix = np.diag(variance) if covariance is None else np.asarray(covariance, dtype=float)
    if matrix.shape != (values.size, values.size):
        raise ValueError('generalized covariance has wrong shape')
    selected = matrix[np.ix_(mask, mask)]
    if not np.isfinite(selected).all() or not np.allclose(selected, selected.T, rtol=1e-8, atol=1e-12):
        raise ValueError('generalized covariance is nonfinite or asymmetric')
    factor = np.linalg.cholesky(selected)
    a = np.linalg.solve(factor, mapping[mask])
    b = np.linalg.solve(factor, values[mask])
    scale = np.linalg.norm(a, axis=0)
    scale[scale == 0.] = 1.
    u, s, vt = np.linalg.svd(a / scale, full_matrices=False)
    rank = int(np.count_nonzero(s > (s[0]*1e-8 if s.size else 0.)))
    result = dict(rank=rank, observations=int(mask.sum()), valid=rank == 6,
                  wrench=np.full(6, np.nan), covariance=np.full((6, 6), np.nan),
                  residual=np.full(10, np.nan))
    if rank != 6:
        return result
    inverse = ((vt.T / s) @ u.T) / scale[:, None]
    result['wrench'] = inverse @ b
    result['covariance'] = inverse @ inverse.T
    result['residual'][mask] = values[mask] - mapping[mask] @ result['wrench']
    return result


class TaskWrenchObserver:
    """Shares the standard disturbance node lifecycle, without control authority."""
    def __init__(self, node, prefix):
        self.node, self.prefix = node, prefix
        self.latest = None
        self.mapper = None
        self.publisher = node.create_publisher(
            TaskWrenchEstimate, '/'+prefix+'/disturbance/task_estimate', 10)
        self.subscription = node.create_subscription(DynamicJointState,
            '/dynamic_joint_states', self.receive, qos_profile_sensor_data)
        self.timer = node.create_timer(.05, self.publish)

    def receive(self, message):
        self.latest = (message, time.monotonic())

    def publish(self):
        output = TaskWrenchEstimate()
        output.header.frame_id = self.prefix+'joint_4'
        output.header.stamp = self.node.get_clock().now().to_msg()
        output.covariance = [float('nan')]*36
        output.generalized_residual = [float('nan')]*10
        output.force_residual_norm = output.torque_residual_norm = float('nan')
        wrench = np.full(6, np.nan)
        try:
            if self.latest is None:
                raise ValueError('joint-state telemetry unavailable')
            message, receipt = self.latest
            output.header.stamp = message.header.stamp
            age = (self.node.get_clock().now()-Time.from_msg(message.header.stamp)).nanoseconds/1e9
            if time.monotonic()-receipt >= .5 or not 0. <= age < .5:
                raise ValueError('joint-state telemetry stale')
            fields = {n: dict(zip(v.interface_names, v.values))
                      for n, v in zip(message.joint_names, message.interface_values)}
            joints = [fields.get(self.prefix+'_axis_'+axis, {}).get('filtered_position', np.nan)
                      for axis in 'edcb']
            estimates, variances = [], []
            for index, _, key in DOFS.values():
                data = fields.get(self.prefix+('IOs' if index < 6 else '_arm_IOs'), {})
                observable = data.get('estimator_valid') == 1.
                if index >= 6:
                    axis = key[-1]
                    observable &= (data.get('estimator_disturbance_observable.'+axis) == 1.
                                   and data.get('estimator_joint_locked.'+axis) == 0.
                                   and data.get('estimator_joint_at_limit.'+axis) == 0.)
                estimates.append(data.get('estimated_disturbance.'+key, np.nan)
                                 if observable else np.nan)
                variances.append(data.get('estimated_disturbance.variance.'+key, np.nan))
            if self.mapper is None:
                self.mapper = ToolWrenchMapper(get_package_share_directory('simlab'),
                                              ReachParams.base_T0_new, ReachParams.tipOffset)
            covariance = np.diag(variances)
            full_blocks = {}
            for label, suffix, start, count in [('vehicle', 'IOs', 0, 6),
                                                ('arm', '_arm_IOs', 6, 4)]:
                data = fields.get(self.prefix+suffix, {})
                keys = [f'estimated_disturbance.covariance.{i}.{j}'
                        for i in range(count) for j in range(count)]
                present = [key in data for key in keys]
                full_blocks[label] = all(present)
                # An unavailable arm must not invalidate a vehicle-only estimate.
                used = any(np.isfinite(estimates[start:start+count]))
                if any(present) and not all(present) and used:
                    raise ValueError('incomplete '+label+' wrench covariance')
                if all(present):
                    covariance[start:start+count, start:start+count] = np.array(
                        [data[key] for key in keys]).reshape(count, count)
            result = infer_wrench(self.mapper, joints, estimates, variances, covariance)
            output.rank, output.observations = result['rank'], result['observations']
            output.valid = result['valid']
            covariance_status = ('full vehicle wrench covariance' if full_blocks['vehicle']
                                 else 'legacy vehicle diagonal covariance approximation')
            if any(np.isfinite(estimates[6:])):
                covariance_status += ('; full arm wrench covariance' if full_blocks['arm']
                                      else '; legacy arm diagonal covariance approximation')
                covariance_status += '; cross-system independence approximation'
            output.status = ('equivalent wrench; '+covariance_status if output.valid
                             else 'insufficient observable wrench directions')
            if output.valid and output.observations == 6:
                output.status += '; no redundant consistency check'
            wrench = result['wrench']
            output.covariance = result['covariance'].ravel().tolist()
            output.generalized_residual = result['residual'].tolist()
            for name, indices in [('force', slice(0, 3)), ('torque', slice(3, 10))]:
                residual = result['residual'][indices]
                finite = residual[np.isfinite(residual)]
                setattr(output, name+'_residual_norm',
                        float(np.linalg.norm(finite)) if finite.size else float('nan'))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            output.status = str(error)
        for i, vector in enumerate((output.wrench.force, output.wrench.torque)):
            for j, axis in enumerate('xyz'):
                setattr(vector, axis, float(wrench[3*i+j]))
        self.publisher.publish(output)
