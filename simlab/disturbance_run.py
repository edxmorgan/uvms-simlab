"""CLI action client for the persistent disturbance injection node."""
import json
import subprocess
import os
import select
import sys
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args
from simlab.action import RunDisturbance
from rcl_interfaces.srv import SetParametersAtomically
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from simlab.disturbance_waveforms import parse_args


def main(argv=None):
    ros_args = sys.argv if argv is None else [sys.argv[0], *argv]
    cli_args = remove_ros_args(args=ros_args)[1:]
    a = parse_args(cli_args)
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = Node('disturbance_run')
    client = ActionClient(node, RunDisturbance, '/'+a.robot_prefix+'/disturbance/run')
    handle, observer, goal_future = None, None, None
    succeeded = False
    try:
        if not client.wait_for_server(timeout_sec=5.):
            raise RuntimeError('Disturbance action unavailable. Launch the standard simulation or disturbance_injection_node.')
        goal = RunDisturbance.Goal(space=a.space, wrench_frame=a.wrench_frame, dofs=a.dof, amplitudes=[c.amplitude for c in a.channels],
            waveforms=[c.waveform for c in a.channels], phases_deg=[c.phase_deg for c in a.channels],
            delays_s=[c.delay_s for c in a.channels], frequency_hz=a.frequency,
            end_frequency_hz=a.end_frequency or 0., cycles=a.cycles, seed=a.seed, correlation_time_s=a.correlation_time)
        if not a.no_plot:
            # Reconfigure the read-only viewer for this robot, or start it once.
            viewer = node.create_client(SetParametersAtomically,
                '/'+a.robot_prefix+'/disturbance/viewer/configure')
            try:
                if viewer.wait_for_service(timeout_sec=1.):
                    request = SetParametersAtomically.Request(parameters=[Parameter(
                        name='argv', value=ParameterValue(type=ParameterType.PARAMETER_STRING,
                                                         string_value=json.dumps(cli_args)))])
                    response = viewer.call_async(request)
                    rclpy.spin_until_future_complete(node, response, timeout_sec=5.)
                    if not response.done() or not response.result().result.successful:
                        reason = response.result().result.reason if response.done() else 'timeout'
                        raise RuntimeError('Viewer could not be reused: '+reason+'; no run started')
                else:
                    read_fd, write_fd = os.pipe()
                    try:
                        observer = subprocess.Popen([sys.executable, '-m', 'simlab.disturbance_plot',
                            *cli_args, '--ready-fd', str(write_fd)], pass_fds=(write_fd,),
                            start_new_session=True)
                        os.close(write_fd); write_fd = None
                        readable, _, _ = select.select([read_fd], [], [], 15.)
                        if not readable or os.read(read_fd, 1) != b'1':
                            raise RuntimeError('Viewer failed to become ready; no run started')
                    finally:
                        os.close(read_fd)
                        if write_fd is not None: os.close(write_fd)
            finally:
                node.destroy_client(viewer)
        future = goal_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.)
        if not future.done():
            raise RuntimeError('Action acceptance timed out; inspect run status before retrying')
        handle = future.result()
        if not handle.accepted:
            raise RuntimeError('Run rejected: check receiver availability, ownership and configuration')
        if a.space == 'task' and not a.no_plot:
            print('Plots show mapped vehicle/arm loads and their EKF estimates, not a tool-wrench estimate.', flush=True)
        print('Disturbance run accepted. Ctrl+C cancels; closing the plot does not.', flush=True)
        result_future = handle.get_result_async()
        while rclpy.ok() and not result_future.done():
            rclpy.spin_once(node, timeout_sec=.1)
        if result_future.done():
            result = result_future.result().result
            succeeded = result.success
            print(f'{result.message}; session={result.session_id}; dynamics elapsed={result.elapsed_s:.3f} s', flush=True)
    except KeyboardInterrupt:
        # Resolve an in-flight acceptance before canceling our specific goal.
        if handle is None and goal_future is not None:
            rclpy.spin_until_future_complete(node, goal_future, timeout_sec=2.)
            if goal_future.done(): handle = goal_future.result()
        if handle is not None and handle.accepted:
            cancel = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(node, cancel, timeout_sec=2.)
            print('Cancellation requested', flush=True)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
    finally:
        if observer is not None and (handle is None or not handle.accepted) and observer.poll() is None:
            observer.terminate()
            try:
                observer.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                observer.kill()
                observer.wait()
        client.destroy()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    # The viewer is independent; recording belongs to the standard MCAP node.
    if not succeeded:
        raise SystemExit(1)
    return 0


if __name__ == '__main__':
    main()
