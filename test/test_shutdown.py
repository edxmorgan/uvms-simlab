"""Exercise real ROS signal handling in isolated owned subprocesses."""
import os
import selectors
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from simlab import shutdown


@pytest.mark.parametrize('stage', ['executor', 'node'])
@pytest.mark.parametrize('error_type', [RuntimeError, KeyboardInterrupt])
def test_cleanup_continues_after_failure(monkeypatch, stage, error_type):
    calls = []
    error = error_type('cleanup interrupted')
    def cleanup(name):
        calls.append(name)
        if name == stage:
            raise error
    executor = SimpleNamespace(shutdown=lambda: cleanup('executor'))
    node = SimpleNamespace(destroy_node=lambda: cleanup('node'))
    monkeypatch.setattr(shutdown.rclpy, 'try_shutdown', lambda: cleanup('context'))
    with pytest.raises(error_type) as caught:
        shutdown.shutdown_node(node, executor)
    assert caught.value is error
    assert calls == ['executor', 'node', 'context']


@pytest.mark.parametrize('with_node', [False, True])
@pytest.mark.parametrize('with_executor', [False, True])
def test_cleanup_optional_resources(monkeypatch, with_node, with_executor):
    calls = []
    node = SimpleNamespace(destroy_node=lambda: calls.append('node')) if with_node else None
    executor = SimpleNamespace(shutdown=lambda: calls.append('executor')) if with_executor else None
    monkeypatch.setattr(shutdown.rclpy, 'try_shutdown', lambda: calls.append('context'))
    shutdown.shutdown_node(node, executor)
    assert calls == (['executor'] if with_executor else []) + (['node'] if with_node else []) + ['context']


@pytest.mark.parametrize('running', [False, True])
def test_shutdown_race_does_not_mask_live_middleware_errors(monkeypatch, running):
    error = shutdown.rclpy_implementation.RCLError('wait set failed')
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(shutdown.rclpy, 'spin', fail)
    node = SimpleNamespace(context=SimpleNamespace(ok=lambda: running))
    if running:
        with pytest.raises(type(error)):
            shutdown.spin_until_shutdown(node)
    else:
        shutdown.spin_until_shutdown(node)


@pytest.mark.parametrize('stop_signal', [signal.SIGINT, signal.SIGTERM])
def test_idle_planner_exits_without_signal_handler_deadlock(stop_signal):
    code = '''
from types import SimpleNamespace
import simlab.planner_action_server as server
server.FCLWorld = lambda **kw: SimpleNamespace(vehicle_radius=.4, env_xyz_bounds=(-5,5,-5,5,-5,0))
server.DEFAULT_PLANNER_CLASSES = ()
original = server.PlannerActionServer.__init__
def initialize(self):
    original(self)
    print("READY", flush=True)
server.PlannerActionServer.__init__ = initialize
server.main(args=['--ros-args', '-p', 'robot_description:=test'])
'''
    process = subprocess.Popen([sys.executable, '-u', '-c', code],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={**os.environ, 'ROS_DOMAIN_ID': '97'})
    output = b''
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + 15
            while b'READY\n' not in output and time.monotonic() < deadline:
                if selector.select(timeout=0.2):
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    output += chunk
        assert b'READY\n' in output, output.decode()
        process.send_signal(stop_signal)
        tail, _ = process.communicate(timeout=4)
        output += tail
        assert process.returncode == 0, output.decode()
        assert b'Traceback' not in output, output.decode()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()
