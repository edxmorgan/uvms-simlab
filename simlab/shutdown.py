import signal
import sys

import rclpy
from rclpy.executors import ExternalShutdownException


_SHUTDOWN_REQUESTED = False


class _ShutdownStderrFilter:
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def write(self, text):
        if (
            _SHUTDOWN_REQUESTED
            and "The following exception was never retrieved: cannot use Destroyable because destruction was requested" in text
        ):
            return len(text)
        return self._wrapped.write(text)

    def flush(self):
        return self._wrapped.flush()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def _mark_shutdown_requested():
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    if not isinstance(sys.stderr, _ShutdownStderrFilter):
        sys.stderr = _ShutdownStderrFilter(sys.stderr)


def install_signal_shutdown_handler():
    def _request_shutdown(signum, frame):
        _mark_shutdown_requested()
        try:
            rclpy.try_shutdown()
        except AttributeError:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)


def spin_until_shutdown(node, executor=None):
    try:
        if executor is None:
            rclpy.spin(node)
        else:
            rclpy.spin(node, executor=executor)
    except (KeyboardInterrupt, ExternalShutdownException):
        _mark_shutdown_requested()
        pass
    except Exception:
        if rclpy.ok():
            raise
        _mark_shutdown_requested()


def shutdown_node(node=None, executor=None):
    if not rclpy.ok():
        _mark_shutdown_requested()
    if executor is not None:
        try:
            executor.shutdown()
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
    if node is not None:
        try:
            node.destroy_node()
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        except Exception:
            pass
    if rclpy.ok():
        try:
            rclpy.shutdown()
        except Exception:
            pass
