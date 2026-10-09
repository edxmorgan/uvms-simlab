"""Use rclpy's deferred SIGINT/SIGTERM handling, never shutdown inside a signal callback."""
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.impl.implementation_singleton import rclpy_implementation


def spin_until_shutdown(node, executor=None):
    try:
        rclpy.spin(node, executor=executor)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except (rclpy_implementation.RCLError, rclpy_implementation.InvalidHandle):
        # A signal can invalidate the context between spin's ok() check and
        # wait-set construction. Do not hide middleware errors while running.
        if node.context.ok():
            raise


def shutdown_node(node=None, executor=None):
    # Complete every cleanup stage even if an earlier stage raises (including
    # KeyboardInterrupt), without hiding the cleanup failure from the caller.
    try:
        if executor is not None:
            executor.shutdown()
    finally:
        try:
            if node is not None:
                node.destroy_node()
        finally:
            rclpy.try_shutdown()
