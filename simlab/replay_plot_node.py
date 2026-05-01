from pathlib import Path

import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node

from simlab.replay_session_plotter import build_plot, resolve_csv_path


class ReplayPlotNode(Node):
    def __init__(self):
        super().__init__("replay_plot_node")
        self.declare_parameter("csv", "")
        self.declare_parameter("time", "replay_time_sec")
        self.declare_parameter("arm", "all")
        self.declare_parameter("vehicle_pose", "all")
        self.declare_parameter("vehicle_wrench", "all")
        self.declare_parameter("only", "all")
        self._timer = self.create_timer(0.1, self._plot_once)

    def _plot_once(self) -> None:
        self._timer.cancel()
        try:
            csv_param = str(self.get_parameter("csv").value).strip()
            csv_path = resolve_csv_path(Path(csv_param) if csv_param else None)
            self.get_logger().info(f"Plotting replay session: {csv_path}")
            build_plot(
                csv_path=csv_path,
                time_col=str(self.get_parameter("time").value),
                arm=str(self.get_parameter("arm").value),
                vehicle_pose=str(self.get_parameter("vehicle_pose").value),
                vehicle_wrench=str(self.get_parameter("vehicle_wrench").value),
                only=str(self.get_parameter("only").value),
            )
            plt.show()
        except Exception as exc:
            self.get_logger().error(f"Failed to plot replay session: {exc}")
        finally:
            rclpy.shutdown()


def main() -> None:
    rclpy.init()
    node = ReplayPlotNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
