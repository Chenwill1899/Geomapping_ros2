"""ROS 2 wrapper for the internal MPPI simulation runner."""

from __future__ import annotations

from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node

from mppi_controller.config import load_config
from mppi_controller.simulation.runner import MppiSimulationRunner, goal_reached


class FdmMppiSimNode(Node):
    def __init__(self) -> None:
        super().__init__("b2_fdm_mppi")
        self.declare_parameter("config_file", "")
        config_path = self._resolve_config_path(
            self.get_parameter("config_file").get_parameter_value().string_value
        )
        self.config = load_config(config_path)
        self.runner = MppiSimulationRunner(self.config, logger=self.get_logger())
        self.steps = 0
        self.max_steps = int(self.config["simulation"]["max_steps"])
        self.finished = False
        period = 1.0 / float(self.config["simulation"]["sampling_rate"])
        self.timer = self.create_timer(period, self._on_timer)
        self.get_logger().info(f"FDM MPPI internal simulation started with {config_path}")

    def _on_timer(self) -> None:
        if self.steps >= self.max_steps or goal_reached(
            self.runner.state,
            self.runner.desired_pose,
            self.runner.minimum_distance,
        ):
            self._finish()
            return

        self.runner.step()
        self.steps += 1
        if self.runner.failed:
            self._finish()

    def _finish(self) -> None:
        if self.finished:
            return
        self.finished = True
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        self.runner._save_results()
        if self.config["results"].get("enable_plots", True):
            self.runner._plot_results()
        self.get_logger().info(
            f"FDM MPPI internal simulation finished after {self.steps} steps; "
            f"results: {self.runner.results_path}"
        )

    def _resolve_config_path(self, value: str) -> Path:
        if value:
            return Path(value)
        try:
            return Path(get_package_share_directory("b2_fdm_mppi")) / "config" / "fdm_mppi.yaml"
        except Exception:
            return Path("config/fdm_mppi.yaml")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FdmMppiSimNode()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
