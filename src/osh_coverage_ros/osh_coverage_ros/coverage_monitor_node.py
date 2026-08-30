from __future__ import annotations

import json

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from std_msgs.msg import String

from osh_coverage_core import CoverageMonitor, DynamicRepairManager
from .ros_utils import bool_mask_message, grid_from_message


class CoverageMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__("coverage_monitor")
        self.declare_parameter("working_width_m", 0.60)
        self.declare_parameter("minimum_residual_area_m2", 0.05)
        self.declare_parameter("max_retries", 2)
        self.monitor = None
        self.repair = DynamicRepairManager(int(self.get_parameter("max_retries").value))
        self.create_subscription(OccupancyGrid, "/coverage/reachable_mask", self._on_reachable, 1)
        self.create_subscription(PoseStamped, "/coverage/actual_pose", self._on_pose, 50)
        self.covered_publisher = self.create_publisher(OccupancyGrid, "/coverage/covered_mask", 1)
        self.residual_publisher = self.create_publisher(OccupancyGrid, "/coverage/residual_mask", 1)
        self.status_publisher = self.create_publisher(String, "/coverage/monitor_status", 10)

    def _on_reachable(self, message: OccupancyGrid) -> None:
        grid = grid_from_message(message)
        reachable = np.asarray(message.data, dtype=np.int16).reshape(grid.shape) < 50
        self.monitor = CoverageMonitor(grid, reachable, float(self.get_parameter("working_width_m").value))
        self.get_logger().info("coverage monitor initialized")

    def _on_pose(self, message: PoseStamped) -> None:
        if self.monitor is None:
            return
        self.monitor.update_pose(message.pose.position.x, message.pose.position.y)
        stamp = self.get_clock().now().to_msg()
        covered = self.monitor.covered_mask
        regions = self.repair.annotate(
            self.monitor.residual_regions(float(self.get_parameter("minimum_residual_area_m2").value))
        )
        residual = self.monitor.reachable_mask & ~covered
        self.covered_publisher.publish(bool_mask_message(covered, self.monitor.grid, stamp, occupied_when_false=False))
        self.residual_publisher.publish(bool_mask_message(residual, self.monitor.grid, stamp, occupied_when_false=False))
        status = String()
        status.data = json.dumps(
            {
                "coverage_ratio": self.monitor.coverage_ratio,
                "overlap_ratio": self.monitor.overlap_ratio,
                "residual_regions": [
                    {
                        "id": region.region_id,
                        "area_m2": region.area_m2,
                        "centroid_xy": region.centroid_xy,
                        "retry_count": region.retry_count,
                        "state": region.state,
                    }
                    for region in regions
                ],
            },
            ensure_ascii=False,
        )
        self.status_publisher.publish(status)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CoverageMonitorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

