"""Project an accumulated, map-frame 3-D cloud into a 2-D OccupancyGrid."""

from __future__ import annotations

import math

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class MapProjectorNode(Node):
    def __init__(self) -> None:
        super().__init__("map_projector")
        self.declare_parameter("input_topic", "/rtabmap/cloud_map")
        self.declare_parameter("output_topic", "/map")
        self.declare_parameter("resolution", 0.05)
        self.declare_parameter("minimum_z", 0.08)
        self.declare_parameter("maximum_z", 1.50)
        self.declare_parameter("minimum_hits", 2)
        self.declare_parameter("padding_m", 0.50)
        self.declare_parameter("minimum_x", 0.0)
        self.declare_parameter("maximum_x", 0.0)
        self.declare_parameter("minimum_y", 0.0)
        self.declare_parameter("maximum_y", 0.0)
        self.declare_parameter("accumulate", True)
        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self.publisher = self.create_publisher(OccupancyGrid, output_topic, 1)
        self.create_subscription(PointCloud2, input_topic, self._on_cloud, 1)
        self._hits: dict[tuple[int, int], int] = {}
        self._fixed_bounds = self._read_bounds()
        self._derived_bounds = None

    def _read_bounds(self):
        minimum_x = float(self.get_parameter("minimum_x").value)
        maximum_x = float(self.get_parameter("maximum_x").value)
        minimum_y = float(self.get_parameter("minimum_y").value)
        maximum_y = float(self.get_parameter("maximum_y").value)
        if maximum_x > minimum_x and maximum_y > minimum_y:
            return minimum_x, maximum_x, minimum_y, maximum_y
        return None

    def _on_cloud(self, message: PointCloud2) -> None:
        minimum_z = float(self.get_parameter("minimum_z").value)
        maximum_z = float(self.get_parameter("maximum_z").value)
        points = np.asarray(
            [
                (x, y)
                for x, y, z in point_cloud2.read_points(message, field_names=("x", "y", "z"), skip_nans=True)
                if minimum_z <= float(z) <= maximum_z and math.isfinite(x) and math.isfinite(y)
            ],
            dtype=float,
        )
        if points.size == 0:
            self.get_logger().warning("received a cloud with no points in the configured obstacle-height interval")
            return
        resolution = float(self.get_parameter("resolution").value)
        padding = float(self.get_parameter("padding_m").value)
        if self._fixed_bounds is None and self._derived_bounds is None:
            minimum_x = math.floor((float(points[:, 0].min()) - padding) / resolution) * resolution
            maximum_x = math.ceil((float(points[:, 0].max()) + padding) / resolution) * resolution
            minimum_y = math.floor((float(points[:, 1].min()) - padding) / resolution) * resolution
            maximum_y = math.ceil((float(points[:, 1].max()) + padding) / resolution) * resolution
            self._derived_bounds = (minimum_x, maximum_x, minimum_y, maximum_y)
            bounds = self._derived_bounds
        elif self._fixed_bounds is None:
            bounds = self._derived_bounds
        else:
            bounds = self._fixed_bounds
        minimum_x, maximum_x, minimum_y, maximum_y = bounds
        width = max(1, int(math.ceil((maximum_x - minimum_x) / resolution)))
        height = max(1, int(math.ceil((maximum_y - minimum_y) / resolution)))
        columns = np.floor((points[:, 0] - minimum_x) / resolution).astype(int)
        rows = np.floor((points[:, 1] - minimum_y) / resolution).astype(int)
        valid = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
        current_cells = set(zip(rows[valid].tolist(), columns[valid].tolist()))
        if not bool(self.get_parameter("accumulate").value):
            self._hits.clear()
        for cell in current_cells:
            self._hits[cell] = self._hits.get(cell, 0) + 1
        minimum_hits = int(self.get_parameter("minimum_hits").value)
        occupied = np.zeros((height, width), dtype=bool)
        for (row, col), hits in self._hits.items():
            if hits >= minimum_hits and 0 <= row < height and 0 <= col < width:
                occupied[row, col] = True
        output = OccupancyGrid()
        output.header.stamp = message.header.stamp
        output.header.frame_id = message.header.frame_id or "airy_map"
        output.info.resolution = resolution
        output.info.width = width
        output.info.height = height
        output.info.origin.position.x = minimum_x
        output.info.origin.position.y = minimum_y
        output.info.origin.orientation.w = 1.0
        output.data = np.where(occupied, 100, 0).astype(np.int8).ravel().tolist()
        self.publisher.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MapProjectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
