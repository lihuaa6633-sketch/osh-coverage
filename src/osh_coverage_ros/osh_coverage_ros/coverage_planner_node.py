from __future__ import annotations

import json

import numpy as np
import rclpy
from geometry_msgs.msg import PolygonStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from std_msgs.msg import String

from osh_coverage_core import CoveragePlanner, PlannerConfig
from osh_coverage_core.rl import MaskedDoubleDQN, make_ddqn_scheduler
from .ros_utils import bool_mask_message, grid_from_message, polygon_mask, quaternion_from_yaw


class CoveragePlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("coverage_planner")
        self.declare_parameter("working_width_m", 0.60)
        self.declare_parameter("overlap_ratio", 0.10)
        self.declare_parameter("safety_margin_m", 0.15)
        self.declare_parameter("holonomic", True)
        self.declare_parameter("start_x", 0.30)
        self.declare_parameter("start_y", 0.30)
        self.declare_parameter("auto_plan", True)
        self.declare_parameter("rl_model_path", "")
        self.declare_parameter("rl_candidate_slots", 6)
        self._map_message = None
        self._roi_message = None
        self._start_xy = None
        self.create_subscription(OccupancyGrid, "/map", self._on_map, 1)
        self.create_subscription(PolygonStamped, "/coverage/roi", self._on_roi, 1)
        self.create_subscription(PoseStamped, "/coverage/start_pose", self._on_start, 1)
        self.path_publisher = self.create_publisher(Path, "/coverage/path", 1)
        self.reachable_publisher = self.create_publisher(OccupancyGrid, "/coverage/reachable_mask", 1)
        self.status_publisher = self.create_publisher(String, "/coverage/planner_status", 10)

    def _on_map(self, message: OccupancyGrid) -> None:
        self._map_message = message
        if bool(self.get_parameter("auto_plan").value):
            self._plan()

    def _on_roi(self, message: PolygonStamped) -> None:
        self._roi_message = message
        if self._map_message is not None and bool(self.get_parameter("auto_plan").value):
            self._plan()

    def _on_start(self, message: PoseStamped) -> None:
        self._start_xy = (message.pose.position.x, message.pose.position.y)
        if self._map_message is not None and bool(self.get_parameter("auto_plan").value):
            self._plan()

    def _publish_status(self, state: str, **details) -> None:
        message = String()
        message.data = json.dumps({"state": state, **details}, ensure_ascii=False)
        self.status_publisher.publish(message)

    def _plan(self) -> None:
        try:
            grid = grid_from_message(self._map_message)
            config = PlannerConfig(
                working_width_m=float(self.get_parameter("working_width_m").value),
                overlap_ratio=float(self.get_parameter("overlap_ratio").value),
                safety_margin_m=float(self.get_parameter("safety_margin_m").value),
                holonomic=bool(self.get_parameter("holonomic").value),
            )
            roi = polygon_mask(grid, self._roi_message) if self._roi_message is not None else None
            start_xy = self._start_xy or (
                float(self.get_parameter("start_x").value),
                float(self.get_parameter("start_y").value),
            )
            planner = CoveragePlanner(grid, config)
            model_path = str(self.get_parameter("rl_model_path").value)
            scheduler = None
            if model_path:
                scheduler = make_ddqn_scheduler(
                    MaskedDoubleDQN.load(model_path),
                    int(self.get_parameter("rl_candidate_slots").value),
                )
            plan = planner.plan(start_xy, roi_mask=roi, scheduler=scheduler)
        except Exception as error:
            self.get_logger().error(f"coverage planning failed: {error}")
            self._publish_status("failed", error=str(error))
            return
        stamp = self.get_clock().now().to_msg()
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = planner.grid.frame_id
        for point in plan.points:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = point.x
            pose.pose.position.y = point.y
            pose.pose.orientation = quaternion_from_yaw(point.yaw)
            path.poses.append(pose)
        self.path_publisher.publish(path)
        self.reachable_publisher.publish(bool_mask_message(plan.reachable_mask, planner.grid, stamp))
        self._publish_status(
            "ready",
            cells=len(plan.cells),
            waypoints=len(plan.points),
            axis=plan.sweep_axis,
            working_length_m=plan.working_length_m,
            transit_length_m=plan.transit_length_m,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CoveragePlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
