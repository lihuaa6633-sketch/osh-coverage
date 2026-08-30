"""Closed-loop residual coverage supervisor.

It waits for the vendor action chain to finish, plans only over the measured
residual mask, retries an unchanged residual twice, and then reports it as
temporarily unreachable.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from std_msgs.msg import String

from osh_coverage_core import CoveragePlanner, PlannerConfig
from osh_coverage_core.rl import MaskedDoubleDQN, make_ddqn_scheduler
from .ros_utils import grid_from_message, quaternion_from_yaw


class CoverageSupervisorNode(Node):
    def __init__(self) -> None:
        super().__init__("coverage_supervisor")
        self.declare_parameter("working_width_m", 0.60)
        self.declare_parameter("overlap_ratio", 0.10)
        self.declare_parameter("safety_margin_m", 0.15)
        self.declare_parameter("completion_ratio", 0.98)
        self.declare_parameter("max_retries", 2)
        self.declare_parameter("auto_repair", True)
        self.declare_parameter("rl_model_path", "")
        self.declare_parameter("rl_candidate_slots", 6)
        self._map_message = None
        self._residual_mask = None
        self._pose_xy = None
        self._coverage_ratio = 0.0
        self._last_signature = None
        self._same_residual_attempts = 0
        self.create_subscription(OccupancyGrid, "/map", self._on_map, 1)
        self.create_subscription(OccupancyGrid, "/coverage/residual_mask", self._on_residual, 1)
        self.create_subscription(PoseStamped, "/coverage/actual_pose", self._on_pose, 50)
        self.create_subscription(String, "/coverage/monitor_status", self._on_monitor_status, 10)
        self.create_subscription(String, "/coverage/woosh_status", self._on_woosh_status, 10)
        self.path_publisher = self.create_publisher(Path, "/coverage/path", 1)
        self.status_publisher = self.create_publisher(String, "/coverage/supervisor_status", 10)

    def _publish_status(self, state: str, **details) -> None:
        message = String()
        message.data = json.dumps({"state": state, **details}, ensure_ascii=False)
        self.status_publisher.publish(message)

    def _on_map(self, message: OccupancyGrid) -> None:
        self._map_message = message

    def _on_residual(self, message: OccupancyGrid) -> None:
        values = np.asarray(message.data, dtype=np.int16).reshape((message.info.height, message.info.width))
        self._residual_mask = values >= 50

    def _on_pose(self, message: PoseStamped) -> None:
        self._pose_xy = (message.pose.position.x, message.pose.position.y)

    def _on_monitor_status(self, message: String) -> None:
        try:
            self._coverage_ratio = float(json.loads(message.data).get("coverage_ratio", 0.0))
        except (ValueError, TypeError, json.JSONDecodeError):
            self.get_logger().warning("ignored malformed monitor status")

    def _on_woosh_status(self, message: String) -> None:
        if not bool(self.get_parameter("auto_repair").value):
            return
        try:
            action = json.loads(message.data).get("action")
        except json.JSONDecodeError:
            return
        if action == "finished":
            self._plan_repair()

    def _plan_repair(self) -> None:
        completion_ratio = float(self.get_parameter("completion_ratio").value)
        if self._coverage_ratio >= completion_ratio:
            self._publish_status("complete", coverage_ratio=self._coverage_ratio)
            return
        if self._map_message is None or self._residual_mask is None or self._pose_xy is None:
            self._publish_status("waiting_for_inputs")
            return
        signature = hashlib.sha1(np.packbits(self._residual_mask).tobytes()).hexdigest()
        if signature == self._last_signature:
            self._same_residual_attempts += 1
        else:
            self._last_signature = signature
            self._same_residual_attempts = 0
        max_retries = int(self.get_parameter("max_retries").value)
        if self._same_residual_attempts > max_retries:
            self._publish_status(
                "temporarily_unreachable",
                coverage_ratio=self._coverage_ratio,
                attempts=self._same_residual_attempts,
            )
            return
        try:
            grid = grid_from_message(self._map_message)
            planner = CoveragePlanner(
                grid,
                PlannerConfig(
                    working_width_m=float(self.get_parameter("working_width_m").value),
                    overlap_ratio=float(self.get_parameter("overlap_ratio").value),
                    safety_margin_m=float(self.get_parameter("safety_margin_m").value),
                    holonomic=True,
                ),
            )
            scheduler = None
            model_path = str(self.get_parameter("rl_model_path").value)
            if model_path:
                scheduler = make_ddqn_scheduler(
                    MaskedDoubleDQN.load(model_path),
                    int(self.get_parameter("rl_candidate_slots").value),
                )
            plan = planner.plan(self._pose_xy, roi_mask=self._residual_mask, scheduler=scheduler)
        except Exception as error:
            self._publish_status("repair_planning_failed", error=str(error))
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
        self._publish_status(
            "repair_dispatched",
            attempts=self._same_residual_attempts,
            waypoints=len(path.poses),
            residual_cells=int(self._residual_mask.sum()),
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CoverageSupervisorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

