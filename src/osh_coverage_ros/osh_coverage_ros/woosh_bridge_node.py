"""Adapter for the proprietary Woosh ROS 2 messages described in woosh_ros2.pdf."""

from __future__ import annotations

import json
import math

import numpy as np

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String

from osh_coverage_core.alignment import SE2
from .ros_utils import quaternion_from_yaw
from .woosh_protocol import move_base_succeeded, transform_planar_pose

try:
    from woosh_common_msgs.msg import Pose2D
    from woosh_robot_msgs.msg import AbnormalCodes, OperationState, PoseSpeed, RobotState
    from woosh_ros_msgs.action import MoveBase

    WOOSH_MESSAGES_AVAILABLE = True
except ImportError:
    WOOSH_MESSAGES_AVAILABLE = False


class WooshBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("woosh_bridge")
        if not WOOSH_MESSAGES_AVAILABLE:
            raise RuntimeError(
                "Woosh proprietary message packages are unavailable. Install the "
                "vendor agent/messages before starting this node."
            )
        self.declare_parameter("namespace", "/woosh_robot")
        self.declare_parameter("dry_run", True)
        self.declare_parameter("require_idle_before_start", True)
        self.declare_parameter("slam_to_woosh_x", 0.0)
        self.declare_parameter("slam_to_woosh_y", 0.0)
        self.declare_parameter("slam_to_woosh_yaw", 0.0)
        self.declare_parameter("path_chunk_size", 200)
        self.declare_parameter("max_path_points", 5000)
        namespace = str(self.get_parameter("namespace").value).rstrip("/")
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.require_idle_before_start = bool(
            self.get_parameter("require_idle_before_start").value
        )
        self.slam_to_woosh = SE2(
            float(self.get_parameter("slam_to_woosh_x").value),
            float(self.get_parameter("slam_to_woosh_y").value),
            float(self.get_parameter("slam_to_woosh_yaw").value),
        )
        self.woosh_to_slam = self.slam_to_woosh.inverse()
        self.path_chunk_size = max(2, int(self.get_parameter("path_chunk_size").value))
        self.max_path_points = max(
            self.path_chunk_size, int(self.get_parameter("max_path_points").value)
        )
        self._pending_chunks = []
        self._goal_in_flight = False
        self._robot_state = None
        self.pose_publisher = self.create_publisher(PoseStamped, "/coverage/actual_pose", 50)
        self.status_publisher = self.create_publisher(String, "/coverage/woosh_status", 10)
        self.create_subscription(Path, "/coverage/path", self._on_path, 1)
        self.create_subscription(
            PoseSpeed, namespace + "/robot/PoseSpeed", self._on_pose, 50
        )
        self.create_subscription(
            RobotState, namespace + "/robot/RobotState", self._on_robot_state, 10
        )
        self.create_subscription(
            OperationState,
            namespace + "/robot/OperationState",
            self._on_operation_state,
            10,
        )
        self.create_subscription(
            AbnormalCodes, namespace + "/robot/AbnormalCodes", self._on_abnormal, 10
        )
        self.move_base = ActionClient(self, MoveBase, namespace + "/ros/MoveBase")
        self._last_status = {}
        self._publish_status(action="ready", dry_run=self.dry_run)

    def _publish_status(self, **values) -> None:
        self._last_status.update(values)
        message = String()
        message.data = json.dumps(self._last_status, ensure_ascii=False, default=str)
        self.status_publisher.publish(message)

    def _on_pose(self, message: PoseSpeed) -> None:
        output = PoseStamped()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = "airy_map"
        slam_xy = self.woosh_to_slam.apply(
            np.asarray(((float(message.pose.x), float(message.pose.y)),))
        )[0]
        output.pose.position.x = float(slam_xy[0])
        output.pose.position.y = float(slam_xy[1])
        output.pose.orientation = quaternion_from_yaw(
            float(message.pose.theta) - self.slam_to_woosh.yaw
        )
        self.pose_publisher.publish(output)

    def _on_robot_state(self, message: RobotState) -> None:
        self._robot_state = int(message.state.value)
        self._publish_status(robot_state=self._robot_state)

    def _on_operation_state(self, message: OperationState) -> None:
        self._publish_status(
            operation_nav_bits=int(message.nav),
            operation_robot_bits=int(message.robot),
        )

    def _on_abnormal(self, message: AbnormalCodes) -> None:
        self._publish_status(abnormal_codes=str(message))

    def _on_path(self, message: Path) -> None:
        if not message.poses:
            self.get_logger().warning("ignored empty coverage path")
            return
        if len(message.poses) > self.max_path_points:
            self.get_logger().error(
                f"rejected path with {len(message.poses)} points; limit is {self.max_path_points}"
            )
            self._publish_status(action="path_too_large", path_points=len(message.poses))
            return
        if self._goal_in_flight or self._pending_chunks:
            self.get_logger().warning("rejected new path while another path is active")
            self._publish_status(action="busy")
            return
        if self.dry_run:
            first = self._to_woosh_pose(message.poses[0])
            last = self._to_woosh_pose(message.poses[-1])
            self.get_logger().warning(
                "dry-run: path transformed but no MoveBase goal was sent"
            )
            self._publish_status(
                action="dry_run",
                path_points=len(message.poses),
                first_pose=[first.x, first.y, first.theta],
                last_pose=[last.x, last.y, last.theta],
            )
            return
        if self.require_idle_before_start and self._robot_state != 2:  # K_IDLE
            self.get_logger().error(
                "rejected path because RobotState is not confirmed as K_IDLE"
            )
            self._publish_status(action="robot_not_idle", robot_state=self._robot_state)
            return
        if not self.move_base.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Woosh MoveBase action server is unavailable")
            self._publish_status(action="server_unavailable")
            return
        self._pending_chunks = [
            message.poses[index: index + self.path_chunk_size]
            for index in range(0, len(message.poses), self.path_chunk_size)
        ]
        self._send_next_chunk()

    def _to_woosh_pose(self, source) -> Pose2D:
        pose = Pose2D()
        q = source.pose.orientation
        slam_yaw = float(2.0 * math.atan2(q.z, q.w))
        pose.x, pose.y, pose.theta = transform_planar_pose(
            self.slam_to_woosh,
            source.pose.position.x,
            source.pose.position.y,
            slam_yaw,
        )
        return pose

    def _send_next_chunk(self) -> None:
        if not self._pending_chunks:
            self._goal_in_flight = False
            self._publish_status(action="finished")
            return
        poses = self._pending_chunks.pop(0)
        goal = MoveBase.Goal()
        goal.arg.poses = [self._to_woosh_pose(source) for source in poses]
        goal.arg.target_pose = goal.arg.poses[-1]
        goal.arg.execution_mode.value = 1  # K_ONE_BY_ONE
        goal.arg.action.value = 1  # K_EXECUTE
        self._goal_in_flight = True
        future = self.move_base.send_goal_async(goal, feedback_callback=self._on_feedback)
        future.add_done_callback(self._on_goal_response)

    def _on_feedback(self, feedback) -> None:
        self._publish_status(action="executing", feedback=str(feedback.feedback))

    def _on_goal_response(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as error:  # rclpy future propagates transport errors here
            self._pending_chunks.clear()
            self._goal_in_flight = False
            self._publish_status(action="goal_error", error=str(error))
            return
        if not goal_handle.accepted:
            self._pending_chunks.clear()
            self._goal_in_flight = False
            self._publish_status(action="rejected")
            return
        self._publish_status(action="accepted")
        goal_handle.get_result_async().add_done_callback(self._on_chunk_result)

    def _on_chunk_result(self, future) -> None:
        try:
            response = future.result()
        except Exception as error:  # rclpy future propagates transport errors here
            self._pending_chunks.clear()
            self._goal_in_flight = False
            self._publish_status(action="result_error", error=str(error))
            return

        feedback_result = getattr(response.result, "ret", None)
        vendor_state = getattr(getattr(feedback_result, "state", None), "value", None)
        succeeded = move_base_succeeded(
            response.status,
            vendor_state,
            ros_succeeded_status=GoalStatus.STATUS_SUCCEEDED,
        )
        self._publish_status(
            action="chunk_succeeded" if succeeded else "chunk_failed",
            remaining_chunks=len(self._pending_chunks),
            ros_goal_status=int(response.status),
            vendor_state=vendor_state,
            result=str(response.result),
        )
        self._goal_in_flight = False
        if not succeeded:
            self._pending_chunks.clear()
            return
        self._send_next_chunk()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WooshBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
