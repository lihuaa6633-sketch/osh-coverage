"""Adapter for the proprietary Woosh ROS 2 messages described in woosh_ros2.pdf."""

from __future__ import annotations

import json
import math

import numpy as np

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String

from osh_coverage_core.alignment import SE2
from .ros_utils import quaternion_from_yaw

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
                "Woosh proprietary message packages are unavailable. Install the vendor agent/messages before starting this node."
            )
        self.declare_parameter("namespace", "/woosh_robot")
        self.declare_parameter("slam_to_woosh_x", 0.0)
        self.declare_parameter("slam_to_woosh_y", 0.0)
        self.declare_parameter("slam_to_woosh_yaw", 0.0)
        self.declare_parameter("path_chunk_size", 200)
        namespace = str(self.get_parameter("namespace").value).rstrip("/")
        self.slam_to_woosh = SE2(
            float(self.get_parameter("slam_to_woosh_x").value),
            float(self.get_parameter("slam_to_woosh_y").value),
            float(self.get_parameter("slam_to_woosh_yaw").value),
        )
        self.woosh_to_slam = self.slam_to_woosh.inverse()
        self.path_chunk_size = max(2, int(self.get_parameter("path_chunk_size").value))
        self._pending_chunks = []
        self.pose_publisher = self.create_publisher(PoseStamped, "/coverage/actual_pose", 50)
        self.status_publisher = self.create_publisher(String, "/coverage/woosh_status", 10)
        self.create_subscription(Path, "/coverage/path", self._on_path, 1)
        self.create_subscription(PoseSpeed, namespace + "/robot/PoseSpeed", self._on_pose, 50)
        self.create_subscription(RobotState, namespace + "/robot/RobotState", self._on_robot_state, 10)
        self.create_subscription(OperationState, namespace + "/robot/OperationState", self._on_operation_state, 10)
        self.create_subscription(AbnormalCodes, namespace + "/robot/AbnormalCodes", self._on_abnormal, 10)
        self.move_base = ActionClient(self, MoveBase, namespace + "/ros/MoveBase")
        self._last_status = {}

    def _publish_status(self, **values) -> None:
        self._last_status.update(values)
        message = String()
        message.data = json.dumps(self._last_status, ensure_ascii=False, default=str)
        self.status_publisher.publish(message)

    def _on_pose(self, message: PoseSpeed) -> None:
        output = PoseStamped()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = "airy_map"
        slam_xy = self.woosh_to_slam.apply(np.asarray(((float(message.pose.x), float(message.pose.y)),)))[0]
        output.pose.position.x = float(slam_xy[0])
        output.pose.position.y = float(slam_xy[1])
        output.pose.orientation = quaternion_from_yaw(float(message.pose.theta) - self.slam_to_woosh.yaw)
        self.pose_publisher.publish(output)

    def _on_robot_state(self, message: RobotState) -> None:
        self._publish_status(robot_state=int(message.state.value))

    def _on_operation_state(self, message: OperationState) -> None:
        self._publish_status(operation_nav_bits=int(message.nav), operation_robot_bits=int(message.robot))

    def _on_abnormal(self, message: AbnormalCodes) -> None:
        self._publish_status(abnormal_codes=str(message))

    def _on_path(self, message: Path) -> None:
        if not message.poses:
            self.get_logger().warning("ignored empty coverage path")
            return
        if not self.move_base.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Woosh MoveBase action server is unavailable")
            self._publish_status(action="server_unavailable")
            return
        self._pending_chunks = [
            message.poses[index : index + self.path_chunk_size]
            for index in range(0, len(message.poses), self.path_chunk_size)
        ]
        self._send_next_chunk()

    def _send_next_chunk(self) -> None:
        if not self._pending_chunks:
            self._publish_status(action="finished")
            return
        poses = self._pending_chunks.pop(0)
        goal = MoveBase.Goal()
        goal.arg.poses = []
        for source in poses:
            pose = Pose2D()
            woosh_xy = self.slam_to_woosh.apply(
                np.asarray(((float(source.pose.position.x), float(source.pose.position.y)),))
            )[0]
            pose.x = float(woosh_xy[0])
            pose.y = float(woosh_xy[1])
            # Coverage planner sends a planar quaternion.
            q = source.pose.orientation
            slam_yaw = float(2.0 * math.atan2(q.z, q.w))
            pose.theta = float(slam_yaw + self.slam_to_woosh.yaw)
            goal.arg.poses.append(pose)
        goal.arg.target_pose = goal.arg.poses[-1]
        goal.arg.execution_mode.value = 1  # K_ONE_BY_ONE
        goal.arg.action.value = 1  # K_EXECUTE
        future = self.move_base.send_goal_async(goal, feedback_callback=self._on_feedback)
        future.add_done_callback(self._on_goal_response)

    def _on_feedback(self, feedback) -> None:
        self._publish_status(action="executing", feedback=str(feedback.feedback))

    def _on_goal_response(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._publish_status(action="rejected")
            return
        self._publish_status(action="accepted")
        goal_handle.get_result_async().add_done_callback(self._on_chunk_result)

    def _on_chunk_result(self, future) -> None:
        result = future.result()
        self._publish_status(
            action="chunk_finished",
            remaining_chunks=len(self._pending_chunks),
            result=str(result.result),
        )
        self._send_next_chunk()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WooshBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
