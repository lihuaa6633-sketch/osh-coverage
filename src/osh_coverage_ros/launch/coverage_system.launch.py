from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(get_package_share_directory("osh_coverage_ros"), "config", "coverage_params.yaml")
    use_woosh = LaunchConfiguration("use_woosh_bridge")
    woosh_dry_run = LaunchConfiguration("woosh_dry_run")
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_woosh_bridge", default_value="false"),
            DeclareLaunchArgument(
                "woosh_dry_run",
                default_value="true",
                description="Transform and report Woosh paths without sending MoveBase goals",
            ),
            Node(package="osh_coverage_ros", executable="coverage_planner_node", parameters=[config], output="screen"),
            Node(package="osh_coverage_ros", executable="coverage_monitor_node", parameters=[config], output="screen"),
            Node(package="osh_coverage_ros", executable="coverage_supervisor_node", parameters=[config], output="screen"),
            Node(
                package="osh_coverage_ros",
                executable="woosh_bridge_node",
                parameters=[config, {"dry_run": woosh_dry_run}],
                output="screen",
                condition=IfCondition(use_woosh),
            ),
        ]
    )
