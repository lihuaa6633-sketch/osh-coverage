"""Airy SLAM integration template for ROS 2 Humble.

Validate topic names with `ros2 topic list` because KISS-ICP/RTAB-Map release
packages can use different default namespaces.
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    kiss_launch = os.path.join(get_package_share_directory("kiss_icp"), "launch", "odometry.launch.py")
    rtab_params = os.path.join(get_package_share_directory("osh_coverage_ros"), "config", "rtabmap_airyslam.yaml")
    coverage_params = os.path.join(get_package_share_directory("osh_coverage_ros"), "config", "coverage_params.yaml")
    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(kiss_launch),
                launch_arguments={"topic": "/rslidar_points", "visualize": "false"}.items(),
            ),
            Node(
                package="rtabmap_slam",
                executable="rtabmap",
                name="rtabmap",
                parameters=[rtab_params],
                remappings=[
                    ("scan_cloud", "/rslidar_points"),
                    ("imu", "/rslidar_imu_data"),
                    ("odom", "/kiss/odometry"),
                ],
                output="screen",
            ),
            Node(
                package="osh_coverage_ros",
                executable="map_projector_node",
                parameters=[coverage_params],
                output="screen",
            ),
        ]
    )
