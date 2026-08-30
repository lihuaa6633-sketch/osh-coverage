from glob import glob
from setuptools import find_packages, setup


package_name = "osh_coverage_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="OSH Coverage Research",
    maintainer_email="research@example.com",
    description="ROS 2 adapters for OSH100 dynamic coverage planning.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "coverage_planner_node = osh_coverage_ros.coverage_planner_node:main",
            "coverage_monitor_node = osh_coverage_ros.coverage_monitor_node:main",
            "coverage_supervisor_node = osh_coverage_ros.coverage_supervisor_node:main",
            "map_projector_node = osh_coverage_ros.map_projector_node:main",
            "woosh_bridge_node = osh_coverage_ros.woosh_bridge_node:main",
        ]
    },
)
