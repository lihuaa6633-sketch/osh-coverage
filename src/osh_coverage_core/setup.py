from setuptools import find_packages, setup


package_name = "osh_coverage_core"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["numpy"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="OSH Coverage Research",
    maintainer_email="research@example.com",
    description="ROS-independent dynamic coverage planning core for OSH100 Ranger Pro.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "osh_coverage_demo = osh_coverage_core.cli:demo_main",
            "osh_train_ddqn = osh_coverage_core.cli:train_main",
            "osh_align_maps = osh_coverage_core.cli:align_main",
        ]
    },
)
