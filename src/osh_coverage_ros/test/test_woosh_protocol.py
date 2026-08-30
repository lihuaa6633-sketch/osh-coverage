import math

from osh_coverage_core.alignment import SE2
from osh_coverage_ros.woosh_protocol import (
    move_base_succeeded,
    transform_planar_pose,
    wrap_to_pi,
)


def test_transform_planar_pose_applies_translation_and_rotation():
    x, y, yaw = transform_planar_pose(
        SE2(2.0, -1.0, math.pi / 2.0),
        1.0,
        0.0,
        math.pi / 4.0,
    )
    assert abs(x - 2.0) < 1e-12
    assert abs(y) < 1e-12
    assert abs(yaw - 3.0 * math.pi / 4.0) < 1e-12


def test_wrap_to_pi_is_bounded():
    assert abs(wrap_to_pi(3.0 * math.pi) + math.pi) < 1e-12
    assert abs(wrap_to_pi(-3.0 * math.pi) + math.pi) < 1e-12


def test_move_base_success_requires_both_statuses():
    assert move_base_succeeded(4, 1, ros_succeeded_status=4)
    assert not move_base_succeeded(5, 1, ros_succeeded_status=4)
    assert not move_base_succeeded(4, -1, ros_succeeded_status=4)
    assert not move_base_succeeded(4, None, ros_succeeded_status=4)
