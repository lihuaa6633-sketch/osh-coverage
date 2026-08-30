"""Pure helpers for the Woosh adapter that can be tested without vendor messages."""

from __future__ import annotations

import math

import numpy as np

from osh_coverage_core.alignment import SE2


def wrap_to_pi(angle: float) -> float:
    """Normalize an angle to the interval [-pi, pi)."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def transform_planar_pose(
    transform: SE2,
    x: float,
    y: float,
    yaw: float,
) -> tuple[float, float, float]:
    """Apply an SE(2) transform to a planar pose."""
    transformed = transform.apply(np.asarray(((float(x), float(y)),)))[0]
    return (
        float(transformed[0]),
        float(transformed[1]),
        wrap_to_pi(float(yaw) + transform.yaw),
    )


def move_base_succeeded(
    ros_goal_status: int,
    vendor_state: int | None,
    *,
    ros_succeeded_status: int,
) -> bool:
    """Fail closed unless both ROS and Woosh report explicit success."""
    return int(ros_goal_status) == int(ros_succeeded_status) and vendor_state == 1
