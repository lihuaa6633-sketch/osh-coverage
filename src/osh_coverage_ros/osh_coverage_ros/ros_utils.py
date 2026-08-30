from __future__ import annotations

import math

import numpy as np

from osh_coverage_core.grid import GridMap


def yaw_from_quaternion(quaternion) -> float:
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def quaternion_from_yaw(yaw: float):
    from geometry_msgs.msg import Quaternion

    result = Quaternion()
    result.z = math.sin(yaw / 2.0)
    result.w = math.cos(yaw / 2.0)
    return result


def grid_from_message(message) -> GridMap:
    origin_yaw = yaw_from_quaternion(message.info.origin.orientation)
    if abs(origin_yaw) > 1e-6:
        raise ValueError("rotated OccupancyGrid origins are not supported; transform the map to an axis-aligned frame")
    return GridMap.from_occupancy_values(
        message.data,
        message.info.width,
        message.info.height,
        message.info.resolution,
        message.info.origin.position.x,
        message.info.origin.position.y,
        frame_id=message.header.frame_id or "map",
    )


def polygon_mask(grid: GridMap, polygon_message) -> np.ndarray:
    points = [(point.x, point.y) for point in polygon_message.polygon.points]
    if len(points) < 3:
        raise ValueError("ROI polygon requires at least three points")
    mask = np.zeros(grid.shape, dtype=bool)
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            x, y = grid.grid_to_world(row, col)
            inside = False
            previous = len(points) - 1
            for current in range(len(points)):
                xi, yi = points[current]
                xj, yj = points[previous]
                intersects = ((yi > y) != (yj > y)) and (
                    x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
                )
                if intersects:
                    inside = not inside
                previous = current
            mask[row, col] = inside
    return mask


def bool_mask_message(mask: np.ndarray, grid: GridMap, stamp, occupied_when_false: bool = True):
    from nav_msgs.msg import OccupancyGrid

    message = OccupancyGrid()
    message.header.stamp = stamp
    message.header.frame_id = grid.frame_id
    message.info.resolution = float(grid.resolution)
    message.info.width = int(grid.shape[1])
    message.info.height = int(grid.shape[0])
    message.info.origin.position.x = float(grid.origin_x)
    message.info.origin.position.y = float(grid.origin_y)
    message.info.origin.orientation.w = 1.0
    values = np.where(mask, 0 if occupied_when_false else 100, 100 if occupied_when_false else 0).astype(np.int8)
    message.data = values.ravel().tolist()
    return message

