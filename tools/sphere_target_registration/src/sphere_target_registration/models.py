from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _as_points(values: ArrayLike, *, name: str = "points") -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return result


@dataclass(frozen=True)
class PointCloud:
    """Numeric point-cloud table with named x/y/z columns."""

    data: FloatArray
    fields: tuple[str, ...] = ("x", "y", "z")

    def __post_init__(self) -> None:
        data = np.asarray(self.data, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(f"point cloud must have shape (N, M>=3), got {data.shape}")
        if len(self.fields) != data.shape[1]:
            raise ValueError("number of field names must match point-cloud columns")
        if len(set(self.fields)) != len(self.fields):
            raise ValueError("point-cloud field names must be unique")
        for axis in ("x", "y", "z"):
            if axis not in self.fields:
                raise ValueError(f"point cloud is missing required field {axis!r}")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "fields", tuple(self.fields))

    @property
    def xyz(self) -> FloatArray:
        indices = [self.fields.index(axis) for axis in ("x", "y", "z")]
        return self.data[:, indices]

    def transformed(self, transform: "RigidTransform") -> "PointCloud":
        output = self.data.copy()
        indices = [self.fields.index(axis) for axis in ("x", "y", "z")]
        output[:, indices] = transform.apply(self.xyz)
        return PointCloud(output, self.fields)


@dataclass(frozen=True)
class SphereFitResult:
    center: FloatArray
    radius: float
    rmse: float
    median_absolute_error: float
    inlier_count: int
    point_count: int
    inlier_ratio: float
    direction_eigenvalue_ratio: float
    iterations: int

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("sphere center must be a finite 3-vector")
        object.__setattr__(self, "center", center)

    def to_dict(self) -> dict[str, Any]:
        return {
            "center": self.center.tolist(),
            "radius": self.radius,
            "rmse": self.rmse,
            "median_absolute_error": self.median_absolute_error,
            "inlier_count": self.inlier_count,
            "point_count": self.point_count,
            "inlier_ratio": self.inlier_ratio,
            "direction_eigenvalue_ratio": self.direction_eigenvalue_ratio,
            "iterations": self.iterations,
        }


@dataclass(frozen=True)
class RigidTransform:
    """A transform ``target = rotation @ source + translation``."""

    rotation: FloatArray
    translation: FloatArray
    rmse: float = 0.0
    max_error: float = 0.0

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("rotation and translation must have shapes (3, 3) and (3,)")
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            raise ValueError("transform contains NaN or infinite values")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
            raise ValueError("rotation matrix is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
            raise ValueError("rotation matrix determinant must be +1")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @property
    def matrix(self) -> FloatArray:
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = self.rotation
        result[:3, 3] = self.translation
        return result

    @property
    def quaternion_xyzw(self) -> FloatArray:
        """Return the rotation as a normalized ROS-compatible x/y/z/w quaternion."""
        matrix = self.rotation
        candidates = np.array(
            [
                1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2],
                1.0 - matrix[0, 0] + matrix[1, 1] - matrix[2, 2],
                1.0 - matrix[0, 0] - matrix[1, 1] + matrix[2, 2],
                1.0 + np.trace(matrix),
            ]
        )
        index = int(np.argmax(candidates))
        quaternion = np.empty(4, dtype=np.float64)
        if index == 0:
            quaternion[:] = [
                candidates[0],
                matrix[0, 1] + matrix[1, 0],
                matrix[2, 0] + matrix[0, 2],
                matrix[2, 1] - matrix[1, 2],
            ]
        elif index == 1:
            quaternion[:] = [
                matrix[0, 1] + matrix[1, 0],
                candidates[1],
                matrix[1, 2] + matrix[2, 1],
                matrix[0, 2] - matrix[2, 0],
            ]
        elif index == 2:
            quaternion[:] = [
                matrix[2, 0] + matrix[0, 2],
                matrix[1, 2] + matrix[2, 1],
                candidates[2],
                matrix[1, 0] - matrix[0, 1],
            ]
        else:
            quaternion[:] = [
                matrix[2, 1] - matrix[1, 2],
                matrix[0, 2] - matrix[2, 0],
                matrix[1, 0] - matrix[0, 1],
                candidates[3],
            ]
        quaternion *= 0.5 / np.sqrt(max(candidates[index], 0.0))
        quaternion /= np.linalg.norm(quaternion)
        return quaternion

    def apply(self, points: ArrayLike) -> FloatArray:
        values = np.asarray(points, dtype=np.float64)
        if values.shape == (3,):
            return self.rotation @ values + self.translation
        values = _as_points(values)
        return values @ self.rotation.T + self.translation

    def inverse(self) -> "RigidTransform":
        inverse_rotation = self.rotation.T
        return RigidTransform(
            inverse_rotation,
            -(inverse_rotation @ self.translation),
            self.rmse,
            self.max_error,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": "registered_xyz = rotation @ scanned_xyz + translation",
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "quaternion_xyzw": self.quaternion_xyzw.tolist(),
            "matrix_4x4": self.matrix.tolist(),
            "rmse": self.rmse,
            "max_error": self.max_error,
        }
