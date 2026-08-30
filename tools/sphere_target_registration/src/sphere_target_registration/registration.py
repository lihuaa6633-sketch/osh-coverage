from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .models import RigidTransform, _as_points


def _geometry_score(points: np.ndarray) -> float:
    """Return triangle area normalized by its longest edge squared."""
    if len(points) < 3:
        return 0.0
    best = 0.0
    for first in range(len(points) - 2):
        for second in range(first + 1, len(points) - 1):
            for third in range(second + 1, len(points)):
                a = points[second] - points[first]
                b = points[third] - points[first]
                edge_squared = max(
                    np.dot(a, a),
                    np.dot(b, b),
                    np.dot(points[third] - points[second], points[third] - points[second]),
                )
                if edge_squared > 0.0:
                    best = max(best, np.linalg.norm(np.cross(a, b)) / edge_squared)
    return float(best)


def pairwise_distance_errors(source: ArrayLike, target: ArrayLike) -> np.ndarray:
    source_points = _as_points(source, name="source")
    target_points = _as_points(target, name="target")
    if source_points.shape != target_points.shape:
        raise ValueError("source and target must have the same shape")
    errors = []
    for first in range(len(source_points) - 1):
        for second in range(first + 1, len(source_points)):
            errors.append(
                abs(
                    np.linalg.norm(source_points[first] - source_points[second])
                    - np.linalg.norm(target_points[first] - target_points[second])
                )
            )
    return np.asarray(errors, dtype=np.float64)


def estimate_rigid_transform(
    source: ArrayLike,
    target: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    minimum_geometry_score: float = 1e-3,
    maximum_pair_distance_error: float | None = None,
) -> RigidTransform:
    """Estimate a proper 3D rigid transform using weighted Kabsch/SVD.

    ``source[i]`` and ``target[i]`` must describe the same physical target.
    At least three non-collinear correspondences are required.
    """
    source_points = _as_points(source, name="source")
    target_points = _as_points(target, name="target")
    if source_points.shape != target_points.shape:
        raise ValueError("source and target must have the same shape")
    if len(source_points) < 3:
        raise ValueError("at least three corresponding targets are required")

    source_score = _geometry_score(source_points)
    target_score = _geometry_score(target_points)
    if min(source_score, target_score) < minimum_geometry_score:
        raise ValueError(
            "target centers are collinear or nearly collinear; move one target farther "
            "away from the line through the other two"
        )

    distance_errors = pairwise_distance_errors(source_points, target_points)
    if (
        maximum_pair_distance_error is not None
        and np.max(distance_errors) > maximum_pair_distance_error
    ):
        raise ValueError(
            "target correspondence/geometry check failed: maximum pair-distance "
            f"difference {np.max(distance_errors):.6g} exceeds "
            f"{maximum_pair_distance_error:.6g}"
        )

    if weights is None:
        normalized_weights = np.full(len(source_points), 1.0 / len(source_points))
    else:
        normalized_weights = np.asarray(weights, dtype=np.float64)
        if normalized_weights.shape != (len(source_points),):
            raise ValueError("weights must contain one value per target")
        if not np.all(np.isfinite(normalized_weights)) or np.any(normalized_weights <= 0.0):
            raise ValueError("weights must be finite and positive")
        normalized_weights = normalized_weights / np.sum(normalized_weights)

    source_centroid = np.sum(source_points * normalized_weights[:, None], axis=0)
    target_centroid = np.sum(target_points * normalized_weights[:, None], axis=0)
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid
    covariance = (source_centered * normalized_weights[:, None]).T @ target_centered

    left, _, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(right_t.T @ left.T))
    rotation = right_t.T @ correction @ left.T
    translation = target_centroid - rotation @ source_centroid

    registered = source_points @ rotation.T + translation
    errors = np.linalg.norm(registered - target_points, axis=1)
    rmse = float(np.sqrt(np.sum(normalized_weights * errors**2)))
    return RigidTransform(rotation, translation, rmse, float(np.max(errors)))
