from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from .models import SphereFitResult


def _sphere_from_four(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    origin = points[0]
    system = 2.0 * (points[1:] - origin)
    rhs = np.sum(points[1:] ** 2, axis=1) - np.sum(origin**2)
    if np.linalg.cond(system) > 1e8:
        return None
    try:
        center = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return None
    radius = float(np.linalg.norm(points[0] - center))
    if not np.all(np.isfinite(center)) or not math.isfinite(radius):
        return None
    return center, radius


def _algebraic_fit(points: np.ndarray) -> tuple[np.ndarray, float]:
    system = np.column_stack((2.0 * points, np.ones(len(points))))
    rhs = np.sum(points**2, axis=1)
    solution, _, rank, _ = np.linalg.lstsq(system, rhs, rcond=None)
    if rank < 4:
        raise ValueError("sphere points do not span a 3D surface")
    center = solution[:3]
    radius_squared = float(solution[3] + np.dot(center, center))
    if radius_squared <= 0.0:
        raise ValueError("sphere fit produced a non-positive radius")
    return center, math.sqrt(radius_squared)


def _geometric_refine(
    points: np.ndarray,
    center: np.ndarray,
    radius: float,
    *,
    max_iterations: int = 30,
) -> tuple[np.ndarray, float]:
    center = center.copy()
    radius = float(radius)
    for _ in range(max_iterations):
        offsets = points - center
        distances = np.linalg.norm(offsets, axis=1)
        valid = distances > np.finfo(float).eps
        if np.count_nonzero(valid) < 4:
            break
        residuals = distances[valid] - radius
        scale = max(1.4826 * np.median(np.abs(residuals - np.median(residuals))), 1e-12)
        huber_limit = 1.5 * scale
        weights = np.minimum(1.0, huber_limit / np.maximum(np.abs(residuals), 1e-12))
        jacobian = np.column_stack(
            (-offsets[valid] / distances[valid, None], -np.ones(np.sum(valid)))
        )
        sqrt_weights = np.sqrt(weights)
        step, _, _, _ = np.linalg.lstsq(
            jacobian * sqrt_weights[:, None],
            -residuals * sqrt_weights,
            rcond=None,
        )
        center += step[:3]
        radius += float(step[3])
        if radius <= 0.0:
            raise ValueError("sphere refinement produced a non-positive radius")
        if np.linalg.norm(step) <= 1e-10 * max(1.0, radius):
            break
    return center, radius


def fit_sphere_ransac(
    points: ArrayLike,
    *,
    distance_threshold: float,
    expected_radius: float | None = None,
    radius_tolerance: float | None = None,
    radius_bounds: tuple[float, float] | None = None,
    max_iterations: int = 2000,
    min_inliers: int = 30,
    min_inlier_ratio: float = 0.25,
    max_ransac_points: int = 20000,
    random_seed: int | None = None,
) -> SphereFitResult:
    """Fit a sphere robustly and return quality metrics.

    RANSAC handles clutter in a rough crop; iteratively reweighted geometric least
    squares then removes the small algebraic-fit bias.
    """
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {values.shape}")
    values = values[np.all(np.isfinite(values), axis=1)]
    if len(values) < 4:
        raise ValueError("at least four finite points are required to fit a sphere")
    if distance_threshold <= 0.0:
        raise ValueError("distance_threshold must be positive")
    if max_iterations < 1 or min_inliers < 4:
        raise ValueError("max_iterations must be >= 1 and min_inliers must be >= 4")
    if not 0.0 < min_inlier_ratio <= 1.0:
        raise ValueError("min_inlier_ratio must be in (0, 1]")
    if expected_radius is not None:
        if expected_radius <= 0.0:
            raise ValueError("expected_radius must be positive")
        if radius_tolerance is None or radius_tolerance <= 0.0:
            raise ValueError("positive radius_tolerance is required with expected_radius")
        radius_bounds = (
            expected_radius - radius_tolerance,
            expected_radius + radius_tolerance,
        )
    if radius_bounds is not None:
        if radius_bounds[0] <= 0.0 or radius_bounds[1] <= radius_bounds[0]:
            raise ValueError("radius_bounds must be two increasing positive values")

    rng = np.random.default_rng(random_seed)
    if len(values) > max_ransac_points:
        sample_indices = rng.choice(len(values), max_ransac_points, replace=False)
        ransac_points = values[sample_indices]
    else:
        ransac_points = values

    best_center: np.ndarray | None = None
    best_radius = 0.0
    best_count = 0
    best_median = math.inf
    completed_iterations = 0
    for iteration in range(max_iterations):
        completed_iterations = iteration + 1
        sample = ransac_points[rng.choice(len(ransac_points), 4, replace=False)]
        candidate = _sphere_from_four(sample)
        if candidate is None:
            continue
        center, radius = candidate
        if radius_bounds is not None and not radius_bounds[0] <= radius <= radius_bounds[1]:
            continue
        residuals = np.abs(np.linalg.norm(ransac_points - center, axis=1) - radius)
        mask = residuals <= distance_threshold
        count = int(np.count_nonzero(mask))
        median = float(np.median(residuals[mask])) if count else math.inf
        if count > best_count or (count == best_count and median < best_median):
            best_center = center
            best_radius = radius
            best_count = count
            best_median = median

    required_ransac_inliers = max(
        4,
        int(math.ceil(min_inlier_ratio * len(ransac_points))),
        int(math.ceil(min_inliers * len(ransac_points) / len(values))),
    )
    if best_center is None or best_count < required_ransac_inliers:
        raise ValueError(
            "RANSAC could not find a sphere with enough support; check the target ROI, "
            "radius, units, and distance_threshold"
        )

    center = best_center
    radius = best_radius
    for _ in range(3):
        residuals = np.abs(np.linalg.norm(values - center, axis=1) - radius)
        inlier_mask = residuals <= distance_threshold
        if np.count_nonzero(inlier_mask) < min_inliers:
            raise ValueError(
                f"sphere has only {np.count_nonzero(inlier_mask)} inliers; "
                f"at least {min_inliers} are required"
            )
        center, radius = _algebraic_fit(values[inlier_mask])
        center, radius = _geometric_refine(values[inlier_mask], center, radius)

    residuals = np.abs(np.linalg.norm(values - center, axis=1) - radius)
    inlier_mask = residuals <= distance_threshold
    inlier_count = int(np.count_nonzero(inlier_mask))
    inlier_ratio = inlier_count / len(values)
    if inlier_count < min_inliers or inlier_ratio < min_inlier_ratio:
        raise ValueError(
            f"final sphere support is too small: {inlier_count}/{len(values)} "
            f"({inlier_ratio:.1%})"
        )
    if radius_bounds is not None and not radius_bounds[0] <= radius <= radius_bounds[1]:
        raise ValueError(
            f"fitted radius {radius:.6g} is outside expected range "
            f"[{radius_bounds[0]:.6g}, {radius_bounds[1]:.6g}]"
        )

    inlier_residuals = residuals[inlier_mask]
    directions = values[inlier_mask] - center
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    direction_eigenvalues = np.linalg.eigvalsh((directions.T @ directions) / len(directions))
    direction_ratio = float(direction_eigenvalues[0] / direction_eigenvalues[-1])

    return SphereFitResult(
        center=center,
        radius=float(radius),
        rmse=float(np.sqrt(np.mean(inlier_residuals**2))),
        median_absolute_error=float(np.median(inlier_residuals)),
        inlier_count=inlier_count,
        point_count=len(values),
        inlier_ratio=inlier_ratio,
        direction_eigenvalue_ratio=direction_ratio,
        iterations=completed_iterations,
    )
