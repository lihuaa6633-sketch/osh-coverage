"""SE(2) alignment for self-built SLAM and the Woosh navigation map."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SE2:
    x: float
    y: float
    yaw: float

    @property
    def rotation(self) -> np.ndarray:
        cosine, sine = math.cos(self.yaw), math.sin(self.yaw)
        return np.asarray(((cosine, -sine), (sine, cosine)), dtype=float)

    def apply(self, points: np.ndarray) -> np.ndarray:
        source = np.asarray(points, dtype=float)
        return source @ self.rotation.T + np.asarray((self.x, self.y))

    def inverse(self) -> "SE2":
        rotation_inv = self.rotation.T
        translation = -rotation_inv @ np.asarray((self.x, self.y))
        return SE2(float(translation[0]), float(translation[1]), -self.yaw)

    def compose(self, other: "SE2") -> "SE2":
        translation = self.apply(np.asarray(((other.x, other.y),)))[0]
        yaw = (self.yaw + other.yaw + math.pi) % (2.0 * math.pi) - math.pi
        return SE2(float(translation[0]), float(translation[1]), yaw)


@dataclass(frozen=True)
class AlignmentResult:
    transform: SE2
    inliers: np.ndarray
    residuals_m: np.ndarray

    @property
    def rmse_m(self) -> float:
        selected = self.residuals_m[self.inliers]
        return float(np.sqrt(np.mean(selected**2))) if selected.size else float("inf")


def estimate_se2(source_xy: np.ndarray, target_xy: np.ndarray) -> SE2:
    """Least-squares rigid transform mapping source points to target points."""
    source = np.asarray(source_xy, dtype=float)
    target = np.asarray(target_xy, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("source_xy and target_xy must both have shape (N, 2)")
    if source.shape[0] < 2:
        raise ValueError("at least two point pairs are required")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u_matrix, _, vt_matrix = np.linalg.svd(covariance)
    rotation = vt_matrix.T @ u_matrix.T
    if np.linalg.det(rotation) < 0:
        vt_matrix[-1, :] *= -1
        rotation = vt_matrix.T @ u_matrix.T
    translation = target_center - rotation @ source_center
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return SE2(float(translation[0]), float(translation[1]), yaw)


def ransac_se2(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    threshold_m: float = 0.10,
    iterations: int = 500,
    min_inliers: int = 3,
    seed: int = 7,
) -> AlignmentResult:
    """Robustly estimate map alignment and refit using all inliers."""
    source = np.asarray(source_xy, dtype=float)
    target = np.asarray(target_xy, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("source_xy and target_xy must both have shape (N, 2)")
    if source.shape[0] < min_inliers:
        raise ValueError("not enough point pairs")
    generator = np.random.default_rng(seed)
    best_mask: Optional[np.ndarray] = None
    best_score = (-1, float("inf"))
    for _ in range(iterations):
        sample = generator.choice(source.shape[0], size=2, replace=False)
        if np.linalg.norm(source[sample[0]] - source[sample[1]]) < 1e-6:
            continue
        transform = estimate_se2(source[sample], target[sample])
        residuals = np.linalg.norm(transform.apply(source) - target, axis=1)
        inliers = residuals <= threshold_m
        score = (int(inliers.sum()), float(residuals[inliers].mean()) if inliers.any() else float("inf"))
        if score[0] > best_score[0] or (score[0] == best_score[0] and score[1] < best_score[1]):
            best_score = score
            best_mask = inliers
    if best_mask is None or int(best_mask.sum()) < min_inliers:
        raise RuntimeError("RANSAC could not find a valid map alignment")
    transform = estimate_se2(source[best_mask], target[best_mask])
    residuals = np.linalg.norm(transform.apply(source) - target, axis=1)
    final_mask = residuals <= threshold_m
    return AlignmentResult(transform, final_mask, residuals)

