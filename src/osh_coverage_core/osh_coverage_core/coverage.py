"""Actual-trajectory coverage accounting and deterministic repair state."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from .grid import GridMap, connected_components, line_cells


@dataclass
class ResidualRegion:
    region_id: str
    mask: np.ndarray
    area_m2: float
    centroid_xy: tuple[float, float]
    retry_count: int = 0
    state: str = "pending"


class CoverageMonitor:
    """Rasterize the swept tool footprint from measured, not planned, poses."""

    def __init__(self, grid: GridMap, reachable_mask: np.ndarray, working_width_m: float = 0.60):
        self.grid = grid
        self.reachable_mask = np.asarray(reachable_mask, dtype=bool)
        if self.reachable_mask.shape != grid.shape:
            raise ValueError("reachable_mask shape does not match grid")
        if working_width_m <= 0:
            raise ValueError("working_width_m must be positive")
        self.working_width_m = working_width_m
        self.coverage_count = np.zeros(grid.shape, dtype=np.uint16)
        self._last_visit_step = np.full(grid.shape, -1_000_000, dtype=np.int64)
        radius_cells = max(0, int(math.ceil((self.working_width_m / 2.0) / self.grid.resolution)))
        self._revisit_gap_steps = max(2, 2 * radius_cells + 2)
        self._step_index = 0
        self._previous_xy: Optional[tuple[float, float]] = None

    @property
    def covered_mask(self) -> np.ndarray:
        return (self.coverage_count > 0) & self.reachable_mask

    @property
    def coverage_ratio(self) -> float:
        denominator = int(self.reachable_mask.sum())
        return float(self.covered_mask.sum() / denominator) if denominator else 1.0

    @property
    def overlap_ratio(self) -> float:
        covered = self.covered_mask
        denominator = int(covered.sum())
        return float(((self.coverage_count > 1) & covered).sum() / denominator) if denominator else 0.0

    def reset(self) -> None:
        self.coverage_count.fill(0)
        self._last_visit_step.fill(-1_000_000)
        self._step_index = 0
        self._previous_xy = None

    def update_pose(self, x: float, y: float) -> None:
        current = (float(x), float(y))
        start = self.grid.world_to_grid(*(self._previous_xy or current), clip=True)
        goal = self.grid.world_to_grid(*current, clip=True)
        radius_cells = max(0, int(math.ceil((self.working_width_m / 2.0) / self.grid.resolution)))
        stamp = np.zeros(self.grid.shape, dtype=bool)
        rows, cols = self.grid.shape
        for row, col in line_cells(start, goal):
            for dr in range(-radius_cells, radius_cells + 1):
                for dc in range(-radius_cells, radius_cells + 1):
                    if dr * dr + dc * dc > radius_cells * radius_cells:
                        continue
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        stamp[nr, nc] = True
        valid = stamp & self.reachable_mask
        eligible = valid & ((self._step_index - self._last_visit_step) > self._revisit_gap_steps)
        values = self.coverage_count[eligible].astype(np.uint32) + 1
        self.coverage_count[eligible] = np.minimum(values, np.iinfo(np.uint16).max).astype(np.uint16)
        self._last_visit_step[valid] = self._step_index
        self._step_index += 1
        self._previous_xy = current

    def residual_regions(self, min_area_m2: float = 0.05) -> list[ResidualRegion]:
        residual = self.reachable_mask & ~self.covered_mask
        min_cells = max(1, int(math.ceil(min_area_m2 / self.grid.resolution**2)))
        result: list[ResidualRegion] = []
        for component in connected_components(residual):
            if int(component.sum()) < min_cells:
                continue
            points = np.argwhere(component)
            row, col = points.mean(axis=0)
            x, y = self.grid.grid_to_world(int(round(row)), int(round(col)))
            region_id = f"r{int(round(row))}_c{int(round(col))}"
            result.append(
                ResidualRegion(
                    region_id=region_id,
                    mask=component,
                    area_m2=float(component.sum() * self.grid.resolution**2),
                    centroid_xy=(x, y),
                )
            )
        return result


class DynamicRepairManager:
    """Retry residual regions twice, then report them as temporarily unreachable."""

    def __init__(self, max_retries: int = 2):
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self.max_retries = max_retries
        self._attempts: dict[str, int] = {}

    def register_failure(self, region: ResidualRegion) -> ResidualRegion:
        attempts = self._attempts.get(region.region_id, 0) + 1
        self._attempts[region.region_id] = attempts
        region.retry_count = attempts
        region.state = "temporarily_unreachable" if attempts > self.max_retries else "retry_pending"
        return region

    def register_success(self, region_id: str) -> None:
        self._attempts.pop(region_id, None)

    def annotate(self, regions: list[ResidualRegion]) -> list[ResidualRegion]:
        for region in regions:
            region.retry_count = self._attempts.get(region.region_id, 0)
            if region.retry_count > self.max_retries:
                region.state = "temporarily_unreachable"
        return regions
