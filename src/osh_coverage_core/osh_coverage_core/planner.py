"""High-level geometric coverage planner."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Optional, Sequence

import numpy as np

from .decomposition import (
    CoverageCell,
    PlanPoint,
    SweepVariant,
    boustrophedon_decompose,
    generate_sweep_variant,
)
from .grid import GridIndex, GridMap


@dataclass
class PlannerConfig:
    working_width_m: float = 0.60
    overlap_ratio: float = 0.10
    safety_margin_m: float = 0.15
    holonomic: bool = True
    min_cell_area_m2: float = 0.0
    axis_candidates: tuple[str, ...] = ("x", "y")
    transit_weight: float = 1.0
    heading_weight: float = 0.25
    lane_change_weight: float = 0.10

    @property
    def lane_spacing_m(self) -> float:
        if not 0.0 <= self.overlap_ratio < 1.0:
            raise ValueError("overlap_ratio must be in [0, 1)")
        return self.working_width_m * (1.0 - self.overlap_ratio)


@dataclass
class CoveragePlan:
    points: list[PlanPoint]
    cells: list[CoverageCell]
    cell_order: list[int]
    direction_order: list[int]
    sweep_axis: str
    working_length_m: float
    transit_length_m: float
    heading_change_rad: float
    lane_changes: int
    objective: float
    reachable_mask: np.ndarray

    @property
    def path_xytheta(self) -> np.ndarray:
        return np.asarray([(point.x, point.y, point.yaw) for point in self.points], dtype=float)


Scheduler = Callable[[dict[int, tuple[SweepVariant, SweepVariant]], GridIndex, GridMap], list[tuple[int, int]]]


def _append_transit(
    result: list[PlanPoint],
    path: Sequence[GridIndex],
    grid: GridMap,
) -> float:
    if not path:
        return 0.0
    length = 0.0
    for index, (row, col) in enumerate(path):
        if result and result[-1].row == row and result[-1].col == col:
            continue
        x, y = grid.grid_to_world(row, col)
        if result:
            length += math.hypot(x - result[-1].x, y - result[-1].y)
            yaw = math.atan2(y - result[-1].y, x - result[-1].x)
        elif index + 1 < len(path):
            nx, ny = grid.grid_to_world(*path[index + 1])
            yaw = math.atan2(ny - y, nx - x)
        else:
            yaw = 0.0
        result.append(PlanPoint(row, col, x, y, yaw, "transit", -1))
    return length


def nearest_neighbor_schedule(
    variants: dict[int, tuple[SweepVariant, SweepVariant]],
    start: GridIndex,
    grid: GridMap,
) -> list[tuple[int, int]]:
    remaining = set(variants)
    current = start
    order: list[tuple[int, int]] = []
    while remaining:
        best: Optional[tuple[float, int, int]] = None
        for cell_id in remaining:
            for direction, variant in enumerate(variants[cell_id]):
                distance = math.hypot(variant.entry[0] - current[0], variant.entry[1] - current[1])
                candidate = (distance, cell_id, direction)
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        _, cell_id, direction = best
        order.append((cell_id, direction))
        current = variants[cell_id][direction].exit
        remaining.remove(cell_id)
    return order


class CoveragePlanner:
    def __init__(self, grid: GridMap, config: Optional[PlannerConfig] = None):
        self.original_grid = grid
        self.config = config or PlannerConfig()
        self.grid = grid.inflate(self.config.safety_margin_m)

    def plan(
        self,
        start_xy: tuple[float, float],
        roi_mask: Optional[np.ndarray] = None,
        scheduler: Optional[Scheduler] = None,
    ) -> CoveragePlan:
        start = self.grid.world_to_grid(*start_xy, clip=True)
        nearest = self.grid.nearest_free(start)
        if nearest is None:
            raise RuntimeError("no free start location after safety inflation")
        start = nearest
        globally_reachable = self.grid.reachable_mask(start)
        coverage_target = globally_reachable.copy()
        if roi_mask is not None:
            if np.asarray(roi_mask).shape != self.grid.shape:
                raise ValueError("roi_mask shape does not match grid")
            coverage_target &= np.asarray(roi_mask, dtype=bool)
        if not coverage_target.any():
            raise ValueError("ROI has no free area reachable from the start")
        planner_scheduler = scheduler or nearest_neighbor_schedule
        candidates = [
            self._plan_axis(axis, start, coverage_target, globally_reachable, planner_scheduler)
            for axis in self.config.axis_candidates
        ]
        return min(candidates, key=lambda plan: plan.objective)

    def _plan_axis(
        self,
        axis: str,
        start: GridIndex,
        coverage_target: np.ndarray,
        transit_mask: np.ndarray,
        scheduler: Scheduler,
    ) -> CoveragePlan:
        # Never silently discard a reachable sliver: full coverage has priority.
        cells = boustrophedon_decompose(coverage_target, axis, min_cells=1)
        if not cells:
            raise RuntimeError("no coverage cells were generated")
        variants: dict[int, tuple[SweepVariant, SweepVariant]] = {}
        for cell in cells:
            forward = generate_sweep_variant(
                cell,
                self.grid,
                coverage_target,
                self.config.lane_spacing_m,
                axis,
                self.config.holonomic,
                reverse=False,
            )
            reverse = generate_sweep_variant(
                cell,
                self.grid,
                coverage_target,
                self.config.lane_spacing_m,
                axis,
                self.config.holonomic,
                reverse=True,
            )
            variants[cell.cell_id] = forward, reverse

        order = scheduler(variants, start, self.grid)
        if {cell_id for cell_id, _ in order} != set(variants):
            raise ValueError("scheduler must select every cell exactly once")
        points: list[PlanPoint] = []
        current = start
        working_length = 0.0
        transit_length = 0.0
        heading_change = 0.0
        lane_changes = 0
        for cell_id, direction in order:
            variant = variants[cell_id][direction]
            transit = self.grid.astar(current, variant.entry, allowed=transit_mask)
            if not transit:
                raise RuntimeError(f"cell {cell_id} is unreachable from the previous cell")
            transit_length += _append_transit(points, transit, self.grid)
            for point in variant.points:
                if points and points[-1].row == point.row and points[-1].col == point.col:
                    continue
                points.append(point)
            working_length += variant.path_length
            lane_changes += sum(
                point.phase == "lane_change" and (index == 0 or variant.points[index - 1].phase != "lane_change")
                for index, point in enumerate(variant.points)
            )
            current = variant.exit
        heading_change = 0.0
        for first, second in zip(points, points[1:]):
            heading_change += abs((second.yaw - first.yaw + math.pi) % (2.0 * math.pi) - math.pi)
        objective = (
            working_length
            + self.config.transit_weight * transit_length
            + self.config.heading_weight * heading_change
            + self.config.lane_change_weight * lane_changes
        )
        return CoveragePlan(
            points=points,
            cells=cells,
            cell_order=[cell_id for cell_id, _ in order],
            direction_order=[direction for _, direction in order],
            sweep_axis=axis,
            working_length_m=working_length,
            transit_length_m=transit_length,
            heading_change_rad=heading_change,
            lane_changes=lane_changes,
            objective=objective,
            reachable_mask=coverage_target,
        )
