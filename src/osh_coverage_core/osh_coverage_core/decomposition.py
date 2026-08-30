"""Grid Boustrophedon decomposition and sweep-path generation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional

import numpy as np

from .grid import GridIndex, GridMap


@dataclass
class CoverageCell:
    cell_id: int
    mask: np.ndarray

    @property
    def area_cells(self) -> int:
        return int(self.mask.sum())

    @property
    def centroid_rc(self) -> tuple[float, float]:
        points = np.argwhere(self.mask)
        return float(points[:, 0].mean()), float(points[:, 1].mean())


@dataclass
class PlanPoint:
    row: int
    col: int
    x: float
    y: float
    yaw: float
    phase: str
    cell_id: int = -1


@dataclass
class SweepVariant:
    cell_id: int
    direction: int
    points: list[PlanPoint]
    path_length: float
    heading_change: float

    @property
    def entry(self) -> GridIndex:
        return self.points[0].row, self.points[0].col

    @property
    def exit(self) -> GridIndex:
        return self.points[-1].row, self.points[-1].col


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(values)
    if indices.size == 0:
        return []
    result: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for raw in indices[1:]:
        value = int(raw)
        if value != previous + 1:
            result.append((start, previous))
            start = value
        previous = value
    result.append((start, previous))
    return result


def _overlap(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return max(first[0], second[0]) <= min(first[1], second[1])


def _decompose_columns(free_mask: np.ndarray) -> list[np.ndarray]:
    """Sweep columns and split cells at connectivity critical events."""
    rows, cols = free_mask.shape
    masks: dict[int, np.ndarray] = {}
    next_cell_id = 0
    previous: list[tuple[tuple[int, int], int]] = []
    for col in range(cols):
        current_runs = _runs(free_mask[:, col])
        previous_to_current: dict[int, list[int]] = {index: [] for index in range(len(previous))}
        current_to_previous: dict[int, list[int]] = {index: [] for index in range(len(current_runs))}
        for previous_index, (previous_run, _) in enumerate(previous):
            for current_index, current_run in enumerate(current_runs):
                if _overlap(previous_run, current_run):
                    previous_to_current[previous_index].append(current_index)
                    current_to_previous[current_index].append(previous_index)
        assigned: list[tuple[tuple[int, int], int]] = []
        for current_index, current_run in enumerate(current_runs):
            overlaps = current_to_previous[current_index]
            continuation = (
                len(overlaps) == 1
                and len(previous_to_current[overlaps[0]]) == 1
            )
            if continuation:
                cell_id = previous[overlaps[0]][1]
            else:
                cell_id = next_cell_id
                next_cell_id += 1
                masks[cell_id] = np.zeros_like(free_mask, dtype=bool)
            masks[cell_id][current_run[0] : current_run[1] + 1, col] = True
            assigned.append((current_run, cell_id))
        previous = assigned
    return [masks[key] for key in sorted(masks) if masks[key].any()]


def boustrophedon_decompose(free_mask: np.ndarray, sweep_axis: str = "x", min_cells: int = 1) -> list[CoverageCell]:
    """Approximate exact BCD on a rectilinear occupancy grid.

    ``sweep_axis='x'`` produces lanes parallel to the x/world-column axis;
    ``'y'`` transposes the grid before decomposition.
    """
    source = np.asarray(free_mask, dtype=bool)
    if sweep_axis not in {"x", "y"}:
        raise ValueError("sweep_axis must be 'x' or 'y'")
    oriented = source if sweep_axis == "x" else source.T
    raw_masks = _decompose_columns(oriented)
    masks = [mask if sweep_axis == "x" else mask.T for mask in raw_masks]
    cells = [CoverageCell(index, mask) for index, mask in enumerate(masks) if int(mask.sum()) >= min_cells]
    cells.sort(key=lambda cell: cell.area_cells, reverse=True)
    for index, cell in enumerate(cells):
        cell.cell_id = index
    return cells


def _append_grid_path(
    result: list[PlanPoint],
    indices: Iterable[GridIndex],
    grid: GridMap,
    yaw: float,
    phase: str,
    cell_id: int,
) -> None:
    for row, col in indices:
        if result and result[-1].row == row and result[-1].col == col:
            continue
        x, y = grid.grid_to_world(row, col)
        result.append(PlanPoint(row, col, x, y, yaw, phase, cell_id))


def _path_metrics(points: list[PlanPoint]) -> tuple[float, float]:
    length = 0.0
    heading_change = 0.0
    for first, second in zip(points, points[1:]):
        length += math.hypot(second.x - first.x, second.y - first.y)
        delta = (second.yaw - first.yaw + math.pi) % (2.0 * math.pi) - math.pi
        heading_change += abs(delta)
    return length, heading_change


def generate_sweep_variant(
    cell: CoverageCell,
    grid: GridMap,
    traversable: np.ndarray,
    lane_spacing_m: float,
    sweep_axis: str,
    holonomic: bool,
    reverse: bool = False,
) -> SweepVariant:
    """Generate collision-checked lanes and A* connectors for a cell."""
    spacing = max(1, int(round(lane_spacing_m / grid.resolution)))
    source = cell.mask if sweep_axis == "x" else cell.mask.T
    row_indices = np.flatnonzero(source.any(axis=1))
    if row_indices.size == 0:
        raise ValueError("cannot sweep an empty cell")
    selected = list(range(int(row_indices.min()), int(row_indices.max()) + 1, spacing))
    if selected[-1] != int(row_indices.max()) and int(row_indices.max()) - selected[-1] > spacing // 2:
        selected.append(int(row_indices.max()))
    lane_segments: list[tuple[GridIndex, GridIndex]] = []
    for lane_index in selected:
        for start, end in _runs(source[lane_index]):
            if sweep_axis == "x":
                lane_segments.append(((lane_index, start), (lane_index, end)))
            else:
                lane_segments.append(((start, lane_index), (end, lane_index)))
    if not lane_segments:
        center = tuple(np.rint(cell.centroid_rc).astype(int))
        lane_segments = [(center, center)]
    ordered: list[tuple[GridIndex, GridIndex]] = []
    for index, segment in enumerate(lane_segments):
        ordered.append(segment if index % 2 == 0 else (segment[1], segment[0]))
    if reverse:
        ordered = [(end, start) for start, end in reversed(ordered)]

    fixed_yaw = 0.0 if sweep_axis == "x" else math.pi / 2.0
    points: list[PlanPoint] = []
    previous_end: Optional[GridIndex] = None
    for start, end in ordered:
        travel_yaw = fixed_yaw if holonomic else math.atan2(end[0] - start[0], end[1] - start[1])
        if previous_end is not None and previous_end != start:
            connector = grid.astar(previous_end, start, allowed=traversable)
            if not connector:
                raise RuntimeError(f"cell {cell.cell_id} contains disconnected lanes")
            connector_yaw = fixed_yaw if holonomic else math.atan2(start[0] - previous_end[0], start[1] - previous_end[1])
            _append_grid_path(points, connector, grid, connector_yaw, "lane_change", cell.cell_id)
        segment_path = grid.astar(start, end, allowed=traversable & cell.mask)
        if not segment_path:
            segment_path = [start, end] if start != end else [start]
        _append_grid_path(points, segment_path, grid, travel_yaw, "work", cell.cell_id)
        previous_end = end
    length, heading_change = _path_metrics(points)
    return SweepVariant(cell.cell_id, 1 if reverse else 0, points, length, heading_change)

