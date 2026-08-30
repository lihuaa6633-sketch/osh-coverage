"""Occupancy-grid primitives used by mapping, planning, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np


GridIndex = tuple[int, int]


def _disk_offsets(radius_cells: int) -> list[GridIndex]:
    radius_cells = max(0, int(radius_cells))
    return [
        (dr, dc)
        for dr in range(-radius_cells, radius_cells + 1)
        for dc in range(-radius_cells, radius_cells + 1)
        if dr * dr + dc * dc <= radius_cells * radius_cells
    ]


def connected_components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    """Return boolean masks for connected true-valued regions."""
    source = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(source, dtype=bool)
    components: list[np.ndarray] = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        neighbors += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    rows, cols = source.shape
    for start_r, start_c in np.argwhere(source & ~visited):
        if visited[start_r, start_c]:
            continue
        component = np.zeros_like(source, dtype=bool)
        stack = [(int(start_r), int(start_c))]
        visited[start_r, start_c] = True
        while stack:
            r, c = stack.pop()
            component[r, c] = True
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and source[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        components.append(component)
    components.sort(key=lambda value: int(value.sum()), reverse=True)
    return components


def line_cells(start: GridIndex, goal: GridIndex) -> list[GridIndex]:
    """Bresenham rasterization including both endpoints."""
    r0, c0 = start
    r1, c1 = goal
    dc = abs(c1 - c0)
    dr = -abs(r1 - r0)
    step_c = 1 if c0 < c1 else -1
    step_r = 1 if r0 < r1 else -1
    error = dc + dr
    result: list[GridIndex] = []
    while True:
        result.append((r0, c0))
        if r0 == r1 and c0 == c1:
            return result
        twice_error = 2 * error
        if twice_error >= dr:
            error += dr
            c0 += step_c
        if twice_error <= dc:
            error += dc
            r0 += step_r


@dataclass
class GridMap:
    """A metric occupancy grid; ``occupied=True`` means unavailable."""

    occupied: np.ndarray
    resolution: float = 0.05
    origin_x: float = 0.0
    origin_y: float = 0.0
    frame_id: str = "map"

    def __post_init__(self) -> None:
        self.occupied = np.asarray(self.occupied, dtype=bool)
        if self.occupied.ndim != 2:
            raise ValueError("occupied must be a 2-D array")
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")

    @property
    def shape(self) -> tuple[int, int]:
        return self.occupied.shape

    @property
    def free(self) -> np.ndarray:
        return ~self.occupied

    @classmethod
    def from_occupancy_values(
        cls,
        values: Sequence[int],
        width: int,
        height: int,
        resolution: float,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
        occupied_threshold: int = 50,
        unknown_is_occupied: bool = True,
        frame_id: str = "map",
    ) -> "GridMap":
        raw = np.asarray(values, dtype=np.int16).reshape((height, width))
        occupied = raw >= occupied_threshold
        if unknown_is_occupied:
            occupied |= raw < 0
        return cls(occupied, resolution, origin_x, origin_y, frame_id)

    def world_to_grid(self, x: float, y: float, clip: bool = False) -> GridIndex:
        col = int(math.floor((x - self.origin_x) / self.resolution))
        row = int(math.floor((y - self.origin_y) / self.resolution))
        if clip:
            row = min(max(row, 0), self.shape[0] - 1)
            col = min(max(col, 0), self.shape[1] - 1)
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        return (
            self.origin_x + (float(col) + 0.5) * self.resolution,
            self.origin_y + (float(row) + 0.5) * self.resolution,
        )

    def contains(self, index: GridIndex) -> bool:
        r, c = index
        return 0 <= r < self.shape[0] and 0 <= c < self.shape[1]

    def inflate(self, radius_m: float) -> "GridMap":
        radius_cells = int(math.ceil(max(0.0, radius_m) / self.resolution))
        if radius_cells == 0:
            return GridMap(self.occupied.copy(), self.resolution, self.origin_x, self.origin_y, self.frame_id)
        inflated = self.occupied.copy()
        occupied_indices = np.argwhere(self.occupied)
        rows, cols = self.shape
        for dr, dc in _disk_offsets(radius_cells):
            shifted_r = occupied_indices[:, 0] + dr
            shifted_c = occupied_indices[:, 1] + dc
            valid = (shifted_r >= 0) & (shifted_r < rows) & (shifted_c >= 0) & (shifted_c < cols)
            inflated[shifted_r[valid], shifted_c[valid]] = True
        return GridMap(inflated, self.resolution, self.origin_x, self.origin_y, self.frame_id)

    def reachable_mask(self, start: GridIndex, allowed: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.contains(start):
            raise ValueError("start is outside the grid")
        traversable = self.free.copy()
        if allowed is not None:
            traversable &= np.asarray(allowed, dtype=bool)
        if not traversable[start]:
            raise ValueError("start is not traversable")
        result = np.zeros_like(traversable, dtype=bool)
        stack = [start]
        result[start] = True
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.shape[0] and 0 <= nc < self.shape[1] and traversable[nr, nc] and not result[nr, nc]:
                    result[nr, nc] = True
                    stack.append((nr, nc))
        return result

    def astar(
        self,
        start: GridIndex,
        goal: GridIndex,
        allowed: Optional[np.ndarray] = None,
    ) -> list[GridIndex]:
        """Eight-connected A* with diagonal corner-cut prevention."""
        if start == goal:
            return [start]
        traversable = self.free if allowed is None else self.free & np.asarray(allowed, dtype=bool)
        if not self.contains(start) or not self.contains(goal) or not traversable[start] or not traversable[goal]:
            return []
        moves = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]
        queue: list[tuple[float, float, GridIndex]] = [(0.0, 0.0, start)]
        came_from: dict[GridIndex, GridIndex] = {}
        g_score = {start: 0.0}
        while queue:
            _, current_cost, current = heapq.heappop(queue)
            if current == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(came_from[path[-1]])
                path.reverse()
                return path
            if current_cost > g_score.get(current, float("inf")):
                continue
            r, c = current
            for dr, dc, move_cost in moves:
                neighbor = (r + dr, c + dc)
                if not self.contains(neighbor) or not traversable[neighbor]:
                    continue
                if dr and dc and (not traversable[r + dr, c] or not traversable[r, c + dc]):
                    continue
                tentative = current_cost + move_cost
                if tentative >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                heuristic = math.hypot(goal[0] - neighbor[0], goal[1] - neighbor[1])
                heapq.heappush(queue, (tentative + heuristic, tentative, neighbor))
        return []

    def nearest_free(self, index: GridIndex, max_radius_cells: int = 20) -> Optional[GridIndex]:
        if self.contains(index) and self.free[index]:
            return index
        r0, c0 = index
        for radius in range(1, max_radius_cells + 1):
            candidates: list[GridIndex] = []
            for r in range(r0 - radius, r0 + radius + 1):
                candidates.extend(((r, c0 - radius), (r, c0 + radius)))
            for c in range(c0 - radius + 1, c0 + radius):
                candidates.extend(((r0 - radius, c), (r0 + radius, c)))
            valid = [candidate for candidate in candidates if self.contains(candidate) and self.free[candidate]]
            if valid:
                return min(valid, key=lambda value: (value[0] - r0) ** 2 + (value[1] - c0) ** 2)
        return None

