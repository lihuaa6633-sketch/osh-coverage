"""Reproducible non-learning baselines for thesis comparisons."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .grid import GridIndex, GridMap
from .rl import RoutingCandidate


def spiral_stc_grid_baseline(grid: GridMap, start: GridIndex, target_mask: np.ndarray) -> list[GridIndex]:
    """Grid STC-style depth-first traversal with clockwise neighbor preference.

    This is a raster baseline rather than the exact polygonal Spiral-STC paper
    implementation.  It visits every connected target cell and explicitly
    backtracks along the spanning tree.
    """
    target = np.asarray(target_mask, dtype=bool) & grid.free
    if not target[start]:
        raise ValueError("start must be inside target_mask for the STC baseline")
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    visited = {start}
    path = [start]
    # Each stack item stores (cell, heading index, next clockwise option).
    stack: list[list] = [[start, 1, 0]]
    while stack:
        current, heading, option_index = stack[-1]
        if option_index >= 4:
            stack.pop()
            if stack:
                path.append(stack[-1][0])
            continue
        stack[-1][2] += 1
        direction_index = (heading + 1 - option_index) % 4
        dr, dc = directions[direction_index]
        neighbor = (current[0] + dr, current[1] + dc)
        if not grid.contains(neighbor) or not target[neighbor] or neighbor in visited:
            continue
        visited.add(neighbor)
        path.append(neighbor)
        stack.append([neighbor, direction_index, 0])
    return path


def routing_cost(
    candidates: Sequence[RoutingCandidate],
    start_xy: tuple[float, float],
    order: Sequence[int],
    directions: Sequence[int],
    heading_weight: float = 0.25,
) -> float:
    by_id = {candidate.cell_id: candidate for candidate in candidates}
    current = np.asarray(start_xy, dtype=float)
    current_yaw = 0.0
    cost = 0.0
    for cell_id, direction in zip(order, directions):
        candidate = by_id[int(cell_id)]
        direction = int(direction)
        cost += float(np.linalg.norm(candidate.entry_xy[direction] - current))
        delta = abs((float(candidate.yaw[direction]) - current_yaw + math.pi) % (2.0 * math.pi) - math.pi)
        cost += heading_weight * delta + 2.0 * candidate.blocked_risk
        current = candidate.exit_xy[direction]
        current_yaw = float(candidate.yaw[direction])
    return cost


def genetic_route(
    candidates: Sequence[RoutingCandidate],
    start_xy: tuple[float, float],
    population_size: int = 80,
    generations: int = 100,
    seed: int = 7,
) -> tuple[list[int], list[int], float]:
    """Small permutation+direction GA used only as an offline baseline."""
    generator = np.random.default_rng(seed)
    cell_ids = np.asarray([candidate.cell_id for candidate in candidates], dtype=int)
    count = len(cell_ids)
    if count == 0:
        return [], [], 0.0

    def random_individual():
        return generator.permutation(cell_ids), generator.integers(0, 2, size=count, dtype=int)

    def crossover(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        if count < 2:
            return first.copy()
        left, right = sorted(generator.choice(count, size=2, replace=False))
        child = np.full(count, -1, dtype=int)
        child[left:right] = first[left:right]
        remaining = [value for value in second if value not in child]
        child[child < 0] = remaining
        return child

    population = [random_individual() for _ in range(max(4, population_size))]
    for _ in range(max(1, generations)):
        ranked = sorted(
            population,
            key=lambda item: routing_cost(candidates, start_xy, item[0], item[1]),
        )
        elite_count = max(2, len(ranked) // 5)
        elites = ranked[:elite_count]
        next_population = [(order.copy(), directions.copy()) for order, directions in elites]
        while len(next_population) < len(population):
            parent_a = elites[int(generator.integers(0, elite_count))]
            parent_b = elites[int(generator.integers(0, elite_count))]
            order = crossover(parent_a[0], parent_b[0])
            directions = np.where(generator.random(count) < 0.5, parent_a[1], parent_b[1]).astype(int)
            if count > 1 and generator.random() < 0.25:
                first, second = generator.choice(count, size=2, replace=False)
                order[first], order[second] = order[second], order[first]
            if generator.random() < 0.25:
                directions[int(generator.integers(0, count))] ^= 1
            next_population.append((order, directions))
        population = next_population
    best_order, best_directions = min(
        population,
        key=lambda item: routing_cost(candidates, start_xy, item[0], item[1]),
    )
    cost = routing_cost(candidates, start_xy, best_order, best_directions)
    return best_order.tolist(), best_directions.tolist(), float(cost)

