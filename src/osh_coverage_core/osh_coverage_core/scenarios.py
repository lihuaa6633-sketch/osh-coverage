"""Deterministic laboratory scenes and procedural routing problems."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .grid import GridMap


def _add_rect(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    rows, cols = occupied.shape
    occupied[max(0, top) : min(rows, top + height), max(0, left) : min(cols, left + width)] = True


def laboratory_scene(kind: str = "plate_shop", resolution: float = 0.10) -> GridMap:
    """Return one of the three thesis-scale laboratory abstractions."""
    occupied = np.zeros((80, 120), dtype=bool)
    occupied[[0, -1], :] = True
    occupied[:, [0, -1]] = True
    if kind == "plate_shop":
        _add_rect(occupied, 15, 22, 12, 26)
        _add_rect(occupied, 48, 68, 14, 32)
        _add_rect(occupied, 31, 54, 8, 12)
    elif kind == "long_aisle":
        _add_rect(occupied, 10, 20, 60, 10)
        _add_rect(occupied, 10, 52, 45, 10)
        _add_rect(occupied, 25, 84, 45, 10)
        _add_rect(occupied, 36, 30, 8, 22)
    elif kind == "irregular":
        _add_rect(occupied, 12, 14, 17, 22)
        _add_rect(occupied, 42, 18, 26, 16)
        _add_rect(occupied, 18, 62, 12, 32)
        _add_rect(occupied, 43, 65, 24, 34)
        occupied[:18, 100:] = True
        occupied[62:, 104:] = True
    elif kind == "train_skin":
        _add_rect(occupied, 29, 12, 22, 96)
        _add_rect(occupied, 8, 42, 12, 8)
        _add_rect(occupied, 60, 72, 12, 8)
    else:
        raise ValueError(f"unknown scene kind: {kind}")
    return GridMap(occupied, resolution=resolution)


def random_warehouse_scene(
    seed: int,
    rows: int = 80,
    cols: int = 120,
    obstacle_count: int = 8,
    resolution: float = 0.10,
) -> GridMap:
    generator = np.random.default_rng(seed)
    occupied = np.zeros((rows, cols), dtype=bool)
    occupied[[0, -1], :] = True
    occupied[:, [0, -1]] = True
    for _ in range(obstacle_count):
        height = int(generator.integers(5, max(6, rows // 4)))
        width = int(generator.integers(5, max(6, cols // 5)))
        top = int(generator.integers(2, max(3, rows - height - 2)))
        left = int(generator.integers(2, max(3, cols - width - 2)))
        _add_rect(occupied, top, left, height, width)
    return GridMap(occupied, resolution=resolution)

