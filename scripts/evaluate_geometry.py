from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from osh_coverage_core.coverage import CoverageMonitor
from osh_coverage_core.planner import CoveragePlanner, PlannerConfig
from osh_coverage_core.scenarios import laboratory_scene


def evaluate(scene: str, holonomic: bool) -> dict[str, float | str | bool]:
    grid = laboratory_scene(scene)
    planner = CoveragePlanner(
        grid,
        PlannerConfig(
            working_width_m=0.60,
            overlap_ratio=0.10,
            safety_margin_m=0.15,
            holonomic=holonomic,
        ),
    )
    start = grid.grid_to_world(grid.shape[0] - 4, 3)
    plan = planner.plan(start)
    monitor = CoverageMonitor(planner.grid, plan.reachable_mask, 0.60)
    for point in plan.points:
        monitor.update_pose(point.x, point.y)
    return {
        "scene": scene,
        "holonomic": holonomic,
        "axis": plan.sweep_axis,
        "cells": len(plan.cells),
        "waypoints": len(plan.points),
        "coverage_ratio": monitor.coverage_ratio,
        "overlap_ratio": monitor.overlap_ratio,
        "working_length_m": plan.working_length_m,
        "transit_length_m": plan.transit_length_m,
        "heading_change_rad": plan.heading_change_rad,
        "lane_changes": plan.lane_changes,
        "objective": plan.objective,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [
        evaluate(scene, holonomic)
        for scene in ("plate_shop", "long_aisle", "irregular", "train_skin")
        for holonomic in (False, True)
    ]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

