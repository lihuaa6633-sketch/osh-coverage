"""Command-line tools for offline reproduction without ROS."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .alignment import ransac_se2
from .coverage import CoverageMonitor
from .planner import CoveragePlanner, PlannerConfig
from .rl import train_curriculum
from .scenarios import laboratory_scene


def demo_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic offline coverage demo")
    parser.add_argument("--scene", choices=("plate_shop", "long_aisle", "irregular", "train_skin"), default="plate_shop")
    parser.add_argument("--width", type=float, default=0.60, help="effective working width in metres")
    parser.add_argument("--overlap", type=float, default=0.10)
    parser.add_argument("--margin", type=float, default=0.15)
    args = parser.parse_args(argv)
    grid = laboratory_scene(args.scene)
    config = PlannerConfig(working_width_m=args.width, overlap_ratio=args.overlap, safety_margin_m=args.margin)
    planner = CoveragePlanner(grid, config)
    start = grid.grid_to_world(grid.shape[0] - 4, 3)
    plan = planner.plan(start)
    monitor = CoverageMonitor(planner.grid, plan.reachable_mask, args.width)
    for point in plan.points:
        monitor.update_pose(point.x, point.y)
    payload = {
        "scene": args.scene,
        "axis": plan.sweep_axis,
        "cells": len(plan.cells),
        "waypoints": len(plan.points),
        "working_length_m": round(plan.working_length_m, 3),
        "transit_length_m": round(plan.transit_length_m, 3),
        "heading_change_rad": round(plan.heading_change_rad, 3),
        "coverage_ratio": round(monitor.coverage_ratio, 5),
        "overlap_ratio": round(monitor.overlap_ratio, 5),
        "residual_regions": len(monitor.residual_regions()),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def train_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the masked Double DQN cell scheduler")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plain-dqn", action="store_true")
    args = parser.parse_args(argv)
    agent, rewards = train_curriculum(args.episodes, args.slots, args.seed, not args.plain_dqn)
    agent.save(args.output)
    summary = {
        "episodes": args.episodes,
        "mean_reward_last_100": float(np.mean(rewards[-100:])),
        "model": str(args.output),
        "double_dqn": not args.plain_dqn,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def align_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Estimate Airy-SLAM to Woosh-map SE(2) alignment")
    parser.add_argument("csv_file", type=Path, help="CSV columns: slam_x,slam_y,woosh_x,woosh_y")
    parser.add_argument("--threshold", type=float, default=0.10)
    args = parser.parse_args(argv)
    rows = []
    with args.csv_file.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows.append([float(row[key]) for key in ("slam_x", "slam_y", "woosh_x", "woosh_y")])
    data = np.asarray(rows, dtype=float)
    result = ransac_se2(data[:, :2], data[:, 2:], threshold_m=args.threshold)
    payload = {
        "x": result.transform.x,
        "y": result.transform.y,
        "yaw_rad": result.transform.yaw,
        "rmse_m": result.rmse_m,
        "inliers": int(result.inliers.sum()),
        "samples": int(result.inliers.size),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

