from __future__ import annotations

import argparse
import json
import sys

from .pipeline import run_registration
from .pointcloud_io import read_point_cloud
from .sphere import fit_sphere_ransac


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sphere-target-register",
        description="Fit sphere targets and register a point cloud to prescribed coordinates.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the complete JSON-configured workflow")
    run_parser.add_argument("config", help="path to the JSON configuration")

    fit_parser = subparsers.add_parser("fit", help="fit and inspect one cropped sphere cloud")
    fit_parser.add_argument("cloud", help="target point cloud (PCD/PLY/XYZ/CSV/NPY)")
    fit_parser.add_argument("--radius", type=float, required=True, help="expected sphere radius")
    fit_parser.add_argument(
        "--radius-tolerance", type=float, required=True, help="allowed radius error"
    )
    fit_parser.add_argument(
        "--distance-threshold", type=float, required=True, help="RANSAC shell threshold"
    )
    fit_parser.add_argument("--max-iterations", type=int, default=2000)
    fit_parser.add_argument("--min-inliers", type=int, default=30)
    fit_parser.add_argument("--min-inlier-ratio", type=float, default=0.25)
    fit_parser.add_argument("--random-seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "run":
            report = run_registration(args.config)
            registration = report["registration"]
            print(f"registered cloud: {report['output_cloud']}")
            print(f"report: {report['report']}")
            print(f"registration RMSE: {registration['rmse']:.6g}")
            print("transform (scanned -> reference):")
            print(json.dumps(registration["matrix_4x4"], indent=2))
            return 0

        cloud = read_point_cloud(args.cloud)
        fit = fit_sphere_ransac(
            cloud.xyz,
            distance_threshold=args.distance_threshold,
            expected_radius=args.radius,
            radius_tolerance=args.radius_tolerance,
            max_iterations=args.max_iterations,
            min_inliers=args.min_inliers,
            min_inlier_ratio=args.min_inlier_ratio,
            random_seed=args.random_seed,
        )
        print(json.dumps(fit.to_dict(), indent=2))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
