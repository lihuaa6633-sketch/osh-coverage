from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .pointcloud_io import read_point_cloud, write_point_cloud
from .registration import estimate_rigid_transform, pairwise_distance_errors
from .sphere import fit_sphere_ransac


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _vector3(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite [x, y, z] array")
    return result


def run_registration(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    base = config_path.parent
    if "environment_cloud" not in config:
        raise ValueError("config is missing environment_cloud")
    environment_path = _resolve(base, str(config["environment_cloud"]))
    environment = read_point_cloud(environment_path)

    target_configs = config.get("targets")
    if not isinstance(target_configs, list) or len(target_configs) < 3:
        raise ValueError("config.targets must contain at least three targets")
    target_ids = [str(item.get("id", "")) for item in target_configs]
    if any(not target_id for target_id in target_ids) or len(set(target_ids)) != len(target_ids):
        raise ValueError("every target needs a unique, non-empty id")

    fitting_defaults = config.get("fitting", {})
    if "distance_threshold" not in fitting_defaults:
        raise ValueError("config.fitting.distance_threshold is required")
    measured_centers = []
    reference_centers = []
    target_reports = []

    for target_index, target_config in enumerate(target_configs):
        target_id = target_ids[target_index]
        if "points" in target_config and "roi" in target_config:
            raise ValueError(f"target {target_id}: use either points or roi, not both")
        if "points" in target_config:
            target_path = _resolve(base, str(target_config["points"]))
            target_points = read_point_cloud(target_path).xyz
            point_source: dict[str, Any] = {"points": str(target_path)}
        elif "roi" in target_config:
            roi = target_config["roi"]
            lower = _vector3(roi.get("min"), f"target {target_id} roi.min")
            upper = _vector3(roi.get("max"), f"target {target_id} roi.max")
            if np.any(upper <= lower):
                raise ValueError(f"target {target_id}: each roi.max value must exceed roi.min")
            xyz = environment.xyz
            mask = np.all((xyz >= lower) & (xyz <= upper), axis=1)
            target_points = xyz[mask]
            point_source = {"roi": {"min": lower.tolist(), "max": upper.tolist()}}
        else:
            raise ValueError(f"target {target_id}: either points or roi is required")

        fit_options = dict(fitting_defaults)
        fit_options.update(target_config.get("fitting", {}))
        fit_options["expected_radius"] = float(target_config["expected_radius"])
        fit_options["radius_tolerance"] = float(target_config["radius_tolerance"])
        fit_options.setdefault("random_seed", int(config.get("random_seed", 0)) + target_index)
        fit = fit_sphere_ransac(target_points, **fit_options)
        reference_center = _vector3(
            target_config.get("reference_center"),
            f"target {target_id} reference_center",
        )
        measured_centers.append(fit.center)
        reference_centers.append(reference_center)
        target_reports.append(
            {
                "id": target_id,
                "source": point_source,
                "reference_center": reference_center.tolist(),
                "fit": fit.to_dict(),
                "warnings": (
                    [
                        "visible sphere surface has weak angular coverage; "
                        "center accuracy may be poor"
                    ]
                    if fit.direction_eigenvalue_ratio < 0.02
                    else []
                ),
            }
        )

    measured_array = np.asarray(measured_centers)
    reference_array = np.asarray(reference_centers)
    registration_config = config.get("registration", {})
    maximum_pair_error = registration_config.get("maximum_pair_distance_error")
    if maximum_pair_error is None:
        maximum_pair_error = 5.0 * float(fitting_defaults["distance_threshold"])
    transform = estimate_rigid_transform(
        measured_array,
        reference_array,
        minimum_geometry_score=float(registration_config.get("minimum_geometry_score", 1e-3)),
        maximum_pair_distance_error=float(maximum_pair_error),
    )
    registered_centers = transform.apply(measured_array)
    center_errors = np.linalg.norm(registered_centers - reference_array, axis=1)
    pair_errors = pairwise_distance_errors(measured_array, reference_array)

    output_cloud = _resolve(base, str(config.get("output_cloud", "registered_cloud.pcd")))
    report_path = _resolve(base, str(config.get("report", "registration_report.json")))
    write_point_cloud(output_cloud, environment.transformed(transform))
    report: dict[str, Any] = {
        "method": "sphere-target landmark rigid registration (6-DoF, no scale)",
        "source_cloud": str(environment_path),
        "output_cloud": str(output_cloud),
        "point_count": len(environment.data),
        "targets": target_reports,
        "registration": {
            **transform.to_dict(),
            "target_ids_in_correspondence_order": target_ids,
            "registered_centers": registered_centers.tolist(),
            "center_errors": center_errors.tolist(),
            "pair_distance_errors": pair_errors.tolist(),
            "maximum_pair_distance_error_limit": float(maximum_pair_error),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report["report"] = str(report_path)
    return report
