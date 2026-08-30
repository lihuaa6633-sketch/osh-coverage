import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sphere_target_registration.models import PointCloud, RigidTransform
from sphere_target_registration.pipeline import run_registration
from sphere_target_registration.pointcloud_io import read_point_cloud, write_point_cloud
from sphere_target_registration.registration import estimate_rigid_transform
from sphere_target_registration.sphere import fit_sphere_ransac


def rotation_xyz(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return z @ y @ x


def noisy_sphere(rng, center, radius, count=500, outlier_count=100):
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    directions = directions[directions[:, 0] > -0.2]
    radial_noise = rng.normal(scale=0.0005, size=len(directions))
    surface = center + directions * (radius + radial_noise)[:, None]
    outliers = center + rng.uniform(-0.13, 0.13, size=(outlier_count, 3))
    return np.vstack((surface, outliers))


class SphereFitTests(unittest.TestCase):
    def test_robust_fit_with_partial_surface_and_outliers(self):
        rng = np.random.default_rng(15)
        center = np.array([1.2, -0.7, 0.4])
        points = noisy_sphere(rng, center, 0.075)
        fit = fit_sphere_ransac(
            points,
            distance_threshold=0.002,
            expected_radius=0.075,
            radius_tolerance=0.006,
            max_iterations=1500,
            min_inliers=100,
            min_inlier_ratio=0.5,
            random_seed=12,
        )
        np.testing.assert_allclose(fit.center, center, atol=4e-4)
        self.assertAlmostEqual(fit.radius, 0.075, delta=2e-4)
        self.assertLess(fit.rmse, 0.001)


class RegistrationTests(unittest.TestCase):
    def test_recovers_rigid_transform(self):
        source = np.array(
            [[-0.4, 0.3, 0.2], [1.3, -0.2, 0.5], [0.1, 1.4, -0.3], [0.8, 0.6, 1.1]]
        )
        rotation = rotation_xyz(0.15, -0.22, 0.63)
        translation = np.array([3.0, -1.2, 0.7])
        target = source @ rotation.T + translation
        transform = estimate_rigid_transform(
            source,
            target,
            maximum_pair_distance_error=1e-8,
        )
        np.testing.assert_allclose(transform.rotation, rotation, atol=1e-12)
        np.testing.assert_allclose(transform.translation, translation, atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(transform.quaternion_xyzw), 1.0, places=12)
        self.assertLess(transform.rmse, 1e-12)

    def test_rejects_collinear_targets(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "collinear"):
            estimate_rigid_transform(points, points)


class PipelineTests(unittest.TestCase):
    def test_end_to_end_files_and_report(self):
        rng = np.random.default_rng(20)
        reference_centers = np.array(
            [[0.0, 0.0, 0.5], [2.0, 0.1, 0.7], [0.3, 1.7, 0.4]]
        )
        true_transform = RigidTransform(
            rotation_xyz(-0.08, 0.17, 0.42),
            np.array([4.0, -2.0, 0.3]),
        )
        measured_centers = true_transform.inverse().apply(reference_centers)
        target_clouds = [noisy_sphere(rng, center, 0.08, 450, 60) for center in measured_centers]
        background = rng.uniform(-2.0, 2.0, size=(250, 4))
        background[:, 3] = rng.uniform(0.0, 255.0, size=250)
        first_target_with_intensity = np.column_stack(
            (target_clouds[0], np.full(len(target_clouds[0]), 180.0))
        )
        scanned_environment = np.vstack((background, first_target_with_intensity))

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_point_cloud(
                root / "environment.pcd",
                PointCloud(scanned_environment, ("x", "y", "z", "intensity")),
            )
            targets = []
            for index, (measured, reference, target_cloud) in enumerate(
                zip(measured_centers, reference_centers, target_clouds), start=1
            ):
                target = {
                    "id": f"T{index}",
                    "reference_center": reference.tolist(),
                    "expected_radius": 0.08,
                    "radius_tolerance": 0.006,
                }
                if index == 1:
                    target["roi"] = {
                        "min": (measured - 0.14).tolist(),
                        "max": (measured + 0.14).tolist(),
                    }
                else:
                    target_path = root / f"target_{index}.ply"
                    write_point_cloud(target_path, PointCloud(target_cloud))
                    target["points"] = target_path.name
                targets.append(target)

            config = {
                "environment_cloud": "environment.pcd",
                "output_cloud": "registered.pcd",
                "report": "report.json",
                "random_seed": 4,
                "fitting": {
                    "distance_threshold": 0.002,
                    "max_iterations": 1200,
                    "min_inliers": 100,
                    "min_inlier_ratio": 0.5,
                },
                "registration": {"maximum_pair_distance_error": 0.005},
                "targets": targets,
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            report = run_registration(config_path)

            output = read_point_cloud(root / "registered.pcd")
            expected_xyz = true_transform.apply(scanned_environment[:, :3])
            np.testing.assert_allclose(output.xyz, expected_xyz, atol=0.0015)
            np.testing.assert_allclose(output.data[:, 3], scanned_environment[:, 3], atol=1e-7)
            self.assertLess(report["registration"]["rmse"], 0.001)
            self.assertTrue((root / "report.json").is_file())


if __name__ == "__main__":
    unittest.main()
