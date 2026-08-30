from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from osh_coverage_core.alignment import SE2, ransac_se2
from osh_coverage_core.baselines import genetic_route, spiral_stc_grid_baseline
from osh_coverage_core.coverage import CoverageMonitor, DynamicRepairManager
from osh_coverage_core.grid import GridMap, connected_components
from osh_coverage_core.planner import CoveragePlanner, PlannerConfig
from osh_coverage_core.rl import MaskedDoubleDQN, make_ddqn_scheduler, random_routing_problem
from osh_coverage_core.scenarios import laboratory_scene


class GridTests(unittest.TestCase):
    def test_inflation_and_reachability(self):
        occupied = np.zeros((12, 12), dtype=bool)
        occupied[6, 6] = True
        grid = GridMap(occupied, resolution=0.1)
        inflated = grid.inflate(0.2)
        self.assertTrue(inflated.occupied[6, 6])
        self.assertTrue(inflated.occupied[6, 8])
        self.assertFalse(inflated.occupied[6, 9])
        reachable = inflated.reachable_mask((1, 1))
        self.assertTrue(reachable[10, 10])

    def test_astar_does_not_cut_obstacle_corners(self):
        occupied = np.zeros((8, 8), dtype=bool)
        occupied[3, 3] = True
        occupied[3, 4] = True
        path = GridMap(occupied, 0.1).astar((1, 1), (6, 6))
        self.assertTrue(path)
        self.assertTrue(all(not occupied[index] for index in path))


class PlannerTests(unittest.TestCase):
    def _make_plan(self, holonomic: bool):
        grid = laboratory_scene("plate_shop", resolution=0.1)
        config = PlannerConfig(
            working_width_m=0.6,
            overlap_ratio=0.1,
            safety_margin_m=0.1,
            holonomic=holonomic,
        )
        planner = CoveragePlanner(grid, config)
        start = grid.grid_to_world(grid.shape[0] - 4, 3)
        return planner, planner.plan(start)

    def test_plan_is_collision_free_and_covers_reachable_area(self):
        planner, plan = self._make_plan(True)
        self.assertGreater(len(plan.cells), 1)
        self.assertEqual(len(plan.cell_order), len(set(plan.cell_order)))
        for point in plan.points:
            self.assertTrue(planner.grid.free[point.row, point.col])
            self.assertTrue(plan.reachable_mask[point.row, point.col])
        monitor = CoverageMonitor(planner.grid, plan.reachable_mask, 0.6)
        for point in plan.points:
            monitor.update_pose(point.x, point.y)
        self.assertGreaterEqual(monitor.coverage_ratio, 0.98)

    def test_holonomic_mode_reduces_heading_changes(self):
        _, holonomic = self._make_plan(True)
        _, traditional = self._make_plan(False)
        self.assertLess(holonomic.heading_change_rad, traditional.heading_change_rad)

    def test_random_agent_scheduler_still_selects_every_cell(self):
        grid = laboratory_scene("irregular", resolution=0.1)
        config = PlannerConfig(safety_margin_m=0.1)
        planner = CoveragePlanner(grid, config)
        start = grid.grid_to_world(grid.shape[0] - 4, 3)
        agent = MaskedDoubleDQN(state_size=6 * 8 + 3, action_size=12, seed=3)
        plan = planner.plan(start, scheduler=make_ddqn_scheduler(agent, 6))
        self.assertEqual(set(plan.cell_order), {cell.cell_id for cell in plan.cells})

    def test_start_may_be_outside_residual_roi(self):
        occupied = np.zeros((30, 40), dtype=bool)
        occupied[[0, -1], :] = True
        occupied[:, [0, -1]] = True
        grid = GridMap(occupied, 0.1)
        roi = np.zeros(grid.shape, dtype=bool)
        roi[10:20, 25:35] = True
        planner = CoveragePlanner(grid, PlannerConfig(safety_margin_m=0.0))
        start = grid.grid_to_world(4, 4)
        plan = planner.plan(start, roi_mask=roi)
        self.assertTrue(np.array_equal(plan.reachable_mask, roi))
        self.assertEqual({cell.cell_id for cell in plan.cells}, set(plan.cell_order))


class AlignmentTests(unittest.TestCase):
    def test_ransac_recovers_transform_with_outlier(self):
        generator = np.random.default_rng(4)
        source = generator.uniform(-3.0, 3.0, (30, 2))
        expected = SE2(1.25, -0.70, 0.35)
        target = expected.apply(source) + generator.normal(0.0, 0.005, source.shape)
        target[2] += np.asarray((2.0, -2.0))
        result = ransac_se2(source, target, threshold_m=0.05, iterations=300)
        self.assertAlmostEqual(result.transform.x, expected.x, delta=0.02)
        self.assertAlmostEqual(result.transform.y, expected.y, delta=0.02)
        self.assertAlmostEqual(result.transform.yaw, expected.yaw, delta=0.01)
        self.assertLess(result.rmse_m, 0.02)
        self.assertFalse(result.inliers[2])


class CoverageTests(unittest.TestCase):
    def test_residual_detection_and_retry_state(self):
        occupied = np.zeros((20, 20), dtype=bool)
        grid = GridMap(occupied, 0.1)
        monitor = CoverageMonitor(grid, grid.free, working_width_m=0.2)
        for col in range(2, 18):
            x, y = grid.grid_to_world(10, col)
            monitor.update_pose(x, y)
        regions = monitor.residual_regions(min_area_m2=0.05)
        self.assertGreaterEqual(len(regions), 1)
        manager = DynamicRepairManager(max_retries=2)
        region = regions[0]
        self.assertEqual(manager.register_failure(region).state, "retry_pending")
        self.assertEqual(manager.register_failure(region).state, "retry_pending")
        self.assertEqual(manager.register_failure(region).state, "temporarily_unreachable")


class ReinforcementLearningTests(unittest.TestCase):
    def test_action_mask_and_training_step(self):
        environment = random_routing_problem(10, cell_count=6, candidate_slots=4)
        state, mask = environment.reset()
        agent = MaskedDoubleDQN(environment.state_size, environment.action_size, hidden_size=24, seed=10)
        for _ in range(80):
            action = agent.select_action(state, mask, epsilon=0.5)
            self.assertTrue(mask[action])
            next_state, next_mask, reward, done, _ = environment.step(action)
            agent.remember(state, action, reward, next_state, next_mask, done)
            state, mask = next_state, next_mask
            if done:
                state, mask = environment.reset()
        loss = agent.train_batch(batch_size=32)
        self.assertIsNotNone(loss)
        self.assertTrue(math.isfinite(loss))

    def test_model_round_trip(self):
        agent = MaskedDoubleDQN(11, 4, hidden_size=8, seed=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.npz"
            agent.save(path)
            loaded = MaskedDoubleDQN.load(path)
            state = np.linspace(0.0, 1.0, 11, dtype=np.float32)
            np.testing.assert_allclose(agent.online.predict(state), loaded.online.predict(state))

    def test_genetic_and_stc_baselines(self):
        environment = random_routing_problem(21, cell_count=7, candidate_slots=4)
        order, directions, cost = genetic_route(
            list(environment.candidates.values()),
            tuple(environment.start_xy),
            population_size=20,
            generations=10,
            seed=21,
        )
        self.assertEqual(set(order), set(environment.candidates))
        self.assertEqual(len(directions), len(order))
        self.assertTrue(math.isfinite(cost))
        grid = GridMap(np.zeros((6, 7), dtype=bool), 0.1)
        path = spiral_stc_grid_baseline(grid, (0, 0), grid.free)
        self.assertEqual(set(path), {(row, col) for row in range(6) for col in range(7)})


if __name__ == "__main__":
    unittest.main()
