"""Small, dependency-free masked DQN/Double-DQN routing module.

The policy only selects a cell and traversal direction. Geometry and collision
checking remain outside the network by design.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math
from pathlib import Path
from typing import Optional

import numpy as np

from .decomposition import SweepVariant
from .grid import GridIndex, GridMap


@dataclass(frozen=True)
class RoutingCandidate:
    cell_id: int
    entry_xy: np.ndarray
    exit_xy: np.ndarray
    area: float
    sweep_length: float
    yaw: np.ndarray
    blocked_risk: float = 0.0


class CellRoutingEnvironment:
    """Discrete cell/direction environment shared by training and inference."""

    feature_count = 8

    def __init__(
        self,
        candidates: list[RoutingCandidate],
        start_xy: tuple[float, float],
        start_yaw: float = 0.0,
        candidate_slots: int = 6,
        map_scale: float = 20.0,
    ):
        self.candidates = {candidate.cell_id: candidate for candidate in candidates}
        self.start_xy = np.asarray(start_xy, dtype=float)
        self.start_yaw = float(start_yaw)
        self.candidate_slots = int(candidate_slots)
        self.map_scale = max(float(map_scale), 1e-6)
        self.reset()

    @property
    def state_size(self) -> int:
        return self.candidate_slots * self.feature_count + 3

    @property
    def action_size(self) -> int:
        return self.candidate_slots * 2

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        self.current_xy = self.start_xy.copy()
        self.current_yaw = self.start_yaw
        self.unvisited = set(self.candidates)
        self._slot_ids: list[Optional[int]] = []
        return self.observe()

    @staticmethod
    def _angle_delta(first: float, second: float) -> float:
        return abs((second - first + math.pi) % (2.0 * math.pi) - math.pi)

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        ranked = sorted(
            self.unvisited,
            key=lambda cell_id: min(
                np.linalg.norm(self.candidates[cell_id].entry_xy[0] - self.current_xy),
                np.linalg.norm(self.candidates[cell_id].entry_xy[1] - self.current_xy),
            ),
        )[: self.candidate_slots]
        self._slot_ids = list(ranked) + [None] * (self.candidate_slots - len(ranked))
        features: list[float] = []
        mask = np.zeros(self.action_size, dtype=bool)
        for slot, cell_id in enumerate(self._slot_ids):
            if cell_id is None:
                features.extend([0.0] * self.feature_count)
                continue
            candidate = self.candidates[cell_id]
            distance_0 = float(np.linalg.norm(candidate.entry_xy[0] - self.current_xy))
            distance_1 = float(np.linalg.norm(candidate.entry_xy[1] - self.current_xy))
            heading_0 = self._angle_delta(self.current_yaw, float(candidate.yaw[0]))
            heading_1 = self._angle_delta(self.current_yaw, float(candidate.yaw[1]))
            features.extend(
                [
                    distance_0 / self.map_scale,
                    distance_1 / self.map_scale,
                    min(candidate.area / (self.map_scale**2), 1.0),
                    candidate.sweep_length / self.map_scale,
                    heading_0 / math.pi,
                    heading_1 / math.pi,
                    float(np.clip(candidate.blocked_risk, 0.0, 1.0)),
                    1.0,
                ]
            )
            if candidate.blocked_risk < 1.0:
                mask[2 * slot : 2 * slot + 2] = True
        covered_ratio = 1.0 - len(self.unvisited) / max(len(self.candidates), 1)
        features.extend(
            [
                self.current_xy[0] / self.map_scale,
                self.current_xy[1] / self.map_scale,
                covered_ratio,
            ]
        )
        return np.asarray(features, dtype=np.float32), mask

    def step(self, action: int) -> tuple[np.ndarray, np.ndarray, float, bool, dict[str, float]]:
        state, mask = self.observe()
        if action < 0 or action >= self.action_size or not mask[action]:
            return state, mask, -25.0, False, {"invalid": 1.0}
        slot, direction = divmod(int(action), 2)
        cell_id = self._slot_ids[slot]
        assert cell_id is not None
        candidate = self.candidates[cell_id]
        transit = float(np.linalg.norm(candidate.entry_xy[direction] - self.current_xy))
        heading = self._angle_delta(self.current_yaw, float(candidate.yaw[direction]))
        reward = -transit - 0.25 * heading - 2.0 * candidate.blocked_risk
        self.current_xy = candidate.exit_xy[direction].copy()
        self.current_yaw = float(candidate.yaw[direction])
        self.unvisited.remove(cell_id)
        done = not self.unvisited
        if done:
            reward += 10.0
        next_state, next_mask = self.observe()
        return next_state, next_mask, reward, done, {"transit": transit, "heading": heading, "cell_id": float(cell_id)}


class _MLP:
    def __init__(self, input_size: int, output_size: int, hidden_size: int, seed: int):
        generator = np.random.default_rng(seed)
        self.w1 = generator.normal(0.0, math.sqrt(2.0 / input_size), (input_size, hidden_size)).astype(np.float32)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.w2 = generator.normal(0.0, math.sqrt(2.0 / hidden_size), (hidden_size, output_size)).astype(np.float32)
        self.b2 = np.zeros(output_size, dtype=np.float32)

    def predict(self, states: np.ndarray) -> np.ndarray:
        source = np.atleast_2d(states).astype(np.float32)
        hidden = np.maximum(source @ self.w1 + self.b1, 0.0)
        return hidden @ self.w2 + self.b2

    def train_chosen(self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray, learning_rate: float) -> float:
        source = np.atleast_2d(states).astype(np.float32)
        pre_hidden = source @ self.w1 + self.b1
        hidden = np.maximum(pre_hidden, 0.0)
        output = hidden @ self.w2 + self.b2
        row_indices = np.arange(source.shape[0])
        error = output[row_indices, actions] - targets
        loss = float(np.mean(error**2))
        grad_output = np.zeros_like(output)
        grad_output[row_indices, actions] = np.clip(2.0 * error / source.shape[0], -5.0, 5.0)
        grad_w2 = hidden.T @ grad_output
        grad_b2 = grad_output.sum(axis=0)
        grad_hidden = grad_output @ self.w2.T
        grad_hidden[pre_hidden <= 0.0] = 0.0
        grad_w1 = source.T @ grad_hidden
        grad_b1 = grad_hidden.sum(axis=0)
        self.w2 -= learning_rate * grad_w2
        self.b2 -= learning_rate * grad_b2
        self.w1 -= learning_rate * grad_w1
        self.b1 -= learning_rate * grad_b1
        return loss

    def copy_from(self, other: "_MLP") -> None:
        self.w1 = other.w1.copy()
        self.b1 = other.b1.copy()
        self.w2 = other.w2.copy()
        self.b2 = other.b2.copy()


@dataclass
class _Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_mask: np.ndarray
    done: bool


class MaskedDoubleDQN:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 96,
        gamma: float = 0.98,
        learning_rate: float = 5e-4,
        replay_capacity: int = 20000,
        double_dqn: bool = True,
        seed: int = 7,
    ):
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.double_dqn = bool(double_dqn)
        self.online = _MLP(state_size, action_size, hidden_size, seed)
        self.target = _MLP(state_size, action_size, hidden_size, seed + 1)
        self.target.copy_from(self.online)
        self.replay: deque[_Transition] = deque(maxlen=replay_capacity)
        self.generator = np.random.default_rng(seed)
        self.training_steps = 0

    def select_action(self, state: np.ndarray, mask: np.ndarray, epsilon: float = 0.0) -> int:
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            return -1
        if self.generator.random() < epsilon:
            return int(self.generator.choice(valid))
        q_values = self.online.predict(state)[0]
        masked = np.where(mask, q_values, -np.inf)
        return int(np.argmax(masked))

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_mask: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.append(
            _Transition(
                np.asarray(state, dtype=np.float32).copy(),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32).copy(),
                np.asarray(next_mask, dtype=bool).copy(),
                bool(done),
            )
        )

    def train_batch(self, batch_size: int = 64) -> Optional[float]:
        if len(self.replay) < batch_size:
            return None
        indices = self.generator.choice(len(self.replay), size=batch_size, replace=False)
        batch = [self.replay[int(index)] for index in indices]
        states = np.stack([item.state for item in batch])
        actions = np.asarray([item.action for item in batch], dtype=int)
        rewards = np.asarray([item.reward for item in batch], dtype=np.float32)
        next_states = np.stack([item.next_state for item in batch])
        next_masks = np.stack([item.next_mask for item in batch])
        dones = np.asarray([item.done for item in batch], dtype=bool)
        online_next = self.online.predict(next_states)
        target_next = self.target.predict(next_states)
        masked_online = np.where(next_masks, online_next, -np.inf)
        if self.double_dqn:
            next_actions = np.argmax(masked_online, axis=1)
            next_values = target_next[np.arange(batch_size), next_actions]
        else:
            next_values = np.max(np.where(next_masks, target_next, -np.inf), axis=1)
        no_valid_action = ~next_masks.any(axis=1)
        next_values[no_valid_action | dones] = 0.0
        targets = rewards + self.gamma * next_values
        loss = self.online.train_chosen(states, actions, targets, self.learning_rate)
        self.training_steps += 1
        return loss

    def update_target(self) -> None:
        self.target.copy_from(self.online)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            state_size=self.state_size,
            action_size=self.action_size,
            gamma=self.gamma,
            learning_rate=self.learning_rate,
            double_dqn=int(self.double_dqn),
            w1=self.online.w1,
            b1=self.online.b1,
            w2=self.online.w2,
            b2=self.online.b2,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MaskedDoubleDQN":
        data = np.load(path)
        agent = cls(
            int(data["state_size"]),
            int(data["action_size"]),
            hidden_size=int(data["b1"].shape[0]),
            gamma=float(data["gamma"]),
            learning_rate=float(data["learning_rate"]),
            double_dqn=bool(int(data["double_dqn"])),
        )
        agent.online.w1 = data["w1"].astype(np.float32)
        agent.online.b1 = data["b1"].astype(np.float32)
        agent.online.w2 = data["w2"].astype(np.float32)
        agent.online.b2 = data["b2"].astype(np.float32)
        agent.update_target()
        return agent


def random_routing_problem(seed: int, cell_count: int, candidate_slots: int = 6) -> CellRoutingEnvironment:
    generator = np.random.default_rng(seed)
    candidates: list[RoutingCandidate] = []
    for cell_id in range(cell_count):
        center = generator.uniform(1.0, 19.0, size=2)
        length = float(generator.uniform(1.0, 6.0))
        yaw = float(generator.choice((0.0, math.pi / 2.0)))
        direction = np.asarray((math.cos(yaw), math.sin(yaw)))
        entry_0 = center - direction * length / 2.0
        exit_0 = center + direction * length / 2.0
        candidates.append(
            RoutingCandidate(
                cell_id=cell_id,
                entry_xy=np.stack((entry_0, exit_0)),
                exit_xy=np.stack((exit_0, entry_0)),
                area=float(generator.uniform(1.0, 12.0)),
                sweep_length=length,
                yaw=np.asarray((yaw, (yaw + math.pi) % (2.0 * math.pi))),
                blocked_risk=float(generator.uniform(0.0, 0.4)),
            )
        )
    return CellRoutingEnvironment(candidates, (1.0, 1.0), candidate_slots=candidate_slots)


def train_curriculum(
    episodes: int = 2000,
    candidate_slots: int = 6,
    seed: int = 7,
    double_dqn: bool = True,
) -> tuple[MaskedDoubleDQN, list[float]]:
    template = random_routing_problem(seed, 4, candidate_slots)
    agent = MaskedDoubleDQN(template.state_size, template.action_size, double_dqn=double_dqn, seed=seed)
    rewards: list[float] = []
    for episode in range(episodes):
        progress = episode / max(episodes - 1, 1)
        maximum_cells = 8 if progress < 0.5 else 20
        cell_count = int(agent.generator.integers(4, maximum_cells + 1))
        environment = random_routing_problem(seed * 100000 + episode, cell_count, candidate_slots)
        state, mask = environment.reset()
        epsilon = max(0.05, 1.0 - 0.95 * progress)
        total_reward = 0.0
        while environment.unvisited:
            action = agent.select_action(state, mask, epsilon)
            next_state, next_mask, reward, done, _ = environment.step(action)
            agent.remember(state, action, reward, next_state, next_mask, done)
            agent.train_batch(64)
            state, mask = next_state, next_mask
            total_reward += reward
            if done:
                break
        if episode % 25 == 0:
            agent.update_target()
        rewards.append(total_reward)
    agent.update_target()
    return agent, rewards


def make_ddqn_scheduler(agent: MaskedDoubleDQN, candidate_slots: int = 6):
    """Adapt a trained policy to :class:`CoveragePlanner`'s scheduler contract.

    The returned callable is intentionally geometry-blind: it can only select
    one of the already collision-checked forward/reverse sweep variants.
    """

    expected_state_size = candidate_slots * CellRoutingEnvironment.feature_count + 3
    expected_action_size = candidate_slots * 2
    if agent.state_size != expected_state_size or agent.action_size != expected_action_size:
        raise ValueError("agent dimensions do not match candidate_slots")

    def scheduler(
        variants: dict[int, tuple[SweepVariant, SweepVariant]],
        start: GridIndex,
        grid: GridMap,
    ) -> list[tuple[int, int]]:
        candidates: list[RoutingCandidate] = []
        for cell_id, pair in variants.items():
            entries = np.asarray([grid.grid_to_world(*variant.entry) for variant in pair], dtype=float)
            exits = np.asarray([grid.grid_to_world(*variant.exit) for variant in pair], dtype=float)
            yaws = np.asarray([variant.points[0].yaw for variant in pair], dtype=float)
            mean_length = float(np.mean([variant.path_length for variant in pair]))
            candidates.append(
                RoutingCandidate(
                    cell_id=cell_id,
                    entry_xy=entries,
                    exit_xy=exits,
                    area=max(mean_length * grid.resolution, grid.resolution**2),
                    sweep_length=mean_length,
                    yaw=yaws,
                )
            )
        start_xy = grid.grid_to_world(*start)
        map_scale = max(grid.shape) * grid.resolution
        environment = CellRoutingEnvironment(candidates, start_xy, candidate_slots=candidate_slots, map_scale=map_scale)
        state, mask = environment.reset()
        order: list[tuple[int, int]] = []
        while environment.unvisited:
            action = agent.select_action(state, mask, epsilon=0.0)
            if action < 0:
                # Safety fallback: nearest feasible entry and direction.
                valid = np.flatnonzero(mask)
                if valid.size == 0:
                    raise RuntimeError("no valid routing action")
                action = int(valid[0])
            next_state, next_mask, _, done, info = environment.step(action)
            order.append((int(info["cell_id"]), int(action % 2)))
            state, mask = next_state, next_mask
            if done:
                break
        return order

    return scheduler


def order_residual_regions(
    agent: MaskedDoubleDQN,
    regions: list,
    current_xy: tuple[float, float],
    candidate_slots: int = 6,
) -> list[str]:
    """Reuse the same high-level policy for residual-region revisit order."""
    candidates: list[RoutingCandidate] = []
    id_lookup: dict[int, str] = {}
    for numeric_id, region in enumerate(regions):
        center = np.asarray(region.centroid_xy, dtype=float)
        id_lookup[numeric_id] = region.region_id
        blocked = 1.0 if region.state == "temporarily_unreachable" else 0.0
        candidates.append(
            RoutingCandidate(
                cell_id=numeric_id,
                entry_xy=np.stack((center, center)),
                exit_xy=np.stack((center, center)),
                area=float(region.area_m2),
                sweep_length=max(float(region.area_m2), 0.01),
                yaw=np.asarray((0.0, math.pi)),
                blocked_risk=blocked,
            )
        )
    if not candidates:
        return []
    map_scale = max(1.0, max(np.linalg.norm(candidate.entry_xy[0] - np.asarray(current_xy)) for candidate in candidates))
    environment = CellRoutingEnvironment(candidates, current_xy, candidate_slots=candidate_slots, map_scale=map_scale)
    state, mask = environment.reset()
    ordered: list[str] = []
    while mask.any():
        action = agent.select_action(state, mask, epsilon=0.0)
        next_state, next_mask, _, done, info = environment.step(action)
        ordered.append(id_lookup[int(info["cell_id"])])
        state, mask = next_state, next_mask
        if done:
            break
    return ordered
