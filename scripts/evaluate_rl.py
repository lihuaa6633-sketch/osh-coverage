from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from osh_coverage_core.rl import MaskedDoubleDQN, random_routing_problem
from osh_coverage_core.baselines import genetic_route


def run_episode(environment, agent=None) -> dict[str, float]:
    state, mask = environment.reset()
    reward_total = transit_total = heading_total = 0.0
    while environment.unvisited:
        if agent is None:
            # Candidates are distance-ranked; choose the closer direction of slot zero.
            action = 0 if state[0] <= state[1] else 1
        else:
            action = agent.select_action(state, mask, epsilon=0.0)
        next_state, next_mask, reward, done, info = environment.step(action)
        reward_total += reward
        transit_total += info.get("transit", 0.0)
        heading_total += info.get("heading", 0.0)
        state, mask = next_state, next_mask
        if done:
            break
    return {"reward": reward_total, "transit": transit_total, "heading": heading_total}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--problems", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--include-ga", action="store_true")
    parser.add_argument("--ga-generations", type=int, default=40)
    args = parser.parse_args()
    agent = MaskedDoubleDQN.load(args.model)
    slots = agent.action_size // 2
    learned = []
    nearest = []
    genetic_costs = []
    for index in range(args.problems):
        cell_count = 4 + index % 17
        environment = random_routing_problem(args.seed + index, cell_count, slots)
        learned.append(run_episode(environment, agent))
        nearest.append(run_episode(environment, None))
        if args.include_ga:
            _, _, cost = genetic_route(
                list(environment.candidates.values()),
                tuple(environment.start_xy),
                population_size=60,
                generations=args.ga_generations,
                seed=args.seed + index,
            )
            genetic_costs.append(cost)
    payload = {
        "problems": args.problems,
        "learned": {key: float(np.mean([row[key] for row in learned])) for key in learned[0]},
        "nearest": {key: float(np.mean([row[key] for row in nearest])) for key in nearest[0]},
    }
    if genetic_costs:
        payload["genetic"] = {"routing_cost": float(np.mean(genetic_costs))}
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
