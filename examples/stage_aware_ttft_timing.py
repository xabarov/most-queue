"""GL3 diagnostic runtime check with matching wall AND process clocks.

Same four laws/budgets and costs as the frozen experiment, load=.5, B=2,
K=6; three repeats, one 12000-job DES per solve. No accuracy parameters fitted.
"""

import json
import time
from pathlib import Path

import numpy as np

from examples.stage_aware_ttft_experiment import COSTS, LAWS
from most_queue.sim.prefill_decode import PrefillDecodeSim
from most_queue.theory.continuous_batching import StageAwareTtftCalc


def main():
    """Report paired same-clock costs, including construction and solve."""
    rows = []
    for budget in (2, 4):
        for law in LAWS:
            probabilities = np.array([item[2] for item in law])
            setup = StageAwareTtftCalc(2, budget, 6)
            setup.set_servers(law, *COSTS[0])
            rate = 0.5 / (probabilities @ setup.isolated_times()[0])
            repetitions = []
            for seed in range(93000, 93003):
                # Generation is excluded for BOTH analytical and simulation cost.
                calc = StageAwareTtftCalc(2, budget, 6)
                calc.set_sources(rate)
                calc.set_servers(law, *COSTS[0])
                rng = np.random.default_rng(seed)
                arrivals = np.cumsum(rng.exponential(1 / rate, 12000))
                marks = np.array([item[:2] for item in law], dtype=int)[rng.choice(len(law), 12000, p=probabilities)]
                sim = PrefillDecodeSim(2, budget)
                sim.set_servers(1, 1, *COSTS[0])
                wall_start, cpu_start = time.perf_counter(), time.process_time()
                result = calc.run()
                solver_wall, solver_cpu = time.perf_counter() - wall_start, time.process_time() - cpu_start
                wall_start, cpu_start = time.perf_counter(), time.process_time()
                sim.replay(arrivals, *marks.T, warmup_fraction=0.2)
                des_wall, des_cpu = time.perf_counter() - wall_start, time.process_time() - cpu_start
                repetitions.append(
                    {
                        "seed": seed,
                        "solver_wall": solver_wall,
                        "solver_cpu": solver_cpu,
                        "des_wall": des_wall,
                        "des_cpu": des_cpu,
                    }
                )
            rows.append(
                {
                    "budget": budget,
                    "law": law,
                    "load": 0.5,
                    "states": result.states,
                    "repetitions": repetitions,
                    "median_wall_solver_over_des": float(
                        np.median([r["solver_wall"] / r["des_wall"] for r in repetitions])
                    ),
                    "median_cpu_solver_over_des": float(
                        np.median([r["solver_cpu"] / r["des_cpu"] for r in repetitions])
                    ),
                }
            )
    output = Path("docs/research/stage-aware-ttft-timing-2026.json")
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "diagnostic_only": True,
                "costs": COSTS[0],
                "queue_capacity": 6,
                "jobs": 12000,
                "rows": rows,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps([{key: value for key, value in row.items() if key != "repetitions"} for row in rows], indent=2))


if __name__ == "__main__":
    main()
