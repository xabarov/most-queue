"""Synthetic reproducible L5 trajectories, finite spectra and paired integer DES.

No Vidur/GPU reproduction; per-iteration units. Run from the repository root.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from most_queue.sim.kv_cache_congestion import KvCacheCongestionSim
from most_queue.theory.continuous_batching import KvCacheCongestionCalc


def make_calc(memory, inputs, decodes, proportions):
    """Build a fixed-class deterministic calculator."""
    model = KvCacheCongestionCalc(memory)
    model.set_sources(proportions)
    model.set_servers(inputs, decodes)
    return model


def serialize(result):
    """Exclude wall-clock execution duration from reproducible records."""
    record = asdict(result)
    record.pop("duration")
    return record


def experiment(iterations=3000, replications=4):
    """Return all settings, seeds, trajectories and finite-horizon observations."""
    if iterations < 100 or replications < 2:
        raise ValueError("iterations>=100 and replications>=2 are required")
    fluid = []
    for name, memory, inputs, decodes, p, initial in [
        ("homogeneous_cascade", 24, [2], [3], [1], [[2.5, 2, 1.7]]),
        ("exact_fixed_point", 24, [2], [3], [1], [[2, 2, 2]]),
        ("balanced_capture", 24, [2], [3], [1], [[48 / 13, 24 / 13, 72 / 65]]),
        ("coprime", 518, [50, 50], [2, 3], [0.5, 0.5], [[2.5, 2], [2.5, 2, 55 / 53]]),
        ("non_coprime", 626, [50, 50], [2, 4], [0.5, 0.5], [[2.5, 2], [2.5, 2, 2, 57 / 54]]),
    ]:
        model = make_calc(memory, inputs, decodes, p)
        analytic = model.run()
        # Unstable balanced orbit is inspected briefly; floating-point roundoff
        # is not an exact arithmetic guarantee of perpetual exceptional capture.
        horizon = 7 if name == "balanced_capture" else 120
        for cap in [None, analytic.nominal_throughput]:
            trajectory = model.evolve(horizon, initial_state=initial, admission_cap=cap, record_trace=True)
            fluid.append(
                {
                    "name": name,
                    "memory": memory,
                    "inputs": inputs,
                    "decodes": decodes,
                    "proportions": p,
                    "initial_state": initial,
                    "iterations": horizon,
                    "cap": cap,
                    "analytic": serialize(analytic),
                    "trajectory": serialize(trajectory),
                }
            )
    spectral = []
    for pair in [(2, 3), (2, 4), (2, 7), (4, 7)]:
        for input_length in [0, 1, 2, 5, 10, 14, 15, 17, 18, 30, 50, 100]:
            model = make_calc(10000, [input_length] * 2, pair, [0.5, 0.5])
            spectral.append(
                {
                    "input_length": input_length,
                    "decode_lengths": pair,
                    "proportions": [0.5, 0.5],
                    "result": serialize(model.run()),
                }
            )
    stochastic = []
    warmup = iterations // 5
    for memory, inputs, decodes, p in [
        (60, [2], [3], [1]),
        (2440, [10], [40], [1]),
        (518, [50, 50], [2, 3], [0.5, 0.5]),
        (626, [50, 50], [2, 4], [0.5, 0.5]),
    ]:
        analytic = make_calc(memory, inputs, decodes, p).run()
        integer_cap = int(np.floor(analytic.nominal_throughput + 1e-12))
        for fraction in [0.65, 0.9, 1.15]:
            rate = fraction * analytic.nominal_throughput
            for cap in [None, integer_cap]:
                runs = []
                for seed in range(8000, 8000 + replications):
                    simulator = KvCacheCongestionSim(memory, seed=seed)
                    simulator.set_sources([rate * pk for pk in p])
                    simulator.set_servers(inputs, decodes)
                    result = simulator.run(iterations, warmup=warmup, admission_cap=cap)
                    assert result.arrived == result.completed + result.unfinished
                    assert result.executed_work == result.completed_work + result.remaining_work + result.discarded_work
                    runs.append({"seed": seed, "result": serialize(result)})
                throughputs = [run["result"]["throughput"] for run in runs]
                stochastic.append(
                    {
                        "memory": memory,
                        "inputs": inputs,
                        "decodes": decodes,
                        "proportions": p,
                        "arrival_rate": rate,
                        "nominal_fraction": fraction,
                        "admission_cap": cap,
                        "analytic": serialize(analytic),
                        "runs": runs,
                        "throughput_mean": float(np.mean(throughputs)),
                        "throughput_se": float(np.std(throughputs, ddof=1) / np.sqrt(replications)),
                    }
                )
    return {
        "schema_version": 1,
        "units": "requests/iteration; memory in reserved KV tokens",
        "source": "arXiv:2606.15555v1",
        "scope": "synthetic; integer FCFS != proportional fluid",
        "iterations": iterations,
        "warmup": warmup,
        "replications": replications,
        "fluid": fluid,
        "spectral": spectral,
        "stochastic": stochastic,
    }


def main():
    """Write reproducible experiment records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/research/kv-cache-congestion-results-2026.json")
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--replications", type=int, default=4)
    args = parser.parse_args()
    output = experiment(args.iterations, args.replications)
    Path(args.output).write_text(
        json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        f"Saved {len(output['fluid'])} fluid, {len(output['spectral'])} spectral, "
        f"{len(output['stochastic'])} DES settings"
    )


if __name__ == "__main__":
    main()
