"""Reproduce L3 inputs and measure model error, not GPU validation error.

python -m examples.prefill_decode_experiment --output path.json --simulate
Paper Table I is milliseconds; this experiment converts all costs to seconds.
No latency traces from the paper are available in this repository.
"""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

from most_queue.sim.prefill_decode import PrefillDecodeSim
from most_queue.theory.continuous_batching import PrefillDecodeCalc

PAPER_COSTS_SECONDS = {
    "Llama-3.1-8B_H100": (0.00668, 0.0000201, 0.0000000552),
    "Qwen2.5-14B_H100": (0.01014, 0.0000368, 0.0000000848),
}
METRICS = ["waiting", "sojourn", "ttft", "all_decode", "itl_after_first", "prefill", "batch"]


def summary(result):
    """Keep PMF tail diagnostics, while omitting runtime and large prefix arrays."""
    return {
        "waiting": result.w[0],
        "sojourn": result.v[0],
        "ttft": result.ttft_mean,
        "itl": result.itl_mean,
        "prefill": result.prefill_mean,
        "service": result.service_mean,
        "batch": result.batch_mean,
        "jobs": result.jobs_mean,
        "rho": result.utilization,
        "busy": result.busy_probability,
        "capacity": result.capacity,
        "p_wait": result.waiting_probability,
        "p_queue_nonempty": result.queue_nonempty_probability,
        "mean_chunks": result.effective_prefill_chunks,
        "valid": result.metrics_valid,
        "diagnostics": result.diagnostics,
        "stationary_residual": result.stationary_residual,
        "omitted_probability": result.omitted_probability,
    }


def calculate(cap, budget, n, m, costs, rho):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Set rate as a fraction of the analytical surrogate's capacity."""
    calc = PrefillDecodeCalc(cap, budget)
    calc.set_servers(n, m, *costs)
    rate = rho * cap / calc.stage_profile(cap)["service"]
    calc.set_sources(rate)
    return calc, rate


def experiment(simulate=False, replications=4, jobs=8000):
    """Published-cost predictions plus explicitly synthetic independent DES."""
    if replications < 2 or jobs < 2:
        raise ValueError("replications and jobs must be >= 2")
    paper, synthetic, heterogeneous = [], [], []
    for name, costs in PAPER_COSTS_SECONDS.items():
        for n in (64, 256, 1024, 4096):
            for m in (64, 256, 1024, 4096):
                for rho in (0.05, 0.35, 0.7):
                    calc, rate = calculate(256, 8192, n, m, costs, rho)
                    paper.append(
                        {"model": name, "input": n, "output": m, "rate": rate, "prediction": summary(calc.run())}
                    )
    costs = (0.1, 0.02, 0.003)
    for (n, m), cap, budget, rho in product(((4, 3), (12, 3)), (1, 4), (None, 4), (0.05, 0.35, 0.7)):
        calc, rate = calculate(cap, budget, n, m, costs, rho)
        row = {"input": n, "output": m, "cap": cap, "budget": budget, "rate": rate, "prediction": summary(calc.run())}
        if simulate:
            sample = []
            for seed in range(replications):
                sim = PrefillDecodeSim(cap, budget, seed=3000 + seed)
                sim.set_sources(rate)
                sim.set_servers(n, m, *costs)
                r = sim.run(jobs)
                sample.append(
                    [r.w[0], r.v[0], r.ttft_mean, r.all_decode_mean, r.itl_mean, r.prefill_mean, r.batch_mean]
                )
            sample = np.asarray(sample)
            means = sample.mean(axis=0)
            row["des"] = {
                "metrics": METRICS,
                "means": means.tolist(),
                "standard_errors": (sample.std(axis=0, ddof=1) / np.sqrt(replications)).tolist(),
            }
            # Relative differences are MODEL error, not confidence-test failures.
            prediction = np.array(
                [row["prediction"][k] for k in ("waiting", "sojourn", "ttft", "itl", "itl", "prefill", "batch")]
            )
            row["relative_model_error"] = {
                k: float((p - d) / d) if d else None for k, p, d in zip(METRICS, prediction, means)
            }
        synthetic.append(row)
    # Same input/output means, different dispersion; independent per-request draws.
    for budget, rho in product((None, 4), (0.35, 0.7)):
        calc, rate = calculate(4, budget, 8, 3, costs, rho)
        row = {
            "input_mean": 8,
            "output_mean": 3,
            "cap": 4,
            "budget": budget,
            "rate": rate,
            "input_pmf": {4: 0.5, 12: 0.5},
            "output_pmf": {1: 0.5, 5: 0.5},
            "prediction": summary(calc.run()),
        }
        if simulate:
            sample = []
            for seed in range(replications):
                sim = PrefillDecodeSim(4, budget, seed=4000 + seed)
                sim.set_sources(rate)
                sim.set_servers(
                    8, 3, *costs, length_sampler=lambda rng: (int(rng.choice([4, 12])), int(rng.choice([1, 5])))
                )
                r = sim.run(jobs)
                sample.append(
                    [r.w[0], r.v[0], r.ttft_mean, r.all_decode_mean, r.itl_mean, r.prefill_mean, r.batch_mean]
                )
            sample = np.asarray(sample)
            row["des"] = {
                "metrics": METRICS,
                "means": sample.mean(axis=0).tolist(),
                "standard_errors": (sample.std(axis=0, ddof=1) / np.sqrt(replications)).tolist(),
            }
        heterogeneous.append(row)
    negative = PrefillDecodeCalc(4, None)
    negative.set_sources(1e-5)
    negative.set_servers(1, 100, 0.001, 0, 0.01)
    return {
        "source": "https://arxiv.org/html/2609.20957v1",
        "units": "seconds",
        "paper_costs_seconds": PAPER_COSTS_SECONDS,
        "paper_cap": 256,
        "paper_budget": 8192,
        "paper_predictions": paper,
        "synthetic_costs": costs,
        "synthetic": synthetic,
        "heterogeneous": heterogeneous,
        "negative_prefill_example": summary(negative.run()),
        "des_configuration": {
            "enabled": simulate,
            "replications": replications,
            "jobs": jobs,
            "seed_first": 3000,
            "warmup_fraction": 0.2,
            "policy": "boundary admission, decode-first then FCFS prefill",
            "stability": "rho labels the surrogate, not a DES stability certificate",
        },
    }


def main():
    """Write a reproducible experiment JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--replications", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=8000)
    args = parser.parse_args()
    result = experiment(args.simulate, args.replications, args.jobs)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(
        f"Saved {len(result['paper_predictions'])} published-cost predictions "
        f"and {len(result['synthetic']) + len(result['heterogeneous'])} synthetic cases"
    )


if __name__ == "__main__":
    main()
