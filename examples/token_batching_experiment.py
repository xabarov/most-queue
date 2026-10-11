"""Reproducible L2 comparison; synthetic costs, not a fresh GPU calibration.

Run: python -m examples.token_batching_experiment --output path.json
Optional independent seeded DES: --simulate --replications 4 --jobs 10000
"""

import argparse
import json
from pathlib import Path

import numpy as np

from most_queue.sim.token_batching import TokenBatchingSim
from most_queue.theory.batch.token_batching import (
    FixedTokenBatchingCalc,
    TokenBatchingCalc,
    optimize_token_limit,
    paper_fixed_formula,
)
from most_queue.theory.batch.token_service import TokenLengthDistribution

COST = (0.1, 0.2, 0.0025, 0.004)


def experiment(simulate=False, replications=4, jobs=10_000):
    """Finite-PMF comparisons and published-formula discrepancy diagnostics."""
    original = TokenLengthDistribution([1, 3, 10, 100], [0.3, 0.4, 0.25, 0.05])
    laws = {
        "sparse_clipped_16": TokenLengthDistribution(original.lengths, original.probabilities, 16),
        "uniform_16": TokenLengthDistribution.uniform(16),
        "normal_clipped_32": TokenLengthDistribution.clipped_normal(12, 3, 32),
        "lognormal_clipped_32": TokenLengthDistribution.clipped_lognormal(2, 0.7, 32),
    }
    rows, fixed = [], []
    for name, tokens in laws.items():
        for rate in (0.5, 1.5, 3.0):
            for strategy in ("dynamic", "elastic", "fixed"):
                calc = FixedTokenBatchingCalc(4) if strategy == "fixed" else TokenBatchingCalc(tolerance=1e-8)
                result = calc.set_sources(rate).set_servers(tokens, *COST, elastic=strategy == "elastic").run()
                row = {
                    "law": name,
                    "rate": rate,
                    "strategy": strategy,
                    "waiting_mean": result.w[0],
                    "latency_mean": result.v[0],
                    "utilization": result.utilization,
                    "converged": result.converged,
                    "flow_residual": result.flow_residual,
                    "moment_residual": result.moment_residual,
                    "clipped_fraction": tokens.clipped_fraction,
                }
                if strategy != "fixed":
                    row["bounds"] = calc.paper_bound_report()
                if simulate:
                    sample = []
                    for seed in range(replications):
                        sim = TokenBatchingSim(strategy, 4 if strategy == "fixed" else None, seed=1000 + seed)
                        run = sim.set_sources(rate).set_servers(tokens, *COST).run(jobs)
                        sample.append([run.w[0], run.v[0], run.utilization])
                    sample = np.asarray(sample)
                    row["des"] = {
                        "replications": replications,
                        "jobs_minimum": jobs,
                        "means": sample.mean(axis=0).tolist(),
                        "standard_errors": (sample.std(axis=0, ddof=1) / np.sqrt(replications)).tolist(),
                    }
                rows.append(row)
    tokens = laws["sparse_clipped_16"]
    for size in (1, 2, 4, 8):
        for surrogate in (False, True):
            c = (
                FixedTokenBatchingCalc(size)
                .set_sources(1.5)
                .set_servers(tokens, *COST, mean_service_surrogate=surrogate)
            )
            r = c.run()
            duration = c.service.moments(size)[0]
            fixed.append(
                {
                    "batch_size": size,
                    "mean_service_surrogate": surrogate,
                    "waiting_mean": r.w[0],
                    "latency_mean": r.v[0],
                    "paper_eq25_terms20": paper_fixed_formula(1.5, duration, size),
                    "paper_eq25_terms1000": paper_fixed_formula(1.5, duration, size, 1000),
                    "formation_mean": (size - 1) / (2 * 1.5),
                    "service_mean": duration,
                }
            )
    clipping = optimize_token_limit(original, [1, 2, 3, 5, 8, 10, 16, 32, 64, 100], 1.5, 0.05, 0.05, theta=0.95)
    caps = []
    if simulate:
        for cap in (1, 4, 8):
            sample = []
            for seed in range(replications):
                r = (
                    TokenBatchingSim("dynamic", cap, seed=2000 + seed)
                    .set_sources(1.5)
                    .set_servers(tokens, *COST)
                    .run(jobs)
                )
                sample.append([r.w[0], r.v[0], r.batch_mean])
            sample = np.asarray(sample)
            caps.append(
                {
                    "physical_cap": cap,
                    "means": sample.mean(axis=0).tolist(),
                    "standard_errors": (sample.std(axis=0, ddof=1) / np.sqrt(replications)).tolist(),
                }
            )
    return {
        "source": "arXiv:2407.05347v1",
        "cost_coefficients": list(COST),
        "cost_calibration": "synthetic; no measured GPU coefficients claimed",
        "token_convention": "total output tokens; decode=total-1",
        "grid": rows,
        "fixed_surrogate_and_equation25": fixed,
        "clipping_search": clipping,
        "clipping_cost": {"rate": 1.5, "a": 0.05, "c": 0.05, "theta": 0.95},
        "capped_dynamic_des": caps,
    }


def main():
    """Write deterministic JSON; CPU timings are deliberately excluded."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--replications", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=10000)
    args = parser.parse_args()
    if args.replications < 2 or args.jobs < 1:
        parser.error("require replications>=2 and jobs>=1")
    result = experiment(args.simulate, args.replications, args.jobs)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(result['grid'])} configurations to {args.output}")


if __name__ == "__main__":
    main()
