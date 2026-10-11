"""Reproduce L1 bound/refinement/capped-CTMC comparisons (no GPU measurements).

Run from the repository root:
    python -m examples.dynamic_batching_experiment --output /tmp/dynamic-batching.json
"""

import argparse
import json
from pathlib import Path

from most_queue.theory.batch.bulk_service import BulkServiceMM1Calc
from most_queue.theory.batch.dynamic_batching import DynamicBatchingCalc


def run_study():
    """Evaluate a 60-point affine-service grid and two independent checks."""
    grid = []
    for alpha in (0.001, 0.0025, 0.01):
        for ratio in (0, 2, 8, 20):
            for rho in (0.1, 0.3, 0.5, 0.7, 0.9):
                calc = DynamicBatchingCalc().set_sources(rho / alpha).set_servers(alpha, ratio * alpha)
                result = calc.run()
                grid.append(
                    {
                        "lambda": calc.rate,
                        "alpha": alpha,
                        "tau0": calc.tau0,
                        "rho": rho,
                        **calc.bound_report(),
                        "max_batch": result.max_batch,
                        "refinement_error": result.refinement_error,
                        "moment_residual": result.moment_residual,
                        "boundary_second_moment_fraction": result.boundary_second_moment_fraction,
                    }
                )
    refinement = []
    reference = DynamicBatchingCalc(max_batch=1024).set_sources(9).set_servers(0.1, 0.2).run()
    for size in (16, 32, 64, 128, 256, 512):
        result = DynamicBatchingCalc(max_batch=size).set_sources(9).set_servers(0.1, 0.2).run()
        refinement.append(
            {
                "max_batch": size,
                "waiting": result.waiting_mean,
                "relative_error_vs_1024": abs(result.waiting_mean / reference.waiting_mean - 1),
                "last_state_mass": float(result.batch_distribution[-1]),
                "boundary_second_moment_fraction": result.boundary_second_moment_fraction,
                "moment_residual": result.moment_residual,
            }
        )
    unbounded = DynamicBatchingCalc().set_sources(1).set_exponential_service(0.1, 0.5).get_waiting_mean()
    capped = []
    for cap in (4, 8, 16, 32):
        calc = BulkServiceMM1Calc(a=1, b=cap, queue_truncation=100)
        calc.set_sources(1)
        calc.set_servers(lambda b: 1 / (0.1 * b + 0.5))
        wait = float(calc.get_w(1)[0])
        capped.append({"batch_cap": cap, "waiting": wait, "absolute_error": abs(wait - unbounded)})
    return {
        "source": "Inoue 2021, doi:10.1016/j.peva.2020.102183; numerical reproduction, not new theory",
        "tolerance": 1e-9,
        "grid": grid,
        "refinement": refinement,
        "exponential_unbounded_waiting": unbounded,
        "finite_cap_ctmc": capped,
    }


def main():
    """Write a reproducible, machine-readable report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_study()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    worst = max(report["grid"], key=lambda row: row["best_bound_ratio"])
    print(f"{len(report['grid'])} grid points; maximum bound excess = {100 * (worst['best_bound_ratio'] - 1):.6f}%")
    print(f"Worst parameters: lambda={worst['lambda']}, alpha={worst['alpha']}, tau0={worst['tau0']}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
