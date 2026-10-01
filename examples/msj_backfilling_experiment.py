"""Reproducible finite-cohort MSJ experiment (EPIC-047).

Run from the repository root:
    python -m examples.msj_backfilling_experiment --jobs 12000 --replications 6 \
        --output works/msj_backfilling/results.json

Synthetic workloads only: noisy-oracle predictions are an error model, not a
trained predictor. CIs describe averages of independent finite-run statistics,
including averages of run-level p99 estimates, not certified population p99.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from most_queue.sim.msj_general import MsjGeneralSim, MsjTraceJob


def interval(values):
    """Mean and Student 95% interval across independent replications."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {"mean": mean, "low": mean - half, "high": mean + half}


def make_workload(shape, coupling, jobs, seed):
    """Preserve the exact empirical K/S marginals when shuffling their pairing."""
    rng = np.random.default_rng(seed)
    component = (rng.random(jobs) < 0.35).astype(int)
    means = np.where(component, 2.2, 0.4)
    if shape == "erlang2":
        services = rng.gamma(2, means / 2)
    elif shape == "exponential":
        services = rng.exponential(means)
    else:
        sigma = np.sqrt(np.log(1 + 1.5**2))
        services = rng.lognormal(np.log(means) - sigma**2 / 2, sigma)
    arrivals = np.cumsum(rng.exponential(1 / 0.9, size=jobs))
    # Shuffle with an independent stream, so arrivals and durations are
    # literally identical in the coupled and shuffled workloads.
    classes = component.copy()
    if coupling == "shuffled":
        np.random.default_rng(seed + 1_000_000).shuffle(classes)
    return tuple(MsjTraceJob(float(a), int(c), float(s), float(s)) for a, c, s in zip(arrivals, classes, services))


def evaluate(trace, policy, coupling, warmup, seed):
    """Replay one fixed workload with an explicit information model."""
    if policy == "easy_noisy":
        rng = np.random.default_rng(seed + 2_000_000)
        # Unbiased multiplicative noise, with under- and overprediction.
        factors = rng.lognormal(-(0.6**2) / 2, 0.6, size=len(trace))
        trace = tuple(replace(job, estimate=job.service * factor) for job, factor in zip(trace, factors))
    elif policy == "easy_class_mean":
        means = [0.4, 2.2] if coupling == "coupled" else [0.65 * 0.4 + 0.35 * 2.2] * 2
        trace = tuple(replace(job, estimate=means[job.cls]) for job in trace)
    sim = MsjGeneralSim(4, "fcfs" if policy == "fcfs" else "easy")
    sim.set_servers([1, 3])
    result = sim.run_trace(trace, warmup_jobs=warmup)
    return {
        "mean_w": result.w[0],
        "mean_t": result.v[0],
        "p99_t": result.v_quantiles[0.99],
        "p99_wide": result.v_quantiles_per_class[1][0.99],
        "idle_with_queue": result.idle_with_queue,
        "utilization": result.utilization,
        "backfilled": result.backfilled,
        "reservations": result.reservations,
        "reservation_violations": result.reservation_violations,
    }


def experiment(jobs=12000, replications=6):
    """Return raw per-seed results and paired EASY-minus-FCFS summaries."""
    if jobs < 100 or replications < 2:
        raise ValueError("at least 100 jobs and 2 independent replications are required")
    policies = ("fcfs", "easy_oracle", "easy_noisy", "easy_class_mean")
    records, summaries = [], []
    warmup = jobs // 10
    for shape in ("erlang2", "exponential", "lognormal_cv1.5"):
        for coupling in ("coupled", "shuffled"):
            samples = {policy: [] for policy in policies}
            for rep in range(replications):
                seed = 47000 + rep
                trace = make_workload(shape, coupling, jobs + warmup, seed)
                needs = np.array([1 if job.cls == 0 else 3 for job in trace])
                durations = np.array([job.service for job in trace])
                for policy in policies:
                    metrics = evaluate(trace, policy, coupling, warmup, seed)
                    samples[policy].append(metrics)
                    records.append(
                        {
                            "shape": shape,
                            "coupling": coupling,
                            "policy": policy,
                            "seed": seed,
                            "sample_offered_load": float(0.9 * np.mean(needs * durations) / 4),
                            **metrics,
                        }
                    )
            for policy in policies:
                keys = ("mean_t", "p99_t", "p99_wide", "idle_with_queue", "utilization", "reservation_violations")
                row = {"shape": shape, "coupling": coupling, "policy": policy}
                row["metrics"] = {key: interval([sample[key] for sample in samples[policy]]) for key in keys}
                row["paired_delta"] = {
                    key: interval([sample[key] - base[key] for sample, base in zip(samples[policy], samples["fcfs"])])
                    for key in ("mean_t", "p99_t", "p99_wide")
                }
                summaries.append(row)
    return {
        "protocol": {
            "jobs_measured": jobs,
            "warmup_arrivals": warmup,
            "replications": replications,
            "first_seed": 47000,
            "k": 4,
            "arrival_rate": 0.9,
            "needs": [1, 3],
            "note": "Finite cohorts; t intervals over run means/quantiles; no stationarity or novelty claim. "
            "K/S marginals preserved across couplings, E[K*S] (hence offered work) is not fixed.",
        },
        "summaries": summaries,
        "replications": records,
    }


def main():
    """Print a compact result table and optionally write machine-readable data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=12000)
    parser.add_argument("--replications", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = experiment(args.jobs, args.replications)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("shape coupling policy mean_T p99_T delta_p99_[95%CI] violations")
    for row in result["summaries"]:
        metrics, delta = row["metrics"], row["paired_delta"]["p99_t"]
        print(
            row["shape"],
            row["coupling"],
            row["policy"],
            f"{metrics['mean_t']['mean']:.4f}",
            f"{metrics['p99_t']['mean']:.4f}",
            f"[{delta['low']:.4f}, {delta['high']:.4f}]",
            f"{metrics['reservation_violations']['mean']:.1f}",
        )


if __name__ == "__main__":
    main()
