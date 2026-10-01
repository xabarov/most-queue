"""Controlled-resource-load MSJ pilot (EPIC-048); no stationarity claim.

    python -m examples.msj_conservative_experiment --jobs 6000 --replications 6 \
        --output works/msj_conservative/results.json

Reuse EPIC-047's synthetic K/S marginals, but set lambda from the known E[K*S]
instead of using one arrival rate for workloads with different offered work.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from examples.msj_backfilling_experiment import interval, make_workload
from most_queue.sim.msj_general import MsjGeneralSim, MsjTraceJob

SHAPES = ("erlang2", "exponential", "lognormal_cv1.5")
POLICIES = ("fcfs", "easy_oracle", "conservative_oracle", "easy_class_mean", "conservative_class_mean")
LOADS = (0.35, 0.55)
METRICS = ("mean_t", "p99_t", "p99_wide", "idle_with_queue", "utilization", "reservation_violations")
PAIRED = ("mean_t", "p99_t", "p99_wide")
GEOMETRIES = {
    "two_class": {"needs": [1, 3], "probabilities": [0.65, 0.35], "means": [0.4, 2.2]},
    "three_class": {"needs": [1, 3, 4], "probabilities": [0.5, 0.3, 0.2], "means": [0.4, 2.2, 2.2]},
}


def expected_work(coupling, count, geometry="two_class"):
    """Known ensemble E[K*S], including finite random-permutation correction."""
    if coupling not in ("coupled", "shuffled") or count < 1 or geometry not in GEOMETRIES:
        raise ValueError("valid coupling and positive trace length required")
    config = GEOMETRIES[geometry]
    weights, needs, means = (np.array(config[key]) for key in ("probabilities", "needs", "means"))
    coupled = float(np.sum(weights * needs * means))
    independent = float(np.dot(weights, needs) * np.dot(weights, means))
    # A uniformly permuted label returns to its own service with chance 1/N.
    return coupled if coupling == "coupled" else independent + (coupled - independent) / count


def three_class_workload(shape, coupling, jobs, seed):
    """A diagnostic geometry with both partly and fully occupying wide jobs."""
    rng = np.random.default_rng(seed)
    config = GEOMETRIES["three_class"]
    component = rng.choice(3, jobs, p=config["probabilities"])
    means = np.array(config["means"])[component]
    if shape == "erlang2":
        services = rng.gamma(2, means / 2)
    elif shape == "exponential":
        services = rng.exponential(means)
    else:
        sigma = np.sqrt(np.log(1 + 1.5**2))
        services = rng.lognormal(np.log(means) - sigma**2 / 2, sigma)
    arrivals = np.cumsum(rng.exponential(1 / 0.9, jobs))
    classes = component.copy()
    if coupling == "shuffled":
        np.random.default_rng(seed + 1_000_000).shuffle(classes)
    return tuple(MsjTraceJob(float(a), int(c), float(s), float(s)) for a, c, s in zip(arrivals, classes, services))


def controlled_workload(shape, coupling, jobs, seed, load, geometry="two_class"):
    """Rescale common interarrival draws using a law-based, not fitted, rate."""
    if shape not in SHAPES or not np.isfinite(load) or not 0 < load < 1:
        raise ValueError("supported shape and resource load in (0, 1) required")
    rate = load * 4 / expected_work(coupling, jobs, geometry)
    generator = make_workload if geometry == "two_class" else three_class_workload
    trace = generator(shape, coupling, jobs, seed)
    return tuple(replace(job, arrival=job.arrival * 0.9 / rate) for job in trace), rate


def evaluate(trace, policy, coupling, warmup, geometry="two_class"):
    """Use the same immutable jobs and the same information for both backfills."""
    if policy not in POLICIES:
        raise ValueError("unknown policy")
    config = GEOMETRIES[geometry]
    if policy.endswith("class_mean"):
        means = np.array(config["means"])
        if coupling == "shuffled":
            # Exact ensemble E[S | permuted class] for this finite trace size.
            means = (1 - 1 / len(trace)) * np.dot(config["probabilities"], means) + means / len(trace)
        trace = tuple(replace(job, estimate=float(means[job.cls])) for job in trace)
    sim = MsjGeneralSim(4, policy.split("_")[0])
    sim.set_servers(config["needs"])
    result = sim.run_trace(trace, warmup_jobs=warmup)
    return {
        "mean_t": result.v[0],
        "p99_t": result.v_quantiles[0.99],
        "p99_wide": result.v_quantiles_per_class[-1][0.99],
        "idle_with_queue": result.idle_with_queue,
        "utilization": result.utilization,
        "backfilled": result.backfilled,
        "reservations": result.reservations,
        "reservation_violations": result.reservation_violations,
    }


def paired_intervals(samples, baseline):
    """Student intervals of within-seed differences, not unpaired error bars."""
    return {key: interval([a[key] - b[key] for a, b in zip(samples, baseline)]) for key in PAIRED}


def experiment(jobs=6000, replications=6, geometry="two_class"):
    """Return complete per-run data and paired policy/coupling comparisons."""
    if jobs < 100 or replications < 2:
        raise ValueError("at least 100 jobs and 2 independent replications are required")
    if geometry not in GEOMETRIES:
        raise ValueError("unknown resource geometry")
    config = GEOMETRIES[geometry]
    warmup = jobs // 10
    records, summaries, coupling_deltas = [], [], []
    for shape in SHAPES:
        for load in LOADS:
            by_coupling = {}
            for coupling in ("coupled", "shuffled"):
                samples = {policy: [] for policy in POLICIES}
                for rep in range(replications):
                    seed = 48000 + rep
                    trace, rate = controlled_workload(shape, coupling, jobs + warmup, seed, load, geometry)
                    realized_load = rate * np.mean([config["needs"][job.cls] * job.service for job in trace]) / 4
                    for policy in POLICIES:
                        metrics = evaluate(trace, policy, coupling, warmup, geometry)
                        samples[policy].append(metrics)
                        records.append(
                            {
                                "shape": shape,
                                "coupling": coupling,
                                "target_load": load,
                                "policy": policy,
                                "seed": seed,
                                "arrival_rate": rate,
                                "sample_offered_load": float(realized_load),
                                **metrics,
                            }
                        )
                by_coupling[coupling] = samples
                for policy in POLICIES:
                    row = {
                        "shape": shape,
                        "coupling": coupling,
                        "target_load": load,
                        "policy": policy,
                        "metrics": {key: interval([sample[key] for sample in samples[policy]]) for key in METRICS},
                        "delta_vs_fcfs": paired_intervals(samples[policy], samples["fcfs"]),
                    }
                    if policy.startswith("conservative"):
                        row["delta_vs_easy"] = paired_intervals(
                            samples[policy], samples[policy.replace("conservative", "easy")]
                        )
                    summaries.append(row)
            for policy in POLICIES:
                coupling_deltas.append(
                    {
                        "shape": shape,
                        "target_load": load,
                        "policy": policy,
                        "coupled_minus_shuffled": paired_intervals(
                            by_coupling["coupled"][policy], by_coupling["shuffled"][policy]
                        ),
                    }
                )
    return {
        "protocol": {
            "jobs_measured": jobs,
            "warmup_arrivals": warmup,
            "replications": replications,
            "first_seed": 48000,
            "k": 4,
            "geometry": geometry,
            "needs": config["needs"],
            "class_probabilities": config["probabilities"],
            "component_service_means": config["means"],
            "loads": LOADS,
            "shapes": SHAPES,
            "policies": POLICIES,
            "load_definition": "lambda = target_load * k / known ensemble E[K*S]; shuffled includes Cov(K,S)/N",
            "information": "oracle=S; class_mean=known ensemble E[S|class], not trained or a time limit",
            "p99_wide_definition": "run-level p99 for the largest resource-demand class (3 or 4 servers)",
            "note": "Finite cohorts with full drain; no stationarity or novelty claim. "
            "95% t intervals across independent run statistics, including run-level p99; "
            "not population quantile CIs and not multiplicity-adjusted. "
            "Within coupling policies see identical jobs; across couplings arrival times differ.",
        },
        "summaries": summaries,
        "coupling_deltas": coupling_deltas,
        "replications": records,
    }


def main():
    """Run the full grid and optionally save strict JSON for reproduction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=6000)
    parser.add_argument("--replications", type=int, default=6)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--geometry", choices=tuple(GEOMETRIES), default="two_class")
    args = parser.parse_args()
    result = experiment(args.jobs, args.replications, args.geometry)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("shape coupling load policy mean_T p99_T p99_wide violations")
    for row in result["summaries"]:
        metrics = row["metrics"]
        print(
            row["shape"],
            row["coupling"],
            row["target_load"],
            row["policy"],
            *(f"{metrics[key]['mean']:.4f}" for key in ("mean_t", "p99_t", "p99_wide", "reservation_violations")),
        )


if __name__ == "__main__":
    main()
