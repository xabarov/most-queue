"""EPIC-053: prespecified checkpoint/resume sensitivity on common MSJ work.

    python -m examples.msj_checkpoint_experiment --regime powers_of_two \
        --shape lognormal_hetero --output works/msj_checkpoint/powers_of_two-lognormal_hetero.json

Overheads change scheduling, not just a post-hoc latency correction. Useful
arrival load stays fixed. Positive-cost SF is an explicit gated extension.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from examples.msj_age_runtime_experiment import age_predictor, array_digest, initial_forecasts
from examples.msj_group_calibration_experiment import summarize
from examples.msj_packing_experiment import WORKLOADS, Workload, fit_history, make_bundle, packing_metrics
from examples.msj_runtime_prediction_experiment import FEATURE_SLOPE, LOADS, SHAPES, fingerprint
from most_queue.sim.msj_checkpoint import MsjCheckpointSim
from most_queue.sim.msj_general import MsjGeneralSim, MsjTraceJob
from most_queue.structs import MsjCheckpointResults

REGIMES = ("one_or_all", "powers_of_two")
OVERHEAD_LEVELS = (0.0, 0.01, 0.05, 0.2, 0.5, 1.0)
COSTS = {f"sf:balanced:{level:g}": (level / 2, level / 2) for level in OVERHEAD_LEVELS}
COSTS.update({"sf:checkpoint:0.2": (0.2, 0.0), "sf:resume:0.2": (0.0, 0.2)})
ZERO = "sf:balanced:0"
POLICIES = ("fcfs", "first_fit", "msf", "easy:km_age", "conservative:km_age", *COSTS)
METRICS = (
    "mean_w",
    "mean_t",
    "p99_t",
    "weighted_mean_t",
    "mean_first_wait",
    "p99_first_wait",
    "mean_interruption",
    "mean_queue_wait",
    "mean_checkpoint",
    "mean_resume",
    "utilization",
    "productive_utilization",
    "checkpoint_utilization",
    "resume_utilization",
    "idle_with_queue",
    "checkpoint_resource_time",
    "resume_resource_time",
    "preemptions",
    "preempted_jobs",
    "backlog_at_last_arrival",
    "throughput",
    "drain_time",
    "cohort_resource_ratio",
    "reservations",
    "reservation_violations",
)


def replay(workload: Workload, trace: tuple[MsjTraceJob, ...], policy: str, warmup: int, models: list) -> dict:
    """Replay once and separate resource occupancy from productive service."""
    mean_service = float(np.dot(workload.probabilities, workload.means))
    predictor = None
    if policy in COSTS:
        checkpoint, resume = (factor * mean_service for factor in COSTS[policy])
        sim = MsjCheckpointSim(4, checkpoint, resume)
    else:
        checkpoint = resume = 0.0
        discipline, _, mode = policy.partition(":")
        sim = MsjGeneralSim(4, discipline)
        predictor = age_predictor(models) if mode == "km_age" else None
    sim.set_servers(workload.needs)
    result = sim.run_trace(trace, warmup_jobs=warmup, remaining_predictor=predictor)
    output = packing_metrics(workload, trace, result)
    first = np.array(result.start_times[warmup:]) - np.array([job.arrival for job in trace[warmup:]])
    if isinstance(result, MsjCheckpointResults):
        interruptions = result.interruption_samples
        queue, checkpoint_samples, resume_samples = (
            result.queue_wait_samples,
            result.checkpoint_samples,
            result.resume_samples,
        )
        productive = result.productive_utilization
        checkpoint_use, resume_use = result.checkpoint_utilization, result.resume_utilization
        checkpoint_work, resume_work = result.checkpoint_resource_time, result.resume_resource_time
        output["service_episodes"] = len(result.service_segments)
    else:
        interruptions = checkpoint_samples = resume_samples = np.zeros(len(first))
        queue = result.wait_samples
        productive, checkpoint_use, resume_use = result.utilization, 0.0, 0.0
        checkpoint_work = resume_work = 0.0
    work = sum(workload.needs[job.cls] * job.service for job in trace)
    output.update(
        {
            "checkpoint_time": checkpoint,
            "resume_time": resume,
            "mean_first_wait": float(first.mean()),
            "p99_first_wait": float(np.quantile(first, 0.99)),
            "mean_interruption": float(np.mean(interruptions)),
            "mean_queue_wait": float(np.mean(queue)),
            "mean_checkpoint": float(np.mean(checkpoint_samples)),
            "mean_resume": float(np.mean(resume_samples)),
            "productive_utilization": productive,
            "checkpoint_utilization": checkpoint_use,
            "resume_utilization": resume_use,
            "checkpoint_resource_time": checkpoint_work,
            "resume_resource_time": resume_work,
            "throughput": result.throughput,
            "backlog_at_last_arrival": int(np.sum(np.array(result.completion_times) > trace[-1].arrival)),
            "drain_time": float(max(result.completion_times) - trace[-1].arrival),
            "cohort_resource_ratio": (work + checkpoint_work + resume_work)
            / (4 * (trace[-1].arrival - trace[0].arrival)),
        }
    )
    return output


def experiment(
    regime: str = "powers_of_two", shape: str = "lognormal_hetero", jobs: int = 4000, replications: int = 8
) -> dict:
    """Common-work cost grid with fixed useful load and paired SF/FirstFit/MSF CIs."""
    if regime not in REGIMES or shape not in SHAPES or jobs < 100 or replications < 2:
        raise ValueError("supported regime/shape, at least 100 jobs and 2 replications required")
    workload, warmup = WORKLOADS[regime], jobs // 10
    histories, runs = [], []
    for rep in range(replications):
        seed = 53000 + rep
        history, cohort, unit_arrivals = make_bundle(regime, shape, jobs + warmup, seed)
        models = fit_history(history, len(workload.needs))
        forecasts = initial_forecasts({"km": models}, cohort.classes)["km"]
        histories.append(
            {
                "seed": seed,
                "history_hash": array_digest(history.classes, history.observed_times, history.completed),
                "test_hash": fingerprint(cohort),
                "unit_arrivals_hash": array_digest(unit_arrivals),
                "forecasts_hash": array_digest(forecasts),
                "completed_fraction": float(history.completed.mean()),
                "initial_quantiles": [model.remaining_quantile(0, 0.9) for model in models],
            }
        )
        for load in LOADS:
            rate = load * 4 / workload.mean_work
            trace = tuple(
                MsjTraceJob(float(a / rate), int(c), float(s), float(e))
                for a, c, s, e in zip(unit_arrivals, cohort.classes, cohort.services, forecasts)
            )
            for policy in POLICIES:
                metrics = replay(workload, trace, policy, warmup, models)
                runs.append(
                    {
                        "seed": seed,
                        "load": load,
                        "policy": policy,
                        "arrival_rate": rate,
                        "sample_offered_load": float(
                            rate * np.mean(np.array(workload.needs)[cohort.classes] * cohort.services) / 4
                        ),
                        **metrics,
                    }
                )
    metrics = METRICS + tuple(f"{metric}_k{need}" for metric in ("mean_t", "p99_t") for need in workload.needs)
    return {
        "protocol": {
            "regime": regime,
            "shape": shape,
            "k": 4,
            **asdict(workload),
            "feature_slope": FEATURE_SLOPE,
            "loads": LOADS,
            "mean_work": workload.mean_work,
            "mean_service": float(np.dot(workload.probabilities, workload.means)),
            "cost_factors": COSTS,
            "policies": POLICIES,
            "jobs_measured": jobs,
            "warmup": warmup,
            "replications": replications,
            "first_seed": 53000,
            "history_jobs": 4000,
            "quantile_target": 0.9,
            "censor_mean_multiple": 5,
            "note": "Same service law and four independent data streams as EPIC-052, with new seeds. "
            "Synthetic log-size mixture, not marginal Erlang; historical KM is fixed before replay. "
            "Costs are deterministic factors of theoretical E[S], never realized job S. "
            "Checkpoint/resume hold K servers without useful progress. New preemptions gated while any overhead "
            "is active; overheads non-cancellable; fitting chosen jobs may still start. This is an explicit "
            "SF extension, not Slurm or the original zero-cost algorithm. Zero costs delegate to the original SF. "
            "Useful arrival load is NOT reduced to compensate for overhead. Same trace/estimates in every mode. "
            "W includes queue/checkpoint/resume; first wait and later interruptions are separate. "
            "Utilization counts allocated servers, productive utilization excludes overhead. "
            "Latency excludes warmup; counters/work include warmup/drain; time averages exclude drain. "
            "Cohort resource ratio uses all work over first-to-last arrival span, not stationary utilization. "
            "Every job drains. Backlog/throughput do not prove stability. No lost useful work or I/O contention. "
            "Student 95% intervals across seeds, paired within seed, without multiplicity adjustment. "
            "No real cost calibration or interpolated universal break-even threshold.",
        },
        "history_runs": histories,
        "replications": runs,
        "zero_contrasts": summarize(runs, ("load",), metrics, "policy", lambda _: ZERO),
        "first_fit_contrasts": summarize(runs, ("load",), metrics, "policy", lambda _: "first_fit"),
        "msf_contrasts": summarize(runs, ("load",), metrics, "policy", lambda _: "msf"),
    }


def main() -> None:
    """Save strict JSON and print weighted-T contrasts against MSF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=REGIMES, default="powers_of_two")
    parser.add_argument("--shape", choices=SHAPES, default="lognormal_hetero")
    parser.add_argument("--jobs", type=int, default=4000)
    parser.add_argument("--replications", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = experiment(args.regime, args.shape, args.jobs, args.replications)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in result["msf_contrasts"]:
        if row["policy"] in COSTS:
            print(row["load"], row["policy"], row["paired_delta"]["weighted_mean_t"])


if __name__ == "__main__":
    main()
