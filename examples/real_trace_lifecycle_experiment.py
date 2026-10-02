"""EPIC-057: prespecified carry-in, observed cancellation and runtime-limit sensitivity.

    python -m examples.real_trace_lifecycle_experiment --output-dir works/real_trace_lifecycle

Only completed target durations are generated. Observed carry-in/cancelled work
is fixed across models; this is retrospective sensitivity, not an online trial.
"""

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy

from examples.real_trace_calibration_experiment import (
    MIRROR_COMMIT,
    POLICIES,
    SOURCE_PAGE,
    SOURCE_SHA256,
    SOURCE_URL,
    fingerprint,
    interval,
    verified_source,
)
from examples.real_trace_temporal_experiment import TemporalConfig, fit_context, implementation_hashes, prepare_origins
from most_queue.sim.msj_lifecycle import MsjCarryIn, MsjLifecycleJob, MsjLifecycleSim
from most_queue.sim.utils.workload_lifecycle import observed_snapshot, parse_swf_lifecycle

SCENARIOS = ("empty_completed", "carry_completed", "carry_cancelled", "requested_limit")
MODELS = ("expanding_coarse", "recent_coarse")
METRICS = (
    "mean_release_t",
    "p99_release_t",
    "mean_w",
    "node_weighted_mean_release_t",
    "success_rate",
    "timeout_rate",
    "mean_success_t",
    "p99_success_t",
    "utilization",
    "idle_with_queue",
    "total_resource_time",
)


@dataclass(frozen=True)
class LifecycleCase:
    """Immutable common workload with fixed target IDs and observation window."""

    arrivals: tuple
    running: tuple
    waiting: tuple
    target_indices: tuple
    target_needs: tuple
    observation_window: tuple
    audit: dict
    trace_sha256: str


def build_case(source, cohort, services, forecasts, scenario, *, warmup):  # pylint: disable=too-many-arguments
    """Merge visible competitors while preserving the completed target cohort.

    Historical wait is used ONLY to identify carry-in state and elapsed service.
    Unknown cancellations are absent from source.jobs, not treated as zero work.
    """
    if scenario not in SCENARIOS:
        raise ValueError("unknown lifecycle scenario")
    values = np.asarray(services, dtype=float)
    if values.shape != (len(cohort),) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("services must match the positive finite target cohort")
    if isinstance(warmup, bool) or not isinstance(warmup, (int, np.integer)) or not 0 <= warmup < len(cohort):
        raise ValueError("warmup must leave measured targets")
    boundary, end = cohort[0].submit, cohort[-1].submit
    cancellations = scenario in ("carry_cancelled", "requested_limit")
    snapshot = (
        observed_snapshot(source, boundary, include_cancelled=cancellations) if scenario != SCENARIOS[0] else None
    )
    target_position = {job.job_id: idx for idx, job in enumerate(cohort)}
    records = [
        job
        for job in source.jobs
        if job.job_id in target_position or (cancellations and job.status == 5 and boundary <= job.submit <= end)
    ]
    if sum(job.job_id in target_position for job in records) != len(cohort):
        raise ValueError("target IDs must be unique and present in lifecycle source")

    def convert(job, arrival, service, limited=False):
        return MsjLifecycleJob(
            float(arrival),
            job.need - 1,
            float(service),
            float(forecasts[job.need]),
            "completed" if job.status == 1 else "cancelled",
            job.requested_time if limited else None,
        )

    running = tuple(
        MsjCarryIn(convert(job, 0, job.runtime), boundary - job.started_at)
        for job in (snapshot.running if snapshot else ())
    )
    waiting = tuple(convert(job, 0, job.runtime) for job in (snapshot.waiting if snapshot else ()))
    arrivals = tuple(
        convert(
            job,
            job.submit - boundary,
            values[target_position[job.job_id]] if job.job_id in target_position else job.runtime,
            scenario == "requested_limit",
        )
        for job in records
    )
    initial = len(running) + len(waiting)
    positions = {job.job_id: initial + idx for idx, job in enumerate(records)}
    targets = tuple(positions[job.job_id] for job in cohort[warmup:])
    window = (cohort[warmup].submit - boundary, end - boundary)
    serialized = []
    for kind, job, age in (
        [(0, item.job, item.age) for item in running]
        + [(1, job, 0) for job in waiting]
        + [(2, job, 0) for job in arrivals]
    ):
        serialized.append(
            [
                kind,
                job.arrival,
                job.cls + 1,
                job.service,
                job.estimate,
                job.outcome == "cancelled",
                job.runtime_limit or -1,
                age,
            ]
        )
    audit = {
        "snapshot_at": boundary,
        "initial_running": len(running),
        "initial_running_nodes": sum(item.job.cls + 1 for item in running),
        "initial_waiting": len(waiting),
        "initial_cancelled": sum(item.job.outcome == "cancelled" for item in running)
        + sum(job.outcome == "cancelled" for job in waiting),
        "cancelled_arrivals": sum(job.status == 5 for job in records),
        "new_arrivals": len(arrivals),
        "new_arrivals_missing_requested_time": sum(job.requested_time is None for job in records),
        "target_count": len(targets),
    }
    return LifecycleCase(
        arrivals,
        running,
        waiting,
        targets,
        tuple(job.need for job in cohort[warmup:]),
        window,
        audit,
        fingerprint(serialized),
    )


def run_case(case, capacity, policy):
    """Extract fixed-target terminal latency and separate success accounting."""
    simulator = MsjLifecycleSim(capacity, policy)
    simulator.set_servers(range(1, capacity + 1))
    result = simulator.run_lifecycle(
        case.arrivals,
        initial_running=case.running,
        initial_waiting=case.waiting,
        observation_window=case.observation_window,
    )
    idx = np.array(case.target_indices)
    target_jobs = [case.arrivals[i - result.initial_count] for i in idx]
    arrival = np.array([job.arrival for job in target_jobs])
    terminal = np.array(result.release_times)[idx] - arrival
    wait = np.array(result.start_times)[idx] - arrival
    outcome = np.array(result.outcomes)[idx]
    successful = terminal[outcome == "completed"]
    groups = []
    labels = np.searchsorted([1, 8, 32], case.target_needs)
    for group in range(4):
        values = terminal[labels == group]
        groups.append(
            {
                "count": int(len(values)),
                "mean_release_t": float(values.mean()) if len(values) else None,
                "p99_release_t": float(np.quantile(values, 0.99)) if len(values) else None,
            }
        )
    return {
        "mean_release_t": float(terminal.mean()),
        "p99_release_t": float(np.quantile(terminal, 0.99)),
        "mean_w": float(wait.mean()),
        "node_weighted_mean_release_t": float(np.average(terminal, weights=case.target_needs)),
        "success_rate": float(np.mean(outcome == "completed")),
        "timeout_rate": float(np.mean(outcome == "timed_out")),
        "mean_success_t": float(successful.mean()) if len(successful) else None,
        "p99_success_t": float(np.quantile(successful, 0.99)) if len(successful) else None,
        "successful_target_count": int(len(successful)),
        "target_count": len(target_jobs),
        "target_consumed_service_mean": float(np.mean(np.array(result.consumed_services)[idx])),
        "utilization": result.utilization,
        "idle_with_queue": result.idle_with_queue,
        "total_resource_time": float(sum(result.resource_time_by_outcome.values())),
        "resource_time_by_outcome": result.resource_time_by_outcome,
        "groups": groups,
        "reservation_violations": result.reservation_violations,
    }


def summarize(rows):
    """MC intervals condition on the recorded carry-in and cancellation tape."""
    lookup = {
        (scenario, model, policy): [
            row for row in rows if (row["scenario"], row["model"], row["policy"]) == (scenario, model, policy)
        ]
        for scenario in SCENARIOS
        for model in ("observed", *MODELS)
        for policy in POLICIES
    }
    summaries, changes, decisions = [], [], []
    for scenario in SCENARIOS:
        for model in MODELS:
            for policy in POLICIES:
                samples, observed = lookup[scenario, model, policy], lookup[scenario, "observed", policy][0]
                metrics = {}
                for metric in METRICS:
                    values = [row[metric] for row in samples]
                    if any(value is None for value in values) or observed[metric] is None:
                        metrics[metric] = None
                        continue
                    estimate = interval(values)
                    estimate["reference"] = observed[metric]
                    estimate["relative_error"] = estimate["mean"] / observed[metric] - 1 if observed[metric] else None
                    estimate["difference"] = estimate["mean"] - observed[metric]
                    metrics[metric] = estimate
                summaries.append({"scenario": scenario, "model": model, "policy": policy, "metrics": metrics})
            for metric in ("mean_release_t", "p99_release_t"):
                predicted = {
                    policy: np.mean([r[metric] for r in lookup[scenario, model, policy]]) for policy in POLICIES
                }
                observed = {policy: lookup[scenario, "observed", policy][0][metric] for policy in POLICIES}
                chosen, best = min(POLICIES, key=predicted.get), min(POLICIES, key=observed.get)
                decisions.append(
                    {
                        "scenario": scenario,
                        "model": model,
                        "metric": metric,
                        "chosen": chosen,
                        "reference_best": best,
                        "relative_regret": observed[chosen] / observed[best] - 1,
                    }
                )
    for scenario, baseline in zip(SCENARIOS[1:], SCENARIOS[:-1]):
        for model in ("observed", *MODELS):
            for policy in POLICIES:
                metrics = {}
                for metric in ("mean_release_t", "p99_release_t", "success_rate", "total_resource_time"):
                    samples, bases = lookup[scenario, model, policy], lookup[baseline, model, policy]
                    values = [new[metric] - old[metric] for new, old in zip(samples, bases)]
                    estimate = interval(values) if len(values) > 1 else {"mean": values[0], "low": None, "high": None}
                    metrics[metric] = estimate
                changes.append(
                    {
                        "scenario": scenario,
                        "baseline": baseline,
                        "model": model,
                        "policy": policy,
                        "differences": metrics,
                    }
                )
    return summaries, changes, decisions


def run_origin(source, prepared, config, *, progress=False):
    """Run all four scenarios on identical observed/generated target durations."""
    history, recent, cohort, split = prepared
    models, _ = fit_context(history, recent, config)
    needs = np.array([job.need for job in cohort])
    forecasts = {
        need: models["expanding_coarse"].distribution(need)[0].parameters()["p90"]
        for need in range(1, source.completed.capacity + 1)
    }
    bundles = [("observed", None, np.array([job.runtime for job in cohort]))]
    words = np.array([split["cutoff"]], dtype="<f8").view("<u4").tolist()
    for seed in range(config.first_seed, config.first_seed + config.replications):
        rng = np.random.default_rng(np.random.SeedSequence([seed, *words]))
        uniforms = (rng.integers(0, 2**52, len(cohort), dtype=np.int64) + 0.5) / 2**52
        bundles.extend((model, seed, models[model].quantiles(needs, uniforms)) for model in MODELS)
    rows, audits = [], {}
    for scenario in SCENARIOS:
        for model, seed, services in bundles:
            case = build_case(source, cohort, services, forecasts, scenario, warmup=config.warmup)
            audits[scenario] = case.audit
            for policy in POLICIES:
                rows.append(
                    {
                        "scenario": scenario,
                        "model": model,
                        "seed": seed,
                        "policy": policy,
                        "trace_sha256": case.trace_sha256,
                        **run_case(case, source.completed.capacity, policy),
                    }
                )
        if progress:
            print(
                json.dumps({"cutoff": split["cutoff"], "scenario": scenario, "completed_runs": len(rows)}), flush=True
            )
    summaries, changes, decisions = summarize(rows)
    return {
        "split": split,
        "history_recent_count": len(recent),
        "cohort_sha256": fingerprint([[j.job_id, j.submit, j.runtime, j.need] for j in cohort]),
        "forecast_sha256": fingerprint([forecasts[job.need] for job in cohort]),
        "scenario_audits": audits,
        "scheduler_runs": len(rows),
        "rows": rows,
        "summaries": summaries,
        "scenario_changes": changes,
        "decisions": decisions,
    }


def aggregate(origins):
    """Descriptive equal-origin/policy weighting; no independent-fold claim."""
    output = []
    for scenario in SCENARIOS:
        for model in MODELS:
            row = {"scenario": scenario, "model": model}
            cells = [
                s for origin in origins for s in origin["summaries"] if (s["scenario"], s["model"]) == (scenario, model)
            ]
            for metric in ("mean_release_t", "p99_release_t"):
                decisions = [
                    d
                    for origin in origins
                    for d in origin["decisions"]
                    if (d["scenario"], d["model"], d["metric"]) == (scenario, model, metric)
                ]
                row[metric] = {
                    "mape_of_mc_means": float(np.mean([abs(s["metrics"][metric]["relative_error"]) for s in cells])),
                    "matching_choices": sum(d["chosen"] == d["reference_best"] for d in decisions),
                    "mean_relative_regret": float(np.mean([d["relative_regret"] for d in decisions])),
                    "max_relative_regret": max(d["relative_regret"] for d in decisions),
                }
            row["model_success_rate"] = float(np.mean([s["metrics"]["success_rate"]["mean"] for s in cells]))
            row["observed_success_rate"] = float(np.mean([s["metrics"]["success_rate"]["reference"] for s in cells]))
            output.append(row)
    return output


def main():
    """Verify pinned data and write strict JSON with code and artifact hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache/real_trace/SDSC-SP2-1998-4.2-cln.swf"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.35, 0.5, 0.65, 0.8])
    parser.add_argument("--jobs", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--replications", type=int, default=8)
    args = parser.parse_args()
    config = TemporalConfig(
        fractions=tuple(args.fractions), jobs=args.jobs, warmup=args.warmup, replications=args.replications
    )
    source = parse_swf_lifecycle(verified_source(args.cache, args.download).decode("ascii").splitlines(), 128)
    prepared = prepare_origins(source.completed, config)
    # Validate all requested snapshots before any expensive scheduling work.
    for _, _, cohort, _ in prepared:
        observed_snapshot(source, cohort[0].submit, include_cancelled=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    origins, artifacts = [], []
    for index, origin in enumerate(prepared):
        result = run_origin(source, origin, config, progress=True)
        result.update({"schema_version": 1, "fraction": config.fractions[index], "config": asdict(config)})
        content = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode("utf-8")
        filename = f"origin-{index}.json"
        (args.output_dir / filename).write_bytes(content)
        artifacts.append({"file": filename, "sha256": hashlib.sha256(content).hexdigest()})
        origins.append(result)
    hashes = implementation_hashes()
    root = Path(__file__).resolve().parents[1]
    for name in (
        "examples/real_trace_lifecycle_experiment.py",
        "most_queue/sim/msj_lifecycle.py",
        "most_queue/sim/utils/workload_lifecycle.py",
    ):
        hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "config": asdict(config),
        "capacity": source.completed.capacity,
        "audit": source.audit,
        "source": {
            "page": SOURCE_PAGE,
            "download": SOURCE_URL,
            "sha256": SOURCE_SHA256,
            "mirror_commit": MIRROR_COMMIT,
            "attribution": "SDSC / Victor Hazlewood; conversion Dror Feitelson; DIR-LAB mirror",
            "usage_conditions": "NPACI JOBLOG research/educational/non-profit conditions; not MIT data",
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "implementation_sha256": hashes,
        "artifacts": artifacts,
        "scheduler_runs": sum(origin["scheduler_runs"] for origin in origins),
        "aggregate": aggregate(origins),
        "interpretation": "Retrospective partial carry-in and service-clock cancellation sensitivity. "
        "Requested-time caps are hypothetical enforcement; terminal latency is not successful-completion latency. "
        "Unknown cancellations and latent completion demand are not imputed.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
