"""EPIC-058: pinned Acme/Kalos completed-service calibration and partial carry-in.

    python -m examples.modern_gpu_trace_experiment --download \
        --output-dir works/modern_gpu_trace

See docs/modern_gpu_trace.md. Raw third-party data is cached, not redistributed.
"""

import argparse
import hashlib
import json
import platform
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import scipy

from examples.real_trace_calibration_experiment import POLICIES, fingerprint, interval
from examples.real_trace_lifecycle_experiment import LifecycleCase, run_case
from examples.real_trace_temporal_experiment import (
    TemporalConfig,
    describe_services,
    implementation_hashes,
    prepare_origins,
)
from most_queue.random.trace_resampling import ConditionalEmpirical
from most_queue.sim.msj_lifecycle import LIFECYCLE_OUTCOMES, MsjCarryIn, MsjLifecycleJob
from most_queue.sim.utils.acme_trace import acme_snapshot, parse_acme_kalos
from most_queue.sim.utils.workload_trace import positive_integer

SOURCE_COMMIT = "f9fdf591b4876c2875a9e3d28adb1bda8120dfcb"
SOURCE_PAGE = f"https://github.com/InternLM/AcmeTrace/tree/{SOURCE_COMMIT}"
SOURCE_URL = f"https://raw.githubusercontent.com/InternLM/AcmeTrace/{SOURCE_COMMIT}/data/job_trace/trace_kalos.csv"
SOURCE_SHA256 = "7c5a7845da1d66448fa668342e4ecb622fd9a005118080660ec3ad2788d4c9bf"
SCENARIOS = ("empty_completed", "carry_completed", "carry_terminal")
MODELS = ("expanding_coarse", "recent_coarse")
METRICS = ("mean_t", "p99_t", "mean_w", "gpu_weighted_mean_t", "utilization", "idle_with_queue", "total_resource_time")
DEFAULT_CONFIG = TemporalConfig(fractions=(0.35, 0.45, 0.50, 0.55), first_seed=58000)


def verified_source(path: Path, download=False):
    """Require exact content; network opt-in, bounded read, no cache replacement."""
    limit = 12_000_000
    if path.exists():
        if path.stat().st_size > limit:
            raise ValueError("source exceeds size limit")
        payload = path.read_bytes()
    elif download:
        with urlopen(SOURCE_URL, timeout=45) as response:
            payload = response.read(limit + 1)
    else:
        raise ValueError("source not cached; review CC-BY-4.0 attribution and use --download")
    if len(payload) > limit:
        raise ValueError("source exceeds size limit")
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
        raise ValueError("Acme SHA-256 mismatch; refusing unverified source")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return payload


def json_hash(value):
    """Hash strings and numbers without converting source IDs to floats."""
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def source_diagnostics(source):
    """Describe observed requested-resource intervals, not physical GPU utilization."""
    if source.submit_range[1] <= source.submit_range[0]:
        raise ValueError("source diagnostics require a positive arrival span")
    events, work = [], dict.fromkeys(LIFECYCLE_OUTCOMES, 0.0)
    for job in source.terminal_jobs:
        events.extend(((job.started_at, job.need), (job.ended_at, -job.need)))
        work[job.outcome] += job.need * job.runtime
    occupied = peak = 0
    for _, delta in sorted(events):
        occupied += delta
        peak = max(peak, occupied)
    return {
        "accepted_request_seconds_by_outcome": work,
        "accepted_peak_simultaneous_gpu_request": peak,
        "completed_service": describe_services(
            [j.runtime for j in source.jobs],
            [j.need for j in source.jobs],
            source.submit_range[1] - source.submit_range[0],
            source.capacity,
        ),
    }


def build_case(source, cohort, services, forecasts, scenario, *, warmup):  # pylint: disable=too-many-arguments
    """Keep fixed completed target IDs and labelled historical environment.

    Only selected completed service is generated. Historical wait determines
    initial state but does not constrain a new arrival's simulated start.
    """
    positive_integer(warmup, "warmup", minimum=0)
    cohort = tuple(cohort)
    if scenario not in SCENARIOS or not cohort or warmup >= len(cohort):
        raise ValueError("unknown scenario or empty measured cohort")
    values = np.asarray(services, dtype=float)
    if values.shape != (len(cohort),) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("services must match the positive finite cohort")
    positions = {job.job_id: idx for idx, job in enumerate(cohort)}
    eligible = {job.job_id: job for job in source.jobs}
    if len(positions) != len(cohort) or any(eligible.get(job.job_id) != job for job in cohort):
        raise ValueError("unique completed target IDs must belong to source")
    if any(a.submit > b.submit for a, b in zip(cohort, cohort[1:])):
        raise ValueError("cohort must be chronological")
    boundary, end = cohort[0].submit, cohort[-1].submit
    terminal = scenario == "carry_terminal"
    running, waiting = (
        acme_snapshot(source, boundary, include_unsuccessful=terminal) if scenario != SCENARIOS[0] else ((), ())
    )
    records = tuple(
        job
        for job in source.terminal_jobs
        if job.job_id in positions or (terminal and job.outcome != "completed" and boundary <= job.submit <= end)
    )

    def convert(job, arrival, service):
        return MsjLifecycleJob(float(arrival), job.need - 1, float(service), float(forecasts[job.need]), job.outcome)

    active = tuple(MsjCarryIn(convert(job, 0, job.runtime), boundary - job.started_at) for job in running)
    queued = tuple(convert(job, 0, job.runtime) for job in waiting)
    arrivals = tuple(
        convert(job, job.submit - boundary, values[positions[job.job_id]] if job.job_id in positions else job.runtime)
        for job in records
    )
    offset = len(active) + len(queued)
    indices = {job.job_id: offset + idx for idx, job in enumerate(records)}
    audit = {
        "boundary": boundary,
        "last_submit": end,
        "measurement_start": cohort[warmup].submit,
        "initial_running": len(active),
        "initial_waiting": len(queued),
        "initial_running_gpu_request": sum(job.need for job in running),
        "initial_outcomes": dict(Counter(job.outcome for job in (*running, *waiting))),
        "new_outcomes": dict(Counter(job.outcome for job in records)),
        "target_count": len(cohort) - warmup,
    }
    tape = {
        "running": [asdict(item) for item in active],
        "waiting": [asdict(job) for job in queued],
        "arrivals": [asdict(job) for job in arrivals],
        "source_ids": [job.job_id for job in (*running, *waiting, *records)],
    }
    return LifecycleCase(
        arrivals,
        active,
        queued,
        tuple(indices[j.job_id] for j in cohort[warmup:]),
        tuple(j.need for j in cohort[warmup:]),
        (cohort[warmup].submit - boundary, end - boundary),
        audit,
        json_hash(tape),
    )


def replay_case(case, capacity, policy):
    """Reuse independently tested lifecycle metrics with explicit GPU terminology."""
    result = run_case(case, capacity, policy)
    if result["successful_target_count"] != result["target_count"]:
        raise RuntimeError("completed-only target cohort unexpectedly lost success")
    output = {
        key: result[key]
        for key in (
            "mean_w",
            "utilization",
            "idle_with_queue",
            "total_resource_time",
            "resource_time_by_outcome",
            "reservation_violations",
            "target_count",
        )
    }
    output.update(
        {
            "mean_t": result["mean_release_t"],
            "p99_t": result["p99_release_t"],
            "gpu_weighted_mean_t": result["node_weighted_mean_release_t"],
            "target_mean_s": result["target_consumed_service_mean"],
        }
    )
    return output


def summarize(rows):
    """Compare each model with same-scenario observed replay; pair changes by seed."""
    lookup = {
        (scenario, model, policy): sorted(
            [r for r in rows if (r["scenario"], r["model"], r["policy"]) == (scenario, model, policy)],
            key=lambda r: -1 if r["seed"] is None else r["seed"],
        )
        for scenario in SCENARIOS
        for model in ("observed", *MODELS)
        for policy in POLICIES
    }
    summaries, changes, decisions = [], [], []
    for scenario in SCENARIOS:
        for model in MODELS:
            for policy in POLICIES:
                observed = lookup[scenario, "observed", policy][0]
                metrics = {}
                for metric in METRICS:
                    estimate = interval([r[metric] for r in lookup[scenario, model, policy]])
                    ref = observed[metric]
                    metrics[metric] = {
                        **estimate,
                        "reference": ref,
                        "difference": estimate["mean"] - ref,
                        "relative_error": estimate["mean"] / ref - 1 if ref else None,
                    }
                summaries.append({"scenario": scenario, "model": model, "policy": policy, "metrics": metrics})
            for metric in ("mean_t", "p99_t"):
                means = {p: np.mean([r[metric] for r in lookup[scenario, model, p]]) for p in POLICIES}
                refs = {p: lookup[scenario, "observed", p][0][metric] for p in POLICIES}
                chosen, best = min(POLICIES, key=means.get), min(POLICIES, key=refs.get)
                decisions.append(
                    {
                        "scenario": scenario,
                        "model": model,
                        "metric": metric,
                        "chosen": chosen,
                        "reference_best": best,
                        "relative_regret": refs[chosen] / refs[best] - 1,
                    }
                )
    for scenario, baseline in zip(SCENARIOS[1:], SCENARIOS[:-1]):
        for model in ("observed", *MODELS):
            for policy in POLICIES:
                new, old = lookup[scenario, model, policy], lookup[baseline, model, policy]
                if [r["seed"] for r in new] != [r["seed"] for r in old]:
                    raise ValueError("scenario seeds must be paired")
                differences = {}
                for metric in ("mean_t", "p99_t", "total_resource_time"):
                    values = [a[metric] - b[metric] for a, b in zip(new, old)]
                    differences[metric] = (
                        interval(values) if len(values) > 1 else {"mean": values[0], "low": None, "high": None}
                    )
                changes.append(
                    {
                        "scenario": scenario,
                        "baseline": baseline,
                        "model": model,
                        "policy": policy,
                        "differences": differences,
                    }
                )
    return summaries, changes, decisions


def run_origin(source, prepared, config, *, progress=False):
    """Fit only historical successes; replay one complete prespecified origin."""
    history, recent, cohort, split = prepared
    models = {
        name: ConditionalEmpirical.fit([j.need for j in jobs], [j.runtime for j in jobs], minimum=config.minimum)
        for name, jobs in zip(MODELS, (history, recent))
    }
    needs = np.array([job.need for job in cohort])
    forecasts = {
        need: models[MODELS[0]].distribution(need)[0].parameters()["p90"]
        for need in {j.need for j in source.terminal_jobs}
    }
    bundles = [("observed", None, np.array([job.runtime for job in cohort]))]
    words = np.array([split["cutoff"]], dtype="<f8").view("<u4").tolist()
    for seed in range(config.first_seed, config.first_seed + config.replications):
        rng = np.random.default_rng(np.random.SeedSequence([seed, *words]))
        uniforms = (rng.integers(0, 2**52, len(cohort), dtype=np.int64) + 0.5) / 2**52
        bundles.extend((name, seed, models[name].quantiles(needs, uniforms)) for name in MODELS)
    rows, audits, diagnostics = [], {}, []
    span = cohort[-1].submit - cohort[config.warmup].submit
    for name, seed, services in bundles:
        diagnostics.append(
            {
                "model": name,
                "seed": seed,
                "service_sha256": fingerprint(services),
                **describe_services(services[config.warmup :], needs[config.warmup :], span, source.capacity),
            }
        )
    for scenario in SCENARIOS:
        for name, seed, services in bundles:
            case = build_case(source, cohort, services, forecasts, scenario, warmup=config.warmup)
            audits[scenario] = case.audit
            for policy in POLICIES:
                rows.append(
                    {
                        "scenario": scenario,
                        "model": name,
                        "seed": seed,
                        "policy": policy,
                        "trace_sha256": case.trace_sha256,
                        **replay_case(case, source.capacity, policy),
                    }
                )
        if progress:
            print(json.dumps({"cutoff": split["cutoff"], "scenario": scenario, "runs": len(rows)}), flush=True)
    summaries, changes, decisions = summarize(rows)
    fits = {}
    for name, model in models.items():
        fits[name] = {
            "pooled": model.pooled.parameters(),
            "coarse": [dist.parameters() if dist else None for dist in model.coarse],
        }
    return {
        "split": split,
        "recent_count": len(recent),
        "fits": fits,
        "cohort_sha256": json_hash([asdict(j) for j in cohort]),
        "history_sha256": json_hash([asdict(j) for j in history]),
        "forecast_sha256": json_hash(sorted(forecasts.items())),
        "scenario_audits": audits,
        "service_diagnostics": diagnostics,
        "scheduler_runs": len(rows),
        "rows": rows,
        "summaries": summaries,
        "scenario_changes": changes,
        "decisions": decisions,
    }


def aggregate(origins):
    """Equal origin/policy descriptive MAPE of MC means, not population intervals."""
    output = []
    for scenario in SCENARIOS:
        for model in MODELS:
            row = {"scenario": scenario, "model": model}
            cells = [
                s for origin in origins for s in origin["summaries"] if (s["scenario"], s["model"]) == (scenario, model)
            ]
            for metric in ("mean_t", "p99_t"):
                decisions = [
                    d
                    for o in origins
                    for d in o["decisions"]
                    if (d["scenario"], d["model"], d["metric"]) == (scenario, model, metric)
                ]
                row[metric] = {
                    "mape_of_mc_means": float(np.mean([abs(s["metrics"][metric]["relative_error"]) for s in cells])),
                    "matching_choices": sum(d["chosen"] == d["reference_best"] for d in decisions),
                    "mean_relative_regret": float(np.mean([d["relative_regret"] for d in decisions])),
                    "max_relative_regret": max(d["relative_regret"] for d in decisions),
                }
            output.append(row)
    return output


def main():
    """Verify source, validate cohorts, then write deterministic results and hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache/real_trace/acme-kalos.csv"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fractions", type=float, nargs="+", default=DEFAULT_CONFIG.fractions)
    parser.add_argument("--jobs", type=int, default=DEFAULT_CONFIG.jobs)
    parser.add_argument("--warmup", type=int, default=DEFAULT_CONFIG.warmup)
    parser.add_argument("--replications", type=int, default=DEFAULT_CONFIG.replications)
    args = parser.parse_args()
    config = TemporalConfig(
        fractions=tuple(args.fractions),
        jobs=args.jobs,
        warmup=args.warmup,
        replications=args.replications,
        first_seed=58000,
    )
    source = parse_acme_kalos(verified_source(args.cache, args.download).decode("utf-8").splitlines())
    prepared = prepare_origins(source, config)
    for _, _, cohort, _ in prepared:
        acme_snapshot(source, cohort[0].submit, include_unsuccessful=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    origins, artifacts = [], []
    for index, origin in enumerate(prepared):
        result = run_origin(source, origin, config, progress=True)
        result.update({"schema_version": 1, "fraction": config.fractions[index], "config": asdict(config)})
        payload = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
        name = f"origin-{index}.json"
        (args.output_dir / name).write_bytes(payload)
        artifacts.append({"file": name, "sha256": hashlib.sha256(payload).hexdigest()})
        origins.append(result)
    hashes = implementation_hashes()
    root = Path(__file__).resolve().parents[1]
    for name in (
        "examples/modern_gpu_trace_experiment.py",
        "examples/real_trace_lifecycle_experiment.py",
        "most_queue/sim/utils/acme_trace.py",
        "most_queue/sim/msj_lifecycle.py",
    ):
        hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "config": asdict(config),
        "capacity": source.capacity,
        "submit_range": source.submit_range,
        "audit": source.audit,
        "source_diagnostics": source_diagnostics(source),
        "source": {
            "page": SOURCE_PAGE,
            "download": SOURCE_URL,
            "sha256": SOURCE_SHA256,
            "commit": SOURCE_COMMIT,
            "license": "CC-BY-4.0",
            "attribution": "Shanghai AI Laboratory; Hu et al., NSDI 2024",
            "transformations": "GPU-only positive end-start intervals; separate terminal labels; "
            "no raw rows redistributed",
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "implementation_sha256": hashes,
        "artifacts": artifacts,
        "scheduler_runs": sum(o["scheduler_runs"] for o in origins),
        "aggregate": aggregate(origins),
        "interpretation": "Partial retrospective GPU-request pool replay, not production reconstruction. "
        "No topology, quotas, CPU/memory, latent success demand, retries or inferred runtime budgets. "
        "Non-success durations are a service-clock sensitivity. Intervals cover conditional MC only.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
