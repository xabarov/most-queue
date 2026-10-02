"""EPIC-060: prespecified GPU/node reservation and capacity sensitivity.

    python -m examples.gpu_resource_envelope_experiment --output-dir works/gpu_resource_envelope

Not an estimate of production capacity. Infeasible cells are retained explicitly,
without shrinking jobs or silently changing the fixed target population.
"""

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import scipy

from examples import modern_gpu_trace_experiment as gpu
from examples.real_trace_calibration_experiment import POLICIES, fingerprint, interval
from examples.real_trace_temporal_experiment import implementation_hashes
from most_queue.random.trace_resampling import ConditionalEmpirical
from most_queue.sim.msj_lifecycle import MsjLifecycleSim
from most_queue.sim.utils.acme_trace import acme_snapshot, parse_acme_kalos
from most_queue.sim.utils.resource_envelope import ResourceRequest, assess_envelope, observed_occupancy
from most_queue.sim.utils.workload_trace import chronological_split, positive_integer

ALLOCATIONS = ("gpu_pool", "exclusive_nodes")
NODES = (302, 226, 151, 113)
METRICS = (
    "mean_t",
    "p99_t",
    "mean_w",
    "p99_w",
    "gpu_weighted_mean_t",
    "reserved_utilization",
    "requested_gpu_utilization",
    "idle_with_queue",
    "reserved_gpu_equivalent_time",
    "requested_gpu_time",
)


@dataclass(frozen=True)
class EnvelopeConfig:
    """Fixed cohort size and independent history fits for one origin."""

    fraction: float
    jobs: int = 1000
    warmup: int = 200
    history_limit: int = 4000
    minimum: int = 20
    replications: int = 8
    first_seed: int = 60000

    def __post_init__(self):
        if isinstance(self.fraction, bool) or not np.isfinite(self.fraction) or not 0 < self.fraction < 1:
            raise ValueError("fraction must be inside (0,1)")
        for name in ("jobs", "warmup", "history_limit", "minimum", "replications", "first_seed"):
            positive_integer(getattr(self, name), name, minimum=0 if name in ("warmup", "first_seed") else 1)
        if min(self.history_limit, self.minimum, self.replications) < 2:
            raise ValueError("history_limit, minimum and replications must be >=2")


CONFIGS = (EnvelopeConfig(0.35), EnvelopeConfig(0.50), EnvelopeConfig(0.70, jobs=300, warmup=100))


def read_requests(lines, source):
    """Join audited node requests to accepted IDs without exporting users."""
    reader = csv.DictReader(lines)
    required = {"job_id", "gpu_num", "node_num"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError("resource metadata columns are missing")
    accepted = {j.job_id: j.need for j in source.terminal_jobs}
    requests, seen = {}, set()
    for row in reader:
        idx = row["job_id"]
        if idx in seen:
            raise ValueError("duplicate metadata ID")
        seen.add(idx)
        if idx not in accepted:
            continue
        values = [float(row[k]) for k in ("gpu_num", "node_num")]
        if not all(np.isfinite(v) and v >= 1 and v == int(v) for v in values) or values[0] != accepted[idx]:
            raise ValueError("resource metadata disagrees with accepted source")
        requests[idx] = ResourceRequest(*(int(v) for v in values))
    if set(requests) != set(accepted):
        raise ValueError("missing resource metadata for accepted jobs")
    return requests


def prepare(source, config):
    """History ends strictly before cutoff; all fixed targets remain present."""
    history, heldout, split = chronological_split(source, config.fraction)
    if len(heldout) < config.jobs + config.warmup or len(history) < 2:
        raise ValueError("not enough jobs for the prespecified origin")
    cohort = heldout[: config.jobs + config.warmup]
    if cohort[-1].submit <= cohort[config.warmup].submit:
        raise ValueError("measured arrivals must span positive time")
    running, waiting = acme_snapshot(source, cohort[0].submit, include_unsuccessful=True)
    selected = {j.job_id for j in cohort}
    arrivals = tuple(
        j
        for j in source.terminal_jobs
        if j.job_id in selected or (j.outcome != "completed" and cohort[0].submit <= j.submit <= cohort[-1].submit)
    )
    return {
        "history": history,
        "cohort": cohort,
        "split": split,
        "running": running,
        "waiting": waiting,
        "arrivals": arrivals,
    }


def envelope(prepared, requests, allocation, nodes):
    """Assess resource feasibility before fitting or scheduling this cell."""
    positive_integer(nodes, "nodes")
    if allocation not in ALLOCATIONS:
        raise ValueError("unknown allocation")

    def demand(jobs):
        return [requests[j.job_id].demand(allocation) for j in jobs]

    result = assess_envelope(
        nodes * (8 if allocation == "gpu_pool" else 1),
        demand(prepared["arrivals"]),
        running=demand(prepared["running"]),
        waiting=demand(prepared["waiting"]),
    )
    return {
        "allocation": allocation,
        "nodes": nodes,
        "gpu_inventory": 8 * nodes,
        "unit": "GPU" if allocation == "gpu_pool" else "node",
        **result,
    }


def project_case(base, prepared, requests, cell):
    """Change scheduler demand only; preserve times, outcomes, targets and forecasts."""
    if not cell["feasible"]:
        raise ValueError("cannot replay an infeasible resource envelope")
    jobs = (*prepared["running"], *prepared["waiting"], *prepared["arrivals"])
    original = (*[r.job for r in base.running], *base.waiting, *base.arrivals)
    if len(jobs) != len(original) or any(j.need != r.cls + 1 for j, r in zip(jobs, original)):
        raise ValueError("base case must align with original GPU source order")
    transformed = tuple(
        replace(r, cls=requests[j.job_id].demand(cell["allocation"]) - 1) for j, r in zip(jobs, original)
    )
    nr, nw = len(base.running), len(base.waiting)
    running = tuple(replace(r, job=transformed[i]) for i, r in enumerate(base.running))
    waiting, arrivals = transformed[nr : nr + nw], transformed[nr + nw :]
    tape = {
        "running": [asdict(r) for r in running],
        "waiting": [asdict(j) for j in waiting],
        "arrivals": [asdict(j) for j in arrivals],
        "source_ids": [j.job_id for j in jobs],
    }
    return replace(base, running=running, waiting=waiting, arrivals=arrivals, trace_sha256=gpu.json_hash(tape))


def run_case(case, prepared, cell, policy):
    """Measure target latency and separate requested-GPU from reserved-unit work."""
    simulator = MsjLifecycleSim(cell["capacity"], policy)
    simulator.set_servers(range(1, cell["capacity"] + 1))
    result = simulator.run_lifecycle(
        case.arrivals,
        initial_running=case.running,
        initial_waiting=case.waiting,
        observation_window=case.observation_window,
    )
    source_jobs = (*prepared["running"], *prepared["waiting"], *prepared["arrivals"])
    jobs = (*[r.job for r in case.running], *case.waiting, *case.arrivals)
    idx = np.array(case.target_indices)
    starts, ends = np.array(result.start_times), np.array(result.release_times)
    arrival = np.array([jobs[i].arrival for i in idx])
    target_t, target_w = ends[idx] - arrival, starts[idx] - arrival
    gpus = np.array([j.need for j in source_jobs])
    gpu_work = {}
    for outcome in result.resource_time_by_outcome:
        gpu_work[outcome] = float(
            sum(gpus[i] * result.consumed_services[i] for i, label in enumerate(result.outcomes) if label == outcome)
        )
    begin, end = case.observation_window
    occupied = np.maximum(0, np.minimum(ends, end) - np.maximum(starts, begin))
    factor = 8 if cell["allocation"] == "exclusive_nodes" else 1
    requested_utilization = float(np.dot(gpus, occupied) / (cell["gpu_inventory"] * (end - begin)))
    if not all(result.outcomes[i] == "completed" for i in idx):
        raise RuntimeError("completed target outcome changed without runtime limits")
    groups = []
    for group in range(4):
        mask = np.searchsorted([1, 8, 32], gpus[idx]) == group
        groups.append(
            {
                "count": int(sum(mask)),
                "mean_t": float(target_t[mask].mean()) if any(mask) else None,
                "mean_w": float(target_w[mask].mean()) if any(mask) else None,
                "p99_t": float(np.quantile(target_t[mask], 0.99)) if any(mask) else None,
            }
        )
    return {
        "target_count": len(idx),
        "success_rate": 1.0,
        "mean_t": float(target_t.mean()),
        "p99_t": float(np.quantile(target_t, 0.99)),
        "mean_w": float(target_w.mean()),
        "p99_w": float(np.quantile(target_w, 0.99)),
        "gpu_weighted_mean_t": float(np.average(target_t, weights=gpus[idx])),
        "target_mean_s": float(np.mean(np.array(result.consumed_services)[idx])),
        "groups": groups,
        "reserved_utilization": result.utilization,
        "requested_gpu_utilization": requested_utilization,
        "idle_with_queue": result.idle_with_queue,
        "reserved_work_by_outcome": result.resource_time_by_outcome,
        "requested_gpu_work_by_outcome": gpu_work,
        "reserved_gpu_equivalent_time": float(factor * sum(result.resource_time_by_outcome.values())),
        "requested_gpu_time": sum(gpu_work.values()),
        "reservation_violations": result.reservation_violations,
    }


def summarize(rows, cells):
    """Within-cell model errors and paired resource changes, never fitted capacity."""
    summaries, contrasts, decisions = [], [], []
    lookup = {
        (c["allocation"], c["nodes"], v, p): sorted(
            [
                r
                for r in rows
                if (r["allocation"], r["nodes"], r["variant"], r["policy"]) == (c["allocation"], c["nodes"], v, p)
            ],
            key=lambda r: -1 if r["seed"] is None else r["seed"],
        )
        for c in cells
        if c["feasible"]
        for v in ("observed", "recent_coarse")
        for p in POLICIES
    }
    for cell in cells:
        if not cell["feasible"]:
            continue
        allocation, nodes = cell["allocation"], cell["nodes"]
        for policy in POLICIES:
            ref = lookup[allocation, nodes, "observed", policy][0]
            model = lookup[allocation, nodes, "recent_coarse", policy]
            metrics = {}
            for metric in METRICS:
                estimate = interval([r[metric] for r in model])
                estimate.update(
                    reference=ref[metric], relative_error=estimate["mean"] / ref[metric] - 1 if ref[metric] else None
                )
                metrics[metric] = estimate
            summaries.append({"allocation": allocation, "nodes": nodes, "policy": policy, "metrics": metrics})
            for variant in ("observed", "recent_coarse"):
                values = lookup[allocation, nodes, variant, policy]
                baseline = lookup["gpu_pool", max(c["nodes"] for c in cells), variant, policy]
                delta = {}
                for metric in ("mean_t", "p99_t", "mean_w"):
                    differences = [r[metric] - b[metric] for r, b in zip(values, baseline)]
                    delta[metric] = (
                        interval(differences)
                        if len(differences) > 1
                        else {"mean": differences[0], "low": None, "high": None}
                    )
                contrasts.append(
                    {
                        "allocation": allocation,
                        "nodes": nodes,
                        "variant": variant,
                        "policy": policy,
                        "difference_from_nominal_gpu": delta,
                    }
                )
        for metric in ("mean_t", "p99_t"):
            reference = {p: lookup[allocation, nodes, "observed", p][0][metric] for p in POLICIES}
            means = {p: np.mean([r[metric] for r in lookup[allocation, nodes, "recent_coarse", p]]) for p in POLICIES}
            chosen, best = min(POLICIES, key=means.get), min(reference.values())
            ties = [p for p in POLICIES if np.isclose(reference[p], best, rtol=1e-12, atol=1e-9)]
            decisions.append(
                {
                    "allocation": allocation,
                    "nodes": nodes,
                    "metric": metric,
                    "chosen": chosen,
                    "reference_best": ties,
                    "reference_tie_count": len(ties),
                    "relative_regret": reference[chosen] / best - 1,
                }
            )
    return {"summaries": summaries, "contrasts": contrasts, "decisions": decisions}


def replay_task(task):
    """Run one isolated schedule; no randomness is drawn inside a worker."""
    case, source_parts, cell, policy, tape = task
    return {**tape, "policy": policy, **run_case(case, source_parts, cell, policy)}


def ordered_replays(tasks, workers):
    """Parallelize independent schedules without changing serialized row order."""
    positive_integer(workers, "workers")
    if workers == 1:
        yield from map(replay_task, tasks)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            yield from pool.map(replay_task, tasks)


def run_origin(
    source, requests, prepared, config, nodes=NODES, *, progress=False, workers=1
):  # pylint: disable=too-many-arguments
    """Replay the fixed resource matrix with a common service and forecast tape."""
    if tuple(nodes) != tuple(sorted(set(nodes), reverse=True)) or max(nodes) * 8 != source.capacity:
        raise ValueError("nodes must be unique descending levels beginning at source nominal capacity")
    cells = [envelope(prepared, requests, a, n) for a in ALLOCATIONS for n in nodes]
    if not cells[0]["feasible"]:
        raise ValueError("nominal GPU reference must be feasible")
    history, cohort = prepared["history"], prepared["cohort"]
    recent = history[-config.history_limit :]

    def fit(jobs):
        return ConditionalEmpirical.fit([j.need for j in jobs], [j.runtime for j in jobs], minimum=config.minimum)

    model, expanding = fit(recent), fit(history)
    forecasts = {
        k: float(np.quantile(expanding.distribution(k)[0].samples, 0.9)) for k in {j.need for j in source.terminal_jobs}
    }
    bundles = [("observed", None, np.array([j.runtime for j in cohort]))]
    for seed in range(config.first_seed, config.first_seed + config.replications):
        words = np.array([prepared["split"]["cutoff"]], dtype="<f8").view("<u4").tolist()
        rng = np.random.default_rng(np.random.SeedSequence([seed, *words]))
        uniforms = (rng.integers(0, 2**52, len(cohort), dtype=np.int64) + 0.5) / 2**52
        bundles.append(("recent_coarse", seed, model.quantiles([j.need for j in cohort], uniforms)))
    rows, tapes, tasks = [], [], []
    source_parts = {key: prepared[key] for key in ("running", "waiting", "arrivals")}
    for variant, seed, services in bundles:
        base = gpu.build_case(source, cohort, services, forecasts, "carry_terminal", warmup=config.warmup)
        for cell in cells:
            if not cell["feasible"]:
                continue
            case = project_case(base, prepared, requests, cell)
            tapes.append(
                {
                    "allocation": cell["allocation"],
                    "nodes": cell["nodes"],
                    "variant": variant,
                    "seed": seed,
                    "services_sha256": fingerprint(services),
                    "trace_sha256": case.trace_sha256,
                }
            )
            for policy in POLICIES:
                tasks.append((case, source_parts, cell, policy, tapes[-1]))
    for row in ordered_replays(tasks, workers):
        rows.append(row)
        if progress and len(rows) % len(POLICIES) == 0:
            print(
                json.dumps(
                    {
                        "fraction": config.fraction,
                        "variant": row["variant"],
                        "seed": row["seed"],
                        "allocation": row["allocation"],
                        "nodes": row["nodes"],
                        "runs": len(rows),
                    }
                ),
                flush=True,
            )
    return {
        "schema_version": 1,
        "config": asdict(config),
        "split": prepared["split"],
        "cells": cells,
        "history_sha256": gpu.json_hash([asdict(j) for j in history]),
        "recent_history_count": len(recent),
        "cohort_sha256": gpu.json_hash([asdict(j) for j in cohort]),
        "first_submit": cohort[0].submit,
        "last_submit": cohort[-1].submit,
        "measurement_start": cohort[config.warmup].submit,
        "forecast_sha256": gpu.json_hash(sorted(forecasts.items())),
        "context_counts": dict(Counter(j.workload_type for j in cohort[config.warmup :])),
        "scheduler_runs": len(rows),
        "tapes": tapes,
        "rows": rows,
        **summarize(rows, cells),
    }


def source_audit(source, requests):
    """Recorded-interval compatibility; do not use historical W to tune a cell."""
    output = {}
    for allocation in ALLOCATIONS:
        output[allocation] = observed_occupancy(
            [(j.started_at, j.ended_at, requests[j.job_id].demand(allocation)) for j in source.terminal_jobs],
            [n * (8 if allocation == "gpu_pool" else 1) for n in NODES],
        )
    waits = np.array([j.wait for j in source.jobs])
    output["completed_historical_wait"] = {
        "count": len(waits),
        "mean": float(waits.mean()),
        "p99": float(np.quantile(waits, 0.99)),
        "positive_fraction": float(np.mean(waits > 0)),
    }
    return output


def code_hashes():
    """Pin all new and reused runtime files without requiring a new git commit."""
    output = implementation_hashes()
    root = Path(__file__).resolve().parents[1]
    for filename in (
        "examples/gpu_resource_envelope_experiment.py",
        "examples/modern_gpu_trace_experiment.py",
        "examples/real_trace_lifecycle_experiment.py",
        "most_queue/sim/utils/resource_envelope.py",
        "most_queue/sim/utils/acme_trace.py",
        "most_queue/sim/msj_lifecycle.py",
    ):
        output[filename] = hashlib.sha256((root / filename).read_bytes()).hexdigest()
    return output


def main():
    """Publish the complete envelope before running any counterfactual schedule."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache/real_trace/acme-kalos.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    positive_integer(args.workers, "workers")
    lines = gpu.verified_source(args.cache, args.download).decode().splitlines()
    source = parse_acme_kalos(lines)
    requests = read_requests(lines, source)
    prepared = [prepare(source, c) for c in CONFIGS]
    if any(a["cohort"][-1].submit >= b["cohort"][0].submit for a, b in zip(prepared, prepared[1:])):
        raise ValueError("prespecified cohorts overlap")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []

    def write(filename, result):
        payload = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
        (args.output_dir / filename).write_bytes(payload)
        artifacts.append({"file": filename, "sha256": hashlib.sha256(payload).hexdigest()})

    write(
        "envelope.json",
        {
            "source_audit": source_audit(source, requests),
            "origins": [
                {"fraction": c.fraction, "cells": [envelope(p, requests, a, n) for a in ALLOCATIONS for n in NODES]}
                for p, c in zip(prepared, CONFIGS)
            ],
        },
    )
    total = 0
    for index, (p, c) in enumerate(zip(prepared, CONFIGS)):
        result = run_origin(source, requests, p, c, progress=True, workers=args.workers)
        write(f"origin-{index}.json", result)
        total += result["scheduler_runs"]
    write(
        "manifest.json",
        {
            "schema_version": 1,
            "source": {
                "page": gpu.SOURCE_PAGE,
                "url": gpu.SOURCE_URL,
                "commit": gpu.SOURCE_COMMIT,
                "sha256": gpu.SOURCE_SHA256,
                "attribution": "Shanghai AI Laboratory / InternLM, AcmeTrace Kalos 2023",
                "license": "CC-BY-4.0, not MIT",
            },
            "audit": source.audit,
            "configs": [asdict(c) for c in CONFIGS],
            "nodes": NODES,
            "allocations": ALLOCATIONS,
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "scheduler_runs": total,
            "implementation_sha256": code_hashes(),
            "artifacts": list(artifacts),
            "interpretation": "Explicit resource sensitivity, not inferred capacity or quotas. Same fixed targets, "
            "hypothetical exclusive nodes, observed environment, conditional MC only. Infeasible cells retained.",
        },
    )


if __name__ == "__main__":
    main()
