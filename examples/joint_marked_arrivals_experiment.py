"""EPIC-062: joint marked arrivals with fixed-service and matched-order controls.

    python -m examples.joint_marked_arrivals_experiment --output-dir works/joint_marked_arrivals

All scenarios are completed-only empty-start replays, not lifecycle reconstruction.
No raw records are redistributed. History and cohorts freeze before replay.
"""

import argparse
import hashlib
import json
import multiprocessing
import platform
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy

from examples import feature_service_experiment as feature
from most_queue.random.marked_arrivals import (
    ArrivalMark,
    MarkedArrivalBootstrap,
    anchored_permutation,
    arrival_times,
    decouple_gaps,
)
from most_queue.random.queue_selection import queue_log_error
from most_queue.random.trace_resampling import ConditionalEmpirical, lag_correlation
from most_queue.sim.msj_general import MsjGeneralSim, MsjTraceJob
from most_queue.sim.utils.workload_trace import chronological_split, positive_integer

POLICIES = feature.swf.POLICIES
VARIANTS = (
    "fixed_coarse",
    "recent_gap_independent",
    "recent_joint_iid",
    "recent_joint_block20",
    "recent_joint_shuffle20",
    "expanding_joint_iid",
)
CONTRASTS = (
    ("recent_joint_iid", "recent_gap_independent"),
    ("recent_joint_block20", "recent_joint_shuffle20"),
    ("recent_joint_iid", "expanding_joint_iid"),
    *((v, "fixed_coarse") for v in VARIANTS[1:]),
)
METRICS = ("mean_t", "p99_t", "mean_w", "weighted_mean_t", "utilization", "idle_with_queue", "resource_time")


@dataclass(frozen=True)
class JointConfig:
    """Fixed temporal origins and sample sizes, never fitted to test queues."""

    fractions: tuple = (0.85, 0.90)
    jobs: int = 1000
    warmup: int = 200
    history_limit: int = 4000
    block_length: int = 20
    minimum: int = 20
    replications: int = 8
    first_seed: int = 63000

    def __post_init__(self):
        for name in ("jobs", "warmup", "history_limit", "block_length", "minimum", "replications", "first_seed"):
            positive_integer(getattr(self, name), name, minimum=0 if name in ("warmup", "first_seed") else 1)
        if min(self.history_limit, self.minimum, self.replications) < 2 or self.block_length > self.history_limit:
            raise ValueError("need history/minimum/replications >=2 and block_length <= history_limit")
        values = np.asarray(self.fractions)
        if values.ndim != 1 or not values.size or values.dtype.kind not in "iuf" or not np.all(np.isfinite(values)):
            raise ValueError("fractions must be a nonempty finite numeric vector")
        if np.any((values <= 0) | (values >= 1)) or np.any(np.diff(values) <= 0):
            raise ValueError("fractions must increase strictly inside (0,1)")


CONFIGS = {"sdsc": JointConfig(), "kalos": JointConfig((0.65, 0.70), jobs=300, warmup=100)}


def prepare(source, name, config):
    """Freeze disjoint held-out cohorts and fully resolved arrival prefixes."""
    completed = feature.completed_source(source, name)
    blocks, previous_end = [], -np.inf
    for index, fraction in enumerate(config.fractions):
        history, heldout, split = chronological_split(completed, fraction)
        count = config.jobs + config.warmup
        if len(heldout) < count or len(history) < 2:
            raise ValueError("not enough completed history or full held-out cohort")
        cohort = heldout[:count]
        if cohort[0].submit <= previous_end:
            raise ValueError("held-out arrival cohorts overlap")
        previous_end = cohort[-1].submit
        past = tuple(j for j in completed.jobs if j.submit < split["cutoff"])
        stop = next((i for i, j in enumerate(past) if j.completed_at >= split["cutoff"]), len(past))
        prefix = past[:stop]
        if len(prefix) <= config.block_length:
            raise ValueError("resolved arrival prefix is too short for fixed blocks")
        audit = {
            "past_arrivals": len(past),
            "prefix_jobs": len(prefix),
            "prefix_donors": len(prefix) - 1,
            "omitted_suffix": len(past) - len(prefix),
            "completed_suffix_omitted": sum(j.completed_at < split["cutoff"] for j in past[stop:]),
            "last_prefix_submit": prefix[-1].submit,
            "prefix_lag": split["cutoff"] - prefix[-1].submit,
        }
        blocks.append(
            {
                "index": index,
                "fraction": fraction,
                "split": split,
                "capacity": completed.capacity,
                "cohort": cohort,
                "history": history[-config.history_limit :],
                "expanding_service": history,
                "prefix": prefix,
                "recent_prefix": prefix[-(config.history_limit + 1) :],
                "prefix_audit": audit,
            }
        )
    return blocks


def fit_arrivals(jobs, name):
    """Fit gap/mark donors from an explicitly contiguous resolved prefix."""
    return MarkedArrivalBootstrap.fit(
        [j.submit for j in jobs],
        [j.need for j in jobs],
        [feature.context(j, name) for j in jobs],
        [j.requested_time if name == "sdsc" else None for j in jobs],
    )


def observed_marks(jobs, name):
    """Represent the recorded cohort relative to its first arrival."""
    gaps = np.diff([j.submit for j in jobs], prepend=jobs[0].submit)
    return tuple(
        ArrivalMark(float(gap), j.need, feature.context(j, name), j.requested_time if name == "sdsc" else None)
        for j, gap in zip(jobs, gaps)
    )


def uniform_streams(name, cutoff, seed, count):
    """Four disjoint spawned streams: donor, service, mark order, block order."""
    words = np.asarray([cutoff], dtype="<f8").view("<u4").tolist()
    children = np.random.SeedSequence([seed, ("sdsc", "kalos").index(name), *words]).spawn(4)
    return tuple(
        (np.random.default_rng(child).integers(0, 2**52, count, dtype=np.int64) + 0.5) / 2**52 for child in children
    )


def service_fit(jobs, minimum):
    """Fit the common conditional service law without arrival-prefix restriction."""
    return ConditionalEmpirical.fit([j.need for j in jobs], [j.runtime for j in jobs], minimum=minimum)


def forecasts_for(block, config):
    """Freeze expanding coarse linear p90 for each original resource request."""
    model, cached, result = service_fit(block["expanding_service"], config.minimum), {}, {}
    for need in range(1, block["capacity"] + 1):
        distribution, _ = model.distribution(need)
        if id(distribution) not in cached:
            cached[id(distribution)] = distribution.parameters()["p90"]
        result[need] = cached[id(distribution)]
    return result


def workloads(block, name, config):
    """Yield paired tapes; shuffle moves S too and never regenerates its work."""
    fixed = observed_marks(block["cohort"], name)
    yield "observed", None, fixed, np.array([j.runtime for j in block["cohort"]])
    recent, expanding = fit_arrivals(block["recent_prefix"], name), fit_arrivals(block["prefix"], name)
    model = service_fit(block["history"], config.minimum)
    for seed in range(config.first_seed, config.first_seed + config.replications):
        donor_u, service_u, mark_u, shuffle_u = uniform_streams(name, block["split"]["cutoff"], seed, len(fixed))
        iid = recent.sample(donor_u)
        block_marks = recent.sample(donor_u, block_length=config.block_length)
        block_s = model.quantiles([r.need for r in block_marks], service_u)
        shuffle = anchored_permutation(shuffle_u, config.warmup)
        marks = {
            "fixed_coarse": fixed,
            "recent_gap_independent": decouple_gaps(iid, anchored_permutation(mark_u, config.warmup)),
            "recent_joint_iid": iid,
            "recent_joint_block20": block_marks,
            "recent_joint_shuffle20": tuple(block_marks[i] for i in shuffle),
            "expanding_joint_iid": expanding.sample(donor_u),
        }
        for variant, records in marks.items():
            services = (
                block_s[shuffle]
                if variant == "recent_joint_shuffle20"
                else (
                    block_s
                    if variant == "recent_joint_block20"
                    else model.quantiles([r.need for r in records], service_u)
                )
            )
            yield variant, seed, records, services


def diagnostics(records, services, warmup, reference):
    """Report generated mixture/horizon, never align synthetic IDs to real jobs."""
    times = arrival_times(records)
    targets, observed = records[warmup:], reference[warmup:]
    gaps, needs = np.diff(times[warmup:]), np.array([r.need for r in targets])
    selected_s = np.array(services[warmup:])
    horizon = float(times[-1] - times[warmup])

    def counts(rows, field):
        return Counter(field(r) for r in rows)

    def tv(field):
        a, b = counts(targets, field), counts(observed, field)
        return float(
            0.5 * sum(abs(a[k] / len(targets) - b[k] / len(observed)) for k in sorted(a.keys() | b.keys(), key=repr))
        )

    def group(record):
        return int(np.searchsorted([1, 8, 32], record.need))

    correlation = None
    if len(gaps) > 1 and np.std(gaps) > 0 and np.std(needs[1:]) > 0:
        correlation = float(np.corrcoef(gaps, needs[1:])[0, 1])
    return {
        "target_count": len(targets),
        "arrival_span": horizon,
        "mean_gap": float(gaps.mean()) if gaps.size else None,
        "gap_scv": float(np.var(gaps) / np.mean(gaps) ** 2) if gaps.size and np.mean(gaps) > 0 else None,
        "zero_gap_fraction": float(np.mean(gaps == 0)) if gaps.size else None,
        "lag1_gap": lag_correlation(gaps),
        "lag1_need": lag_correlation(needs),
        "gap_need_correlation": correlation,
        "need_group_tv": tv(group),
        "context_tv": tv(lambda r: r.context),
        "joint_mark_tv": tv(lambda r: (group(r), r.context)),
        "group_counts": [int(sum(group(r) == g for r in targets)) for g in range(4)],
        "context_counts": dict(Counter(str(r.context) for r in targets)),
        "mean_s": float(selected_s.mean()),
        "p99_s": float(np.quantile(selected_s, 0.99)),
        "target_resource_work": float(np.dot(needs, selected_s)),
        "all_resource_work": float(np.dot([r.need for r in records], services)),
        "transition_rate": (len(targets) - 1) / horizon if horizon else None,
    }


def trace_for(records, services, forecasts):
    """Create a rigid-job tape with exact K and common frozen forecasts."""
    needs = tuple(sorted({r.need for r in records}))
    classes = {need: i for i, need in enumerate(needs)}
    trace = tuple(
        MsjTraceJob(float(t), classes[r.need], float(s), float(forecasts[r.need]))
        for t, r, s in zip(arrival_times(records), records, services)
    )
    return needs, trace


def replay(task):
    """Drain a common tape and independently integrate full allocated work."""
    metadata, capacity, needs, trace, warmup = task
    simulator = MsjGeneralSim(capacity, metadata["policy"])
    simulator.set_servers(needs)
    result = simulator.run_trace(trace, warmup_jobs=warmup)
    weights = np.array([needs[j.cls] for j in trace[warmup:]])
    times = np.array(result.sojourn_samples)
    labels = np.searchsorted([1, 8, 32], weights)
    groups = []
    for group in range(4):
        sample = times[labels == group]
        groups.append(
            {
                "count": len(sample),
                "mean_t": float(sample.mean()) if len(sample) else None,
                "p99_t": float(np.quantile(sample, 0.99)) if len(sample) else None,
            }
        )
    work = float(np.dot([needs[j.cls] for j in trace], np.array(result.completion_times) - result.start_times))
    return {
        **metadata,
        "mean_t": result.v[0],
        "p99_t": result.v_quantiles[0.99],
        "mean_w": result.w[0],
        "weighted_mean_t": float(np.average(times, weights=weights)),
        "utilization": result.utilization,
        "idle_with_queue": result.idle_with_queue,
        "resource_time": work,
        "groups": groups,
        "target_count": len(weights),
        "target_mean_s": float(np.mean([j.service for j in trace[warmup:]])),
        "reservation_violations": result.reservation_violations,
    }


def run_tasks(tasks, workers):
    """Keep deterministic row order; no random draws occur in workers."""
    positive_integer(workers, "workers")
    if workers == 1:
        return list(map(replay, tasks))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(replay, tasks))


def summarize(rows):
    """Separate error of MC means from paired individual-replication errors."""
    lookup = {
        (v, p): sorted(
            [r for r in rows if (r["variant"], r["policy"]) == (v, p)],
            key=lambda r: -1 if r["seed"] is None else r["seed"],
        )
        for v in ("observed", *VARIANTS)
        for p in POLICIES
    }
    summaries, decisions, losses = [], [], []
    refs = [[lookup["observed", p][0][k] for k in ("mean_t", "p99_t")] for p in POLICIES]
    per_seed = {}
    for variant in VARIANTS:
        for policy in POLICIES:
            metrics = {}
            for metric in METRICS:
                ref = lookup["observed", policy][0][metric]
                values = [r[metric] for r in lookup[variant, policy] if r[metric] is not None]
                estimate = (
                    feature.swf.interval(values) if len(values) >= 2 else {"mean": None, "low": None, "high": None}
                )
                estimate.update(reference=ref, valid_replications=len(values))
                estimate["relative_error"] = (
                    estimate["mean"] / ref - 1 if estimate["mean"] is not None and ref else None
                )
                metrics[metric] = estimate
            summaries.append({"variant": variant, "policy": policy, "metrics": metrics})
        samples = np.array(
            [
                [[lookup[variant, p][i][k] for k in ("mean_t", "p99_t")] for p in POLICIES]
                for i in range(len(lookup[variant, POLICIES[0]]))
            ]
        )
        per_seed[variant] = [queue_log_error(sample, refs) for sample in samples]
        losses.append(
            {
                "variant": variant,
                "loss_of_mc_means": queue_log_error(samples.mean(axis=0), refs),
                "seed_losses": per_seed[variant],
            }
        )
        for metric in ("mean_t", "p99_t"):
            predicted = {p: np.mean([r[metric] for r in lookup[variant, p]]) for p in POLICIES}
            observed = {p: lookup["observed", p][0][metric] for p in POLICIES}
            chosen, best = min(POLICIES, key=predicted.get), min(observed.values())
            ties = [p for p in POLICIES if np.isclose(observed[p], best, rtol=1e-12, atol=1e-9)]
            decisions.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "chosen": chosen,
                    "reference_best": ties,
                    "reference_tie_count": len(ties),
                    "relative_regret": observed[chosen] / best - 1,
                }
            )
    contrasts = [
        {"left": a, "right": b, "paired_loss_change": feature.swf.interval(np.array(per_seed[a]) - per_seed[b])}
        for a, b in CONTRASTS
    ]
    return {"summaries": summaries, "losses": losses, "contrasts": contrasts, "decisions": decisions}


def block_audit(block, name):
    """Freeze only aggregates and hashes, not redistributable raw records."""
    return {
        "index": block["index"],
        "fraction": block["fraction"],
        "split": block["split"],
        "capacity": block["capacity"],
        "history": feature.cohort_audit(block["history"], name),
        "prefix": feature.cohort_audit(block["prefix"], name),
        "recent_prefix": feature.cohort_audit(block["recent_prefix"], name),
        "cohort": feature.cohort_audit(block["cohort"], name),
        "prefix_audit": block["prefix_audit"],
    }


def run_block(block, name, config, *, workers=1):
    """Replay the fixed matrix; no tuning, model selection or rate matching."""
    tasks, tapes = [], []
    forecasts = forecasts_for(block, config)
    reference = observed_marks(block["cohort"], name)
    for variant, seed, records, services in workloads(block, name, config):
        needs, trace = trace_for(records, services, forecasts)
        tape_hash = feature.gpu.json_hash({"needs": needs, "trace": [asdict(j) for j in trace]})
        tapes.append(
            {
                "variant": variant,
                "seed": seed,
                "marks_sha256": feature.gpu.json_hash([asdict(r) for r in records]),
                "services_sha256": feature.swf.fingerprint(services),
                "trace_sha256": tape_hash,
                **diagnostics(records, services, config.warmup, reference),
            }
        )
        for policy in POLICIES:
            tasks.append(
                (
                    {"variant": variant, "seed": seed, "policy": policy, "trace_sha256": tape_hash},
                    block["capacity"],
                    needs,
                    trace,
                    config.warmup,
                )
            )
    rows = run_tasks(tasks, workers)
    return {
        "schema_version": 1,
        "source": name,
        "config": asdict(config),
        **block_audit(block, name),
        "forecast_sha256": feature.swf.fingerprint(list(forecasts.values())),
        "scheduler_runs": len(rows),
        "tapes": tapes,
        "rows": rows,
        **summarize(rows),
    }


def code_hashes():
    """Pin prior adapters/service/schedulers and new joint generator code."""
    hashes = feature.code_hashes()
    root = Path(__file__).resolve().parents[1]
    for filename in (
        "most_queue/random/marked_arrivals.py",
        "most_queue/random/queue_selection.py",
        "examples/joint_marked_arrivals_experiment.py",
    ):
        hashes[filename] = hashlib.sha256((root / filename).read_bytes()).hexdigest()
    return hashes


def main():
    """Write the entire cohort/history protocol before scheduling any workload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/real_trace"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    positive_integer(args.workers, "workers")
    sources = {
        "sdsc": feature.parse_swf_lifecycle(
            feature.swf.verified_source(args.cache_dir / "SDSC-SP2-1998-4.2-cln.swf", args.download)
            .decode("ascii")
            .splitlines(),
            128,
        ),
        "kalos": feature.parse_acme_kalos(
            feature.gpu.verified_source(args.cache_dir / "acme-kalos.csv", args.download).decode().splitlines()
        ),
    }
    blocks = {name: prepare(source, name, CONFIGS[name]) for name, source in sources.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []

    def write(filename, value):
        payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        (args.output_dir / filename).write_bytes(payload)
        artifacts.append({"file": filename, "sha256": hashlib.sha256(payload).hexdigest()})

    write(
        "protocol.json",
        {
            "configs": {n: asdict(c) for n, c in CONFIGS.items()},
            "variants": VARIANTS,
            "contrasts": CONTRASTS,
            "blocks": {n: [block_audit(b, n) for b in origin] for n, origin in blocks.items()},
        },
    )
    total = 0
    for name, origin in blocks.items():
        for block in origin:
            result = run_block(block, name, CONFIGS[name], workers=args.workers)
            write(f"{name}-{block['index']}.json", result)
            total += result["scheduler_runs"]
            print(json.dumps({"source": name, "origin": block["fraction"], "runs": total}), flush=True)
    write(
        "manifest.json",
        {
            "schema_version": 1,
            "scheduler_runs": total,
            "implementation_sha256": code_hashes(),
            "artifacts": list(artifacts),
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "sources": {
                "sdsc": {
                    "page": feature.swf.SOURCE_PAGE,
                    "sha256": feature.swf.SOURCE_SHA256,
                    "commit": feature.swf.MIRROR_COMMIT,
                    "conditions": "NPACI JOBLOG research/educational/non-profit; not MIT",
                    "attribution": "SDSC / Victor Hazlewood; conversion Dror Feitelson; DIR-LAB mirror",
                    "audit": sources["sdsc"].audit,
                },
                "kalos": {
                    "page": feature.gpu.SOURCE_PAGE,
                    "sha256": feature.gpu.SOURCE_SHA256,
                    "commit": feature.gpu.SOURCE_COMMIT,
                    "conditions": "CC-BY-4.0, not MIT",
                    "attribution": "Shanghai AI Laboratory / InternLM AcmeTrace",
                    "audit": sources["kalos"].audit,
                },
            },
            "interpretation": "Completed-only empty-start marked arrivals, retrospective prefix eligibility; "
            "fixed coarse service mechanism, synthetic target identities, conditional MC, "
            "no production reconstruction.",
        },
    )


if __name__ == "__main__":
    main()
