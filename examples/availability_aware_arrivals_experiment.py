"""EPIC-065: availability-aware arrival history, without completed-prefix staleness.

    python -m examples.availability_aware_arrivals_experiment \
        --output-dir works/availability_aware_arrivals

EPIC-062 found that the completed-prefix donor history used for arrival-mark
fitting stalls at the first job still unresolved at cutoff, even though its
submit/need/context marks need no completion (prefix_lag up to 8.2 days on
SDSC .90). This reuses the SAME four origins, policies, replications and
completed-only service fit as EPIC-062; only the arrival-mark donor
construction changes. ``fixed_coarse`` and the OLD completed-prefix
``recent_joint_iid_stale`` remain mandatory controls;
``recent_availability_iid``/``expanding_availability_iid`` are new, built from
``most_queue.sim.utils.workload_trace.availability_prefix`` over the already-
parsed richer populations (SwfLifecycleTrace.jobs / AcmeTrace.terminal_jobs).
No new source, download path or ingestion change.
"""

import argparse
import hashlib
import json
import multiprocessing
import platform
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy

from examples import feature_service_experiment as feature
from examples.joint_marked_arrivals_experiment import (
    CONFIGS,
    POLICIES,
    diagnostics,
    fit_arrivals,
    forecasts_for,
    observed_marks,
)
from examples.joint_marked_arrivals_experiment import prepare as stale_prepare
from examples.joint_marked_arrivals_experiment import (
    replay,
    service_fit,
    trace_for,
    uniform_streams,
)
from most_queue.random.queue_selection import queue_log_error
from most_queue.sim.utils.workload_trace import availability_prefix, positive_integer

VARIANTS = ("fixed_coarse", "recent_joint_iid_stale", "recent_availability_iid", "expanding_availability_iid")
CONTRASTS = (
    ("recent_availability_iid", "recent_joint_iid_stale"),
    ("recent_availability_iid", "fixed_coarse"),
    ("expanding_availability_iid", "fixed_coarse"),
)
METRICS = ("mean_t", "p99_t", "mean_w", "weighted_mean_t", "utilization", "idle_with_queue", "resource_time")


def prepare(source, name, config):
    """Augment EPIC-062's completed-prefix blocks with an availability-aware pool.

    The rich, already-parsed population (status in {1,5} for SWF; every
    terminal outcome for Acme) supplies submit/need/context marks regardless
    of whether the job itself had resolved by cutoff; no new ingestion.
    """
    blocks = stale_prepare(source, name, config)
    rich = source.jobs if name == "sdsc" else source.terminal_jobs
    augmented = []
    for block in blocks:
        avail_prefix, avail_audit = availability_prefix(rich, block["split"]["cutoff"])
        if len(avail_prefix) <= len(block["prefix"]):
            raise ValueError("availability-aware prefix must not be shorter than the completed-only prefix")
        augmented.append(
            {
                **block,
                "avail_prefix": avail_prefix,
                "avail_recent_prefix": avail_prefix[-(config.history_limit + 1) :],
                "avail_prefix_audit": avail_audit,
            }
        )
    return augmented


def workloads(block, name, config):
    """Yield paired tapes; same donor_u stream per seed isolates the pool effect."""
    fixed = observed_marks(block["cohort"], name)
    yield "observed", None, fixed, np.array([j.runtime for j in block["cohort"]])
    stale_recent = fit_arrivals(block["recent_prefix"], name)
    avail_recent = fit_arrivals(block["avail_recent_prefix"], name)
    avail_expanding = fit_arrivals(block["avail_prefix"], name)
    model = service_fit(block["history"], config.minimum)
    for seed in range(config.first_seed, config.first_seed + config.replications):
        donor_u, service_u, _, _ = uniform_streams(name, block["split"]["cutoff"], seed, len(fixed))
        marks = {
            "fixed_coarse": fixed,
            "recent_joint_iid_stale": stale_recent.sample(donor_u),
            "recent_availability_iid": avail_recent.sample(donor_u),
            "expanding_availability_iid": avail_expanding.sample(donor_u),
        }
        for variant, records in marks.items():
            services = model.quantiles([r.need for r in records], service_u)
            yield variant, seed, records, services


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
    """Freeze only aggregates/hashes and the stale-vs-availability lag contrast."""
    return {
        "index": block["index"],
        "fraction": block["fraction"],
        "split": block["split"],
        "capacity": block["capacity"],
        "history": feature.cohort_audit(block["history"], name),
        "prefix_stale": feature.cohort_audit(block["prefix"], name),
        "recent_prefix_stale": feature.cohort_audit(block["recent_prefix"], name),
        "avail_prefix": feature.cohort_audit(block["avail_prefix"], name),
        "avail_recent_prefix": feature.cohort_audit(block["avail_recent_prefix"], name),
        "cohort": feature.cohort_audit(block["cohort"], name),
        "prefix_audit_stale": block["prefix_audit"],
        "prefix_audit_availability": block["avail_prefix_audit"],
        "prefix_lag_reduction": block["prefix_audit"]["prefix_lag"] - block["avail_prefix_audit"]["prefix_lag"],
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
    """Pin prior adapters/arrivals/service/schedulers and this new runner."""
    hashes = feature.code_hashes()
    root = Path(__file__).resolve().parents[1]
    for filename in (
        "most_queue/random/marked_arrivals.py",
        "most_queue/random/queue_selection.py",
        "most_queue/sim/utils/workload_trace.py",
        "examples/joint_marked_arrivals_experiment.py",
        "examples/availability_aware_arrivals_experiment.py",
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
            "fixed coarse and old completed-prefix donor pools remain mandatory controls. Availability-aware "
            "donor pools use submit/need/context marks available before completion, never runtime/outcome of "
            "an unresolved job. No production reconstruction, no new ingestion, no tuning by this run's W/T.",
        },
    )


if __name__ == "__main__":
    main()
