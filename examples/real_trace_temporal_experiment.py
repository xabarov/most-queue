"""EPIC-056: prespecified rolling origins and empirical dependence ablations.

    python -m examples.real_trace_temporal_experiment \
        --output-dir works/real_trace_temporal

The pinned EPIC-055 cache is reused; --download is explicitly opt-in.
See docs/real_trace_temporal.md for population and information limitations.
"""

import argparse
import hashlib
import json
import platform
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy

from examples.real_trace_calibration_experiment import (
    METRICS,
    MIRROR_COMMIT,
    POLICIES,
    SOURCE_PAGE,
    SOURCE_SHA256,
    SOURCE_URL,
    fingerprint,
    interval,
    replay,
    verified_source,
)
from most_queue.random.trace_resampling import ConditionalEmpirical, circular_block_indices, lag_correlation
from most_queue.sim.msj_general import MsjTraceJob
from most_queue.sim.utils.workload_trace import chronological_split, parse_swf, positive_integer

VARIANTS = (
    "expanding_coarse",
    "recent_mean_only",
    "recent_coarse",
    "recent_exact",
    "recent_rank_iid",
    "recent_block20",
    "recent_block60",
)
CONTRASTS = (
    ("recent_mean_only", "expanding_coarse"),
    ("recent_coarse", "recent_mean_only"),
    ("recent_exact", "recent_coarse"),
    ("recent_rank_iid", "recent_exact"),
    ("recent_block20", "recent_rank_iid"),
    ("recent_block60", "recent_rank_iid"),
)


@dataclass(frozen=True)
class TemporalConfig:
    """Fixed origins, sample sizes and fitting thresholds; no test-set tuning."""

    fractions: tuple[float, ...] = (0.35, 0.5, 0.65, 0.8)
    jobs: int = 1000
    warmup: int = 200
    replications: int = 8
    history_limit: int = 4000
    minimum: int = 20
    first_seed: int = 56000

    def __post_init__(self):
        for name in ("jobs", "warmup", "replications", "history_limit", "minimum", "first_seed"):
            positive_integer(getattr(self, name), name, minimum=0 if name in ("warmup", "first_seed") else 1)
        if self.replications < 2 or self.minimum < 2 or self.history_limit < 60:
            raise ValueError("need replications>=2, minimum>=2 and history_limit>=60")
        values = np.asarray(self.fractions, dtype=float)
        if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
            raise ValueError("fractions must be a finite nonempty vector")
        if np.any((values <= 0) | (values >= 1)) or np.any(np.diff(values) <= 0):
            raise ValueError("fractions must be strictly increasing inside (0,1)")


def prepare_origins(source, config):
    """Validate all cohorts before replay; training uses original log outcomes."""
    origins = []
    previous_end = -1
    for fraction in config.fractions:
        history, heldout, audit = chronological_split(source, fraction)
        count = config.jobs + config.warmup
        if len(heldout) < count:
            raise ValueError("not enough held-out jobs for a full origin")
        jobs = heldout[:count]
        if jobs[0].submit <= previous_end:
            raise ValueError("held-out origin blocks overlap")
        if jobs[-1].submit <= jobs[config.warmup].submit:
            raise ValueError("measured arrivals must have positive span")
        recent = history[-config.history_limit :]
        if len(recent) < 60:
            raise ValueError("history is too short for block60")
        origins.append((history, recent, jobs, audit))
        previous_end = jobs[-1].submit
    return origins


def fit_context(history, recent, config):
    """Fit three distribution sets without taking any held-out data."""

    def fit(jobs, exact):
        return ConditionalEmpirical.fit(
            [job.need for job in jobs], [job.runtime for job in jobs], exact=exact, minimum=config.minimum
        )

    models = {
        "expanding_coarse": fit(history, False),
        "recent_coarse": fit(recent, False),
        "recent_exact": fit(recent, True),
    }
    ranks = models["recent_exact"].ranks([job.need for job in recent], [job.runtime for job in recent])
    return models, ranks


def service_variants(target_needs, models, ranks, uniforms):
    """Generate paired ablations; only historical mean ratios may rescale S."""
    target_needs = np.asarray(target_needs)
    expanding = models["expanding_coarse"]
    recent = models["recent_coarse"]
    exact = models["recent_exact"]
    old = expanding.quantiles(target_needs, uniforms)
    ratios = {
        need: recent.distribution(need)[0].mean / expanding.distribution(need)[0].mean
        for need in np.unique(target_needs)
    }
    generated = {
        "expanding_coarse": old,
        "recent_mean_only": old * np.array([ratios[need] for need in target_needs]),
        "recent_coarse": recent.quantiles(target_needs, uniforms),
        "recent_exact": exact.quantiles(target_needs, uniforms),
    }
    for variant, length in (("recent_rank_iid", 1), ("recent_block20", 20), ("recent_block60", 60)):
        indices = circular_block_indices(len(ranks), uniforms, length)
        generated[variant] = exact.quantiles(target_needs, ranks[indices])
    return generated


def fit_audit(model, capacity):
    """Summarize every admissible K; no raw historical samples are exported."""
    output, cached = [], {}
    for need in range(1, capacity + 1):
        dist, level = model.distribution(need)
        if id(dist) not in cached:
            params = dist.parameters()
            cached[id(dist)] = {key: params[key] for key in ("count", "mean", "cv2", "p90", "p99")}
        output.append({"need": need, "level": level, **cached[id(dist)]})
    return output


def describe_services(services, needs, span, capacity):
    """Diagnostics of the measured cohort; offered work is not stability rho."""
    values = np.asarray(services, dtype=float)
    return {
        "mean_s": float(values.mean()),
        "p99_s": float(np.quantile(values, 0.99)),
        "cv2_s": float(np.var(values) / values.mean() ** 2),
        "offered_work_ratio": float(np.dot(needs, values) / (capacity * span)),
        "log_s_lag1": lag_correlation(np.log(values)),
        "log_s_lag20": lag_correlation(np.log(values), 20),
    }


def summarize_origin(rows):
    """Conditional MC errors and paired absolute-error changes, never fold CIs."""
    lookup = {
        (variant, policy): sorted(
            [row for row in rows if row["variant"] == variant and row["policy"] == policy],
            key=lambda row: -1 if row["seed"] is None else row["seed"],
        )
        for variant in ("observed", *VARIANTS)
        for policy in POLICIES
    }
    summaries, contrasts, decisions = [], [], []
    for variant in VARIANTS:
        for policy in POLICIES:
            ref = lookup["observed", policy][0]
            metrics = {}
            for metric in METRICS:
                value = interval([row[metric] for row in lookup[variant, policy]])
                value["reference"] = ref[metric]
                value["relative_error"] = value["mean"] / ref[metric] - 1 if ref[metric] else None
                metrics[metric] = value
            summaries.append({"variant": variant, "policy": policy, "metrics": metrics})
        for metric in ("mean_t", "p99_t"):
            means = {policy: np.mean([row[metric] for row in lookup[variant, policy]]) for policy in POLICIES}
            reference = {policy: lookup["observed", policy][0][metric] for policy in POLICIES}
            chosen, best = min(POLICIES, key=means.get), min(POLICIES, key=reference.get)
            decisions.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "chosen": chosen,
                    "reference_best": best,
                    "relative_regret": reference[chosen] / reference[best] - 1,
                }
            )
    for variant, baseline in CONTRASTS:
        for policy in POLICIES:
            metrics = {}
            for metric in ("mean_t", "p99_t"):
                ref = lookup["observed", policy][0][metric]
                differences = [
                    abs(row[metric] / ref - 1) - abs(base[metric] / ref - 1)
                    for row, base in zip(lookup[variant, policy], lookup[baseline, policy])
                ]
                metrics[metric] = interval(differences)
            contrasts.append(
                {"variant": variant, "baseline": baseline, "policy": policy, "absolute_relative_error_change": metrics}
            )
    return summaries, contrasts, decisions


def run_origin(source, prepared, config):
    """Replay one fixed-history origin, with a common forecast for all variants."""
    history, recent, jobs, split = prepared
    models, ranks = fit_context(history, recent, config)
    needs = tuple(sorted({job.need for job in jobs}))
    classes = {need: idx for idx, need in enumerate(needs)}
    target_needs = np.array([job.need for job in jobs])
    arrivals = np.array([job.submit - jobs[0].submit for job in jobs])
    forecast_values = {need: models["expanding_coarse"].distribution(need)[0].parameters()["p90"] for need in needs}
    forecasts = np.array([forecast_values[need] for need in target_needs])
    rows, diagnostics = [], []
    span = arrivals[-1] - arrivals[config.warmup]
    bundles = [("observed", None, np.array([job.runtime for job in jobs]))]
    for seed in range(config.first_seed, config.first_seed + config.replications):
        # Cutoff, not loop index, identifies the origin; subset runs remain reproducible.
        cutoff_words = np.asarray([split["cutoff"]], dtype="<f8").view("<u4").tolist()
        rng = np.random.default_rng(np.random.SeedSequence([seed, *cutoff_words]))
        uniforms = (rng.integers(0, 2**52, size=len(jobs), dtype=np.int64) + 0.5) / 2**52
        bundles.extend(
            (name, seed, values) for name, values in service_variants(target_needs, models, ranks, uniforms).items()
        )
    for variant, seed, services in bundles:
        trace = tuple(
            MsjTraceJob(float(arrival), classes[int(need)], float(service), float(forecast))
            for arrival, need, service, forecast in zip(arrivals, target_needs, services, forecasts)
        )
        digest = fingerprint(np.column_stack((arrivals, target_needs, services, forecasts)))
        diagnostics.append(
            {
                "variant": variant,
                "seed": seed,
                "trace_sha256": digest,
                **describe_services(services[config.warmup :], target_needs[config.warmup :], span, source.capacity),
            }
        )
        for policy in POLICIES:
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "policy": policy,
                    "trace_sha256": digest,
                    **replay(source.capacity, needs, trace, policy, config.warmup),
                }
            )
    summaries, contrasts, decisions = summarize_origin(rows)
    recent_ids = {job.job_id for job in recent}
    recent_positions = [idx for idx, job in enumerate(source.jobs) if job.job_id in recent_ids]
    recent_needs = [job.need for job in recent]
    observed_ranks = models["recent_exact"].ranks(target_needs, [job.runtime for job in jobs])
    return {
        "split": split,
        "history": {
            "expanding_count": len(history),
            "recent_count": len(recent),
            "recent_first_submit": recent[0].submit,
            "recent_last_submit": recent[-1].submit,
            "max_training_completion": max(job.completed_at for job in history),
            "recent_gaps_in_accepted_sequence": int(np.sum(np.diff(recent_positions) != 1)),
            "recent_rank_sha256": fingerprint(ranks),
            "recent_rank_lag1": lag_correlation(ranks),
            "recent_rank_lag20": lag_correlation(ranks, 20),
            "recent_needs_counts": dict(sorted(Counter(recent_needs).items())),
        },
        "cohort": {
            "first_submit": jobs[0].submit,
            "last_submit": jobs[-1].submit,
            "input_sha256": fingerprint([[job.job_id, job.submit, job.runtime, job.need] for job in jobs]),
            "forecast_sha256": fingerprint(forecasts),
            "test_rank_lag1": lag_correlation(observed_ranks[config.warmup :]),
            "exact_fallback_counts": dict(
                Counter(models["recent_exact"].distribution(need)[1] for need in target_needs[config.warmup :])
            ),
        },
        "fits": {name: fit_audit(model, source.capacity) for name, model in models.items()},
        "scheduler_runs": len(rows),
        "rows": rows,
        "service_diagnostics": diagnostics,
        "summaries": summaries,
        "contrasts": contrasts,
        "decisions": decisions,
    }


def aggregate(origins):
    """Descriptive errors over every origin/policy; origins are not iid folds."""
    output = []
    for variant in VARIANTS:
        entry = {"variant": variant}
        for metric in ("mean_t", "p99_t"):
            cells = [
                s["metrics"][metric]["relative_error"]
                for origin in origins
                for s in origin["summaries"]
                if s["variant"] == variant
            ]
            decisions = [
                s
                for origin in origins
                for s in origin["decisions"]
                if s["variant"] == variant and s["metric"] == metric
            ]
            entry[metric] = {
                "mape_of_mc_means": float(np.mean(np.abs(cells))),
                "matching_choices": sum(s["chosen"] == s["reference_best"] for s in decisions),
                "mean_relative_regret": float(np.mean([s["relative_regret"] for s in decisions])),
                "max_relative_regret": max(s["relative_regret"] for s in decisions),
            }
        output.append(entry)
    return output


def implementation_hashes():
    """Pin code files used by this experiment without requiring a git commit."""
    root = Path(__file__).resolve().parents[1]
    names = (
        "examples/real_trace_temporal_experiment.py",
        "examples/real_trace_calibration_experiment.py",
        "most_queue/random/trace_resampling.py",
        "most_queue/random/service_calibration.py",
        "most_queue/sim/utils/workload_trace.py",
        "most_queue/sim/msj_general.py",
        "most_queue/sim/utils/msj_calendar.py",
        "most_queue/sim/utils/msj_packing.py",
        "most_queue/sim/base_core.py",
        "most_queue/structs.py",
    )
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def main():
    """Write one result per origin and a manifest containing byte hashes."""
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
    payload = verified_source(args.cache, args.download)
    source = parse_swf(payload.decode("ascii").splitlines(), capacity=128)
    prepared = prepare_origins(source, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    origins, artifacts = [], []
    for index, origin in enumerate(prepared):
        result = run_origin(source, origin, config)
        result.update({"schema_version": 1, "fraction": config.fractions[index], "config": asdict(config)})
        filename = f"origin-{index}.json"
        serialized = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode("utf-8")
        (args.output_dir / filename).write_bytes(serialized)
        artifacts.append({"file": filename, "sha256": hashlib.sha256(serialized).hexdigest()})
        origins.append(result)
        print(
            json.dumps(
                {"origin": index, "fraction": config.fractions[index], "scheduler_runs": result["scheduler_runs"]}
            ),
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "config": asdict(config),
        "capacity": source.capacity,
        "audit": source.audit,
        "source": {
            "page": SOURCE_PAGE,
            "download": SOURCE_URL,
            "sha256": SOURCE_SHA256,
            "mirror_commit": MIRROR_COMMIT,
            "attribution": "SDSC / Victor Hazlewood; conversion Dror Feitelson; DIR-LAB mirror",
            "usage_conditions": "NPACI JOBLOG research/educational/non-profit conditions; data is not MIT-licensed",
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "implementation_sha256": implementation_hashes(),
        "artifacts": artifacts,
        "scheduler_runs": sum(origin["scheduler_runs"] for origin in origins),
        "aggregate": aggregate(origins),
        "interpretation": "Retrospective selected-cohort workload ablations; conditional MC intervals, "
        "not population or causal guarantees. Rank blocks use circular seams and close omitted-history gaps.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
