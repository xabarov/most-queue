"""EPIC-059: select a conditional service CDF before late lifecycle replay.

    python -m examples.feature_service_experiment --output-dir works/feature_service

Network access is opt-in. Raw third-party traces are never redistributed.
The written selection.json precedes ALL test scoring and scheduler runs.
"""

import argparse
import hashlib
import json
import platform
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import scipy

from examples import modern_gpu_trace_experiment as gpu
from examples import real_trace_calibration_experiment as swf
from examples.real_trace_lifecycle_experiment import METRICS, build_case, run_case
from examples.real_trace_temporal_experiment import describe_services, implementation_hashes
from most_queue.random.feature_service import EmpiricalPrediction, FeatureConditionalEmpirical, request_bucket
from most_queue.random.trace_resampling import ConditionalEmpirical
from most_queue.sim.utils.acme_trace import parse_acme_kalos
from most_queue.sim.utils.workload_lifecycle import parse_swf_lifecycle
from most_queue.sim.utils.workload_trace import chronological_split, positive_integer

CANDIDATES = {"sdsc": ("coarse", "request_bin", "request_ratio"), "kalos": ("coarse", "type_coarse", "type_exact")}
SCENARIOS = {"sdsc": ("carry_cancelled", "requested_limit"), "kalos": ("carry_terminal",)}


@dataclass(frozen=True)
class FeatureConfig:
    """Prespecified counts and temporal boundaries; no test tuning."""

    validation_fraction: float = 0.70
    test_fraction: float = 0.90
    validation_jobs: int = 400
    jobs: int = 1000
    warmup: int = 200
    history_limit: int = 4000
    minimum: int = 20
    replications: int = 8
    first_seed: int = 59000

    def __post_init__(self):
        for name in ("validation_jobs", "jobs", "warmup", "history_limit", "minimum", "replications", "first_seed"):
            positive_integer(getattr(self, name), name, minimum=0 if name in ("warmup", "first_seed") else 1)
        if self.minimum < 2 or self.history_limit < 2 or self.replications < 2:
            raise ValueError("minimum, history_limit and replications must be >=2")
        if not 0 < self.validation_fraction < self.test_fraction < 1:
            raise ValueError("require 0 < validation_fraction < test_fraction < 1")


CONFIGS = {"sdsc": FeatureConfig(), "kalos": FeatureConfig(0.60, 0.70, jobs=300, warmup=100)}


def completed_source(source, name):
    """Retain requested-time metadata while preserving accepted completed IDs."""
    if name == "sdsc":
        rich = {job.job_id: job for job in source.jobs if job.status == 1}
        return replace(source.completed, jobs=tuple(rich[j.job_id] for j in source.completed.jobs))
    if name == "kalos":
        return source
    raise ValueError("unknown source")


def prepare(source, name, config):
    """Select full fixed-size blocks, then reject any unfinished validation job."""
    completed = completed_source(source, name)
    early, validation, val_audit = chronological_split(completed, config.validation_fraction)
    late, test, test_audit = chronological_split(completed, config.test_fraction)
    if len(validation) < config.validation_jobs or len(test) < config.jobs + config.warmup:
        raise ValueError("not enough jobs for the prespecified full cohorts")
    validation, test = validation[: config.validation_jobs], test[: config.jobs + config.warmup]
    if max(j.completed_at for j in validation) >= test_audit["cutoff"]:
        raise ValueError("all validation outcomes must be known strictly before test cutoff")
    if test[-1].submit <= test[config.warmup].submit:
        raise ValueError("measured test arrivals must have positive span")
    if min(len(early), len(late)) < 2:
        raise ValueError("not enough completed history")
    return {
        "early": early[-config.history_limit :],
        "validation": validation,
        "late": late[-config.history_limit :],
        "expanding": late,
        "test": test,
        "validation_split": val_audit,
        "test_split": test_audit,
        "capacity": completed.capacity,
    }


def context(job, name):
    """Use request at submission or explicitly retrospective exported type."""
    return request_bucket(job.requested_time) if name == "sdsc" else job.workload_type


def fit_models(history, name, minimum):
    """Fit only supplied completed history; targets are not an input."""
    needs, services = [j.need for j in history], [j.runtime for j in history]
    contexts = [context(j, name) for j in history]
    models = {"coarse": ConditionalEmpirical.fit(needs, services, minimum=minimum)}
    for variant in CANDIDATES[name][1:]:
        models[variant] = FeatureConditionalEmpirical.fit(
            needs,
            services,
            contexts,
            scales=[j.requested_time for j in history] if variant == "request_ratio" else None,
            exact=variant == "type_exact",
            minimum=minimum,
        )
    return models


def predict(model, job, name):
    """Build a distribution from K/context only, never from target service."""
    if isinstance(model, ConditionalEmpirical):
        distribution, level = model.distribution(job.need)
        return EmpiricalPrediction(distribution, level=level)
    return model.predict(job.need, context(job, name), scale=job.requested_time if model.ratio else None)


def score(model, jobs, name):
    """Exact full-CDF scores; no MC error bar or held-out parameter fit."""
    predictions = [predict(model, job, name) for job in jobs]
    crps = [p.crps(j.runtime) for p, j in zip(predictions, jobs)]
    groups = {}
    for label in dict.fromkeys(context(j, name) for j in jobs):
        indices = [i for i, j in enumerate(jobs) if context(j, name) == label]
        groups[str(label)] = {"count": len(indices), "crps": float(np.mean([crps[i] for i in indices]))}
    result = {
        "count": len(jobs),
        "crps": float(np.mean(crps)),
        "mae_predictive_mean": float(np.mean([abs(p.mean - j.runtime) for p, j in zip(predictions, jobs)])),
        "p90_coverage": float(np.mean([j.runtime <= p.quantile(0.9) for p, j in zip(predictions, jobs)])),
        "mean_prediction": float(np.mean([p.mean for p in predictions])),
        "observed_mean": float(np.mean([j.runtime for j in jobs])),
        "fallback_counts": dict(Counter(p.level for p in predictions)),
        "contexts": groups,
    }
    if name == "sdsc":
        pairs = [
            (p.probability_exceeding(j.requested_time), j.runtime > j.requested_time)
            for p, j in zip(predictions, jobs)
            if j.requested_time is not None
        ]
        result["request_exceedance"] = {
            "count": len(pairs),
            "predicted": float(np.mean([p for p, _ in pairs])) if pairs else None,
            "observed": float(np.mean([y for _, y in pairs])) if pairs else None,
            "brier": float(np.mean([(p - y) ** 2 for p, y in pairs])) if pairs else None,
        }
    return result


def cohort_audit(jobs, name):
    """Keep stable input hashes and coverage counts, not raw source rows."""
    return {
        "count": len(jobs),
        "first_submit": jobs[0].submit,
        "last_submit": jobs[-1].submit,
        "max_completion": max(j.completed_at for j in jobs),
        "input_sha256": gpu.json_hash([asdict(j) for j in jobs]),
        "context_counts": dict(Counter(str(context(j, name)) for j in jobs)),
        "need_group_counts": dict(Counter(str(int(np.searchsorted([1, 8, 32], j.need))) for j in jobs)),
    }


def select(prepared, name, config):
    """Select by early validation CRPS in candidate order; no test read here."""
    models = fit_models(prepared["early"], name, config.minimum)
    scores = {variant: score(model, prepared["validation"], name) for variant, model in models.items()}
    return {
        "selected": min(CANDIDATES[name], key=lambda variant: scores[variant]["crps"]),
        "criterion": "minimum mean validation CRPS in seconds; ties in candidate order",
        "candidate_order": CANDIDATES[name],
        "scores": scores,
        "history": cohort_audit(prepared["early"], name),
        "validation": cohort_audit(prepared["validation"], name),
        "validation_split": prepared["validation_split"],
    }


def uniform_tape(name, cutoff, seed, count):
    """Pair candidates by source, cutoff and seed, not by model or loop index."""
    words = np.asarray([cutoff], dtype="<f8").view("<u4").tolist()
    rng = np.random.default_rng(np.random.SeedSequence([seed, ("sdsc", "kalos").index(name), *words]))
    return (rng.integers(0, 2**52, size=count, dtype=np.int64) + 0.5) / 2**52


def summarize(rows, name):
    """Conditional MC errors, paired changes and policy regret with reference ties."""
    lookup = {
        (scenario, variant, policy): sorted(
            [r for r in rows if (r["scenario"], r["variant"], r["policy"]) == (scenario, variant, policy)],
            key=lambda r: -1 if r["seed"] is None else r["seed"],
        )
        for scenario in SCENARIOS[name]
        for variant in ("observed", *CANDIDATES[name])
        for policy in swf.POLICIES
    }
    summaries, contrasts, decisions = [], [], []
    for scenario in SCENARIOS[name]:
        for variant in CANDIDATES[name]:
            for policy in swf.POLICIES:
                reference = lookup[scenario, "observed", policy][0]
                samples = lookup[scenario, variant, policy]
                metrics = {}
                for metric in METRICS:
                    values = [r[metric] for r in samples if r[metric] is not None]
                    estimate = swf.interval(values) if len(values) >= 2 else {"mean": None, "low": None, "high": None}
                    estimate.update({"valid_replications": len(values), "reference": reference[metric]})
                    estimate["relative_error"] = (
                        estimate["mean"] / reference[metric] - 1
                        if estimate["mean"] is not None and reference[metric]
                        else None
                    )
                    metrics[metric] = estimate
                summaries.append({"scenario": scenario, "variant": variant, "policy": policy, "metrics": metrics})
                if variant != "coarse":
                    changes = {}
                    for metric in ("mean_release_t", "p99_release_t", "success_rate"):
                        ref = reference[metric]
                        changes[metric] = swf.interval(
                            [
                                abs(r[metric] - ref) - abs(b[metric] - ref)
                                for r, b in zip(samples, lookup[scenario, "coarse", policy])
                            ]
                        )
                    contrasts.append(
                        {"scenario": scenario, "variant": variant, "policy": policy, "absolute_error_change": changes}
                    )
            for metric in ("mean_release_t", "p99_release_t"):
                means = {p: np.mean([r[metric] for r in lookup[scenario, variant, p]]) for p in swf.POLICIES}
                refs = {p: lookup[scenario, "observed", p][0][metric] for p in swf.POLICIES}
                chosen = min(swf.POLICIES, key=means.get)
                best = min(refs.values())
                ties = [p for p in swf.POLICIES if np.isclose(refs[p], best, rtol=1e-12, atol=1e-9)]
                decisions.append(
                    {
                        "scenario": scenario,
                        "variant": variant,
                        "metric": metric,
                        "chosen": chosen,
                        "reference_best": ties,
                        "reference_tie_count": len(ties),
                        "relative_regret": refs[chosen] / best - 1,
                    }
                )
    return {"summaries": summaries, "contrasts": contrasts, "decisions": decisions}


def run_test(source, prepared, name, config, selection, *, progress=False):  # pylint: disable=too-many-arguments
    """Refit all prespecified candidates; selection is a frozen pointer only."""
    jobs, capacity = prepared["test"], prepared["capacity"]
    models = fit_models(prepared["late"], name, config.minimum)
    if selection["selected"] not in models:
        raise ValueError("invalid selected candidate")
    expanding = ConditionalEmpirical.fit(
        [j.need for j in prepared["expanding"]], [j.runtime for j in prepared["expanding"]], minimum=config.minimum
    )
    cached = {}
    forecasts = {}
    for need in range(1, capacity + 1):
        distribution, _ = expanding.distribution(need)
        if id(distribution) not in cached:
            cached[id(distribution)] = distribution.parameters()["p90"]
        forecasts[need] = cached[id(distribution)]
    predictions = {v: [predict(m, j, name) for j in jobs] for v, m in models.items()}
    bundles = [("observed", None, np.array([j.runtime for j in jobs]))]
    for seed in range(config.first_seed, config.first_seed + config.replications):
        uniforms = uniform_tape(name, prepared["test_split"]["cutoff"], seed, len(jobs))
        bundles.extend(
            (v, seed, np.array([p.quantile(u) for p, u in zip(pred, uniforms)])) for v, pred in predictions.items()
        )
    rows, diagnostics = [], []
    span = jobs[-1].submit - jobs[config.warmup].submit
    builder = build_case if name == "sdsc" else gpu.build_case
    for variant, seed, services in bundles:
        diagnostics.append(
            {
                "variant": variant,
                "seed": seed,
                "services_sha256": swf.fingerprint(services),
                **describe_services(services[config.warmup :], [j.need for j in jobs[config.warmup :]], span, capacity),
            }
        )
        for scenario in SCENARIOS[name]:
            case = builder(source, jobs, services, forecasts, scenario, warmup=config.warmup)
            for policy in swf.POLICIES:
                rows.append(
                    {
                        "scenario": scenario,
                        "variant": variant,
                        "seed": seed,
                        "policy": policy,
                        "trace_sha256": case.trace_sha256,
                        **run_case(case, capacity, policy),
                    }
                )
        if progress:
            print(json.dumps({"source": name, "variant": variant, "seed": seed, "runs": len(rows)}), flush=True)
    observed_cases = {
        s: builder(source, jobs, bundles[0][2], forecasts, s, warmup=config.warmup).audit for s in SCENARIOS[name]
    }
    return {
        "schema_version": 1,
        "source": name,
        "config": asdict(config),
        "capacity": capacity,
        "resource_unit": "SDSC allocated processor" if name == "sdsc" else "requested GPU",
        "selected": selection["selected"],
        "selection_sha256": gpu.json_hash(selection),
        "test_split": prepared["test_split"],
        "history": cohort_audit(prepared["late"], name),
        "warmup_and_targets": cohort_audit(jobs, name),
        "measured_targets": cohort_audit(jobs[config.warmup :], name),
        "forecast_sha256": swf.fingerprint(list(forecasts.values())),
        "observed_cases": observed_cases,
        "fit_audits": {
            v: m.audit  # pylint: disable=no-member
            for v, m in models.items()
            if isinstance(m, FeatureConditionalEmpirical)
        },
        "test_scores": {v: score(m, jobs[config.warmup :], name) for v, m in models.items()},
        "scheduler_runs": len(rows),
        "rows": rows,
        "service_diagnostics": diagnostics,
        **summarize(rows, name),
    }


def code_hashes():
    """Pin the runner, feature model, adapters and reused scheduler code."""
    hashes = implementation_hashes()
    root = Path(__file__).resolve().parents[1]
    for filename in (
        "examples/feature_service_experiment.py",
        "examples/modern_gpu_trace_experiment.py",
        "examples/real_trace_lifecycle_experiment.py",
        "most_queue/random/feature_service.py",
        "most_queue/sim/msj_lifecycle.py",
        "most_queue/sim/utils/acme_trace.py",
        "most_queue/sim/utils/workload_lifecycle.py",
    ):
        hashes[filename] = hashlib.sha256((root / filename).read_bytes()).hexdigest()
    return hashes


def main():
    """Write validation selection before evaluating either late test period."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/real_trace"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    sources = {
        "sdsc": parse_swf_lifecycle(
            swf.verified_source(args.cache_dir / "SDSC-SP2-1998-4.2-cln.swf", args.download)
            .decode("ascii")
            .splitlines(),
            128,
        ),
        "kalos": parse_acme_kalos(
            gpu.verified_source(args.cache_dir / "acme-kalos.csv", args.download).decode().splitlines()
        ),
    }
    prepared = {n: prepare(sources[n], n, c) for n, c in CONFIGS.items()}
    selections = {n: select(prepared[n], n, c) for n, c in CONFIGS.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []

    def write(filename, value):
        payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        (args.output_dir / filename).write_bytes(payload)
        artifacts.append({"file": filename, "sha256": hashlib.sha256(payload).hexdigest()})

    write("selection.json", selections)
    total = 0
    for name, config in CONFIGS.items():
        result = run_test(sources[name], prepared[name], name, config, selections[name], progress=True)
        write(f"{name}.json", result)
        total += result["scheduler_runs"]
    write(
        "manifest.json",
        {
            "schema_version": 1,
            "scheduler_runs": total,
            "configs": {n: asdict(c) for n, c in CONFIGS.items()},
            "sources": {
                "sdsc": {
                    "page": swf.SOURCE_PAGE,
                    "download": swf.SOURCE_URL,
                    "sha256": swf.SOURCE_SHA256,
                    "commit": swf.MIRROR_COMMIT,
                    "audit": sources["sdsc"].audit,
                    "conditions": "NPACI JOBLOG research/educational/non-profit; not MIT",
                    "attribution": "SDSC / Victor Hazlewood; conversion Dror Feitelson; DIR-LAB mirror",
                },
                "kalos": {
                    "page": gpu.SOURCE_PAGE,
                    "download": gpu.SOURCE_URL,
                    "sha256": gpu.SOURCE_SHA256,
                    "commit": gpu.SOURCE_COMMIT,
                    "audit": sources["kalos"].audit,
                    "conditions": "CC-BY-4.0",
                    "attribution": "InternLM / AcmeTrace, Kalos 2023",
                },
            },
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "implementation_sha256": code_hashes(),
            "artifacts": list(artifacts),
            "interpretation": "Completed-only conditional CDFs, frozen validation selection, retrospective type and "
            "partial carry-in. Conditional MC intervals, not population inference or online validation.",
        },
    )


if __name__ == "__main__":
    main()
