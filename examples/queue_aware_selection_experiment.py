"""EPIC-061: freeze model selection from early queue replay before late tests.

    python -m examples.queue_aware_selection_experiment --output-dir works/queue_aware_selection

All source data remain in the opt-in, hash-pinned cache. The late periods were
previously explored: this is retrospective temporal evaluation, not a blind trial.
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
from most_queue.random.queue_selection import queue_log_error, select_queue_model
from most_queue.random.trace_resampling import ConditionalEmpirical
from most_queue.sim.utils.workload_trace import chronological_split, positive_integer

QUEUE_METRICS = ("mean_release_t", "p99_release_t")
POLICIES = feature.swf.POLICIES


@dataclass(frozen=True)
class QueueSelectionConfig:
    """Prespecified temporal blocks, fitting rules and independent seed ranges."""

    validation_fractions: tuple = (0.60, 0.70)
    test_fractions: tuple = (0.85, 0.90)
    validation_jobs: int = 400
    jobs: int = 1000
    warmup: int = 200
    history_limit: int = 4000
    minimum: int = 20
    replications: int = 8
    validation_seed: int = 61000
    test_seed: int = 62000

    def __post_init__(self):
        for name in (
            "validation_jobs",
            "jobs",
            "warmup",
            "history_limit",
            "minimum",
            "replications",
            "validation_seed",
            "test_seed",
        ):
            positive_integer(
                getattr(self, name), name, minimum=0 if name in ("warmup", "validation_seed", "test_seed") else 1
            )
        if min(self.minimum, self.history_limit, self.replications) < 2:
            raise ValueError("minimum, history_limit and replications must be >=2")
        if len(self.validation_fractions) < 2 or not self.test_fractions:
            raise ValueError("need at least two validation origins and one test origin")
        fractions = np.asarray((*self.validation_fractions, *self.test_fractions))
        if fractions.dtype.kind not in "iuf" or fractions.ndim != 1 or not np.all(np.isfinite(fractions)):
            raise ValueError("fractions must be finite numeric values")
        if np.any((fractions <= 0) | (fractions >= 1)) or np.any(np.diff(fractions) <= 0):
            raise ValueError("fractions must increase strictly within (0,1)")
        validation_seeds = set(range(self.validation_seed, self.validation_seed + self.replications))
        if validation_seeds.intersection(range(self.test_seed, self.test_seed + self.replications)):
            raise ValueError("validation and test seeds must be disjoint")


CONFIGS = {
    "sdsc": QueueSelectionConfig(),
    "kalos": QueueSelectionConfig((0.35, 0.50), (0.65, 0.70), validation_jobs=300, jobs=300, warmup=100),
}


def main_scenario(name):
    """Use the uncapped scenario for selection, never reward target timeouts."""
    return {"sdsc": "carry_cancelled", "kalos": "carry_terminal"}[name]


def prepare(source, name, config):
    """Audit full disjoint cohorts and availability of all validation outcomes."""
    completed = feature.completed_source(source, name)
    blocks, previous_end = [], -np.inf
    terminal = source.jobs if name == "sdsc" else source.terminal_jobs
    for phase, fractions in (("validation", config.validation_fractions), ("test", config.test_fractions)):
        for index, fraction in enumerate(fractions):
            history, heldout, split = chronological_split(completed, fraction)
            count = (config.validation_jobs if phase == "validation" else config.jobs) + config.warmup
            if len(heldout) < count or len(history) < 2:
                raise ValueError("not enough jobs for full cohorts and completed history")
            cohort = heldout[:count]
            if cohort[0].submit <= previous_end:
                raise ValueError("selected arrival blocks overlap")
            if cohort[-1].submit <= cohort[config.warmup].submit:
                raise ValueError("measured targets require a positive arrival span")
            previous_end = cohort[-1].submit
            identifiers = {job.job_id for job in cohort}
            environment = [
                j
                for j in terminal
                if j.job_id in identifiers
                or j.submit < cohort[0].submit < j.completed_at
                or (
                    cohort[0].submit <= j.submit <= cohort[-1].submit
                    and (j.status != 1 if name == "sdsc" else j.outcome != "completed")
                )
            ]
            blocks.append(
                {
                    "phase": phase,
                    "index": index,
                    "fraction": fraction,
                    "split": split,
                    "history": history[-config.history_limit :],
                    "expanding": history,
                    "cohort": cohort,
                    "capacity": completed.capacity,
                    "environment_max_completion": max(j.completed_at for j in environment),
                }
            )
    first_test = next(b["split"]["cutoff"] for b in blocks if b["phase"] == "test")
    if any(b["environment_max_completion"] >= first_test for b in blocks if b["phase"] == "validation"):
        raise ValueError("validation cohort and environment must finish before first test cutoff")
    return blocks


def forecasts_for(history, capacity, minimum):
    """Freeze expanding-history coarse linear p90 for every original need."""
    fit = ConditionalEmpirical.fit([j.need for j in history], [j.runtime for j in history], minimum=minimum)
    cached, forecasts = {}, {}
    for need in range(1, capacity + 1):
        distribution, _ = fit.distribution(need)
        if id(distribution) not in cached:
            cached[id(distribution)] = distribution.parameters()["p90"]
        forecasts[need] = cached[id(distribution)]
    return forecasts


def coverage(history, targets, name, model):
    """Describe context/need drift and support coverage without tuning on them."""

    def groups(jobs):
        return Counter(str(int(np.searchsorted([1, 8, 32], j.need))) for j in jobs)

    def contexts(jobs):
        return Counter(str(feature.context(j, name)) for j in jobs)

    def tv(a, b):
        return float(0.5 * sum(abs(a[k] / len(history) - b[k] / len(targets)) for k in set(a) | set(b)))

    history_groups, target_groups = groups(history), groups(targets)
    history_contexts, target_contexts = contexts(history), contexts(targets)
    predictions = [feature.predict(model, j, name) for j in targets]
    details = []
    for group in range(4):
        selected = [p for j, p in zip(targets, predictions) if int(np.searchsorted([1, 8, 32], j.need)) == group]
        sizes = [len(p.distribution.samples) for p in selected]
        details.append(
            {
                "group": group,
                "history_count": history_groups[str(group)],
                "target_count": len(selected),
                "levels": dict(Counter(p.level for p in selected)),
                "support_min": min(sizes) if sizes else None,
                "support_median": float(np.median(sizes)) if sizes else None,
                "support_max": max(sizes) if sizes else None,
            }
        )
    return {
        "groups": details,
        "need_total_variation": tv(history_groups, target_groups),
        "context_total_variation": tv(history_contexts, target_contexts),
        "history_context_counts": dict(history_contexts),
        "target_context_counts": dict(target_contexts),
        "unseen_context_targets": sum(count for label, count in target_contexts.items() if not history_contexts[label]),
    }


def _replay(task):
    metadata, case, capacity = task
    return {**metadata, "trace_sha256": case.trace_sha256, **feature.run_case(case, capacity, metadata["policy"])}


def replay_tasks(tasks, workers):
    """Ordered deterministic execution with no random draws in workers."""
    positive_integer(workers, "workers")
    if workers == 1:
        return list(map(_replay, tasks))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(_replay, tasks))


def run_block(source, name, config, block, *, workers=1):
    """Replay all fixed candidates for one origin; no selection in this function."""
    jobs, history, capacity = block["cohort"], block["history"], block["capacity"]
    models = feature.fit_models(history, name, config.minimum)
    forecasts = forecasts_for(block["expanding"], capacity, config.minimum)
    predictions = {v: [feature.predict(m, j, name) for j in jobs] for v, m in models.items()}
    seed_start = config.validation_seed if block["phase"] == "validation" else config.test_seed
    bundles = [("observed", None, np.array([j.runtime for j in jobs]))]
    for seed in range(seed_start, seed_start + config.replications):
        uniforms = feature.uniform_tape(name, block["split"]["cutoff"], seed, len(jobs))
        bundles.extend(
            (v, seed, np.array([p.quantile(u) for p, u in zip(pred, uniforms)])) for v, pred in predictions.items()
        )
    scenarios = feature.SCENARIOS[name] if block["phase"] == "test" else (main_scenario(name),)
    builder = feature.build_case if name == "sdsc" else feature.gpu.build_case
    tasks, diagnostics, observed_cases = [], [], {}
    for variant, seed, services in bundles:
        diagnostics.append({"variant": variant, "seed": seed, "services_sha256": feature.swf.fingerprint(services)})
        for scenario in scenarios:
            case = builder(source, jobs, services, forecasts, scenario, warmup=config.warmup)
            if variant == "observed":
                observed_cases[scenario] = case.audit
            for policy in POLICIES:
                tasks.append(
                    ({"scenario": scenario, "variant": variant, "seed": seed, "policy": policy}, case, capacity)
                )
    rows = replay_tasks(tasks, workers)
    # The existing summary routine expects both SDSC scenarios. Validation is
    # intentionally uncapped, so summarize one scenario at a time locally.
    summaries = summarize_rows(rows, name, scenarios)
    targets = jobs[config.warmup :]
    return {
        "schema_version": 1,
        "source": name,
        "phase": block["phase"],
        "index": block["index"],
        "fraction": block["fraction"],
        "split": block["split"],
        "config": asdict(config),
        "capacity": capacity,
        "resource_unit": "allocated processor" if name == "sdsc" else "requested GPU",
        "environment_max_completion": block["environment_max_completion"],
        "history": feature.cohort_audit(history, name),
        "warmup_and_targets": feature.cohort_audit(jobs, name),
        "measured_targets": feature.cohort_audit(targets, name),
        "forecast_sha256": feature.swf.fingerprint(list(forecasts.values())),
        "observed_cases": observed_cases,
        "scores": {v: feature.score(m, targets, name) for v, m in models.items()},
        "coverage": {v: coverage(history, targets, name, m) for v, m in models.items()},
        "scheduler_runs": len(rows),
        "rows": rows,
        "service_diagnostics": diagnostics,
        **summaries,
    }


def summarize_rows(rows, name, scenarios):
    """Use the EPIC-059 equations for an explicit subset of source scenarios."""
    output = {key: [] for key in ("summaries", "contrasts", "decisions")}
    for scenario in scenarios:
        lookup = {
            (v, p): sorted(
                [r for r in rows if (r["scenario"], r["variant"], r["policy"]) == (scenario, v, p)],
                key=lambda r: -1 if r["seed"] is None else r["seed"],
            )
            for v in ("observed", *feature.CANDIDATES[name])
            for p in POLICIES
        }
        for variant in feature.CANDIDATES[name]:
            for policy in POLICIES:
                reference, samples = lookup["observed", policy][0], lookup[variant, policy]
                metrics = {}
                for metric in feature.METRICS:
                    values = [r[metric] for r in samples if r[metric] is not None]
                    estimate = (
                        feature.swf.interval(values) if len(values) >= 2 else {"mean": None, "low": None, "high": None}
                    )
                    estimate.update({"valid_replications": len(values), "reference": reference[metric]})
                    estimate["relative_error"] = (
                        estimate["mean"] / reference[metric] - 1
                        if estimate["mean"] is not None and reference[metric]
                        else None
                    )
                    metrics[metric] = estimate
                output["summaries"].append(
                    {"scenario": scenario, "variant": variant, "policy": policy, "metrics": metrics}
                )
                if variant != "coarse":
                    changes = {
                        metric: feature.swf.interval(
                            [
                                abs(r[metric] - reference[metric]) - abs(b[metric] - reference[metric])
                                for r, b in zip(samples, lookup["coarse", policy])
                            ]
                        )
                        for metric in (*QUEUE_METRICS, "success_rate")
                    }
                    output["contrasts"].append(
                        {"scenario": scenario, "variant": variant, "policy": policy, "absolute_error_change": changes}
                    )
            for metric in QUEUE_METRICS:
                means = {p: float(np.mean([r[metric] for r in lookup[variant, p]])) for p in POLICIES}
                refs = {p: lookup["observed", p][0][metric] for p in POLICIES}
                chosen, best = min(POLICIES, key=means.get), min(refs.values())
                ties = [p for p in POLICIES if np.isclose(refs[p], best, rtol=1e-12, atol=1e-9)]
                output["decisions"].append(
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
    return output


def checked_rows(result, name, scenario):
    """Reject missing or duplicate policy/candidate/seed cells before scoring."""
    rows = [r for r in result["rows"] if r["scenario"] == scenario]
    keys = [(r["variant"], r["policy"], r["seed"]) for r in rows]
    seeds = {r["seed"] for r in rows if r["variant"] != "observed"}
    for seed in seeds:
        positive_integer(seed, "seed", minimum=0)
    if len(seeds) < 2:
        raise ValueError("need at least two model seeds")
    expected = {("observed", p, None) for p in POLICIES} | {
        (v, p, s) for v in feature.CANDIDATES[name] for p in POLICIES for s in seeds
    }
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("incomplete or duplicate candidate/policy/seed grid")
    return rows


def metric_arrays(results, name, *, omit_seed=None):
    """Stack origin/policy/metric summaries, averaging seeds before scoring."""
    reference, predictions = [], {v: [] for v in feature.CANDIDATES[name]}
    for result in results:
        rows = checked_rows(result, name, main_scenario(name))
        reference.append(
            [
                [next(r[k] for r in rows if r["variant"] == "observed" and r["policy"] == p) for k in QUEUE_METRICS]
                for p in POLICIES
            ]
        )
        for variant in predictions:
            predictions[variant].append(
                [
                    [
                        float(
                            np.mean(
                                [
                                    r[k]
                                    for r in rows
                                    if r["variant"] == variant and r["policy"] == p and r["seed"] != omit_seed
                                ]
                            )
                        )
                        for k in QUEUE_METRICS
                    ]
                    for p in POLICIES
                ]
            )
    return reference, predictions


def select(validation_results, name):
    """Accept validation artifacts only; never read late models or outcomes."""
    if len(validation_results) < 2 or any(
        r["phase"] != "validation" or r["source"] != name for r in validation_results
    ):
        raise ValueError("selection requires at least two validation blocks of one source")
    if len({r["fraction"] for r in validation_results}) != len(validation_results):
        raise ValueError("validation origins must be distinct")
    order = feature.CANDIDATES[name]
    refs, predictions = metric_arrays(validation_results, name)
    queue = select_queue_model(predictions, refs, candidate_order=order)
    crps = {v: float(np.mean([r["scores"][v]["crps"] for r in validation_results])) for v in order}
    seed_sets = [set(r["seed"] for r in result["rows"] if r["variant"] != "observed") for result in validation_results]
    if any(s != seed_sets[0] for s in seed_sets[1:]) or len(seed_sets[0]) < 2:
        raise ValueError("validation origins must share at least two seeds")
    leave_origin = []
    for index in range(len(validation_results)):
        reference, predicted = metric_arrays([r for i, r in enumerate(validation_results) if i != index], name)
        leave_origin.append(select_queue_model(predicted, reference, candidate_order=order))
    leave_seed = []
    for seed in sorted(seed_sets[0]):
        reference, predicted = metric_arrays(validation_results, name, omit_seed=seed)
        leave_seed.append({"omitted_seed": seed, **select_queue_model(predicted, reference, candidate_order=order)})
    return {
        "queue": queue,
        "crps": {"selected": min(order, key=crps.get), "losses": crps},
        "baseline": "coarse",
        "validation_sha256": [feature.gpu.json_hash(r) for r in validation_results],
        "validation_fractions": [r["fraction"] for r in validation_results],
        "leave_one_origin_out": leave_origin,
        "leave_one_seed_out": leave_seed,
        "criterion": "equal origin/policy/mean-T,p99-T absolute log error of MC means; exact ties in candidate order",
    }


def evaluate_selection(result, selection):
    """Report frozen choices and paired test errors; do not reselect on test."""
    if result["phase"] != "test":
        raise ValueError("selection evaluation requires a test block")
    choices = {"queue": selection["queue"]["selected"], "crps": selection["crps"]["selected"], "coarse": "coarse"}
    output = []
    for scenario in result["observed_cases"]:
        rows = checked_rows(result, result["source"], scenario)
        reference = [
            [next(r[k] for r in rows if r["variant"] == "observed" and r["policy"] == p) for k in QUEUE_METRICS]
            for p in POLICIES
        ]
        seeds = sorted({r["seed"] for r in rows if r["variant"] != "observed"})
        losses, means = {}, {}
        for method, variant in choices.items():
            samples = np.array(
                [
                    [
                        [
                            next(r[k] for r in rows if (r["variant"], r["policy"], r["seed"]) == (variant, p, seed))
                            for k in QUEUE_METRICS
                        ]
                        for p in POLICIES
                    ]
                    for seed in seeds
                ]
            )
            means[method] = queue_log_error(samples.mean(axis=0), reference)
            losses[method] = [queue_log_error(sample, reference) for sample in samples]
        output.append(
            {
                "scenario": scenario,
                "choices": choices,
                "loss_of_mc_means": means,
                "seed_losses": losses,
                "paired_queue_minus": {
                    method: feature.swf.interval(np.array(losses["queue"]) - losses[method])
                    for method in ("crps", "coarse")
                },
            }
        )
    return {"selection_sha256": feature.gpu.json_hash(selection), "selection_evaluation": output}


def code_hashes():
    """Pin reused implementation and the new objective/runner."""
    hashes = feature.code_hashes()
    root = Path(__file__).resolve().parents[1]
    for filename in ("examples/queue_aware_selection_experiment.py", "most_queue/random/queue_selection.py"):
        hashes[filename] = hashlib.sha256((root / filename).read_bytes()).hexdigest()
    return hashes


def main():
    """Persist all validation and selection before any test scoring or replay."""
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
    artifacts, validations, total = [], {}, 0

    def write(filename, value):
        payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        (args.output_dir / filename).write_bytes(payload)
        artifacts.append({"file": filename, "sha256": hashlib.sha256(payload).hexdigest()})

    for name, config in CONFIGS.items():
        validations[name] = []
        for block in blocks[name]:
            if block["phase"] != "validation":
                continue
            result = run_block(sources[name], name, config, block, workers=args.workers)
            validations[name].append(result)
            write(f"{name}-validation-{block['index']}.json", result)
            total += result["scheduler_runs"]
            print(
                json.dumps({"source": name, "phase": "validation", "origin": block["fraction"], "runs": total}),
                flush=True,
            )
    selections = {name: select(results, name) for name, results in validations.items()}
    write("selection.json", selections)
    for name, config in CONFIGS.items():
        for block in blocks[name]:
            if block["phase"] != "test":
                continue
            result = run_block(sources[name], name, config, block, workers=args.workers)
            result.update(evaluate_selection(result, selections[name]))
            write(f"{name}-test-{block['index']}.json", result)
            total += result["scheduler_runs"]
            print(json.dumps({"source": name, "phase": "test", "origin": block["fraction"], "runs": total}), flush=True)
    write(
        "manifest.json",
        {
            "schema_version": 1,
            "scheduler_runs": total,
            "configs": {n: asdict(c) for n, c in CONFIGS.items()},
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
                    "attribution": "Shanghai AI Laboratory / InternLM, AcmeTrace Kalos 2023",
                    "audit": sources["kalos"].audit,
                },
            },
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "implementation_sha256": code_hashes(),
            "artifacts": list(artifacts),
            "interpretation": "Retrospective temporal selection, completed-only service, partial observed environment; "
            "no blind test, production quotas or independent-origin uncertainty.",
        },
    )


if __name__ == "__main__":
    main()
