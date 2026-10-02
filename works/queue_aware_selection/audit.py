"""Read-only independent EPIC-061 temporal/selection/replay audit.

Reuses the EPIC-059 independent raw parser and event-metric checker, not the
experiment runner, source adapters, fitting code or queue-selection objective.
Observed schedules reuse the dispatch engine; no second scheduler is claimed.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from works.feature_service import audit as base

POLICIES = ("fcfs", "first_fit", "msf", "adaptive_quickswap", "easy", "conservative")
CANDIDATES = {"sdsc": ("coarse", "request_bin", "request_ratio"), "kalos": ("coarse", "type_coarse", "type_exact")}
METRICS = ("mean_release_t", "p99_release_t")


def loss(predicted, observed):
    """Independent scalar-cell log-ratio calculation for ordinary finite data."""
    pairs = zip(np.asarray(predicted).ravel(), np.asarray(observed).ravel())
    errors = [abs(np.log(a / b)) for a, b in pairs]
    return float(sum(errors) / len(errors))


def arrays(results, name, excluded_seed=None):
    reference, predictions = [], {v: [] for v in CANDIDATES[name]}
    scenario = "carry_cancelled" if name == "sdsc" else "carry_terminal"
    for result in results:
        rows = [r for r in result["rows"] if r["scenario"] == scenario]
        index = {(r["variant"], r["policy"], r["seed"]): r for r in rows}
        assert len(index) == len(rows)
        seeds = sorted({r["seed"] for r in rows if r["variant"] != "observed" and r["seed"] != excluded_seed})
        reference.append([[index["observed", p, None][k] for k in METRICS] for p in POLICIES])
        for variant in predictions:
            predictions[variant].append(
                [[np.mean([index[variant, p, s][k] for s in seeds]) for k in METRICS] for p in POLICIES]
            )
    return reference, predictions


def check_choice(results, name, reported, excluded_seed=None):
    reference, predictions = arrays(results, name, excluded_seed)
    losses = {v: loss(values, reference) for v, values in predictions.items()}
    for v, value in losses.items():
        base.close(value, reported["losses"][v])
    assert reported["selected"] == min(CANDIDATES[name], key=losses.get)
    assert reported["candidate_order"] == list(CANDIDATES[name])


def check_selection(results, name, reported, counter):
    assert all(r["phase"] == "validation" and r["source"] == name for r in results)
    assert [base.digest(r) for r in results] == reported["validation_sha256"]
    check_choice(results, name, reported["queue"])
    scores = {v: np.mean([r["scores"][v]["crps"] for r in results]) for v in CANDIDATES[name]}
    for v, value in scores.items():
        base.close(value, reported["crps"]["losses"][v])
    assert reported["crps"]["selected"] == min(CANDIDATES[name], key=scores.get)
    assert reported["baseline"] == "coarse"
    for index, item in enumerate(reported["leave_one_origin_out"]):
        check_choice([r for i, r in enumerate(results) if i != index], name, item)
    for item in reported["leave_one_seed_out"]:
        check_choice(results, name, item, item["omitted_seed"])
    counter["frozen_selections"] += 1


def check_evaluation(result, selection, counter):
    assert result["selection_sha256"] == base.digest(selection)
    choices = {"queue": selection["queue"]["selected"], "crps": selection["crps"]["selected"], "coarse": "coarse"}
    for evaluation in result["selection_evaluation"]:
        assert evaluation["choices"] == choices
        rows = [r for r in result["rows"] if r["scenario"] == evaluation["scenario"]]
        index = {(r["variant"], r["policy"], r["seed"]): r for r in rows}
        seeds = sorted({r["seed"] for r in rows if r["variant"] != "observed"})
        reference = [[index["observed", p, None][k] for k in METRICS] for p in POLICIES]
        losses = {}
        for method, variant in choices.items():
            samples = np.array([[[index[variant, p, s][k] for k in METRICS] for p in POLICIES] for s in seeds])
            losses[method] = [loss(sample, reference) for sample in samples]
            np.testing.assert_allclose(losses[method], evaluation["seed_losses"][method], atol=1e-12)
            base.close(loss(samples.mean(axis=0), reference), evaluation["loss_of_mc_means"][method])
        for method in ("crps", "coarse"):
            base.confidence(np.array(losses["queue"]) - losses[method], evaluation["paired_queue_minus"][method])
            counter["selection_contrasts"] += 1


def check_coverage(history, targets, name, variant, minimum, reported):
    hg, tg = Counter(str(base.group(j)) for j in history), Counter(str(base.group(j)) for j in targets)
    hc = Counter(str(base.context(j, name)) for j in history)
    tc = Counter(str(base.context(j, name)) for j in targets)
    for a, b, metric in ((hg, tg, "need_total_variation"), (hc, tc, "context_total_variation")):
        value = sum(abs(a[k] / len(history) - b[k] / len(targets)) for k in a.keys() | b.keys()) / 2
        base.close(value, reported[metric])
    assert dict(hc) == reported["history_context_counts"] and dict(tc) == reported["target_context_counts"]
    assert sum(n for k, n in tc.items() if not hc[k]) == reported["unseen_context_targets"]
    for group, detail in enumerate(reported["groups"]):
        selected = [j for j in targets if base.group(j) == group]
        supports = [base.support(history, j, name, variant, minimum) for j in selected]
        sizes = [len(x) for x, _ in supports]
        assert detail["history_count"] == hg[str(group)] and detail["target_count"] == len(selected)
        assert detail["levels"] == dict(Counter(level for _, level in supports))
        assert detail["support_min"] == (min(sizes) if sizes else None)
        assert detail["support_max"] == (max(sizes) if sizes else None)
        assert detail["support_median"] == (float(np.median(sizes)) if sizes else None)


def check_block(terminal, bounds, result, counter):
    name, config = result["source"], result["config"]
    cutoff = bounds[0] + result["fraction"] * (bounds[1] - bounds[0])
    base.close(cutoff, result["split"]["cutoff"])
    completed = [j for j in terminal if base.label(j) == "completed"]
    expanding = [j for j in completed if base.end(j) < cutoff]
    history = expanding[-config["history_limit"] :]
    count = config["validation_jobs"] if result["phase"] == "validation" else config["jobs"]
    cohort = [j for j in completed if j["submit"] >= cutoff][: config["warmup"] + count]
    assert len(cohort) == config["warmup"] + count
    targets = cohort[config["warmup"] :]
    for jobs, audit in (
        (history, result["history"]),
        (cohort, result["warmup_and_targets"]),
        (targets, result["measured_targets"]),
    ):
        assert base.digest(jobs) == audit["input_sha256"]
    ids = {j["job_id"] for j in cohort}
    begin, last = cohort[0]["submit"], cohort[-1]["submit"]
    environment = [
        j
        for j in terminal
        if j["job_id"] in ids
        or j["submit"] < begin < base.end(j)
        or (begin <= j["submit"] <= last and base.label(j) != "completed")
    ]
    base.close(max(base.end(j) for j in environment), result["environment_max_completion"])
    for variant in CANDIDATES[name]:
        base.check_scores(history, targets, name, variant, config["minimum"], result["scores"][variant])
        check_coverage(history, targets, name, variant, config["minimum"], result["coverage"][variant])
        counter["score_and_coverage_blocks"] += 1
    forecasts = {
        k: float(np.quantile(base.support(expanding, {"need": k}, name, "coarse", config["minimum"])[0], 0.9))
        for k in range(1, result["capacity"] + 1)
    }
    assert base.fingerprint(list(forecasts.values())) == result["forecast_sha256"]
    scenarios = ("carry_cancelled",) if name == "sdsc" else ("carry_terminal",)
    if name == "sdsc" and result["phase"] == "test":
        scenarios += ("requested_limit",)
    assert list(result["observed_cases"]) == list(scenarios)
    cases = {}
    seed_start = config["validation_seed"] if result["phase"] == "validation" else config["test_seed"]
    expected = {("observed", None)} | {
        (v, s) for v in CANDIDATES[name] for s in range(seed_start, seed_start + config["replications"])
    }
    assert {(d["variant"], d["seed"]) for d in result["service_diagnostics"]} == expected
    for diagnostic in result["service_diagnostics"]:
        variant, seed = diagnostic["variant"], diagnostic["seed"]
        values = [base.service(j) for j in cohort]
        if variant != "observed":
            words = np.array([cutoff], dtype="<f8").view("<u4").tolist()
            rng = np.random.default_rng(np.random.SeedSequence([seed, ("sdsc", "kalos").index(name), *words]))
            uniforms = (rng.integers(0, 2**52, len(cohort), dtype=np.int64) + 0.5) / 2**52
            values = []
            for job, u in zip(cohort, uniforms):
                x, _ = base.support(history, job, name, variant, config["minimum"])
                values.append(x[int(np.ceil(u * len(x))) - 1])
        assert base.fingerprint(values) == diagnostic["services_sha256"]
        counter["service_tapes"] += 1
        for scenario in scenarios:
            cases[scenario, variant, seed] = base.make_case(
                terminal, cohort, values, forecasts, scenario, config["warmup"]
            )
            counter["workload_tapes"] += 1
    expected_rows = {(s, v, p, seed) for s in scenarios for v, seed in expected for p in POLICIES}
    actual_rows = [(r["scenario"], r["variant"], r["policy"], r["seed"]) for r in result["rows"]]
    assert len(actual_rows) == len(set(actual_rows)) == result["scheduler_runs"]
    assert set(actual_rows) == expected_rows
    for row in result["rows"]:
        case, tape_hash = cases[row["scenario"], row["variant"], row["seed"]]
        assert tape_hash == row["trace_sha256"]
        assert row["target_count"] == count
        if row["variant"] == "observed":
            base.check_schedule(case, result["capacity"], row)
            counter["observed_schedules"] += 1
    base.check_summaries(result, counter)
    return ids


def main():
    """Verify provenance, all artifacts and independent observed event metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/real_trace"))
    parser.add_argument("--output-dir", type=Path, default=Path("works/queue_aware_selection"))
    args = parser.parse_args()
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    for filename, expected in manifest["implementation_sha256"].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
    for artifact in manifest["artifacts"]:
        assert hashlib.sha256((args.output_dir / artifact["file"]).read_bytes()).hexdigest() == artifact["sha256"]
    selections = json.loads((args.output_dir / "selection.json").read_text())
    counter = Counter()
    for name, filename in (("sdsc", "SDSC-SP2-1998-4.2-cln.swf"), ("kalos", "acme-kalos.csv")):
        base.SUPPORTS.clear()
        path = args.cache_dir / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sources"][name]["sha256"]
        terminal, bounds = base.load_source(path, name)
        validations = [json.loads((args.output_dir / f"{name}-validation-{i}.json").read_text()) for i in range(2)]
        tests = [json.loads((args.output_dir / f"{name}-test-{i}.json").read_text()) for i in range(2)]
        ids = set()
        for result in validations + tests:
            new_ids = check_block(terminal, bounds, result, counter)
            assert not ids.intersection(new_ids)
            ids.update(new_ids)
            print(
                json.dumps(
                    {"source": name, "phase": result["phase"], "origin": result["fraction"], "verified": dict(counter)}
                ),
                flush=True,
            )
        assert max(r["environment_max_completion"] for r in validations) < tests[0]["split"]["cutoff"]
        check_selection(validations, name, selections[name], counter)
        for result in tests:
            check_evaluation(result, selections[name], counter)
    assert counter["rows"] == manifest["scheduler_runs"] == 1500
    print(json.dumps(dict(counter), indent=2))


if __name__ == "__main__":
    main()
