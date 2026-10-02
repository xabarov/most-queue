"""Independent EPIC-059 source/CDF/selection/tape/summary audit; writes nothing.

No experiment, adapter or calibration imports. Reuses the dispatch engine for
18 observed-control schedules; this is not a second scheduler implementation.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import t

from most_queue.sim.msj_lifecycle import MsjCarryIn, MsjLifecycleJob, MsjLifecycleSim

SUPPORTS = {}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def fingerprint(value):
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()


def close(a, b):
    assert np.isclose(a, b, rtol=1e-11, atol=1e-8), (a, b)


def service(job):
    return job["runtime"] if "runtime" in job else job["ended_at"] - job["started_at"]


def start(job):
    return job["submit"] + job["wait"] if "wait" in job else job["started_at"]


def end(job):
    return start(job) + service(job)


def label(job):
    return ("completed" if job["status"] == 1 else "cancelled") if "status" in job else job["outcome"]


def group(job):
    return sum(job["need"] > boundary for boundary in (1, 8, 32))


def context(job, name):
    if name == "kalos":
        return job["workload_type"]
    request = job["requested_time"]
    return None if request is None else str(sum(request > b for b in (300, 1800, 7200, 28800, 86400)))


def load_source(path, name):
    """Independently apply the pinned-source eligibility rules."""
    jobs, submits = [], []
    if name == "sdsc":
        for line in path.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith(";"):
                continue
            v = list(map(float, line.split()))
            submits.append(v[1])
            if v[10] not in (1, 5) or v[3] <= 0 or v[2] < 0 or not 1 <= v[4] <= 128 or v[7] != v[4] or v[16] != -1:
                continue
            jobs.append(
                dict(
                    job_id=int(v[0]),
                    submit=v[1],
                    wait=v[2],
                    runtime=v[3],
                    need=int(v[4]),
                    status=int(v[10]),
                    requested_time=v[8] if v[8] > 0 else None,
                )
            )
    else:
        labels = {"COMPLETED": "completed", "CANCELLED": "cancelled", "FAILED": "failed"}
        for row in csv.DictReader(path.read_text().splitlines()):
            times = [
                datetime.fromisoformat(row[k]).timestamp() if row[k] else None
                for k in ("submit_time", "start_time", "end_time")
            ]
            submit, started, ended = times
            submits.append(submit)
            if int(row["gpu_num"]) > 0 and row["state"] in labels and ended is not None and ended > started >= submit:
                jobs.append(
                    dict(
                        job_id=row["job_id"],
                        submit=submit,
                        started_at=started,
                        ended_at=ended,
                        need=int(row["gpu_num"]),
                        outcome=labels[row["state"]],
                        workload_type=row["type"],
                    )
                )
    return jobs, (min(submits), max(submits))


def support(history, target, name, variant, minimum):
    """Memoize identical feature queries; retain history to prevent ID reuse."""
    key = (
        id(history),
        name,
        variant,
        minimum,
        target["need"] if variant == "type_exact" else group(target),
        context(target, name) if variant != "coarse" else None,
        target.get("requested_time") if variant == "request_ratio" else None,
    )
    if key not in SUPPORTS:
        SUPPORTS[key] = history, reconstruct_support(history, target, name, variant, minimum)
    return SUPPORTS[key][1]


def reconstruct_support(history, target, name, variant, minimum):
    """Reconstruct the selected full ECDF without the production fit classes."""
    usable = history
    ratio = variant == "request_ratio" and target["requested_time"] is not None
    if ratio:
        usable = [j for j in history if j["requested_time"] is not None]
        ratio = len(usable) >= 2
    if not ratio:
        usable = history
    prefix = "absolute_" if variant == "request_ratio" and not ratio else ""
    feature = variant != "coarse" and not prefix
    cells = []
    if variant == "type_exact":
        cells.append(
            (
                [j for j in usable if j["need"] == target["need"] and context(j, name) == context(target, name)],
                "exact_context",
            )
        )
    if feature:
        cells.append(
            (
                [j for j in usable if group(j) == group(target) and context(j, name) == context(target, name)],
                "coarse_context",
            )
        )
    cells.append(([j for j in usable if group(j) == group(target)], "coarse"))
    chosen, level = next(((cell, level) for cell, level in cells if len(cell) >= minimum), (usable, "pooled"))
    values = [service(j) / j["requested_time"] * target["requested_time"] if ratio else service(j) for j in chosen]
    return np.sort(values), prefix + level


def check_scores(history, jobs, name, variant, minimum, reported):
    """Integrate (F(x)-1[x>=y])² directly, independently of the energy formula."""
    scores, errors, covered, means, levels, probabilities, actual = [], [], [], [], [], [], []
    for job in jobs:
        x, level = support(history, job, name, variant, minimum)
        y = service(job)
        grid = np.unique(np.append(x, y))
        cdf = np.searchsorted(x, grid[:-1], side="right") / len(x)
        scores.append(np.sum(np.diff(grid) * (cdf - (grid[:-1] >= y)) ** 2))
        errors.append(abs(np.mean(x) - y))
        covered.append(y <= x[int(np.ceil(0.9 * len(x))) - 1])
        means.append(np.mean(x))
        levels.append(level)
        if name == "sdsc" and job["requested_time"] is not None:
            probabilities.append(np.mean(x > job["requested_time"]))
            actual.append(y > job["requested_time"])
    for key, values in (
        ("crps", scores),
        ("mae_predictive_mean", errors),
        ("p90_coverage", covered),
        ("mean_prediction", means),
    ):
        close(np.mean(values), reported[key])
    assert dict(Counter(levels)) == reported["fallback_counts"]
    if probabilities:
        close(np.mean(probabilities), reported["request_exceedance"]["predicted"])
        close(np.mean((np.array(probabilities) - actual) ** 2), reported["request_exceedance"]["brier"])
    return float(np.mean(scores))


def make_case(terminal, cohort, services, forecasts, scenario, warmup):
    boundary, last = cohort[0]["submit"], cohort[-1]["submit"]
    carry = [j for j in terminal if j["submit"] < boundary < end(j)]
    active, queued = [j for j in carry if start(j) <= boundary], [j for j in carry if start(j) > boundary]
    generated = {j["job_id"]: float(s) for j, s in zip(cohort, services)}
    new = [
        j for j in terminal if j["job_id"] in generated or (label(j) != "completed" and boundary <= j["submit"] <= last)
    ]

    def convert(job, arrival, runtime, limited=False):
        return MsjLifecycleJob(
            float(arrival),
            job["need"] - 1,
            float(runtime),
            float(forecasts[job["need"]]),
            label(job),
            job["requested_time"] if limited else None,
        )

    running = [MsjCarryIn(convert(j, 0, service(j)), boundary - start(j)) for j in active]
    waiting = [convert(j, 0, service(j)) for j in queued]
    arrivals = [
        convert(j, j["submit"] - boundary, generated.get(j["job_id"], service(j)), scenario == "requested_limit")
        for j in new
    ]
    position = {j["job_id"]: len(carry) + i for i, j in enumerate(new)}
    targets = [position[j["job_id"]] for j in cohort[warmup:]]
    if scenario == "carry_terminal":
        tape = {
            "running": [asdict(j) for j in running],
            "waiting": [asdict(j) for j in waiting],
            "arrivals": [asdict(j) for j in arrivals],
            "source_ids": [j["job_id"] for j in (*active, *queued, *new)],
        }
        tape_hash = digest(tape)
    else:
        serialized = [
            [kind, j.arrival, j.cls + 1, j.service, j.estimate, j.outcome == "cancelled", j.runtime_limit or -1, age]
            for kind, j, age in (
                [(0, c.job, c.age) for c in running] + [(1, j, 0) for j in waiting] + [(2, j, 0) for j in arrivals]
            )
        ]
        tape_hash = fingerprint(serialized)
    return (running, waiting, arrivals, targets, (cohort[warmup]["submit"] - boundary, last - boundary)), tape_hash


def check_schedule(case, capacity, row):
    """Recompute target metrics and time-integrated resources from event intervals."""
    running, waiting, arrivals, targets, window = case
    simulator = MsjLifecycleSim(capacity, row["policy"])
    simulator.set_servers(range(1, capacity + 1))
    result = simulator.run_lifecycle(
        arrivals, initial_running=running, initial_waiting=waiting, observation_window=window
    )
    jobs = [c.job for c in running] + waiting + arrivals
    starts, ends = np.array(result.start_times), np.array(result.release_times)
    target_arrivals = np.array([jobs[i].arrival for i in targets])
    latency, wait = ends[targets] - target_arrivals, starts[targets] - target_arrivals
    successful = np.array(result.outcomes)[targets] == "completed"
    close(np.mean(latency), row["mean_release_t"])
    close(np.quantile(latency, 0.99), row["p99_release_t"])
    close(np.mean(wait), row["mean_w"])
    close(np.average(latency, weights=[jobs[i].cls + 1 for i in targets]), row["node_weighted_mean_release_t"])
    close(np.mean(successful), row["success_rate"])
    assert sum(successful) == row["successful_target_count"]
    if np.any(successful):
        close(np.mean(latency[successful]), row["mean_success_t"])
        close(np.quantile(latency[successful], 0.99), row["p99_success_t"])
    ledger = defaultdict(float)
    events = defaultdict(lambda: [0, 0])
    for i, job in enumerate(jobs):
        need = job.cls + 1
        ledger[result.outcomes[i]] += need * (ends[i] - max(starts[i], 0))
        events[max(starts[i], 0)][0] += need
        events[ends[i]][0] -= need
        if i >= len(running):
            events[job.arrival][1] += 1
            events[starts[i]][1] -= 1
    for outcome, work in row["resource_time_by_outcome"].items():
        close(ledger[outcome], work)
    occupied = queue = busy = idle = previous = 0
    for instant, (resource_delta, queue_delta) in sorted(events.items()):
        duration = max(0, min(instant, window[1]) - max(previous, window[0]))
        busy += occupied * duration
        idle += (capacity - occupied) * duration if queue else 0
        occupied += resource_delta
        queue += queue_delta
        assert 0 <= occupied <= capacity and queue >= 0
        previous = instant
    assert occupied == queue == 0
    close(busy / (capacity * (window[1] - window[0])), row["utilization"])
    close(idle / (capacity * (window[1] - window[0])), row["idle_with_queue"])


def confidence(values, reported):
    mean = np.mean(values)
    half = t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values))
    for key, value in (("mean", mean), ("low", mean - half), ("high", mean + half)):
        close(value, reported[key])


def check_summaries(result, counter):
    rows = result["rows"]
    for row in rows:
        close(row["total_resource_time"], sum(row["resource_time_by_outcome"].values()))
        close(row["mean_release_t"] - row["mean_w"], row["target_consumed_service_mean"])
        close(row["success_rate"] + row["timeout_rate"], 1)
        counter["rows"] += 1
    for summary in result["summaries"]:
        selected = [r for r in rows if all(r[k] == summary[k] for k in ("scenario", "variant", "policy"))]
        ref = next(
            r for r in rows if r["variant"] == "observed" and all(r[k] == summary[k] for k in ("scenario", "policy"))
        )
        for metric, reported in summary["metrics"].items():
            values = [r[metric] for r in selected if r[metric] is not None]
            assert len(values) == reported["valid_replications"]
            if len(values) >= 2:
                confidence(values, reported)
                if ref[metric]:
                    close(np.mean(values) / ref[metric] - 1, reported["relative_error"])
            counter["intervals"] += 1
    for contrast in result["contrasts"]:
        subset = [r for r in rows if all(r[k] == contrast[k] for k in ("scenario", "policy"))]
        a = [r for r in subset if r["variant"] == contrast["variant"]]
        b = [r for r in subset if r["variant"] == "coarse"]
        ref = next(r for r in subset if r["variant"] == "observed")
        for metric, reported in contrast["absolute_error_change"].items():
            confidence([abs(x[metric] - ref[metric]) - abs(y[metric] - ref[metric]) for x, y in zip(a, b)], reported)
            counter["paired_intervals"] += 1
    policies = list(dict.fromkeys(r["policy"] for r in rows))
    for decision in result["decisions"]:
        metric, scenario, variant = decision["metric"], decision["scenario"], decision["variant"]
        refs = {r["policy"]: r[metric] for r in rows if r["scenario"] == scenario and r["variant"] == "observed"}
        means = {
            p: np.mean(
                [r[metric] for r in rows if (r["scenario"], r["variant"], r["policy"]) == (scenario, variant, p)]
            )
            for p in policies
        }
        assert min(policies, key=means.get) == decision["chosen"]
        best = min(refs.values())
        ties = [p for p in policies if np.isclose(refs[p], best, rtol=1e-12, atol=1e-9)]
        assert ties == decision["reference_best"] and len(ties) == decision["reference_tie_count"]
        close(refs[decision["chosen"]] / best - 1, decision["relative_regret"])
        counter["decisions"] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/real_trace"))
    parser.add_argument("--output-dir", type=Path, default=Path("works/feature_service"))
    args = parser.parse_args()
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    for filename, expected in manifest["implementation_sha256"].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
    for artifact in manifest["artifacts"]:
        assert hashlib.sha256((args.output_dir / artifact["file"]).read_bytes()).hexdigest() == artifact["sha256"]
    selections = json.loads((args.output_dir / "selection.json").read_text())
    counter = Counter()
    for name, filename in (("sdsc", "SDSC-SP2-1998-4.2-cln.swf"), ("kalos", "acme-kalos.csv")):
        SUPPORTS.clear()
        path = args.cache_dir / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sources"][name]["sha256"]
        terminal, bounds = load_source(path, name)
        completed = [j for j in terminal if label(j) == "completed"]
        result = json.loads((args.output_dir / f"{name}.json").read_text())
        config, selection = result["config"], selections[name]
        assert digest(selection) == result["selection_sha256"]
        cut = lambda f: bounds[0] + f * (bounds[1] - bounds[0])
        early_cut, late_cut = cut(config["validation_fraction"]), cut(config["test_fraction"])
        early = [j for j in completed if end(j) < early_cut][-config["history_limit"] :]
        expanding = [j for j in completed if end(j) < late_cut]
        late = expanding[-config["history_limit"] :]
        validation = [j for j in completed if j["submit"] >= early_cut][: config["validation_jobs"]]
        cohort = [j for j in completed if j["submit"] >= late_cut][: config["jobs"] + config["warmup"]]
        assert max(end(j) for j in validation) < late_cut
        for jobs, audit in (
            (early, selection["history"]),
            (validation, selection["validation"]),
            (late, result["history"]),
            (cohort, result["warmup_and_targets"]),
        ):
            assert digest(jobs) == audit["input_sha256"]
        scores = {}
        for variant in selection["candidate_order"]:
            scores[variant] = check_scores(
                early, validation, name, variant, config["minimum"], selection["scores"][variant]
            )
            check_scores(
                late, cohort[config["warmup"] :], name, variant, config["minimum"], result["test_scores"][variant]
            )
            counter["score_blocks"] += 2
        assert min(selection["candidate_order"], key=scores.get) == selection["selected"] == result["selected"]
        forecasts = {
            k: float(np.quantile(support(expanding, {"need": k}, name, "coarse", config["minimum"])[0], 0.9))
            for k in range(1, result["capacity"] + 1)
        }
        assert fingerprint(list(forecasts.values())) == result["forecast_sha256"]
        cases = {}
        for diagnostic in result["service_diagnostics"]:
            variant, seed = diagnostic["variant"], diagnostic["seed"]
            values = [service(j) for j in cohort]
            if variant != "observed":
                words = np.array([late_cut], dtype="<f8").view("<u4").tolist()
                rng = np.random.default_rng(np.random.SeedSequence([seed, ("sdsc", "kalos").index(name), *words]))
                uniforms = (rng.integers(0, 2**52, len(cohort), dtype=np.int64) + 0.5) / 2**52
                values = []
                for job, u in zip(cohort, uniforms):
                    x, _ = support(late, job, name, variant, config["minimum"])
                    values.append(x[int(np.ceil(u * len(x))) - 1])
            assert fingerprint(values) == diagnostic["services_sha256"]
            counter["service_tapes"] += 1
            for scenario in result["observed_cases"]:
                cases[scenario, variant, seed] = make_case(
                    terminal, cohort, values, forecasts, scenario, config["warmup"]
                )
                counter["workload_tapes"] += 1
        for row in result["rows"]:
            case, tape_hash = cases[row["scenario"], row["variant"], row["seed"]]
            assert row["trace_sha256"] == tape_hash
            if row["variant"] == "observed":
                check_schedule(case, result["capacity"], row)
                counter["observed_schedules"] += 1
        check_summaries(result, counter)
        print(json.dumps({"source": name, "verified": dict(counter)}), flush=True)
    assert counter["rows"] == manifest["scheduler_runs"] == 450
    print(json.dumps(dict(counter), indent=2))


if __name__ == "__main__":
    main()
