"""Independent EPIC-060 source, projection, interval and control-schedule audit.

No experiment/adapter/calibration imports. Reuses the EPIC-059 independent raw
parser and the dispatcher, not an independent scheduling algorithm. Writes nothing.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from most_queue.sim.msj_lifecycle import MsjCarryIn, MsjLifecycleJob, MsjLifecycleSim
from works.feature_service.audit import confidence, digest, fingerprint, load_source


def close(left, right):
    assert np.isclose(left, right, rtol=1e-11, atol=1e-8), (left, right)


def observed_integrals(terminal, node_counts, reported):
    for allocation in ("gpu_pool", "exclusive_nodes"):
        audit = reported[allocation]
        need = lambda j: j["need"] if allocation == "gpu_pool" else node_counts[j["job_id"]]
        moments = sorted({j[k] for j in terminal for k in ("started_at", "ended_at")})
        positions = {instant: i for i, instant in enumerate(moments)}
        changes = np.zeros(len(moments), dtype=np.int64)
        work = 0
        for job in terminal:
            changes[positions[job["started_at"]]] += need(job)
            changes[positions[job["ended_at"]]] -= need(job)
            work += need(job) * (job["ended_at"] - job["started_at"])
        occupied, dt = np.cumsum(changes)[:-1], np.diff(moments)
        assert np.max(occupied) == audit["peak_concurrent_demand"]
        assert max(need(j) for j in terminal) == audit["maximum_individual_demand"]
        close(work, audit["resource_time"])
        close(np.dot(occupied, dt), work)
        for cell in audit["capacities"]:
            capacity = cell["capacity"]
            assert cell["compatible_with_recorded_intervals"] == bool(np.max(occupied) <= capacity)
            close(np.sum(dt[occupied > capacity]), cell["seconds_above_capacity"])
            close(np.dot(np.maximum(occupied - capacity, 0), dt), cell["excess_resource_time"])


def make_case(parts, cohort, services, forecasts, node_counts, cell, warmup):
    running_source, waiting_source, arrivals_source = parts
    generated = {j["job_id"]: float(s) for j, s in zip(cohort, services)}
    boundary = cohort[0]["submit"]

    def convert(job, arrival, runtime):
        demand = job["need"] if cell["allocation"] == "gpu_pool" else node_counts[job["job_id"]]
        return MsjLifecycleJob(float(arrival), demand - 1, float(runtime), forecasts[job["need"]], job["outcome"])

    active = [
        MsjCarryIn(convert(j, 0, j["ended_at"] - j["started_at"]), boundary - j["started_at"]) for j in running_source
    ]
    waiting = [convert(j, 0, j["ended_at"] - j["started_at"]) for j in waiting_source]
    arrivals = [
        convert(j, j["submit"] - boundary, generated.get(j["job_id"], j["ended_at"] - j["started_at"]))
        for j in arrivals_source
    ]
    source_jobs = [*running_source, *waiting_source, *arrivals_source]
    ids = {j["job_id"]: i for i, j in enumerate(source_jobs)}
    targets = [ids[j["job_id"]] for j in cohort[warmup:]]
    tape = {
        "running": [asdict(j) for j in active],
        "waiting": [asdict(j) for j in waiting],
        "arrivals": [asdict(j) for j in arrivals],
        "source_ids": [j["job_id"] for j in source_jobs],
    }
    window = cohort[warmup]["submit"] - boundary, cohort[-1]["submit"] - boundary
    return (active, waiting, arrivals, source_jobs, targets, window), digest(tape)


def replay(case, cell, row):
    active, waiting, arrivals, source, targets, window = case
    simulator = MsjLifecycleSim(cell["capacity"], row["policy"])
    simulator.set_servers(range(1, cell["capacity"] + 1))
    result = simulator.run_lifecycle(
        arrivals, initial_running=active, initial_waiting=waiting, observation_window=window
    )
    jobs = [j.job for j in active] + waiting + arrivals
    starts, ends = np.array(result.start_times), np.array(result.release_times)
    initial = len(active)
    waits = np.array([starts[i] - jobs[i].arrival for i in targets])
    latency = np.array([ends[i] - jobs[i].arrival for i in targets])
    for key, value in (
        ("mean_t", latency.mean()),
        ("p99_t", np.quantile(latency, 0.99)),
        ("mean_w", waits.mean()),
        ("p99_w", np.quantile(waits, 0.99)),
        ("gpu_weighted_mean_t", np.average(latency, weights=[source[i]["need"] for i in targets])),
    ):
        close(value, row[key])
    events = defaultdict(lambda: [0, 0, 0])  # reserved, requested GPU, queued
    reserved_work, gpu_work = defaultdict(float), defaultdict(float)
    for i, job in enumerate(jobs):
        reserved, requested = job.cls + 1, source[i]["need"]
        begin = max(0, starts[i])
        assert result.outcomes[i] == source[i]["outcome"]
        close(ends[i] - starts[i], job.service)
        reserved_work[job.outcome] += reserved * (ends[i] - begin)
        gpu_work[job.outcome] += requested * (ends[i] - begin)
        events[begin][0] += reserved
        events[ends[i]][0] -= reserved
        events[begin][1] += requested
        events[ends[i]][1] -= requested
        if i >= initial:
            events[job.arrival][2] += 1
            events[starts[i]][2] -= 1
    for outcome in row["reserved_work_by_outcome"]:
        close(reserved_work[outcome], row["reserved_work_by_outcome"][outcome])
        close(gpu_work[outcome], row["requested_gpu_work_by_outcome"][outcome])
    reserved = requested = queue = last = 0
    reserved_area = gpu_area = idle_area = 0
    for instant, (dr, dg, dq) in sorted(events.items()):
        dt = max(0, min(instant, window[1]) - max(last, window[0]))
        reserved_area += reserved * dt
        gpu_area += requested * dt
        idle_area += (cell["capacity"] - reserved) * dt if queue else 0
        reserved += dr
        requested += dg
        queue += dq
        assert 0 <= reserved <= cell["capacity"] and 0 <= requested <= cell["gpu_inventory"] and queue >= 0
        last = instant
    assert reserved == requested == queue == 0
    close(reserved_area / (cell["capacity"] * (window[1] - window[0])), row["reserved_utilization"])
    close(gpu_area / (cell["gpu_inventory"] * (window[1] - window[0])), row["requested_gpu_utilization"])
    close(idle_area / (cell["capacity"] * (window[1] - window[0])), row["idle_with_queue"])


def summaries(result, counter):
    rows = result["rows"]
    for row in rows:
        close(row["mean_t"] - row["mean_w"], row["target_mean_s"])
        assert row["target_count"] == result["config"]["jobs"]
        assert row["success_rate"] == 1
        assert row["requested_gpu_time"] <= row["reserved_gpu_equivalent_time"]
        assert row["requested_gpu_utilization"] <= row["reserved_utilization"] + 1e-12
        close(sum(row["requested_gpu_work_by_outcome"].values()), row["requested_gpu_time"])
        factor = 1 if row["allocation"] == "gpu_pool" else 8
        close(factor * sum(row["reserved_work_by_outcome"].values()), row["reserved_gpu_equivalent_time"])
        counter["rows"] += 1
    for summary in result["summaries"]:
        subset = [r for r in rows if all(r[k] == summary[k] for k in ("allocation", "nodes", "policy"))]
        ref = next(r for r in subset if r["variant"] == "observed")
        for metric, estimate in summary["metrics"].items():
            values = [r[metric] for r in subset if r["variant"] == "recent_coarse"]
            confidence(values, estimate)
            close(ref[metric], estimate["reference"])
            if ref[metric]:
                close(np.mean(values) / ref[metric] - 1, estimate["relative_error"])
            else:
                assert estimate["relative_error"] is None
            counter["model_intervals"] += 1
    for contrast in result["contrasts"]:
        selected = [r for r in rows if all(r[k] == contrast[k] for k in ("allocation", "nodes", "variant", "policy"))]
        baseline = [
            r
            for r in rows
            if r["allocation"] == "gpu_pool"
            and r["nodes"] == 302
            and all(r[k] == contrast[k] for k in ("variant", "policy"))
        ]
        for metric, estimate in contrast["difference_from_nominal_gpu"].items():
            values = [a[metric] - b[metric] for a, b in zip(selected, baseline)]
            if len(values) == 1:
                close(values[0], estimate["mean"])
                assert estimate["low"] is None and estimate["high"] is None
            else:
                confidence(values, estimate)
            counter["paired_or_deterministic_contrasts"] += 1
    policies = list(dict.fromkeys(r["policy"] for r in rows))
    for decision in result["decisions"]:
        subset = [r for r in rows if all(r[k] == decision[k] for k in ("allocation", "nodes"))]
        metric = decision["metric"]
        refs = {r["policy"]: r[metric] for r in subset if r["variant"] == "observed"}
        means = {
            p: np.mean([r[metric] for r in subset if r["variant"] == "recent_coarse" and r["policy"] == p])
            for p in policies
        }
        chosen, best = min(policies, key=means.get), min(refs.values())
        ties = [p for p in policies if np.isclose(refs[p], best, rtol=1e-12, atol=1e-9)]
        assert chosen == decision["chosen"] and ties == decision["reference_best"]
        assert len(ties) == decision["reference_tie_count"]
        close(refs[chosen] / best - 1, decision["relative_regret"])
        counter["decisions"] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache/real_trace/acme-kalos.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("works/gpu_resource_envelope"))
    args = parser.parse_args()
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert hashlib.sha256(args.cache.read_bytes()).hexdigest() == manifest["source"]["sha256"]
    for filename, expected in manifest["implementation_sha256"].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
    for artifact in manifest["artifacts"]:
        assert hashlib.sha256((args.output_dir / artifact["file"]).read_bytes()).hexdigest() == artifact["sha256"]
    terminal, bounds = load_source(args.cache, "kalos")
    node_counts = {r["job_id"]: int(float(r["node_num"])) for r in csv.DictReader(args.cache.read_text().splitlines())}
    completed = [j for j in terminal if j["outcome"] == "completed"]
    envelope = json.loads((args.output_dir / "envelope.json").read_text())
    observed_integrals(terminal, node_counts, envelope["source_audit"])
    historical_wait = [j["started_at"] - j["submit"] for j in completed]
    close(np.mean(historical_wait), envelope["source_audit"]["completed_historical_wait"]["mean"])
    close(np.quantile(historical_wait, 0.99), envelope["source_audit"]["completed_historical_wait"]["p99"])
    counter = Counter()
    for index, config in enumerate(manifest["configs"]):
        result = json.loads((args.output_dir / f"origin-{index}.json").read_text())
        assert result["config"] == config
        cutoff = bounds[0] + config["fraction"] * (bounds[1] - bounds[0])
        history = [j for j in completed if j["ended_at"] < cutoff]
        cohort = [j for j in completed if j["submit"] >= cutoff][: config["jobs"] + config["warmup"]]
        assert digest(history) == result["history_sha256"] and digest(cohort) == result["cohort_sha256"]
        distributions = []
        for jobs in (history, history[-config["history_limit"] :]):
            pooled = np.sort([j["ended_at"] - j["started_at"] for j in jobs])
            groups = []
            for group in range(4):
                values = np.sort(
                    [j["ended_at"] - j["started_at"] for j in jobs if sum(j["need"] > b for b in (1, 8, 32)) == group]
                )
                groups.append(values if len(values) >= config["minimum"] else pooled)
            distributions.append(groups)
        forecasts = {
            k: float(np.quantile(distributions[0][sum(k > b for b in (1, 8, 32))], 0.9))
            for k in {j["need"] for j in terminal}
        }
        assert digest(sorted(forecasts.items())) == result["forecast_sha256"]
        boundary, last = cohort[0]["submit"], cohort[-1]["submit"]
        carry = [j for j in terminal if j["submit"] < boundary < j["ended_at"]]
        active = [j for j in carry if j["started_at"] <= boundary]
        waiting = [j for j in carry if j["started_at"] > boundary]
        selected = {j["job_id"] for j in cohort}
        arrivals = [
            j
            for j in terminal
            if j["job_id"] in selected or (j["outcome"] != "completed" and boundary <= j["submit"] <= last)
        ]
        parts = active, waiting, arrivals
        assert result["cells"] == envelope["origins"][index]["cells"]
        for cell in result["cells"]:
            demand = lambda j: j["need"] if cell["allocation"] == "gpu_pool" else node_counts[j["job_id"]]
            maximum = max(demand(j) for jobs in parts for j in jobs)
            initial = sum(demand(j) for j in active)
            assert cell["maximum_individual_demand"] == maximum and cell["initial_running_demand"] == initial
            assert cell["feasible"] == (maximum <= cell["capacity"] and initial <= cell["capacity"])
            if not cell["feasible"]:
                assert not any(
                    (r["allocation"], r["nodes"]) == (cell["allocation"], cell["nodes"]) for r in result["rows"]
                )
                counter["infeasible_cells"] += 1
        bundles = [("observed", None, [j["ended_at"] - j["started_at"] for j in cohort])]
        for seed in range(config["first_seed"], config["first_seed"] + config["replications"]):
            words = np.array([cutoff], dtype="<f8").view("<u4").tolist()
            rng = np.random.default_rng(np.random.SeedSequence([seed, *words]))
            uniforms = (rng.integers(0, 2**52, len(cohort), dtype=np.int64) + 0.5) / 2**52
            samples = [distributions[1][sum(j["need"] > b for b in (1, 8, 32))] for j in cohort]
            bundles.append(
                ("recent_coarse", seed, [x[int(np.ceil(u * len(x))) - 1] for x, u in zip(samples, uniforms)])
            )
        cases = {}
        for variant, seed, services in bundles:
            for cell in result["cells"]:
                if not cell["feasible"]:
                    continue
                key = cell["allocation"], cell["nodes"], variant, seed
                case, tape_hash = make_case(parts, cohort, services, forecasts, node_counts, cell, config["warmup"])
                cases[key] = case, cell
                tape = next(x for x in result["tapes"] if (x["allocation"], x["nodes"], x["variant"], x["seed"]) == key)
                assert tape["trace_sha256"] == tape_hash and tape["services_sha256"] == fingerprint(services)
                counter["workload_tapes"] += 1
        for row in result["rows"]:
            key = row["allocation"], row["nodes"], row["variant"], row["seed"]
            tape = next(x for x in result["tapes"] if (x["allocation"], x["nodes"], x["variant"], x["seed"]) == key)
            assert row["trace_sha256"] == tape["trace_sha256"] and row["services_sha256"] == tape["services_sha256"]
            if row["variant"] == "observed":
                replay(*cases[key], row)
                counter["observed_schedules"] += 1
        summaries(result, counter)
        print(json.dumps({"origin": index, "verified": dict(counter)}), flush=True)
    assert counter["rows"] == manifest["scheduler_runs"] == 1080
    print(json.dumps(dict(counter), indent=2))


if __name__ == "__main__":
    main()
