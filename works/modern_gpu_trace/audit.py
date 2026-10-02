"""Independent EPIC-058 artifact audit, without experiment/adapter/fit helpers.

Run: python -m works.modern_gpu_trace.audit --cache .cache/real_trace/acme-kalos.csv
Reconstructs 204 tapes, all summaries, and 72 observed schedules. Also compares
408 empty-state rows with ordinary MsjGeneralSim. Writes nothing; prints counts.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import t

from most_queue.sim.msj_general import MsjGeneralSim, MsjTraceJob
from most_queue.sim.msj_lifecycle import MsjCarryIn, MsjLifecycleJob, MsjLifecycleSim


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def close(a, b):
    assert np.isclose(a, b, rtol=1e-12, atol=1e-9), (a, b)


def confidence(values, reported):
    mean = np.mean(values)
    close(mean, reported["mean"])
    if len(values) == 1:
        assert reported["low"] is None and reported["high"] is None
    else:
        half = t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values))
        close(mean - half, reported["low"])
        close(mean + half, reported["high"])


def schedule_metrics(case, capacity):
    """Integrate resources from output event intervals, independently of engine areas."""
    simulator = MsjLifecycleSim(capacity, case["policy"])
    simulator.set_servers(range(1, capacity + 1))
    running = [MsjCarryIn(MsjLifecycleJob(**item["job"]), item["age"]) for item in case["running"]]
    waiting = [MsjLifecycleJob(**job) for job in case["waiting"]]
    arrivals = [MsjLifecycleJob(**job) for job in case["arrivals"]]
    result = simulator.run_lifecycle(
        arrivals, initial_running=running, initial_waiting=waiting, observation_window=case["window"]
    )
    jobs = [item.job for item in running] + waiting + arrivals
    starts, ends = np.array(result.start_times), np.array(result.release_times)
    needs = np.array([j.cls + 1 for j in jobs])
    new_offset = len(running) + len(waiting)
    assert np.all(starts[len(running) :] >= np.array([j.arrival for j in jobs[len(running) :]]))
    consumed = ends - np.maximum(starts, 0)
    close(np.dot(consumed, needs), sum(result.resource_time_by_outcome.values()))
    for label in result.resource_time_by_outcome:
        close(
            sum(consumed[i] * needs[i] for i, job in enumerate(jobs) if job.outcome == label),
            result.resource_time_by_outcome[label],
        )
    for i, job in enumerate(jobs):
        close(ends[i] - starts[i], job.service)
        assert result.outcomes[i] == job.outcome
    events = defaultdict(lambda: [0, 0])
    for i, job in enumerate(jobs):
        events[max(starts[i], 0)][0] += needs[i]
        events[ends[i]][0] -= needs[i]
        if i >= len(running):
            events[job.arrival][1] += 1
            events[starts[i]][1] -= 1
    occupied = queue = busy = idle = 0
    previous = 0
    begin, end = case["window"]
    for time, (resource_delta, queue_delta) in sorted(events.items()):
        duration = max(0, min(time, end) - max(previous, begin))
        busy += occupied * duration
        idle += (capacity - occupied) * duration if queue else 0
        occupied += resource_delta
        queue += queue_delta
        assert 0 <= occupied <= capacity and queue >= 0
        previous = time
    assert occupied == queue == 0
    close(result.utilization, busy / (capacity * (end - begin)))
    close(result.idle_with_queue, idle / (capacity * (end - begin)))
    targets = np.array(case["targets"])
    arrival = np.array([arrivals[i - new_offset].arrival for i in targets])
    latency = ends[targets] - arrival
    return {
        "mean_t": np.mean(latency),
        "p99_t": np.quantile(latency, 0.99),
        "mean_w": np.mean(starts[targets] - arrival),
        "gpu_weighted_mean_t": np.average(latency, weights=needs[targets]),
        "utilization": result.utilization,
        "idle_with_queue": result.idle_with_queue,
        "total_resource_time": sum(result.resource_time_by_outcome.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("works/modern_gpu_trace"))
    args = parser.parse_args()
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    raw = args.cache.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest["source"]["sha256"]
    for name, value in manifest["implementation_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == value, name
    table = list(csv.DictReader(raw.decode().splitlines()))
    labels = {"COMPLETED": "completed", "CANCELLED": "cancelled", "FAILED": "failed"}
    terminal, raw_submits = [], []
    for row in table:
        timestamp = lambda name: datetime.fromisoformat(row[name]).timestamp() if row[name] else None
        submit, start, end = (timestamp(name) for name in ("submit_time", "start_time", "end_time"))
        raw_submits.append(submit)
        if int(row["gpu_num"]) > 0 and row["state"] in labels and end is not None and end > start >= submit:
            terminal.append(
                {
                    "job_id": row["job_id"],
                    "submit": submit,
                    "started_at": start,
                    "ended_at": end,
                    "need": int(row["gpu_num"]),
                    "outcome": labels[row["state"]],
                    "workload_type": row["type"],
                }
            )
    completed = [j for j in terminal if j["outcome"] == "completed"]
    actual_counts = Counter(j["outcome"] for j in terminal)
    for outcome, count in actual_counts.items():
        assert manifest["audit"]["counts"][f"accepted_{outcome}"] == count
    for outcome in labels.values():
        close(
            sum(j["need"] * (j["ended_at"] - j["started_at"]) for j in terminal if j["outcome"] == outcome),
            manifest["source_diagnostics"]["accepted_request_seconds_by_outcome"][outcome],
        )
    counter = Counter()
    all_origins = []
    config = manifest["config"]
    for artifact, fraction in zip(manifest["artifacts"], config["fractions"]):
        payload = (args.output_dir / artifact["file"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == artifact["sha256"]
        origin = json.loads(payload)
        all_origins.append(origin)
        cutoff = min(raw_submits) + fraction * (max(raw_submits) - min(raw_submits))
        history = [j for j in completed if j["ended_at"] < cutoff]
        cohort = [j for j in completed if j["submit"] >= cutoff][: config["jobs"] + config["warmup"]]
        assert digest(history) == origin["history_sha256"] and digest(cohort) == origin["cohort_sha256"]
        assert len(history) == origin["split"]["training"]
        distributions = {}
        for name, jobs in (("expanding_coarse", history), ("recent_coarse", history[-config["history_limit"] :])):
            pooled = np.sort([j["ended_at"] - j["started_at"] for j in jobs])
            groups = []
            for group in range(4):
                values = np.sort(
                    [j["ended_at"] - j["started_at"] for j in jobs if np.searchsorted([1, 8, 32], j["need"]) == group]
                )
                groups.append(values if len(values) >= config["minimum"] else pooled)
            distributions[name] = groups
        forecasts = {
            k: float(np.quantile(distributions["expanding_coarse"][np.searchsorted([1, 8, 32], k)], 0.9))
            for k in {j["need"] for j in terminal}
        }
        assert digest(sorted(forecasts.items())) == origin["forecast_sha256"]
        boundary, end = cohort[0]["submit"], cohort[-1]["submit"]
        window = (cohort[config["warmup"]]["submit"] - boundary, end - boundary)
        policy_order = list(dict.fromkeys(r["policy"] for r in origin["rows"]))
        checked = set()
        for row in origin["rows"]:
            key = row["scenario"], row["model"], row["seed"]
            services = np.array([j["ended_at"] - j["started_at"] for j in cohort])
            if row["model"] != "observed":
                words = np.array([cutoff], dtype="<f8").view("<u4").tolist()
                rng = np.random.default_rng(np.random.SeedSequence([row["seed"], *words]))
                uniforms = (rng.integers(0, 2**52, len(cohort), dtype=np.int64) + 0.5) / 2**52
                services = np.array(
                    [
                        distributions[row["model"]][np.searchsorted([1, 8, 32], j["need"])][
                            int(np.ceil(u * len(distributions[row["model"]][np.searchsorted([1, 8, 32], j["need"])])))
                            - 1
                        ]
                        for j, u in zip(cohort, uniforms)
                    ]
                )
            selected = {j["job_id"]: float(s) for j, s in zip(cohort, services)}
            carry = (
                []
                if row["scenario"] == "empty_completed"
                else [
                    j
                    for j in terminal
                    if j["submit"] < boundary < j["ended_at"]
                    and (j["outcome"] == "completed" or row["scenario"] == "carry_terminal")
                ]
            )
            running = [j for j in carry if j["started_at"] <= boundary]
            waiting = [j for j in carry if j["started_at"] > boundary]
            arrivals = [
                j
                for j in terminal
                if j["job_id"] in selected
                or (
                    row["scenario"] == "carry_terminal"
                    and j["outcome"] != "completed"
                    and boundary <= j["submit"] <= end
                )
            ]

            def job(j, initial=False):
                return {
                    "arrival": 0.0 if initial else j["submit"] - boundary,
                    "cls": j["need"] - 1,
                    "service": (
                        float(j["ended_at"] - j["started_at"])
                        if initial or j["job_id"] not in selected
                        else selected[j["job_id"]]
                    ),
                    "estimate": forecasts[j["need"]],
                    "outcome": j["outcome"],
                    "runtime_limit": None,
                }

            tape = {
                "running": [{"job": job(j, True), "age": boundary - j["started_at"]} for j in running],
                "waiting": [job(j, True) for j in waiting],
                "arrivals": [job(j) for j in arrivals],
                "source_ids": [j["job_id"] for j in running + waiting + arrivals],
            }
            assert digest(tape) == row["trace_sha256"]
            if key not in checked:
                counter["tapes"] += 1
                checked.add(key)
            close(np.mean(services[config["warmup"] :]), row["target_mean_s"])
            close(row["mean_t"] - row["mean_w"], row["target_mean_s"])
            ledger = defaultdict(float)
            for j in running:
                ledger[j["outcome"]] += j["need"] * (j["ended_at"] - boundary)
            for j in waiting:
                ledger[j["outcome"]] += j["need"] * (j["ended_at"] - j["started_at"])
            for j in arrivals:
                ledger[j["outcome"]] += j["need"] * job(j)["service"]
            for label, work in row["resource_time_by_outcome"].items():
                close(work, ledger[label])
            close(sum(ledger.values()), row["total_resource_time"])
            if row["model"] == "observed":
                ids = {j["job_id"]: len(running) + len(waiting) + i for i, j in enumerate(arrivals)}
                case = {
                    **tape,
                    "policy": row["policy"],
                    "window": window,
                    "targets": [ids[j["job_id"]] for j in cohort[config["warmup"] :]],
                }
                for name, value in schedule_metrics(case, manifest["capacity"]).items():
                    close(value, row[name])
                counter["control_replays"] += 1
            if row["scenario"] == "empty_completed":
                sim = MsjGeneralSim(manifest["capacity"], row["policy"])
                sim.set_servers(range(1, manifest["capacity"] + 1))
                trace = tuple(
                    MsjTraceJob(j["arrival"], j["cls"], j["service"], j["estimate"]) for j in tape["arrivals"]
                )
                result = sim.run_trace(trace, warmup_jobs=config["warmup"])
                for actual, name in (
                    (result.v[0], "mean_t"),
                    (result.w[0], "mean_w"),
                    (result.v_quantiles[0.99], "p99_t"),
                    (result.utilization, "utilization"),
                    (result.idle_with_queue, "idle_with_queue"),
                ):
                    assert actual == row[name], (name, actual, row[name])
                counter["legacy_rows"] += 1
            counter["rows"] += 1
        for summary in origin["summaries"]:
            matching = [r for r in origin["rows"] if all(r[k] == summary[k] for k in ("scenario", "model", "policy"))]
            observed = next(
                r
                for r in origin["rows"]
                if r["model"] == "observed" and all(r[k] == summary[k] for k in ("scenario", "policy"))
            )
            for metric, estimate in summary["metrics"].items():
                confidence([r[metric] for r in matching], estimate)
                close(estimate["reference"], observed[metric])
                close(estimate["difference"], estimate["mean"] - observed[metric])
                if observed[metric]:
                    close(estimate["relative_error"], estimate["mean"] / observed[metric] - 1)
                else:
                    assert estimate["relative_error"] is None
                counter["intervals"] += 1
        for change in origin["scenario_changes"]:
            a = [r for r in origin["rows"] if all(r[k] == change[k] for k in ("scenario", "model", "policy"))]
            b = [
                r
                for r in origin["rows"]
                if r["scenario"] == change["baseline"] and all(r[k] == change[k] for k in ("model", "policy"))
            ]
            assert [r["seed"] for r in a] == [r["seed"] for r in b]
            for metric, estimate in change["differences"].items():
                confidence([x[metric] - y[metric] for x, y in zip(a, b)], estimate)
                counter["intervals" if len(a) > 1 else "deterministic_changes"] += 1
        for decision in origin["decisions"]:
            metric = decision["metric"]
            means = {
                p: np.mean(
                    [
                        r[metric]
                        for r in origin["rows"]
                        if all(r[k] == decision[k] for k in ("scenario", "model")) and r["policy"] == p
                    ]
                )
                for p in policy_order
            }
            refs = {
                p: next(
                    r[metric]
                    for r in origin["rows"]
                    if r["scenario"] == decision["scenario"] and r["model"] == "observed" and r["policy"] == p
                )
                for p in policy_order
            }
            assert (
                min(means, key=means.get) == decision["chosen"]
                and min(refs, key=refs.get) == decision["reference_best"]
            )
            close(refs[decision["chosen"]] / min(refs.values()) - 1, decision["relative_regret"])
            counter["decisions"] += 1
        print(json.dumps({"audited_origin": fraction, **counter}), flush=True)
    for cell in manifest["aggregate"]:
        for metric in ("mean_t", "p99_t"):
            summaries = [
                s for o in all_origins for s in o["summaries"] if all(s[k] == cell[k] for k in ("scenario", "model"))
            ]
            decisions = [
                d
                for o in all_origins
                for d in o["decisions"]
                if all(d[k] == cell[k] for k in ("scenario", "model")) and d["metric"] == metric
            ]
            close(
                cell[metric]["mape_of_mc_means"],
                np.mean([abs(s["metrics"][metric]["relative_error"]) for s in summaries]),
            )
            assert cell[metric]["matching_choices"] == sum(d["chosen"] == d["reference_best"] for d in decisions)
            close(cell[metric]["mean_relative_regret"], np.mean([d["relative_regret"] for d in decisions]))
            close(cell[metric]["max_relative_regret"], max(d["relative_regret"] for d in decisions))
    assert counter["rows"] == manifest["scheduler_runs"]
    print(json.dumps({"status": "passed", **counter}), flush=True)


if __name__ == "__main__":
    main()
