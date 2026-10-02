"""Independent EPIC-062 prefix, marked tapes, matched controls and metrics audit.

No runner, generator, adapter, fitting or objective imports. The independent
EPIC-059 raw parser is reused, as is the dispatch engine for observed replays.
This is not a second implementation of a scheduling discipline.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from most_queue.sim.msj_general import MsjGeneralSim, MsjTraceJob
from works.feature_service import audit as base

POLICIES = ("fcfs", "first_fit", "msf", "adaptive_quickswap", "easy", "conservative")
VARIANTS = (
    "fixed_coarse",
    "recent_gap_independent",
    "recent_joint_iid",
    "recent_joint_block20",
    "recent_joint_shuffle20",
    "expanding_joint_iid",
)


def mark(job, gap, name):
    return {
        "gap": float(gap),
        "need": job["need"],
        "context": base.context(job, name),
        "requested_time": job["requested_time"] if name == "sdsc" else None,
    }


def donors(prefix, name):
    return [mark(b, b["submit"] - a["submit"], name) for a, b in zip(prefix, prefix[1:])]


def permutation(keys, warmup):
    result = list(range(len(keys)))
    for begin, end in ((0, warmup), (warmup, len(keys))):
        result[begin + 1 : end] = sorted(range(begin + 1, end), key=lambda i: keys[i])
    return result


def sample(records, uniforms, length):
    output = []
    for position in range(0, len(uniforms), length):
        start = int(uniforms[position] * len(records))
        output.extend(records[(start + j) % len(records)] for j in range(length))
    return output[: len(uniforms)]


def tapes(prefix, history, cohort, name, cutoff, config):
    recorded = [mark(j, 0 if i == 0 else j["submit"] - cohort[i - 1]["submit"], name) for i, j in enumerate(cohort)]
    result = {("observed", None): (recorded, [base.service(j) for j in cohort])}
    full = donors(prefix, name)
    recent = full[-config["history_limit"] :]
    words = np.array([cutoff], dtype="<f8").view("<u4").tolist()
    for seed in range(config["first_seed"], config["first_seed"] + config["replications"]):
        children = np.random.SeedSequence([seed, ("sdsc", "kalos").index(name), *words]).spawn(4)
        streams = [
            (np.random.default_rng(c).integers(0, 2**52, len(cohort), dtype=np.int64) + 0.5) / 2**52 for c in children
        ]
        du, su, mu, bu = streams
        iid = sample(recent, du, 1)
        blocked = sample(recent, du, config["block_length"])
        order = permutation(mu, config["warmup"])
        shuffled = permutation(bu, config["warmup"])
        records = {
            "fixed_coarse": recorded,
            "recent_joint_iid": iid,
            "recent_gap_independent": [dict(iid[j], gap=iid[i]["gap"]) for i, j in enumerate(order)],
            "recent_joint_block20": blocked,
            "expanding_joint_iid": sample(full, du, 1),
        }
        for variant, rows in records.items():
            services = []
            for job, u in zip(rows, su):
                x, _ = base.support(history, job, name, "coarse", config["minimum"])
                services.append(float(x[int(np.ceil(u * len(x))) - 1]))
            result[variant, seed] = (rows, services)
        block_services = result["recent_joint_block20", seed][1]
        result["recent_joint_shuffle20", seed] = ([blocked[i] for i in shuffled], [block_services[i] for i in shuffled])
    return result


def arrival_times(records):
    return np.r_[0, np.cumsum([j["gap"] for j in records[1:]])]


def correlation(left, right):
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    x, y = np.asarray(left) - np.mean(left), np.asarray(right) - np.mean(right)
    return float(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y)))


def same(actual, expected):
    if expected is None:
        assert actual is None
    else:
        base.close(actual, expected)


def check_diagnostics(records, services, warmup, observed, reported):
    times = arrival_times(records)
    target = records[warmup:]
    ref = observed[warmup:]
    gaps = np.diff(times[warmup:])
    needs = np.array([j["need"] for j in target])
    svc = np.array(services[warmup:])
    horizon = times[-1] - times[warmup]
    expected = {
        "target_count": len(target),
        "arrival_span": horizon,
        "mean_gap": np.mean(gaps) if len(gaps) else None,
        "gap_scv": np.var(gaps) / np.mean(gaps) ** 2 if len(gaps) and np.mean(gaps) > 0 else None,
        "zero_gap_fraction": np.mean(gaps == 0) if len(gaps) else None,
        "lag1_gap": correlation(gaps[:-1], gaps[1:]),
        "lag1_need": correlation(needs[:-1], needs[1:]),
        "gap_need_correlation": correlation(gaps, needs[1:]),
        "mean_s": np.mean(svc),
        "p99_s": np.quantile(svc, 0.99),
        "target_resource_work": np.dot(needs, svc),
        "all_resource_work": np.dot([j["need"] for j in records], services),
        "transition_rate": (len(target) - 1) / horizon if horizon else None,
    }
    for key, value in expected.items():
        same(reported[key], value)
    fields = {
        "need_group_tv": base.group,
        "context_tv": lambda j: j["context"],
        "joint_mark_tv": lambda j: (base.group(j), j["context"]),
    }
    for field, key in fields.items():
        a, b = Counter(key(j) for j in target), Counter(key(j) for j in ref)
        total = sum(abs(a[k] / len(target) - b[k] / len(ref)) for k in a.keys() | b.keys()) / 2
        base.close(total, reported[field])
    assert dict(Counter(str(j["context"]) for j in target)) == reported["context_counts"]
    assert [sum(base.group(j) == g for j in target) for g in range(4)] == reported["group_counts"]


def trace(records, services, forecasts):
    needs = sorted({j["need"] for j in records})
    jobs = [
        MsjTraceJob(float(a), needs.index(j["need"]), float(s), float(forecasts[j["need"]]))
        for a, j, s in zip(arrival_times(records), records, services)
    ]
    return needs, jobs


def check_schedule(records, services, forecasts, capacity, warmup, row):
    needs, jobs = trace(records, services, forecasts)
    simulator = MsjGeneralSim(capacity, row["policy"])
    simulator.set_servers(needs)
    result = simulator.run_trace(jobs, warmup_jobs=warmup)
    arrivals = arrival_times(records)
    starts = np.array(result.start_times)
    ends = np.array(result.completion_times)
    waits = starts[warmup:] - arrivals[warmup:]
    times = ends[warmup:] - arrivals[warmup:]
    weights = np.array([j["need"] for j in records[warmup:]])
    for k, value in {
        "mean_t": np.mean(times),
        "p99_t": np.quantile(times, 0.99),
        "mean_w": np.mean(waits),
        "weighted_mean_t": np.average(times, weights=weights),
        "resource_time": np.dot(ends - starts, [j["need"] for j in records]),
    }.items():
        base.close(value, row[k])
    groups = np.array([base.group(j) for j in records[warmup:]])
    for g, reported in enumerate(row["groups"]):
        x = times[groups == g]
        assert len(x) == reported["count"]
        same(reported["mean_t"], np.mean(x) if len(x) else None)
        same(reported["p99_t"], np.quantile(x, 0.99) if len(x) else None)
    events = defaultdict(lambda: [0, 0])
    for i, job in enumerate(records):
        events[arrivals[i]][1] += 1
        events[starts[i]][1] -= 1
        events[starts[i]][0] += job["need"]
        events[ends[i]][0] -= job["need"]
    occupied = queue = busy = idle = previous = 0
    for instant, (resource, waiting) in sorted(events.items()):
        dt = max(0, min(instant, arrivals[-1]) - max(previous, arrivals[warmup]))
        busy += occupied * dt
        idle += (capacity - occupied) * dt if queue else 0
        occupied += resource
        queue += waiting
        previous = instant
        assert 0 <= occupied <= capacity and queue >= 0
    assert occupied == queue == 0
    span = arrivals[-1] - arrivals[warmup]
    same(row["utilization"], busy / (capacity * span) if span else None)
    same(row["idle_with_queue"], idle / (capacity * span) if span else None)


def log_loss(samples, reference):
    return float(np.mean(np.abs(np.log(np.array(samples) / reference))))


def check_summaries(result, counter):
    rows = result["rows"]
    index = {(r["variant"], r["policy"], r["seed"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows if r["variant"] != "observed"})
    assert len(index) == len(rows) == result["scheduler_runs"]
    assert set(index) == {("observed", p, None) for p in POLICIES} | {
        (v, p, s) for v in VARIANTS for p in POLICIES for s in seeds
    }
    for summary in result["summaries"]:
        v, p = summary["variant"], summary["policy"]
        for metric, reported in summary["metrics"].items():
            values = [index[v, p, s][metric] for s in seeds if index[v, p, s][metric] is not None]
            assert len(values) == reported["valid_replications"]
            if len(values) >= 2:
                base.confidence(values, reported)
                ref = index["observed", p, None][metric]
                if ref:
                    base.close(np.mean(values) / ref - 1, reported["relative_error"])
            counter["model_intervals"] += 1
    reference = [[index["observed", p, None][k] for k in ("mean_t", "p99_t")] for p in POLICIES]
    loss_by_variant = {}
    for record in result["losses"]:
        v = record["variant"]
        samples = np.array([[[index[v, p, s][k] for k in ("mean_t", "p99_t")] for p in POLICIES] for s in seeds])
        loss_by_variant[v] = [log_loss(x, reference) for x in samples]
        np.testing.assert_allclose(record["seed_losses"], loss_by_variant[v], rtol=1e-11, atol=1e-12)
        base.close(log_loss(samples.mean(axis=0), reference), record["loss_of_mc_means"])
        counter["queue_losses"] += 1
    for contrast in result["contrasts"]:
        values = np.array(loss_by_variant[contrast["left"]]) - loss_by_variant[contrast["right"]]
        base.confidence(values, contrast["paired_loss_change"])
        counter["paired_contrasts"] += 1
    for record in result["decisions"]:
        v, k = record["variant"], record["metric"]
        predictions = {p: np.mean([index[v, p, s][k] for s in seeds]) for p in POLICIES}
        ref = {p: index["observed", p, None][k] for p in POLICIES}
        best = min(ref.values())
        ties = [p for p in POLICIES if np.isclose(ref[p], best, rtol=1e-12, atol=1e-9)]
        assert record["chosen"] == min(POLICIES, key=predictions.get)
        assert record["reference_best"] == ties and record["reference_tie_count"] == len(ties)
        base.close(ref[record["chosen"]] / best - 1, record["relative_regret"])
        counter["decisions"] += 1


def check_block(terminal, bounds, result, protocol, counter):
    name, c = result["source"], result["config"]
    cutoff = bounds[0] + result["fraction"] * (bounds[1] - bounds[0])
    completed = [j for j in terminal if base.label(j) == "completed"]
    expanding = [j for j in completed if base.end(j) < cutoff]
    history = expanding[-c["history_limit"] :]
    past = [j for j in completed if j["submit"] < cutoff]
    stop = next((i for i, j in enumerate(past) if base.end(j) >= cutoff), len(past))
    prefix = past[:stop]
    cohort = [j for j in completed if j["submit"] >= cutoff][: c["warmup"] + c["jobs"]]
    for label, jobs in (
        ("history", history),
        ("prefix", prefix),
        ("recent_prefix", prefix[-c["history_limit"] - 1 :]),
        ("cohort", cohort),
    ):
        assert base.digest(jobs) == result[label]["input_sha256"]
    for key, value in protocol.items():
        assert result[key] == value
    a = result["prefix_audit"]
    assert a["prefix_jobs"] == len(prefix) and a["prefix_donors"] == len(prefix) - 1
    assert a["past_arrivals"] == len(past) and a["omitted_suffix"] == len(past) - len(prefix)
    assert a["completed_suffix_omitted"] == sum(base.end(j) < cutoff for j in past[stop:])
    base.close(cutoff - prefix[-1]["submit"], a["prefix_lag"])
    forecasts = {
        k: float(np.quantile(base.support(expanding, {"need": k}, name, "coarse", c["minimum"])[0], 0.9))
        for k in range(1, result["capacity"] + 1)
    }
    assert base.fingerprint(list(forecasts.values())) == result["forecast_sha256"]
    workloads = tapes(prefix, history, cohort, name, cutoff, c)
    reference = workloads["observed", None][0]
    for record in result["tapes"]:
        key = record["variant"], record["seed"]
        marks, services = workloads[key]
        assert base.digest(marks) == record["marks_sha256"]
        assert base.fingerprint(services) == record["services_sha256"]
        needs, jobs = trace(marks, services, forecasts)
        serialized = {
            "needs": needs,
            "trace": [{"arrival": j.arrival, "cls": j.cls, "service": j.service, "estimate": j.estimate} for j in jobs],
        }
        assert base.digest(serialized) == record["trace_sha256"]
        check_diagnostics(marks, services, c["warmup"], reference, record)
        counter["workload_tapes"] += 1
    for seed in range(c["first_seed"], c["first_seed"] + c["replications"]):
        for begin, end in ((0, c["warmup"]), (c["warmup"], len(cohort))):
            bags = []
            for variant in ("recent_joint_block20", "recent_joint_shuffle20"):
                marks, services = workloads[variant, seed]
                bags.append(
                    Counter(
                        (j["gap"], j["need"], j["context"], j["requested_time"], s)
                        for j, s in zip(marks[begin:end], services[begin:end])
                    )
                )
            assert bags[0] == bags[1]
        counter["matched_block_pairs"] += 1
    by = {(r["variant"], r["seed"]): r for r in result["tapes"]}
    for row in result["rows"]:
        key = row["variant"], row["seed"]
        marks, services = workloads[key]
        assert row["trace_sha256"] == by[key]["trace_sha256"]
        base.close(row["resource_time"], by[key]["all_resource_work"])
        base.close(row["mean_t"] - row["mean_w"], row["target_mean_s"])
        assert row["target_count"] == c["jobs"] == sum(g["count"] for g in row["groups"])
        if row["variant"] == "observed":
            check_schedule(marks, services, forecasts, result["capacity"], c["warmup"], row)
            counter["observed_schedules"] += 1
        counter["rows"] += 1
    check_summaries(result, counter)
    return {j["job_id"] for j in cohort}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/real_trace"))
    parser.add_argument("--output-dir", type=Path, default=Path("works/joint_marked_arrivals"))
    args = parser.parse_args()
    folder = args.output_dir
    m = json.loads((folder / "manifest.json").read_text())
    protocol = json.loads((folder / "protocol.json").read_text())
    for p, h in m["implementation_sha256"].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h, p
    for a in m["artifacts"]:
        assert hashlib.sha256((folder / a["file"]).read_bytes()).hexdigest() == a["sha256"]
    counter = Counter()
    for name, filename in (("sdsc", "SDSC-SP2-1998-4.2-cln.swf"), ("kalos", "acme-kalos.csv")):
        base.SUPPORTS.clear()
        path = args.cache_dir / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == m["sources"][name]["sha256"]
        terminal, bounds = base.load_source(path, name)
        ids = set()
        for index in range(2):
            result = json.loads((folder / f"{name}-{index}.json").read_text())
            new = check_block(terminal, bounds, result, protocol["blocks"][name][index], counter)
            assert not ids.intersection(new)
            ids.update(new)
            print(json.dumps({"source": name, "origin": result["fraction"], "verified": dict(counter)}), flush=True)
    assert counter["rows"] == m["scheduler_runs"] == 1176
    print(json.dumps(dict(counter), indent=2))


if __name__ == "__main__":
    main()
