"""EPIC-064: opt-in MSJ capacity calendar over the audited Helios Venus trace.

Reuses the EPIC-063 raw Helios auditor (examples/resource_observability_audit.py)
for archive verification and parsing; no new source or download path. Three
prescribed, non-tuned scenarios over one prescribed window -- the first seven
calendar days covered by Venus's daily VC table (chosen before looking at any
replay result): fixed aggregate pool (control), daily aggregate pool, and the
single largest VC observed in that window, isolated. Grandfathering, explicit
infeasible/unresolved outcomes and reservation-calendar mismatch detection are
exercised by most_queue.sim.msj_lifecycle.MsjLifecycleSim.run_capacity_calendar;
see EPIC-063's go/no-go: daily VC counts are a bounded-scenario hypothesis, not
a reconstructed production quota. Oracle estimates (service == estimate) stand
in for an unavailable requested-runtime field in the Helios schema; this is a
simplifying assumption, not a forecast-quality claim.

    python -m examples.msj_capacity_calendar_experiment --download \
        --output-dir works/msj_capacity_calendar

See docs/epics/EPIC-064-msj-capacity-calendar.md.
"""

import argparse
import csv
import io
import json
import platform
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from examples.resource_observability_audit import (
    DAY,
    FIELDS,
    TERMINAL,
    digest,
    integer,
    read_capacities,
    timestamp,
    verified_archive,
)
from most_queue.sim.msj_lifecycle import LIFECYCLE_POLICIES, MsjLifecycleJob, MsjLifecycleSim
from most_queue.sim.utils.msj_capacity_calendar import constant_calendar, daily_calendar

CLUSTER = "Venus"
WINDOW_DAYS = 7
OUTCOME_BY_STATE = {
    "COMPLETED": "completed",
    "CANCELLED": "cancelled",
    "FAILED": "failed",
    "TIMEOUT": "timed_out",
    "NODE_FAIL": "node_failed",
}
SCENARIOS = ("fixed_aggregate", "daily_aggregate", "isolated_vc")


def extract_jobs(lines):
    """Every valid closed-terminal GPU row; same validity rule as the EPIC-063 audit."""
    reader = csv.DictReader(lines)
    if tuple(reader.fieldnames or ()) != FIELDS:
        raise ValueError("unexpected Helios job schema")
    rows = []
    for row in reader:
        if None in row or any(v is None for v in row.values()):
            continue
        need = integer(row["gpu_num"])
        if need is None or need == 0:
            continue
        submit, start, end = (timestamp(row[k]) for k in ("submit_time", "start_time", "end_time"))
        if submit is None or start is None or end is None or not submit <= start < end:
            continue
        if row["state"] not in TERMINAL:
            continue
        rows.append(
            {
                "job_id": row["job_id"],
                "vc": row["vc"],
                "need": need,
                "submit": submit,
                "start": start,
                "end": end,
                "outcome": OUTCOME_BY_STATE[row["state"]],
            }
        )
    return rows


def load_cluster(payload, cluster):
    """Read only the two allowlisted members for one cluster, with input hashes."""
    inputs = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        texts = {}
        for table in ("cluster_gpu_number", "cluster_log"):
            filename = f"data/{cluster}/{table}.csv"
            raw = archive.read(filename)
            inputs.append({"file": filename, "bytes": len(raw), "sha256": digest(raw)})
            texts[table] = io.StringIO(raw.decode("utf-8-sig"))
        capacities, _ = read_capacities(texts["cluster_gpu_number"])
        jobs = extract_jobs(texts["cluster_log"])
    return capacities, jobs, inputs


def select_window(capacities, days):
    """The first ``days`` consecutive calendar days covered by the daily table."""
    first_day = min(capacities)
    window_days = list(range(first_day, first_day + days))
    if any(day not in capacities for day in window_days):
        raise ValueError("prescribed window is not fully covered by the daily capacity table")
    return window_days


def window_jobs(jobs, window_days):
    """Jobs whose submission falls inside the window; sorted, ties broken by ID."""
    begin, end = window_days[0] * DAY, (window_days[-1] + 1) * DAY
    selected = [job for job in jobs if begin <= job["submit"] < end]
    selected.sort(key=lambda job: (job["submit"], job["job_id"]))
    return selected, float(begin), float(end)


def pick_largest_vc(selected_jobs, capacities, window_days):
    """The VC with the most in-window jobs among VCs with a known daily column."""
    counts = Counter(job["vc"] for job in selected_jobs)
    known = {vc for day in window_days for vc in capacities[day] if vc != "total"}
    candidates = {vc: n for vc, n in counts.items() if vc in known}
    if not candidates:
        raise ValueError("no known VC observed in the prescribed window")
    return max(sorted(candidates), key=lambda vc: candidates[vc])


def build_trace(selected_jobs, vc_filter=None):
    """Oracle-estimate lifecycle jobs for one scenario's job subset."""
    rows = [job for job in selected_jobs if vc_filter is None or job["vc"] == vc_filter]
    needs_sorted = sorted({job["need"] for job in rows})
    cls_of = {need: idx for idx, need in enumerate(needs_sorted)}
    trace = tuple(
        MsjLifecycleJob(
            float(job["submit"]),
            cls_of[job["need"]],
            float(job["end"] - job["start"]),
            estimate=float(job["end"] - job["start"]),
            outcome=job["outcome"],
        )
        for job in rows
    )
    return trace, needs_sorted


def run_scenario(trace, needs_sorted, calendar):
    """Replay one scenario's trace under all six lifecycle policies."""
    if not trace:
        raise ValueError("scenario trace must not be empty")
    capacity = max(needs_sorted)
    policies = {}
    for policy in LIFECYCLE_POLICIES:
        sim = MsjLifecycleSim(capacity, policy)
        sim.set_servers(needs_sorted)
        result = sim.run_capacity_calendar(trace, calendar)
        policies[policy] = {
            "status_counts": dict(sorted(Counter(result.status).items())),
            "control_outcome_counts": dict(sorted(Counter(result.control_outcomes).items())),
            "infeasible_count": len(result.infeasible_indices),
            "reservation_mismatch_time": result.reservation_mismatch_time,
            "utilization": result.utilization,
            "idle_with_queue": result.idle_with_queue,
            "observation_time": result.observation_time,
            "resource_time_by_status": result.resource_time_by_status,
            "backfilled": result.backfilled,
            "reservation_violations": result.reservation_violations,
            "horizon_end": result.horizon_end,
        }
    return policies


def main():
    """Verify source, replay three prescribed scenarios, write hashes and results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/resource_observability"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    payload = verified_archive(args.cache_dir / "helios-data.zip", args.download)
    capacities, jobs, inputs = load_cluster(payload, CLUSTER)
    window_days = select_window(capacities, WINDOW_DAYS)
    selected, begin, end = window_jobs(jobs, window_days)
    if not selected:
        raise ValueError("prescribed window has no valid closed-terminal GPU rows")
    vc = pick_largest_vc(selected, capacities, window_days)

    fixed_capacity = capacities[window_days[0]]["total"]
    calendars = {
        "fixed_aggregate": constant_calendar(fixed_capacity, begin, end),
        "daily_aggregate": daily_calendar({day: capacities[day]["total"] for day in window_days}, day_seconds=DAY),
        "isolated_vc": daily_calendar({day: capacities[day][vc] for day in window_days}, day_seconds=DAY),
    }
    result = {
        "schema_version": 1,
        "cluster": CLUSTER,
        "window_days": [window_days[0], window_days[-1]],
        "window_seconds": [begin, end],
        "isolated_vc": vc,
        "window_job_count": len(selected),
        "scenarios": {},
    }
    for name in SCENARIOS:
        vc_filter = vc if name == "isolated_vc" else None
        trace, needs_sorted = build_trace(selected, vc_filter)
        calendar = calendars[name]
        result["scenarios"][name] = {
            "job_count": len(trace),
            "needs": needs_sorted,
            "calendar": {
                "times": list(calendar.times),
                "capacities": list(calendar.capacities),
                "domain_end": calendar.domain_end,
            },
            "policies": run_scenario(trace, needs_sorted, calendar),
        }

    output = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "capacity-calendar-helios.json").write_bytes(output)
    hashes = {
        name: digest((Path(__file__).resolve().parents[1] / name).read_bytes())
        for name in (
            "examples/msj_capacity_calendar_experiment.py",
            "examples/resource_observability_audit.py",
            "most_queue/sim/msj_lifecycle.py",
            "most_queue/sim/utils/msj_capacity_calendar.py",
        )
    }
    manifest = {
        "archive_sha256": digest(payload),
        "inputs": inputs,
        "attribution": "SenseTime / S-Lab-System-Group; Hu et al., SC 2021, doi:10.1145/3458817.3476223",
        "source_license": "CC-BY-4.0, not the project's MIT license",
        "implementation_sha256": hashes,
        "numpy": np.__version__,
        "environment": {"python": platform.python_version()},
        "output": "capacity-calendar-helios.json",
        "output_sha256": digest(output),
        "scheduler_runs": len(SCENARIOS) * len(LIFECYCLE_POLICIES),
        "interpretation": "Bounded capacity-calendar sensitivity, not production reconstruction. Oracle estimates; "
        "no intraday quota, borrowing, placement or real initial-state observation. Daily VC counts are a "
        "modeled scenario assumption per EPIC-063, not a confirmed production quota.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({name: result["scenarios"][name]["job_count"] for name in SCENARIOS}))


if __name__ == "__main__":
    main()
