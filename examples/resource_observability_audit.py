"""EPIC-063: audit Helios timestamps and daily VC counts, without replay.

Source: SenseTime / S-Lab-System-Group, HeliosData (CC-BY-4.0).
https://github.com/S-Lab-System-Group/HeliosData
Daily GPU counts are not assumed to be exact intraday hard quotas.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np

COMMIT = "159f0caeec16600b9b6017862952a36aae01c43f"
URL = f"https://raw.githubusercontent.com/S-Lab-System-Group/HeliosData/{COMMIT}/data.zip"
SHA256 = "3d22a5f6c0ae669e2fcbfe4200fa9c48664507bc397c677bad8f085222c032ac"
CLUSTERS = ("Earth", "Saturn", "Uranus", "Venus")
FIELDS = (
    "job_id",
    "user",
    "vc",
    "gpu_num",
    "cpu_num",
    "node_num",
    "state",
    "submit_time",
    "start_time",
    "end_time",
    "duration",
    "queue",
)
TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL"}
DAY = 86400


def digest(data):
    """Return the SHA-256 of an exact input or output payload."""
    return hashlib.sha256(data).hexdigest()


def verified_archive(path, download=False):
    """Fetch only on explicit opt-in, verify before saving, never replace cache."""
    if path.exists():
        payload = path.read_bytes()
    elif download:
        with urllib.request.urlopen(URL, timeout=60) as response:
            payload = response.read(50_000_001)
        if len(payload) > 50_000_000:
            raise ValueError("archive exceeds the fixed download limit")
    else:
        raise FileNotFoundError("Helios cache missing; pass --download to obtain the pinned source")
    if digest(payload) != SHA256:
        raise ValueError("Helios archive SHA-256 mismatch")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return payload


def integer(value):
    """Parse a nonnegative decimal integer, without silently rounding demand."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9]+", value) is None:
        return None
    return int(value)


def number(value):
    """Return a finite numeric exported duration, or None for invalid data."""
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def timestamp(value):
    """Parse naive source clock seconds; do not assign an unreported timezone."""
    if not isinstance(value, str) or len(value) != 19 or value[10] != " ":
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        return None
    return parsed.toordinal() * DAY + parsed.hour * 3600 + parsed.minute * 60 + parsed.second


def read_capacities(lines):
    """Read exact daily configurations; fail on ambiguous keys or malformed rows."""
    reader = csv.DictReader(lines)
    fields = reader.fieldnames
    if not fields or fields[0] != "date" or fields[-1] != "total" or len(set(fields)) != len(fields) or len(fields) < 3:
        raise ValueError("invalid capacity header")
    if any(not x.startswith("vc") for x in fields[1:-1]):
        raise ValueError("invalid capacity VC header")
    rows = {}
    for row in reader:
        if None in row or any(v is None for v in row.values()):
            raise ValueError("invalid capacity row width")
        day = date.fromisoformat(row["date"]).toordinal()
        if day in rows:
            raise ValueError("duplicate capacity date")
        values = {k: integer(row[k]) for k in fields[1:]}
        if any(v is None for v in values.values()):
            raise ValueError("invalid capacity count")
        rows[day] = values
    if not rows:
        raise ValueError("empty capacity table")
    days = sorted(rows)
    vcs = fields[1:-1]
    report = {
        "days": len(days),
        "first_date": date.fromordinal(days[0]).isoformat(),
        "last_date": date.fromordinal(days[-1]).isoformat(),
        "vc_columns": len(vcs),
        "missing_days": days[-1] - days[0] + 1 - len(days),
        "total_min": min(r["total"] for r in rows.values()),
        "total_max": max(r["total"] for r in rows.values()),
        "sum_vc_not_total_days": sum(sum(r[k] for k in vcs) != r["total"] for r in rows.values()),
        "total_changes": sum(rows[b]["total"] != rows[a]["total"] for a, b in zip(days, days[1:])),
        "vc_day_changes": sum(rows[b][k] != rows[a][k] for a, b in zip(days, days[1:]) for k in vcs),
        "zero_vc_days": sum(r[k] == 0 for r in rows.values() for k in vcs),
    }
    return rows, report


def occupancy(events, daily):
    """Integrate half-open intervals against a hypothetical constant daily count.

    Missing dates are excluded, never filled forward/backward. Events on equal
    timestamps are combined. The count is a diagnostic assumption, not a quota.
    """
    changes = defaultdict(int)
    for instant, delta in events:
        changes[instant] += delta
    for day in daily:
        changes[day * DAY] += 0
        changes[(day + 1) * DAY] += 0
    occupied = work = covered = excess_work = excess_seconds = peak = 0
    previous = None
    for instant in sorted(changes):
        if previous is not None:
            dt = instant - previous
            work += occupied * dt
            limit = daily.get(previous // DAY)
            if limit is not None:
                covered += occupied * dt
                excess = max(0, occupied - limit)
                excess_work += excess * dt
                excess_seconds += dt if excess else 0
        occupied += changes[instant]
        if occupied < 0:
            raise ValueError("negative recorded occupancy")
        peak = max(peak, occupied)
        previous = instant
    if occupied:
        raise ValueError("unclosed occupancy intervals")
    return {
        "peak_requested_gpu": peak,
        "full_gpu_seconds": work,
        "covered_gpu_seconds": covered,
        "excess_gpu_seconds": excess_work,
        "excess_seconds": excess_seconds,
        "compared_seconds": len(daily) * DAY,
    }


def capacity_at(capacities, vc, instant, need):
    """Audit same-date lookup separately for date, VC, zero and exceeded counts."""
    if instant is None:
        return {"invalid_time": 1}
    row = capacities.get(instant // DAY)
    if row is None:
        return {"missing_date": 1}
    if vc not in row or vc == "total":
        return {"unknown_vc": 1}
    return {"compared": 1, "zero_count": int(row[vc] == 0), "need_above_daily_count": int(need > row[vc])}


def audit_jobs(lines, capacities):
    """Audit every job, preserving all states and explicit invalid-record counts."""
    reader = csv.DictReader(lines)
    if tuple(reader.fieldnames or ()) != FIELDS:
        raise ValueError("unexpected Helios job schema")
    counts, states, gpu_states, matches = Counter(), Counter(), Counter(), Counter()
    seen, vcs = set(), set()
    events = defaultdict(list)
    waits, service = [], []
    bounds = {}
    for row in reader:
        counts["rows"] += 1
        if None in row or any(v is None for v in row.values()):
            counts["malformed_width"] += 1
            continue
        counts["duplicate_job_ids"] += row["job_id"] in seen
        seen.add(row["job_id"])
        counts["empty_job_id"] += not row["job_id"]
        counts["empty_user"] += not row["user"]
        states[row["state"]] += 1
        vc = row["vc"]
        vcs.add(vc)
        need = integer(row["gpu_num"])
        counts["invalid_cpu"] += integer(row["cpu_num"]) is None
        counts["invalid_node"] += integer(row["node_num"]) is None
        counts["invalid_gpu"] += need is None
        counts["cpu_only"] += need == 0
        counts["gpu_jobs"] += need is not None and need > 0
        submit, start, end = (timestamp(row[k]) for k in ("submit_time", "start_time", "end_time"))
        for key, value in (("submit", submit), ("start", start), ("end", end)):
            counts[f"invalid_{key}"] += value is None
            if value is not None:
                bounds[f"first_{key}"] = min(bounds.get(f"first_{key}", value), value)
                bounds[f"last_{key}"] = max(bounds.get(f"last_{key}", value), value)
        duration, queue = number(row["duration"]), number(row["queue"])
        counts["invalid_duration"] += duration is None
        counts["invalid_queue"] += queue is None
        if start is not None and end is not None:
            counts["closed_timestamps"] += 1
            counts["negative_execution"] += end < start
            counts["zero_execution"] += end == start
            matches["duration_checked"] += duration is not None
            matches["duration_not_execution"] += duration is not None and duration != end - start
            if submit is not None and duration is not None:
                matches["duration_equals_sojourn"] += duration == end - submit
        if submit is not None and start is not None:
            counts["negative_wait"] += start < submit
            matches["queue_checked"] += queue is not None
            matches["queue_not_wait"] += queue is not None and queue != start - submit
        if need is None or need == 0:
            continue
        gpu_states[row["state"]] += 1
        for stage, instant in (("gpu_submit", submit), ("gpu_start", start)):
            counts.update({f"{stage}_{k}": v for k, v in capacity_at(capacities, vc, instant, need).items()})
        valid = submit is not None and start is not None and end is not None and submit <= start < end
        if not valid or row["state"] not in TERMINAL:
            counts["gpu_excluded_from_occupancy"] += 1
            continue
        counts["gpu_closed_terminal"] += 1
        events[vc].extend(((start, need), (end, -need)))
        if row["state"] == "COMPLETED":
            counts["gpu_completed_positive"] += 1
            waits.append(start - submit)
            service.append(end - start)
    counts["unique_job_ids"] = len(seen)
    known = set(next(iter(capacities.values()))) - {"total"}
    counts["observed_vcs"] = len(vcs)
    counts["unknown_vcs"] = len(vcs - known)
    vc_reports = [
        occupancy(items, {d: r[vc] for d, r in capacities.items() if vc in r and vc != "total"})
        for vc, items in sorted(events.items())
    ]
    all_events = (event for items in events.values() for event in items)
    total = occupancy(all_events, {d: r["total"] for d, r in capacities.items()})
    total["vc_excess_gpu_seconds"] = sum(r["excess_gpu_seconds"] for r in vc_reports)
    total["vc_excess_seconds_sum"] = sum(r["excess_seconds"] for r in vc_reports)
    total["vcs_with_excess"] = sum(r["excess_seconds"] > 0 for r in vc_reports)
    total["vcs_with_intervals"] = len(vc_reports)
    total["vc_covered_gpu_seconds"] = sum(r["covered_gpu_seconds"] for r in vc_reports)
    completed = {
        "count": len(service),
        "mean_s": float(np.mean(service)) if service else None,
        "p99_s": float(np.quantile(service, 0.99)) if service else None,
        "mean_recorded_w": float(np.mean(waits)) if waits else None,
        "positive_recorded_waits": sum(w > 0 for w in waits),
    }
    clocks = {
        k: f"{date.fromordinal(v // DAY).isoformat()} {(v % DAY) // 3600:02d}:{(v % 3600) // 60:02d}:{v % 60:02d}"
        for k, v in bounds.items()
    }
    return {
        "counts": dict(sorted(counts.items())),
        "states": dict(sorted(states.items())),
        "gpu_states": dict(sorted(gpu_states.items())),
        "derived_fields": dict(sorted(matches.items())),
        "source_clock_bounds": clocks,
        "closed_terminal_occupancy": total,
        "completed_gpu_descriptive": completed,
    }


def audit_archive(payload):
    """Read only the eight allowlisted members, without filesystem extraction."""
    results, inputs = {}, []
    expected = {f"data/{c}/{f}.csv" for c in CLUSTERS for f in ("cluster_log", "cluster_gpu_number")}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [i for i in archive.infolist() if not i.is_dir()]
        if {i.filename for i in members} != expected or len(members) != len(expected):
            raise ValueError("archive members differ from the fixed allowlist")
        if sum(i.file_size for i in members) > 500_000_000:
            raise ValueError("archive expanded size exceeds limit")
        for cluster in CLUSTERS:
            texts = {}
            for table in ("cluster_gpu_number", "cluster_log"):
                filename = f"data/{cluster}/{table}.csv"
                raw = archive.read(filename)
                inputs.append({"file": filename, "bytes": len(raw), "sha256": digest(raw)})
                texts[table] = io.StringIO(raw.decode("utf-8-sig"))
            capacities, report = read_capacities(texts["cluster_gpu_number"])
            results[cluster] = {"configuration": report, **audit_jobs(texts["cluster_log"], capacities)}
    return {
        "schema_version": 1,
        "source_commit": COMMIT,
        "archive_sha256": digest(payload),
        "inputs": inputs,
        "clusters": results,
        "interpretation": "Full raw audit, not replay. Naive source clock; daily counts are diagnostic assumptions, "
        "not measured intraday hard quotas. Occupancy is requested GPU over valid closed terminal intervals.",
    }


def main():
    """Write deterministic aggregates and a provenance manifest, never raw rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/resource_observability"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    result = audit_archive(verified_archive(args.cache_dir / "helios-data.zip", args.download))
    output = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    code = Path(__file__)
    manifest = {
        "source_url": URL,
        "source_commit": COMMIT,
        "archive_sha256": SHA256,
        "attribution": "SenseTime / S-Lab-System-Group; Hu et al., SC 2021, doi:10.1145/3458817.3476223",
        "source_license": "CC-BY-4.0, not the project's MIT license",
        "implementation": "examples/resource_observability_audit.py",
        "implementation_sha256": digest(code.read_bytes()),
        "numpy": np.__version__,
        "output": "helios-audit.json",
        "output_sha256": digest(output),
        "scheduler_runs": 0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "helios-audit.json").write_bytes(output)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({c: r["counts"]["rows"] for c, r in result["clusters"].items()}))


if __name__ == "__main__":
    main()
