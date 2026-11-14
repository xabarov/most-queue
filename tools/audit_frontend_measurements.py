"""Audit the complete frontend experiment, retaining treatment/condition failures."""

# Exact int/float checks intentionally reject bool-valued schema and timestamps.
# pylint: disable=unidiomatic-typecheck

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.runpod_frontend_workload import generate_plan  # pylint: disable=wrong-import-position

ROOT = Path(__file__).resolve().parents[1]
MARKS = ("http_enter_ts", "body_received_ts", "handler_enter_ts", "render_enter_ts", "render_exit_ts")


def read_jsonl(path):
    """Read complete records; a torn line is a failed audit, not a dropped row."""
    return [json.loads(line) for line in path.read_text().splitlines()]


def finite(value):
    """Reject booleans, absent marks and nonfinite clocks."""
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def intervals(record):
    """Independently derive all frontend intervals for an instrumented success."""
    trace, engine = record["frontend_trace"], record["server_timestamps"]
    if engine.get("timestamp_clock") != "monotonic":
        raise ValueError("engine clock differs")
    if trace.get("instrumentation_error") is not None or trace.get("render_exception") is not None:
        raise ValueError("frontend instrumentation failed")
    if type(trace["schema_version"]) is not int or trace["schema_version"] != 1:
        raise ValueError("frontend schema differs")
    if (
        trace["request_id"] != record["request_id"]
        or trace["timestamp_clock"] != "monotonic"
        or trace["response_status"] != 200
        or trace["completed_response_body"] is not True
        or trace.get("exception")
    ):
        raise ValueError("frontend result contract failed")
    clock = [record["client_started_monotonic_s"], *(trace[key] for key in MARKS), engine["queued_ts"]]
    clock += [engine["scheduled_ts"], engine["first_token_ts"], engine["last_token_ts"]]
    if not all(finite(value) for value in clock) or clock != sorted(clock):
        raise ValueError("common-clock order failed")
    if not finite(record["client_ttft_s"]) or not finite(record["latency_to_done_s"]):
        raise ValueError("finite client durations required")
    first = record["client_started_monotonic_s"] + record["client_ttft_s"]
    done = record["client_started_monotonic_s"] + record["latency_to_done_s"]
    if not engine["first_token_ts"] <= first <= done or not engine["last_token_ts"] <= done:
        raise ValueError("client output clock order failed")
    body = trace["first_response_body_ts"]
    if not finite(body) or not trace["render_exit_ts"] <= body <= done:
        raise ValueError("first response-body transport mark failed")
    ready, headers = record["client_json_ready_ts"], record["client_headers_received_ts"]
    if not finite(ready) or not finite(headers):
        raise ValueError("finite client request/headers clocks required")
    if not clock[0] <= ready <= trace["http_enter_ts"] or not trace["http_enter_ts"] <= headers <= first:
        raise ValueError("client request/headers clock order failed")
    values = {
        "client_to_http": trace["http_enter_ts"] - clock[0],
        "receive_body": trace["body_received_ts"] - trace["http_enter_ts"],
        "parse_validate_route": trace["handler_enter_ts"] - trace["body_received_ts"],
        "handler_to_render": trace["render_enter_ts"] - trace["handler_enter_ts"],
        "render": trace["render_exit_ts"] - trace["render_enter_ts"],
        "render_to_engine_queue": engine["queued_ts"] - trace["render_exit_ts"],
        "queue": engine["scheduled_ts"] - engine["queued_ts"],
        "scheduled_to_first": engine["first_token_ts"] - engine["scheduled_ts"],
        "first_output_delivery": first - engine["first_token_ts"],
    }
    if not math.isclose(sum(values.values()), record["client_ttft_s"], rel_tol=0, abs_tol=1e-8):
        raise ValueError("frontend TTFT interval identity failed")
    return values


def clean_spec(item):
    """Only the offered-status field can change in the prespecified schedule."""
    return {key: value for key, value in item.items() if key != "offer_status"}


def validate_engine(row):
    """Verify token counts and all metric durations against the raw engine clocks."""
    engine = row["server_timestamps"]
    if engine["timestamp_clock"] != "monotonic" or engine["arrival_time_clock"] != "wall":
        raise ValueError("engine clock labels differ")
    if engine["num_generation_tokens"] != row["output_tokens"] or engine["num_preemptions"] != 0:
        raise ValueError("engine token/preemption contract differs")
    stamps = [engine[k] for k in ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts")]
    if not all(finite(v) for v in stamps) or stamps != sorted(stamps):
        raise ValueError("engine clock order failed")
    for metric, left, right in zip(
        ("queue_time_ms", "time_to_first_token_ms", "generation_time_ms"), stamps, stamps[1:]
    ):
        value = row["metrics"][metric]
        if (
            type(value) not in (float, int)
            or not math.isfinite(value)
            or not math.isclose(value, (right - left) * 1000, rel_tol=0, abs_tol=1e-5)
        ):
            raise ValueError("engine/SSE metric differs")


def audit(directory, config, seed):
    """Check identities, raw traces, temporal conditions and every offered result."""
    expected = generate_plan(config, seed)
    specs = {r["request_index"]: r for series in expected for r in series["requests"]}
    errors, derived, condition_failures = [], [], []
    files = {
        name: directory / name
        for name in ("plan.json", "requests.jsonl", "series.json", "summary.json", "progress.json", "manifest.json")
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        return {
            "status": "not_collected_or_incomplete",
            "planned": len(specs),
            "errors": ["missing:" + name for name in missing],
        }
    missing_traces = [
        name
        for name in ("server_timestamps.jsonl", "frontend_timestamps.jsonl")
        if not (directory.parent / name).is_file()
    ]
    if missing_traces:
        return {
            "status": "not_collected_or_incomplete",
            "planned": len(specs),
            "errors": ["missing:" + name for name in missing_traces],
        }
    records = read_jsonl(files["requests.jsonl"])
    saved_plan, summary, series_results, manifest = [
        json.loads(files[name].read_text()) for name in ("plan.json", "summary.json", "series.json", "manifest.json")
    ]
    actual_specs = [r for series in saved_plan["series"] for r in series["requests"]]
    if saved_plan["seed"] != seed or saved_plan["expected"] != len(specs):
        errors.append("plan seed/count differs")
    if [clean_spec(r) for r in actual_specs] != [clean_spec(r) for r in specs.values()]:
        errors.append("plan request specification/order differs")

    def series_spec(item):
        return {key: value for key, value in item.items() if key != "requests"}

    if [series_spec(s) for s in saved_plan["series"]] != [series_spec(s) for s in expected]:
        errors.append("plan series specification/order differs")
    artifact_names = {"plan.json", "requests.jsonl", "series.json", "progress.json", "summary.json"}
    if set(manifest["sha256"]) != artifact_names:
        errors.append("manifest artifact set differs")
    for name, sha in manifest["sha256"].items():
        if name not in {"plan.json", "requests.jsonl", "series.json", "progress.json", "summary.json"}:
            errors.append("unexpected manifest file")
        elif hashlib.sha256((directory / name).read_bytes()).hexdigest() != sha:
            errors.append("hash mismatch:" + name)
    by_id = {row["request_index"]: row for row in records}
    if len(by_id) != len(records) or set(by_id) != set(specs):
        errors.append("missing/duplicate/unexpected raw identities")
    engine_rows = read_jsonl(directory.parent / "server_timestamps.jsonl")
    frontend_rows = read_jsonl(directory.parent / "frontend_timestamps.jsonl")
    engine_counts = Counter(r.get("request_id") for r in engine_rows)
    frontend_counts = Counter(r.get("request_id") for r in frontend_rows)
    engine_by_id = {r.get("request_id"): r for r in engine_rows}
    frontend_by_id = {r.get("request_id"): r for r in frontend_rows}
    if engine_counts != Counter({"cmpl-" + r["request_id"] + "-0": 1 for r in specs.values()}):
        errors.append("engine identity/count mismatch")
    if frontend_counts != Counter({r["request_id"]: 1 for r in specs.values() if r["trace_enabled"]}):
        errors.append("frontend identity/count mismatch")
    for index, row in by_id.items():
        try:
            if any(row[key] != value for key, value in clean_spec(specs[index]).items()):
                raise ValueError("raw specification differs")
            if row["success"] is not True or row["offer_status"] != "OFFERED" or not row["done"] or row.get("error"):
                raise ValueError("request failed or was not offered")
            usage = row["usage"]
            if (usage["prompt_tokens"], usage["completion_tokens"]) != (row["input_tokens"], row["output_tokens"]):
                raise ValueError("actual lengths differ")
            engine = row["server_timestamps"]
            if engine_by_id.get("cmpl-" + row["request_id"] + "-0") != engine:
                raise ValueError("joined engine trace differs")
            validate_engine(row)
            if row["trace_enabled"]:
                if frontend_by_id.get(row["request_id"]) != row["frontend_trace"]:
                    raise ValueError("joined frontend trace differs")
                derived.append({"request_index": index, "intervals": intervals(row)})
            elif row.get("frontend_trace") is not None:
                raise ValueError("untraced control has frontend observations")
        except (KeyError, ValueError, TypeError) as exc:
            errors.append(f"request {index}: {exc}")
    results_by_id = {s["series_id"]: s for s in series_results}
    if len(results_by_id) != len(expected) or len(series_results) != len(expected):
        errors.append("series completeness mismatch")
    previous_end = None
    for series in expected:
        result = results_by_id.get(series["series_id"])
        if result is None:
            continue
        n = len(series["requests"])
        if (
            any(result.get(k) != v for k, v in {"planned": n, "offered": n, "successful": n, "failed": 0}.items())
            or result.get("not_offered_request_indices")
            or result.get("deadline_exhausted")
        ):
            errors.append("series failed:" + series["series_id"])
        if previous_end is not None and result["scheduled_start_monotonic_s"] < previous_end:
            errors.append("series overlap:" + series["series_id"])
        previous_end = result["completed_monotonic_s"]
        if (
            not finite(previous_end)
            or not finite(result["scheduled_start_monotonic_s"])
            or previous_end < result["scheduled_start_monotonic_s"] + series["duration_s"]
        ):
            errors.append("series window invalid:" + series["series_id"])
        for spec in series["requests"]:
            row = by_id.get(spec["request_index"])
            if row and row.get("success") is True:
                if row["client_started_monotonic_s"] + row["latency_to_done_s"] > previous_end:
                    errors.append("series not drained:" + series["series_id"])
                if (
                    row["scheduled_arrival_monotonic_s"]
                    != result["scheduled_start_monotonic_s"] + spec["arrival_offset_s"]
                ):
                    errors.append("offered schedule clock differs:" + series["series_id"])
        if series["condition"] == "loaded" and all(r["request_index"] in by_id for r in series["requests"]):
            selected = [by_id[r["request_index"]] for r in series["requests"]]
            try:
                background = next(r for r in selected if r["role"] == "background")["server_timestamps"]
                probes = sorted((r for r in selected if r["role"] == "probe"), key=lambda r: r["arrival_offset_s"])
                if not all(
                    background["first_token_ts"] <= r["client_started_monotonic_s"]
                    and r["server_timestamps"]["queued_ts"] < background["last_token_ts"]
                    for r in probes
                ):
                    condition_failures.append(series["series_id"] + ":background_not_decoding_at_probe")
                if probes[0]["server_timestamps"]["last_token_ts"] > probes[1]["client_started_monotonic_s"]:
                    condition_failures.append(series["series_id"] + ":probes_overlap")
            except (KeyError, TypeError):
                condition_failures.append(series["series_id"] + ":condition_unobservable")
    if (
        summary.get("expected") != len(specs)
        or summary.get("completed") != len(records)
        or summary.get("successful") != len(specs)
        or summary.get("experiment_valid") is not True
        or summary.get("not_offered") != 0
    ):
        errors.append("summary inconsistent/incomplete")
    if summary.get("config") != config or summary.get("seed") != seed:
        errors.append("summary config/seed differs")
    return {
        "status": (
            "complete_success" if not errors and not condition_failures else "incomplete_failed_or_condition_mismatch"
        ),
        "planned": len(specs),
        "completed": len(records),
        "successful": sum(row.get("success") is True for row in records),
        "instrumented_expected": sum(r["trace_enabled"] for r in specs.values()),
        "instrumented_validated": len(derived),
        "errors": errors,
        "condition_failures": condition_failures,
        "intervals": derived,
        "data_role": "diagnostic repeatability and instrumentation study; no model-v3 validation",
    }


def main():
    """Audit exported measurements locally without allocating resources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/runpod/frontend-v1-workload.json")
    parser.add_argument("--seed", type=int, default=2026111404)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.directory, json.loads(args.config.read_text()), args.seed)
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(json.dumps({key: value for key, value in result.items() if key != "intervals"}, indent=2))
    return 0 if result["status"] == "complete_success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
