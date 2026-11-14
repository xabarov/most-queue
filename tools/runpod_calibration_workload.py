#!/usr/bin/env python3
"""Execute a deterministic bounded calibration plan against loopback vLLM."""

# Explicit token-count type checks reject bool, which isinstance accepts as int.
# Request execution carries independent clocks and deadline settings explicitly.
# pylint: disable=unidiomatic-typecheck,too-many-arguments,too-many-positional-arguments

# The workload intentionally retains independent scheduling, worker and engine
# clocks; engine arrival is never inferred from client dispatch.
import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

try:
    from tools.runpod_vllm_server import validate_timestamps
    from tools.runpod_workload import ENDPOINT, atomic_json, prompt_ids, summarize_events
except ModuleNotFoundError:
    from runpod_vllm_server import validate_timestamps
    from runpod_workload import ENDPOINT, atomic_json, prompt_ids, summarize_events

SERIES_GRACE_SECONDS = 5.0


def generate_plan(config, seed):
    """Generate every offered request before starting the first measurement."""
    if config["schema_version"] != 1:
        raise ValueError("Unsupported workload schema")
    series = []
    counter = 0
    cells = [(n, m) for n in config["input_lengths"] for m in config["output_lengths"]]

    def add(split, mode, repetition, group, concurrency=1, rate=None, offsets=None, duration=0, arrival_seed=None):
        nonlocal counter
        series_id = f"s{len(series) + 1:04d}-{split}-{mode}"
        requests = []
        for (n, m), offset in zip(group, offsets if offsets is not None else [0.0] * len(group)):
            counter += 1
            requests.append(
                {
                    "series_id": series_id,
                    "request_index": counter,
                    "request_id": f"mqcal-{counter}",
                    "seed": seed + counter,
                    "phase": "warmup" if split == "warmup" else "measured",
                    "split": split,
                    "mode": mode,
                    "repetition": repetition,
                    "concurrency": concurrency,
                    "request_rate": rate,
                    "input_tokens": n,
                    "output_tokens": m,
                    "arrival_offset_s": offset,
                    "offer_status": "NOT_OFFERED",
                }
            )
        series.append(
            {
                "series_id": series_id,
                "duration_s": duration,
                "arrival_seed": arrival_seed,
                "requests": requests,
            }
        )

    for cell in cells:
        add("warmup", "isolated", 0, [cell])
    for n in config["input_lengths"]:
        add("warmup", "burst", 0, [(n, 32)] * 4, concurrency=4)
    for split, repetitions in (
        ("train", config["train_repetitions"]),
        ("validation", config["isolated_validation_repetitions"]),
    ):
        for repetition in repetitions:
            ordered = cells.copy()
            random.Random(seed + repetition).shuffle(ordered)
            for cell in ordered:
                add(split, "isolated", repetition, [cell])
    for repetition in config["burst_repetitions"]:
        ordered = cells.copy()
        random.Random(seed + repetition).shuffle(ordered)
        for cell in ordered:
            for concurrency in config["burst_concurrencies"]:
                add("validation", "burst", repetition, [cell] * concurrency, concurrency=concurrency)
    for rate_index, rate in enumerate(config["open_loop_rates"]):
        for arrival_seed in config["open_loop_seeds"]:
            generator = random.Random(seed + rate_index * 1000 + arrival_seed)
            offsets = []
            offset = generator.expovariate(rate)
            while offset < config["open_loop_seconds"]:
                offsets.append(offset)
                offset += generator.expovariate(rate)
                if len(offsets) > config["max_requests"]:
                    raise ValueError("Plan exceeds max_requests")
            add(
                "validation",
                "open_loop",
                arrival_seed,
                [(1024, 128)] * len(offsets),
                concurrency=None,
                rate=float(rate),
                offsets=offsets,
                duration=config["open_loop_seconds"],
                arrival_seed=arrival_seed,
            )
    if counter > config["max_requests"]:
        raise ValueError(f"Plan has {counter} requests, exceeding max_requests")
    return series


def failed_record(spec, scheduled, error, dispatch=None):
    """Retain an offered request even when no child can be launched."""
    record = {
        **spec,
        "scheduled_arrival_monotonic_s": scheduled,
        "dispatch_started_monotonic_s": time.monotonic() if dispatch is None else dispatch,
        "client_started_monotonic_s": None,
        "events": [],
        "server_timestamps": None,
        "status": "failed",
    }
    record.update(summarize_events([], error))
    return record


def join_server_trace(record, trace_path):
    """Join the pinned single-prompt completion ID and verify the timing contract."""
    try:
        server_id = record.get("server_request_id")
        if not isinstance(server_id, str) or server_id != f"cmpl-{record['request_id']}":
            raise ValueError("SSE request ID does not match the explicitly supplied request ID")
        expected = server_id + "-0"
        matches = []
        with Path(trace_path).open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                if item.get("request_id") == expected:
                    matches.append(item)
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one engine timestamp record for {expected}; found {len(matches)}")
        stamps = matches[0]
        record["server_timestamps"] = stamps
        validate_timestamps(stamps)
        if stamps.get("timestamp_clock") != "monotonic" or stamps.get("arrival_time_clock") != "wall":
            raise ValueError("Engine timestamp clock labels differ from the pinned contract")
        arrival = stamps.get("arrival_time")
        if type(arrival) not in (int, float) or not math.isfinite(arrival) or arrival <= 0:
            raise ValueError("Engine wall-clock arrival time is missing or invalid")
        if (
            type(stamps.get("num_generation_tokens")) is not int
            or stamps["num_generation_tokens"] != record["completion_tokens"]
        ):
            raise ValueError("Engine generated-token count differs from actual usage")
        if type(stamps.get("num_preemptions")) is not int or stamps["num_preemptions"] != 0:
            raise ValueError("Engine request was preempted or preemption count is invalid")
        for field, left, right in (
            ("queue_time_ms", "queued_ts", "scheduled_ts"),
            ("time_to_first_token_ms", "scheduled_ts", "first_token_ts"),
            ("generation_time_ms", "first_token_ts", "last_token_ts"),
        ):
            reported = (record.get("metrics") or {}).get(field)
            if type(reported) not in (int, float) or not math.isfinite(reported):
                raise ValueError(f"Missing finite server metric: {field}")
            if abs((stamps[right] - stamps[left]) * 1000 - reported) > 1e-5:
                raise ValueError(f"Engine timestamp duration differs from SSE metric: {field}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        record["success"] = False
        record["error"] = record.get("error") or f"Engine timestamp contract: {exc}"
    return record


def execute_request(spec, model, deadline, timeout_s, trace_path, scheduled, dispatch):
    """Run one killable legacy SSE worker with this request's exact token lengths."""
    record = failed_record(spec, scheduled, "Request was not launched", dispatch)
    tokens, digest = prompt_ids(spec["seed"], spec["input_tokens"])
    record["prompt_sha256"] = digest
    payload = {
        "model": model,
        "request_id": spec["request_id"],
        "prompt": tokens,
        "max_tokens": spec["output_tokens"],
        "ignore_eos": True,
        "temperature": 0,
        "n": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    output = ""
    error = None
    process = None
    try:
        remaining = min(dispatch + timeout_s, deadline) - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Request or series deadline exhausted before dispatch")
        process = subprocess.Popen(  # pylint: disable=consider-using-with
            [sys.executable, str(Path(__file__).with_name("runpod_workload.py")), "--request-worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            output, stderr = process.communicate(
                json.dumps(payload), timeout=max(0.001, min(dispatch + timeout_s, deadline) - time.monotonic())
            )
            if process.returncode:
                error = f"Worker exit {process.returncode}: {stderr[:2000]}"
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
            error = "Overall request deadline exceeded"
    except (OSError, TimeoutError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()
    for line in output.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            error = error or "Incomplete worker output"
            continue
        if "event" in item:
            record["events"].append(item["event"])
        if "worker_started_monotonic_s" in item:
            record["client_started_monotonic_s"] = item["worker_started_monotonic_s"]
        error = error or item.get("error")
    record.update(summarize_events(record["events"], error))
    usage = record["usage"] or {}
    if any(
        type(usage.get(field)) is not int or usage[field] != spec[length]
        for field, length in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens"))
    ):
        record["error"] = record["error"] or "Reported token lengths differ from requested lengths"
        record["success"] = False
    if record["success"]:
        worker_started = record["client_started_monotonic_s"]
        if type(worker_started) not in (int, float) or not math.isfinite(worker_started) or worker_started <= 0:
            record["error"] = "Missing finite positive child start timestamp"
            record["success"] = False
    if record["success"]:
        join_server_trace(record, trace_path)
    record["status"] = "succeeded" if record["success"] else "failed"
    record["client_elapsed_s"] = time.monotonic() - dispatch
    return record


def run_series(series, model, config, trace_path, global_deadline, emit):
    """Offer absolute-clock arrivals without response-dependent queueing or dropping."""
    started = time.monotonic()
    deadline = min(
        global_deadline, started + series["duration_s"] + config["request_timeout_seconds"] + SERIES_GRACE_SECONDS
    )
    pending = {}
    completed = []
    not_offered = []

    def save(record):
        completed.append(record)
        emit(record)

    def reap():
        for future in list(pending):
            if future.done():
                spec, scheduled, dispatch = pending.pop(future)
                try:
                    save(future.result())
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    save(failed_record(spec, scheduled, f"Worker exception: {type(exc).__name__}: {exc}", dispatch))

    with ThreadPoolExecutor(max_workers=config["max_in_flight"]) as executor:
        for spec in series["requests"]:
            scheduled = started + spec["arrival_offset_s"]
            while time.monotonic() < min(scheduled, deadline):
                reap()
                time.sleep(max(0.0, min(0.02, min(scheduled, deadline) - time.monotonic())))
            reap()
            dispatch = time.monotonic()
            if dispatch >= deadline:
                not_offered.append(spec["request_index"])
            elif len(pending) >= config["max_in_flight"]:
                save(failed_record(spec, scheduled, "Client overload: max_in_flight reached", dispatch))
            else:
                future = executor.submit(
                    execute_request,
                    spec,
                    model,
                    deadline,
                    config["request_timeout_seconds"],
                    trace_path,
                    scheduled,
                    dispatch,
                )
                pending[future] = (spec, scheduled, dispatch)
        while pending:
            wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
            reap()
        # The offered-arrival observation window remains the full configured duration.
        while time.monotonic() < min(started + series["duration_s"], global_deadline):
            time.sleep(max(0.0, min(0.05, min(started + series["duration_s"], global_deadline) - time.monotonic())))
    ended = time.monotonic()
    return {
        "series_id": series["series_id"],
        "scheduled_start_monotonic_s": started,
        "scheduled_end_monotonic_s": started + series["duration_s"],
        "completed_monotonic_s": ended,
        "planned": len(series["requests"]),
        "offered": len(completed),
        "not_offered_request_indices": not_offered,
        "successful": sum(row["success"] for row in completed),
        "failed": sum(not row["success"] for row in completed),
        "deadline_exhausted": ended >= deadline,
    }


def run_workload(output_dir, model, seed, config, trace_path):
    """Persist the complete plan, incremental evidence, summaries and checksums."""
    plan = generate_plan(config, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = sum(len(series["requests"]) for series in plan)
    records = []
    series_results = []
    started = time.monotonic()
    deadline = started + config["workload_timeout_seconds"]
    profile = {"model": model, "seed": seed, "endpoint": ENDPOINT, "config": config, "python": sys.version}
    by_index = {spec["request_index"]: spec for series in plan for spec in series["requests"]}
    abort_reason = None
    with (output_dir / "requests.jsonl").open("x", encoding="utf-8") as stream:
        atomic_json(output_dir / "plan.json", {"seed": seed, "expected": expected, "series": plan})

        def emit(record):
            record["offer_status"] = "OFFERED"
            by_index[record["request_index"]]["offer_status"] = "OFFERED"
            records.append(record)
            # Match the legacy evidence writer: even malformed/nonfinite parsed
            # server data must survive as a failed record with its raw SSE text.
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            measured = [row for row in records if row["phase"] == "measured"]
            progress = {
                "completed": len(records),
                "expected": expected,
                "measured_completed": len(measured),
                "measured_successful": sum(row["success"] for row in measured),
                "successful": sum(row["success"] for row in records),
                "elapsed_s": time.monotonic() - started,
                "deadline_exhausted": time.monotonic() >= deadline,
                "series": series_results,
            }
            atomic_json(output_dir / "progress.json", progress)
            atomic_json(output_dir / "summary.json", {**profile, **progress})

        for series in plan:
            if time.monotonic() >= deadline:
                abort_reason = "Global workload deadline exhausted"
                break
            series_results.append(run_series(series, model, config, trace_path, deadline, emit))
            atomic_json(output_dir / "series.json", series_results)
            if series["requests"] and series["requests"][0]["split"] == "warmup" and series_results[-1]["failed"]:
                abort_reason = f"Warmup contract failed in {series['series_id']}"
                break
        # Include the final series' end/success metadata in final summaries.
        progress_path = output_dir / "progress.json"
        progress = (
            json.loads(progress_path.read_text(encoding="utf-8"))
            if progress_path.exists()
            else {
                "completed": 0,
                "expected": expected,
                "successful": 0,
                "measured_completed": 0,
                "measured_successful": 0,
            }
        )
        valid = len(records) == expected and all(row["success"] for row in records) and not abort_reason
        progress.update(
            series=series_results,
            offered=len(records),
            not_offered=expected - len(records),
            experiment_valid=bool(valid),
            abort_reason=abort_reason,
            elapsed_s=time.monotonic() - started,
            deadline_exhausted=time.monotonic() >= deadline,
        )
        atomic_json(output_dir / "plan.json", {"seed": seed, "expected": expected, "series": plan})
        atomic_json(output_dir / "series.json", series_results)
        atomic_json(output_dir / "progress.json", progress)
        atomic_json(output_dir / "summary.json", {**profile, **progress})
    artifacts = ("plan.json", "requests.jsonl", "series.json", "progress.json", "summary.json")
    atomic_json(
        output_dir / "manifest.json",
        {"sha256": {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest() for name in artifacts}},
    )
    return 0 if valid else 1


def main():
    """Run the preconfigured bounded calibration experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--server-trace", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    return run_workload(args.output_dir, args.model, args.seed, config, args.server_trace)


if __name__ == "__main__":
    sys.exit(main())
