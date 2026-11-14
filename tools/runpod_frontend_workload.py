#!/usr/bin/env python3
"""Measure paired frontend tracing overhead against a fixed loopback vLLM server."""

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
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

try:
    from tools.runpod_calibration_workload import join_server_trace
    from tools.runpod_workload import ENDPOINT, atomic_json, parse_sse_event, prompt_ids, summarize_events
except ModuleNotFoundError:
    from runpod_calibration_workload import join_server_trace
    from runpod_workload import ENDPOINT, atomic_json, parse_sse_event, prompt_ids, summarize_events

SERIES_GRACE_SECONDS = 5.0


def validate_config(config, seed):
    """Reject malformed or unbounded profiles before generating or offering work."""

    def number(value, name, high, integer=False):
        valid_types = (int,) if integer else (int, float)
        if type(value) not in valid_types or not 0 < value <= high or not math.isfinite(value):
            raise ValueError(f"Invalid or excessive {name}")

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(config) is not dict or type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("Unsupported workload schema")
    for field, maximum in (("input_lengths", 32768), ("open_loop_rates", 100)):
        values = config.get(field)
        if type(values) is not list or not values or len(values) > 16:
            raise ValueError(f"Invalid {field}")
        for value in values:
            number(value, field, maximum, integer=field == "input_lengths")
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate {field}")
    for field, maximum in (
        ("probe_output_tokens", 4096),
        ("background_input_tokens", 32768),
        ("background_output_tokens", 4096),
        ("idle_pairs_per_block", 100),
        ("loaded_pairs_per_block", 100),
        ("max_in_flight", 64),
        ("max_requests", 1000),
    ):
        number(config.get(field), field, maximum, integer=True)
    for field, maximum in (
        ("open_loop_seconds", 1800),
        ("workload_timeout_seconds", 2400),
        ("request_timeout_seconds", 120),
    ):
        number(config.get(field), field, maximum)
    offsets = config.get("probe_offsets_s")
    if type(offsets) is not list or len(offsets) != 2:
        raise ValueError("probe_offsets_s must have two entries")
    for value in offsets:
        number(value, "probe_offsets_s", config["request_timeout_seconds"])
    if offsets[0] >= offsets[1]:
        raise ValueError("probe_offsets_s must strictly increase")
    cells = config.get("warmup_cells")
    if type(cells) is not list or not cells or len(cells) > 16:
        raise ValueError("Invalid warmup_cells")
    for cell in cells:
        if type(cell) is not list or len(cell) != 2:
            raise ValueError("Invalid warmup cell")
        number(cell[0], "warmup input", 32768, integer=True)
        number(cell[1], "warmup output", 4096, integer=True)
    if len({tuple(cell) for cell in cells}) != len(cells):
        raise ValueError("Duplicate warmup cells")
    blocks = config.get("blocks")
    if type(blocks) is not list or not blocks or len(blocks) > 16:
        raise ValueError("Invalid blocks")
    names, seeds = set(), set()
    for block in blocks:
        if type(block) is not dict or not isinstance(block.get("block_id"), str) or not block["block_id"]:
            raise ValueError("Invalid block_id")
        if type(block.get("arrival_seed")) is not int or block["arrival_seed"] < 0:
            raise ValueError("Invalid arrival_seed")
        if block["block_id"] in names or block["arrival_seed"] in seeds:
            raise ValueError("Duplicate block ID or arrival seed")
        names.add(block["block_id"])
        seeds.add(block["arrival_seed"])


def generate_plan(config, seed):
    """Return paired diagnostic series; generation never contacts a server."""
    validate_config(config, seed)
    series = []
    counter = 0

    def add(block, condition, mode, requests, pair_id=None, duration=0, rate=None, repetition=0):
        nonlocal counter
        series_id = f"s{len(series) + 1:04d}-{condition}"
        rows = []
        for n, m, offset, trace, role in requests:
            counter += 1
            if counter > config["max_requests"]:
                raise ValueError("Plan exceeds max_requests")
            prompt_seed = seed + n
            rows.append(
                {
                    "series_id": series_id,
                    "request_index": counter,
                    "request_id": f"mqfront-{counter}",
                    "seed": prompt_seed,
                    "prompt_seed": prompt_seed,
                    "phase": "warmup" if condition == "warmup" else "measured",
                    "split": "warmup" if condition == "warmup" else "diagnostic",
                    "mode": mode,
                    "concurrency": {"isolated": 1, "burst": 3, "open_loop": None}[mode],
                    "request_rate": rate,
                    "input_tokens": n,
                    "output_tokens": m,
                    "arrival_offset_s": offset,
                    "offer_status": "NOT_OFFERED",
                    "trace_enabled": trace,
                    "role": role,
                    "block_id": block,
                    "condition": condition,
                    "repetition": repetition,
                    **({"pair_id": pair_id} if pair_id else {}),
                }
            )
        item = {
            "series_id": series_id,
            "duration_s": duration,
            "block_id": block,
            "condition": condition,
            "requests": rows,
        }
        if pair_id:
            item["pair_id"] = pair_id
        series.append(item)

    for n, m in config["warmup_cells"]:
        add(None, "warmup", "isolated", [(n, m, 0.0, True, "warmup")])
    for block_index, block in enumerate(config["blocks"]):
        rng = random.Random(seed + block_index)
        block_id = block["block_id"]
        for condition, repetitions in (
            ("idle", config["idle_pairs_per_block"]),
            ("loaded", config["loaded_pairs_per_block"]),
        ):
            groups = []
            for n in config["input_lengths"]:
                for repetition in range(repetitions):
                    treatments = [True, False]
                    rng.shuffle(treatments)
                    groups.append((n, repetition, treatments))
            rng.shuffle(groups)
            for n, repetition, treatments in groups:
                pair_id = f"{block_id}-{condition}-{n}-{repetition}"
                if condition == "idle":
                    for trace in treatments:
                        add(
                            block_id,
                            condition,
                            "isolated",
                            [(n, config["probe_output_tokens"], 0.0, trace, "probe")],
                            pair_id,
                            repetition=repetition,
                        )
                else:
                    requests = [
                        (
                            config["background_input_tokens"],
                            config["background_output_tokens"],
                            0.0,
                            False,
                            "background",
                        )
                    ]
                    requests.extend(
                        (n, config["probe_output_tokens"], offset, trace, "probe")
                        for offset, trace in zip(config["probe_offsets_s"], treatments)
                    )
                    add(block_id, condition, "burst", requests, pair_id, repetition=repetition)
        rate_indices = list(enumerate(config["open_loop_rates"]))
        if block_index % 2:
            rate_indices.reverse()
        for rate_index, rate in rate_indices:
            rng = random.Random(seed + block["arrival_seed"] + rate_index * 1000)
            requests = []
            offset = rng.expovariate(rate)
            while offset < config["open_loop_seconds"]:
                requests.append((1024, 128, offset, True, "arrival"))
                if len(requests) + counter > config["max_requests"]:
                    raise ValueError("Plan exceeds max_requests")
                offset += rng.expovariate(rate)
            add(block_id, "open_loop", "open_loop", requests, duration=config["open_loop_seconds"], rate=rate)
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
        "frontend_trace": None,
        "status": "failed",
    }
    record.update(summarize_events([], error))
    return record


def attach_frontend_trace(record, path):
    """Validate one exact-ID frontend response without replacing engine evidence."""
    record["frontend_trace"] = None
    record["frontend_trace_join_pending"] = False
    if not record["trace_enabled"]:
        return record
    try:
        matches = []
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("Frontend trace row is not an object")
                if item.get("request_id") == record["request_id"]:
                    matches.append(item)
        if not matches:
            record["frontend_trace_join_pending"] = True
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one frontend record; found {len(matches)}")
        stamps = matches[0]
        record["frontend_trace"] = stamps
        if type(stamps.get("schema_version")) is not int or stamps["schema_version"] != 1:
            raise ValueError("Invalid frontend schema")
        if stamps.get("timestamp_clock") != "monotonic":
            raise ValueError("Invalid frontend clock")
        if (
            stamps.get("exception") is not None
            or "instrumentation_error" in stamps
            or stamps.get("render_exception")
            or stamps.get("response_status") != 200
            or stamps.get("completed_response_body") is not True
        ):
            raise ValueError("Frontend response failed")
        fields = (
            "http_enter_ts",
            "body_received_ts",
            "handler_enter_ts",
            "render_enter_ts",
            "render_exit_ts",
            "first_response_body_ts",
        )
        for name in fields:
            value = stamps.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid frontend mark: {name}")
        engine = record["server_timestamps"]
        ordered = [record["client_started_monotonic_s"]] + [stamps[name] for name in fields[:5]]
        ordered += [engine["queued_ts"], engine["scheduled_ts"], engine["first_token_ts"]]
        for value in ordered + [record["latency_to_done_s"]]:
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid client/engine clock mark")
        if any(left > right for left, right in zip(ordered, ordered[1:])):
            raise ValueError("Frontend/engine marks are out of order")
        done = record["client_started_monotonic_s"] + record["latency_to_done_s"]
        if not stamps["http_enter_ts"] <= stamps["first_response_body_ts"] <= done:
            raise ValueError("Frontend response marks are out of order")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, FileNotFoundError):
            record["frontend_trace_join_pending"] = True
        record["success"] = False
        record["error"] = record.get("error") or f"Frontend timestamp contract: {exc}"
    return record


def request_worker():
    """Read private metadata locally and emit only timing marks and legacy raw SSE."""
    payload = json.load(sys.stdin)
    started = time.monotonic()
    print(json.dumps({"worker_started_monotonic_s": started}), flush=True)
    trace = payload.pop("_mq_trace")
    request_id = payload.pop("_mq_request_id")
    headers = {"Content-Type": "application/json"}
    if trace:
        headers["X-MQ-Trace-ID"] = request_id
    body = json.dumps(payload).encode()
    print(json.dumps({"client_json_ready_ts": time.monotonic()}), flush=True)
    request = urllib.request.Request(ENDPOINT, body, headers, method="POST")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        """Keep all requests on the fixed loopback endpoint."""

        def redirect_request(self, req, fp, code, msg, headers, newurl):
            """Reject redirects."""
            return None

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=120) as response:
            print(json.dumps({"client_headers_received_ts": time.monotonic()}), flush=True)
            data_lines = []
            for line in response:
                line = line.decode("utf-8").rstrip("\r\n")
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip(" "))
                elif not line and data_lines:
                    event = parse_sse_event("\n".join(data_lines), time.monotonic() - started)
                    print(json.dumps({"event": event}), flush=True)
                    data_lines = []
                    if event.get("done"):
                        return
            if data_lines:
                print(
                    json.dumps({"event": parse_sse_event("\n".join(data_lines), time.monotonic() - started)}),
                    flush=True,
                )
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)


def execute_request(spec, model, deadline, timeout_s, trace_path, scheduled, dispatch):
    """Run one killable legacy SSE worker with this request's exact token lengths."""
    record = failed_record(spec, scheduled, "Request was not launched", dispatch)
    tokens, digest = prompt_ids(spec["seed"], spec["input_tokens"])
    record["prompt_sha256"] = digest
    payload = {
        "model": model,
        "request_id": spec["request_id"],
        "_mq_trace": spec["trace_enabled"],
        "_mq_request_id": spec["request_id"],
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
            [sys.executable, str(Path(__file__).resolve()), "--request-worker"],
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
        for mark in ("client_json_ready_ts", "client_headers_received_ts"):
            if mark in item:
                record[mark] = item[mark]
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
    if record["trace_enabled"] and record["done"]:
        frontend_path = Path(trace_path).with_name("frontend_timestamps.jsonl")
        join_deadline = min(deadline, dispatch + timeout_s, time.monotonic() + 2.0)
        while True:
            candidate = {**record, "error": None}
            attach_frontend_trace(candidate, frontend_path)
            if candidate.get("error") is None:
                record["frontend_trace"] = candidate["frontend_trace"]
                break
            if not candidate["frontend_trace_join_pending"] or time.monotonic() >= join_deadline:
                attach_frontend_trace(record, frontend_path)
                break
            time.sleep(min(0.02, max(0, join_deadline - time.monotonic())))
    record.pop("frontend_trace_join_pending", None)
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
                not_offered.append(
                    {
                        "request_index": spec["request_index"],
                        "request_id": spec["request_id"],
                        "offer_status": "NOT_OFFERED",
                        "reason": "Series deadline exhausted",
                    }
                )
            elif len(pending) >= config["max_in_flight"]:
                not_offered.append(
                    {
                        "request_index": spec["request_index"],
                        "request_id": spec["request_id"],
                        "offer_status": "NOT_OFFERED",
                        "reason": "max_in_flight reached",
                    }
                )
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
        "not_offered": not_offered,
        "not_offered_request_indices": [item["request_index"] for item in not_offered],
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
    profile = {
        "model": model,
        "seed": seed,
        "endpoint": ENDPOINT,
        "config": config,
        "python": sys.version,
        "frontend_trace_path": str(Path(trace_path).with_name("frontend_timestamps.jsonl")),
    }
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
            if (
                series["requests"]
                and series["requests"][0]["split"] == "warmup"
                and (series_results[-1]["failed"] or series_results[-1]["offered"] != series_results[-1]["planned"])
            ):
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
    """Generate a local plan or run the bounded frontend measurement."""
    if sys.argv[1:] == ["--request-worker"]:
        request_worker()
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, nargs="?")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--server-trace", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    plan = generate_plan(config, args.seed)
    if args.plan_only:
        value = {"seed": args.seed, "expected": sum(len(s["requests"]) for s in plan), "series": plan}
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(args.output_dir / "plan.json", value)
        else:
            print(json.dumps(value, indent=2))
        return 0
    if args.output_dir is None or not args.model or args.server_trace is None:
        parser.error("output_dir, --model and --server-trace are required for execution")
    return run_workload(args.output_dir, args.model, args.seed, config, args.server_trace)


if __name__ == "__main__":
    sys.exit(main())
