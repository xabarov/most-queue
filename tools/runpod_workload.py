#!/usr/bin/env python3
"""Bounded synthetic vLLM pilot; run inside the GPU container without provider credentials.

Client TTFT measures arrival to first nonempty content event. Server
metrics.time_to_first_token_ms measures scheduling to first generated token.
They are deliberately retained as separate quantities. No text-to-token estimates
are made. Child processes make deadlines enforceable even for trickling HTTP.
"""

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ENDPOINT = "http://127.0.0.1:8000/v1/completions"
INPUT_LENGTHS = (128, 1024, 4096)
CONCURRENCIES = (1, 4, 8)
REQUEST_TIMEOUT = 120.0
BATCH_TIMEOUT = 150.0
WORKLOAD_TIMEOUT = 900.0


def prompt_ids(seed, length):
    """Return exact-length synthetic token IDs and their canonical JSON hash."""
    generator = random.Random(seed)
    tokens = [generator.randint(1000, 20000) for _ in range(length)]
    digest = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
    return tokens, digest


def parse_sse_event(data, elapsed_s):
    """Preserve one SSE data payload, including malformed events and DONE."""
    event = {"elapsed_s": elapsed_s, "raw_data": data}
    if data == "[DONE]":
        event["done"] = True
    else:
        try:
            event["chunk"] = json.loads(data)
        except json.JSONDecodeError as exc:
            event["parse_error"] = str(exc)
    return event


def summarize_events(events, error=None):
    """Extract reported counts and timing without counting text chunks as tokens."""
    result = {
        "usage": None,
        "metrics": None,
        "server_request_id": None,
        "client_ttft_s": None,
        "latency_to_done_s": None,
        "completion_tokens": None,
        "done": False,
        "error": error,
    }
    for event in events:
        if event.get("parse_error"):
            result["error"] = result["error"] or event["parse_error"]
        if event.get("done"):
            result["done"] = True
            result["latency_to_done_s"] = event["elapsed_s"]
        chunk = event.get("chunk")
        if not isinstance(chunk, dict):
            continue
        if chunk.get("error"):
            result["error"] = result["error"] or chunk["error"]
        if chunk.get("id"):
            result["server_request_id"] = chunk["id"]
        if isinstance(chunk.get("usage"), dict):
            result["usage"] = chunk["usage"]
            result["completion_tokens"] = chunk["usage"].get("completion_tokens")
        if isinstance(chunk.get("metrics"), dict):
            result["metrics"] = chunk["metrics"]
        for choice in chunk.get("choices") or []:
            if isinstance(choice, dict) and choice.get("text") and result["client_ttft_s"] is None:
                result["client_ttft_s"] = event["elapsed_s"]
    required_timings = ("queue_time_ms", "time_to_first_token_ms", "generation_time_ms")
    metrics: dict = result["metrics"] or {}
    valid_timings = all(
        type(metrics.get(field)) in (int, float) and math.isfinite(metrics[field]) and metrics[field] >= 0
        for field in required_timings
    )
    result["success"] = bool(
        not result["error"]
        and result["done"]
        and result["usage"]
        and valid_timings
        and type(result["completion_tokens"]) is int  # pylint: disable=unidiomatic-typecheck
        and result["completion_tokens"] > 0
        and result["client_ttft_s"] is not None
    )
    return result


def _emit(value):
    print(json.dumps(value), flush=True)


def request_worker():
    """Read one request on stdin and flush raw events for timeout-safe recovery."""
    payload = json.load(sys.stdin)
    started = time.monotonic()
    _emit({"worker_started_monotonic_s": started})
    request = urllib.request.Request(
        ENDPOINT, json.dumps(payload).encode(), {"Content-Type": "application/json"}, method="POST"
    )

    # Explicitly bypass proxy environment variables, and never follow redirects.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        """Keep every HTTP request on the fixed loopback endpoint."""

        # urllib requires this exact override signature.
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            """Reject redirects instead of contacting another endpoint."""
            return None

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=REQUEST_TIMEOUT) as response:
            data_lines = []
            for line in response:
                line = line.decode("utf-8").rstrip("\r\n")
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip(" "))
                elif not line and data_lines:
                    event = parse_sse_event("\n".join(data_lines), time.monotonic() - started)
                    _emit({"event": event})
                    data_lines = []
                    if event.get("done"):
                        return
            if data_lines:
                _emit({"event": parse_sse_event("\n".join(data_lines), time.monotonic() - started)})
    except (OSError, ValueError) as exc:
        _emit({"error": f"{type(exc).__name__}: {exc}"})


def execute_request(spec, model, barrier, deadline):
    """Run a killable HTTP reader; retain emitted events even when its deadline expires."""
    tokens, digest = prompt_ids(spec["seed"], spec["input_tokens"])
    record = {**spec, "prompt_sha256": digest, "events": []}
    payload = {
        "model": model,
        "prompt": tokens,
        "max_tokens": 128,
        "ignore_eos": True,
        "temperature": 0,
        "n": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    error = None
    process = None
    output = ""
    started = time.monotonic()
    try:
        barrier.wait(timeout=max(0.001, deadline - time.monotonic()))
        started = time.monotonic()
        timeout = min(REQUEST_TIMEOUT, deadline - started)
        if timeout <= 0:
            raise TimeoutError("Workload or batch deadline exhausted")
        process = subprocess.Popen(  # pylint: disable=consider-using-with
            [sys.executable, str(Path(__file__).resolve()), "--request-worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            output, stderr = process.communicate(
                json.dumps(payload), timeout=max(0.001, timeout - (time.monotonic() - started))
            )
            if process.returncode:
                error = f"Worker exit {process.returncode}: {stderr[:2000]}"
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
            error = "Overall request deadline exceeded"
    except (OSError, TimeoutError, threading.BrokenBarrierError) as exc:
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
    if (
        type(usage.get("prompt_tokens")) is not int  # pylint: disable=unidiomatic-typecheck
        or usage.get("prompt_tokens") != spec["input_tokens"]
        or type(usage.get("completion_tokens")) is not int  # pylint: disable=unidiomatic-typecheck
        or usage.get("completion_tokens") != 128
    ):
        record["error"] = record["error"] or "Reported token lengths differ from the fixed request profile"
        record["success"] = False
    record["client_elapsed_s"] = time.monotonic() - started
    return record


def atomic_json(path, value):
    """Replace a JSON artifact atomically on the local filesystem."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def pilot_batches(seed):
    """Yield three sequential warmups and eighteen measured batches (78 requests)."""
    counter = 0
    for length in INPUT_LENGTHS:
        counter += 1
        yield [
            {
                "request_index": counter,
                "phase": "warmup",
                "input_tokens": length,
                "concurrency": 1,
                "repetition": 0,
                "seed": seed + counter,
            }
        ]
    for repetition in (1, 2):
        for length in INPUT_LENGTHS:
            for concurrency in CONCURRENCIES:
                batch = []
                for _ in range(concurrency):
                    counter += 1
                    batch.append(
                        {
                            "request_index": counter,
                            "phase": "measured",
                            "input_tokens": length,
                            "concurrency": concurrency,
                            "repetition": repetition,
                            "seed": seed + counter,
                        }
                    )
                yield batch


def run_pilot(output_dir, model, seed):
    """Write every completed or skipped request and final artifact checksums."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite evidence from an earlier run.
    records_path = output_dir / "requests.jsonl"
    records = []
    started = time.monotonic()
    deadline = started + WORKLOAD_TIMEOUT
    profile = {
        "model": model,
        "seed": seed,
        "endpoint": ENDPOINT,
        "input_lengths": INPUT_LENGTHS,
        "concurrencies": CONCURRENCIES,
        "max_tokens": 128,
        "request_timeout_s": REQUEST_TIMEOUT,
        "batch_timeout_s": BATCH_TIMEOUT,
        "workload_timeout_s": WORKLOAD_TIMEOUT,
        "python": sys.version,
        "platform": sys.platform,
        "ttft_definition": "Client arrival to first nonempty SSE text; server TTFT is scheduling to first token",
    }
    with records_path.open("x", encoding="utf-8") as stream:
        for batch in pilot_batches(seed):
            batch_deadline = min(deadline, time.monotonic() + BATCH_TIMEOUT)
            barrier = threading.Barrier(len(batch))
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = [executor.submit(execute_request, spec, model, barrier, batch_deadline) for spec in batch]
                for future in as_completed(futures):
                    record = future.result()
                    records.append(record)
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    measured = [row for row in records if row["phase"] == "measured"]
                    progress = {
                        "completed": len(records),
                        "expected": 81,
                        "measured_completed": len(measured),
                        "measured_successful": sum(row["success"] for row in measured),
                        "elapsed_s": time.monotonic() - started,
                        "deadline_exhausted": time.monotonic() >= deadline,
                    }
                    atomic_json(output_dir / "progress.json", progress)
                    atomic_json(output_dir / "summary.json", {**profile, **progress})
    artifacts = ("requests.jsonl", "progress.json", "summary.json")
    atomic_json(
        output_dir / "manifest.json",
        {"sha256": {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest() for name in artifacts}},
    )
    measured = [row for row in records if row["phase"] == "measured"]
    return 0 if len(measured) == 78 and all(row["success"] for row in measured) else 1


def main():
    """Run the fixed pilot or its internal single-request child."""
    if sys.argv[1:] == ["--request-worker"]:
        request_worker()
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", nargs="?", type=Path)
    parser.add_argument("--output-dir", dest="output_dir_option", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    destination = args.output_dir_option or args.output_dir
    if destination is None or (args.output_dir_option and args.output_dir):
        parser.error("Provide exactly one output directory, positional or --output-dir")
    return run_pilot(destination, args.model, args.seed)


if __name__ == "__main__":
    sys.exit(main())
