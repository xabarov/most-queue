"""Read audited request JSONL traces from the Runpod serving experiments.

Client TTFT ends at the first nonempty SSE text; final latency ends at DONE.
Server queue, scheduled-to-first-token and generation spans remain distinct.
This adapter uses recorded usage, never the number of streamed events.
"""

import gzip
import json
import math
from pathlib import Path

from most_queue.io.serving import ServingBenchmark, ServingRequest


def _number(value, name, *, optional=False):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _records(source):
    if isinstance(source, (str, Path)):
        path = Path(source)
        opener = gzip.open if path.suffix == ".gz" else open
        try:
            with opener(path, "rt", encoding="utf-8") as stream:
                for line_index, line in enumerate(stream, 1):
                    try:
                        yield json.loads(line)
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f"invalid JSON record at line {line_index}") from exc
        except (OSError, EOFError, UnicodeError) as exc:
            raise ValueError(f"cannot read request trace: {path}") from exc
    else:
        try:
            yield from source
        except TypeError as exc:
            raise ValueError("source must be a JSONL path or iterable of records") from exc


def _usage(record, requested_input, requested_output):
    usage = record.get("usage")
    if record["success"]:
        if not isinstance(usage, dict):
            raise ValueError("successful requests require usage")
        inputs = _integer(usage.get("prompt_tokens"), "usage.prompt_tokens", minimum=1)
        outputs = _integer(usage.get("completion_tokens"), "usage.completion_tokens", minimum=1)
        if (inputs, outputs) != (requested_input, requested_output):
            raise ValueError("successful usage disagrees with requested lengths")
        if "total_tokens" in usage and _integer(usage["total_tokens"], "usage.total_tokens") != inputs + outputs:
            raise ValueError("usage.total_tokens disagrees with token counts")
        if "completion_tokens" in record:
            if _integer(record["completion_tokens"], "completion_tokens") != outputs:
                raise ValueError("completion_tokens disagrees with usage")
        return inputs, outputs
    # Requested lengths describe the offered workload, not unknown actual usage.
    usage = usage if isinstance(usage, dict) else {}
    counts = []
    for key in ("prompt_tokens", "completion_tokens"):
        value = usage.get(key)
        counts.append(value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0)
    return tuple(counts)


def _server_arrival(record, outputs, timings):
    if "server_timestamps" not in record:
        return None
    stamps = record["server_timestamps"]
    request_id = record.get("server_request_id")
    expected_id = request_id + "-0" if isinstance(request_id, str) and request_id else None
    if not record["success"]:
        # Incomplete/malformed engine traces must not remove failed offers.
        if not isinstance(stamps, dict) or expected_id is None or stamps.get("request_id") != expected_id:
            return None
        queued = stamps.get("queued_ts")
        if isinstance(queued, bool) or not isinstance(queued, (int, float)) or not math.isfinite(queued) or queued <= 0:
            return None
        return float(queued)
    if not isinstance(stamps, dict) or expected_id is None or stamps.get("request_id") != expected_id:
        raise ValueError("server_timestamps request_id must match the exact engine request")
    values = [_number(stamps.get(key), key) for key in ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts")]
    if any(value <= 0 for value in values) or any(right < left for left, right in zip(values, values[1:])):
        raise ValueError("server_timestamps must be positive and chronologically ordered")
    if _integer(stamps.get("num_generation_tokens"), "num_generation_tokens", minimum=1) != outputs:
        raise ValueError("server_timestamps generation count disagrees with actual usage")
    if _integer(stamps.get("num_preemptions"), "num_preemptions") != 0:
        raise ValueError("completed server_timestamps require zero preemptions")
    for left, right, metric in zip(values, values[1:], timings):
        if not math.isclose((right - left) * 1000, metric, rel_tol=0, abs_tol=1e-5):
            raise ValueError("server_timestamps duration disagrees with server metrics")
    return values[0]


def _parse(record):
    if not isinstance(record, dict):
        raise ValueError("each trace record must be an object")
    index = _integer(record.get("request_index"), "request_index")
    requested_input = _integer(record.get("input_tokens"), "input_tokens", minimum=1)
    requested_output = _integer(record.get("output_tokens", 128), "output_tokens", minimum=1)
    concurrency = record.get("concurrency")
    if record.get("mode") != "open_loop" or concurrency is not None:
        concurrency = _integer(concurrency, "concurrency", minimum=1)
    repetition = _integer(record.get("repetition"), "repetition")
    phase = record.get("phase")
    if not isinstance(phase, str) or not phase.strip():
        raise ValueError("phase must be a nonempty string")
    series = record.get("series_id")
    if "series_id" in record and (not isinstance(series, str) or not series.strip()):
        raise ValueError("series_id must be a nonempty string")
    split = record.get("split", "warmup" if phase == "warmup" else None)
    mode = record.get("mode", "isolated" if concurrency == 1 else "burst")
    if mode not in {"isolated", "burst", "open_loop"}:
        raise ValueError("unknown serving mode")
    if split not in {None, "train", "validation", "warmup"}:
        raise ValueError("unknown serving split")
    if series is not None and ("split" not in record or "mode" not in record or "request_rate" not in record):
        raise ValueError("explicit series require split, mode and request_rate")
    if series is not None and "output_tokens" not in record:
        raise ValueError("explicit series require requested output_tokens")
    if series is not None and split is None:
        raise ValueError("explicit series require a named split")
    rate = _number(record.get("request_rate"), "request_rate", optional=mode != "open_loop")
    if rate == 0:
        raise ValueError("request_rate must be positive when provided")
    if not isinstance(record.get("success"), bool):
        raise ValueError("success must be a boolean")
    success = record["success"]
    if "done" in record and not isinstance(record["done"], bool):
        raise ValueError("done must be a boolean")
    if success and (record.get("done") is not True or record.get("error") not in (None, "")):
        raise ValueError("successful requests require DONE and no error")
    start = _number(record.get("client_started_monotonic_s"), "client_started_monotonic_s", optional=True)
    dispatch = _number(record.get("dispatch_started_monotonic_s"), "dispatch_started_monotonic_s", optional=True)
    scheduled = _number(record.get("scheduled_arrival_monotonic_s"), "scheduled_arrival_monotonic_s", optional=True)
    if start is None:
        if success or dispatch is None:
            raise ValueError("request has no actual start timestamp for the measurement window")
        start = dispatch
    ttft = _number(record.get("client_ttft_s"), "client_ttft_s", optional=not success)
    latency = _number(record.get("latency_to_done_s"), "latency_to_done_s", optional=not success)
    elapsed = _number(record.get("client_elapsed_s"), "client_elapsed_s", optional=success)
    if ttft is not None and latency is not None and latency < ttft:
        raise ValueError("DONE cannot precede the first token")
    if ttft is not None and elapsed is not None and elapsed < ttft:
        raise ValueError("client elapsed time cannot precede the first token")
    metrics = record.get("metrics")
    if metrics is None and not success:
        metrics = {}
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be an object")
    timings = [
        _number(metrics.get(key), key, optional=not success)
        for key in ("queue_time_ms", "time_to_first_token_ms", "generation_time_ms")
    ]
    if "mean_itl_ms" in metrics:
        _number(metrics["mean_itl_ms"], "mean_itl_ms", optional=True)
    queue, scheduled_ttft, generation = [None if value is None else value / 1000 for value in timings]
    inputs, outputs = _usage(record, requested_input, requested_output)
    server_arrival = _server_arrival(record, outputs, timings)
    request = ServingRequest(
        input_tokens=inputs,
        output_tokens=outputs,
        status="completed" if success else "failed",
        ttft=ttft,
        latency=latency if success else None,
        server_queue_time=queue,
        server_ttft=queue + scheduled_ttft if queue is not None and scheduled_ttft is not None else None,
        server_scheduled_to_first_token=scheduled_ttft,
        server_generation_time=generation,
    )
    attributes = {
        "phase": phase,
        "repetition": repetition,
        "split": split,
        "mode": mode,
        "series_id": series,
        "request_rate": rate,
    }
    attributes["max_concurrency"] = concurrency if mode != "open_loop" else None
    group = ("series", series) if series is not None else ("legacy", phase, repetition, requested_input, concurrency)
    return {
        "group": group,
        "attributes": attributes,
        "consistency": (phase, split, mode, repetition, rate, concurrency),
        "request": request,
        "index": index,
        "start": start,
        "end": start + latency if success else (dispatch if dispatch is not None else start) + elapsed,
        "requested_input": requested_input,
        "requested_output": requested_output,
        "request_id": record.get("server_request_id"),
        "error": record.get("error"),
        "dispatch": dispatch,
        "scheduled": scheduled,
        "server_arrival": server_arrival,
    }


def read_runpod_trace(source, *, measurement_profile, include_warmup=True) -> list[ServingBenchmark]:
    """Read JSONL/gzip or record objects into complete finite offered cohorts.

    Explicit series IDs define cohorts, including mixed request lengths. Legacy
    pilot cohorts use (phase, repetition, input_tokens, concurrency). Durations
    span actual client starts through DONE or failed-request client elapsed time.
    Optional engine timestamps are checked against request IDs, usage and server
    duration metrics. Engine arrivals share the client cohort's time origin.
    Warmup is retained by default; invalid records and duplicate indices raise.
    """
    if not isinstance(measurement_profile, str) or not measurement_profile.strip():
        raise ValueError("measurement_profile must be an explicit nonempty string")
    groups, indices = {}, set()
    for record in _records(source):
        parsed = _parse(record)
        if parsed["index"] in indices:
            raise ValueError("duplicate request_index")
        indices.add(parsed["index"])
        cohort = groups.setdefault(parsed["group"], [])
        if cohort and cohort[0]["consistency"] != parsed["consistency"]:
            raise ValueError("series contains inconsistent cohort attributes")
        if cohort and parsed["group"][0] == "legacy" and cohort[0]["requested_output"] != parsed["requested_output"]:
            raise ValueError("mixed requested lengths require an explicit series_id")
        cohort.append(parsed)
    if not indices:
        raise ValueError("request trace is empty")
    benchmarks = []
    for cohort in groups.values():
        cohort.sort(key=lambda row: (row["start"], row["index"]))
        attributes = cohort[0]["attributes"]
        first = cohort[0]["start"]
        duration = max(row["end"] for row in cohort) - first
        metadata = dict(
            attributes,
            measurement_profile=measurement_profile,
            schema="runpod-request-jsonl",
            latency_basis="client_to_done",
            server_ttft_basis="queue_entry_to_first_token",
            arrival_times=[row["start"] - first for row in cohort],
            server_arrival_times=[
                row["server_arrival"] - first if row["server_arrival"] is not None else None for row in cohort
            ],
            request_indices=[row["index"] for row in cohort],
            requested_inputs=[row["requested_input"] for row in cohort],
            requested_outputs=[row["requested_output"] for row in cohort],
            request_ids=[row["request_id"] for row in cohort],
            errors=[row["error"] for row in cohort],
            scheduled_arrival_monotonic_s=[row["scheduled"] for row in cohort],
            dispatch_started_monotonic_s=[row["dispatch"] for row in cohort],
            dispatch_lateness_s=[
                (
                    row["dispatch"] - row["scheduled"]
                    if row["dispatch"] is not None and row["scheduled"] is not None
                    else None
                )
                for row in cohort
            ],
        )
        benchmark = ServingBenchmark(tuple(row["request"] for row in cohort), duration, metadata)
        if include_warmup or (attributes["phase"] != "warmup" and attributes["split"] != "warmup"):
            benchmarks.append(benchmark)
    return benchmarks
