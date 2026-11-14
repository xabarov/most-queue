"""Pure descriptive arithmetic for frontend diagnostic records, in seconds.

No records are read here. Pair summaries retain every supplied eligible probe and
use linear interpolation at (n - 1) * p for descriptive sample quartiles.
"""

import math
from collections import defaultdict
from statistics import fmean

TOLERANCE = 1e-8
PAIR_METRICS = ("client_ttft", "server_ttft", "client_tpot", "server_tpot", "queue", "outside_engine")


def _finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Expected a finite number")
    return float(value)


def _duration(value):
    value = _finite(value)
    if value < -TOLERANCE:
        raise ValueError("Negative duration or inconsistent clock order")
    return max(0.0, value)


def request_metrics(record):
    """Compute timing components for one successful row on a common clock.

    Clock differences down to -1e-8 seconds are treated as rounding noise and
    returned as zero. Larger negative durations or clock inversions are errors.
    The outside_engine key means outside QUEUED-to-first-token timing, not
    necessarily outside the EngineCore process or its input queue.
    """
    if record.get("success") is not True:
        raise ValueError("A successful request is required")
    count = record["output_tokens"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 2:
        raise ValueError("output_tokens must be an integer >= 2")
    client_ttft = _duration(record["client_ttft_s"])
    latency = _duration(record["latency_to_done_s"])
    arrival, dispatch, started = (
        _finite(record[key])
        for key in ("scheduled_arrival_monotonic_s", "dispatch_started_monotonic_s", "client_started_monotonic_s")
    )
    queued, scheduled, first, last = (
        _finite(record["server_timestamps"][key])
        for key in ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts")
    )
    ordered = (arrival, dispatch, started, queued, scheduled, first, last, started + latency)
    for left, right in zip(ordered, ordered[1:]):
        _duration(right - left)
    _duration(latency - client_ttft)
    server_ttft = _duration(first - queued)
    return {
        "client_ttft": client_ttft,
        "server_ttft": server_ttft,
        "client_tpot": _duration(latency - client_ttft) / (count - 1),
        "server_tpot": _duration(last - first) / (count - 1),
        "queue": _duration(scheduled - queued),
        "outside_engine": _duration(client_ttft - server_ttft),
        "pre_engine": _duration(queued - started),
        "first_output_delivery": _duration(started + client_ttft - first),
        "dispatch_to_worker_start": _duration(started - dispatch),
        "dispatch_lateness": _duration(dispatch - arrival),
    }


def summarize(values):
    """Describe every supplied finite value with linearly interpolated quartiles."""
    ordered = sorted(_finite(value) for value in values)
    if not ordered:
        raise ValueError("At least one value is required")

    def quantile(fraction):
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))

    result = {
        "count": len(ordered),
        "minimum": ordered[0],
        "q25": quantile(0.25),
        "median": quantile(0.5),
        "q75": quantile(0.75),
        "maximum": ordered[-1],
        "mean": float(fmean(ordered)),
    }
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError("Summary arithmetic must remain finite")
    return result


def pair_comparisons(records):
    """Require complete on/off probe pairs and return signed on-minus-off metrics."""
    grouped = defaultdict(list)
    for row in records:
        if row.get("phase") == "warmup" or row.get("role") != "probe":
            continue
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("Every eligible probe requires pair_id")
        grouped[pair_id].append(row)
    result = []
    for pair_id, rows in sorted(grouped.items()):
        if len(rows) != 2 or any(not isinstance(row.get("trace_enabled"), bool) for row in rows):
            raise ValueError("Every pair requires exactly two rows with Boolean trace controls")
        if {row["trace_enabled"] for row in rows} != {False, True}:
            raise ValueError("Every pair requires one trace-on and one trace-off row")
        on = next(row for row in rows if row["trace_enabled"])
        off = next(row for row in rows if not row["trace_enabled"])
        for key in ("input_tokens", "output_tokens", "prompt_sha256", "block_id", "condition"):
            if key not in on or key not in off or on[key] != off[key]:
                raise ValueError(f"Pair differs or lacks {key}")
        if on["condition"] not in ("idle", "loaded"):
            raise ValueError("Probe condition must be idle or loaded")
        metrics_on, metrics_off = request_metrics(on), request_metrics(off)
        result.append(
            {
                "pair_id": pair_id,
                **{key: on[key] for key in ("input_tokens", "output_tokens", "prompt_sha256", "block_id", "condition")},
                "request_id_on": on["request_id"],
                "request_id_off": off["request_id"],
                "metrics_on": metrics_on,
                "metrics_off": metrics_off,
                "difference": {key: metrics_on[key] - metrics_off[key] for key in PAIR_METRICS},
            }
        )
    return result


def summarize_pairs(pairs):
    """Pool supplied pairs by condition and input length, retaining block metadata.

    Expected counts come from the supplied pairs, not a hard-coded design. The
    fixed warning uses a strict > comparison; equality does not raise a warning.
    To describe one block, callers can filter pairs by block_id before this call.
    """
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[(pair["condition"], pair["input_tokens"])].append(pair)
    result = []
    for (condition, input_tokens), group in sorted(grouped.items()):
        differences = {key: summarize(pair["difference"][key] for pair in group) for key in PAIR_METRICS}
        mean_off_ttft = summarize(pair["metrics_off"]["client_ttft"] for pair in group)["mean"]
        threshold = max(0.005, 0.05 * mean_off_ttft)
        result.append(
            {
                "condition": condition,
                "input_tokens": input_tokens,
                "blocks": sorted({pair["block_id"] for pair in group}),
                "pair_ids": [pair["pair_id"] for pair in group],
                "pair_count": len(group),
                "expected_request_count": 2 * len(group),
                "difference": differences,
                "mean_off_ttft": mean_off_ttft,
                "warning_threshold": threshold,
                "warning": abs(differences["client_ttft"]["median"]) > threshold,
            }
        )
    return result
