"""Audited request-level serving measurements and joint TTFT/TPOT SLOs.

Reference serializer: vLLM v0.33.0rc3, commit
4ba787ec9959e566d3b587a796dff07673324da5, vllm/benchmarks/serve.py.
TTFT and latency are client measurements; queue_times is CLIENT semaphore
waiting. Server timings are separate optional measurements, never inferred.
The SGLang array reader explicitly uses the span of streamed output events
when full per-request latencies are absent. See docs/serving_measurements.md.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ServingRequest:
    """One offered request; all durations are seconds and counts are actual.

    A pending request has an unknown final latency/TPOT. Failed and cancelled
    requests remain in the offered denominator. An absent measurement is None.
    """

    input_tokens: int
    output_tokens: int
    status: str
    ttft: float | None = None
    latency: float | None = None
    client_queue_time: float | None = None
    server_queue_time: float | None = None
    server_ttft: float | None = None
    server_scheduled_to_first_token: float | None = None
    server_generation_time: float | None = None

    def __post_init__(self):
        if self.status not in {"completed", "failed", "cancelled", "pending"}:
            raise ValueError("unknown request status")
        for count in (self.input_tokens, self.output_tokens):
            if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
                raise ValueError("token counts must be nonnegative integers")
        for value in (
            self.ttft,
            self.latency,
            self.client_queue_time,
            self.server_queue_time,
            self.server_ttft,
            self.server_scheduled_to_first_token,
            self.server_generation_time,
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("measured durations must be finite and nonnegative, or None")
        if self.status == "completed":
            if min(self.input_tokens, self.output_tokens) < 1 or self.ttft is None or self.latency is None:
                raise ValueError("completed text requests require positive lengths, TTFT and latency")
            if self.latency < self.ttft:
                raise ValueError("latency cannot precede the first token")
        elif self.latency is not None:
            raise ValueError("only completed requests have a final latency")

    @property
    def tpot(self):
        """Per-request TPOT, excluding the first token; None for <=1 token."""
        if self.status != "completed" or self.output_tokens <= 1:
            return None
        return (self.latency - self.ttft) / (self.output_tokens - 1)

    @property
    def server_tpot(self):
        """Server generation span per token after the first, when observed."""
        if self.status != "completed" or self.output_tokens <= 1 or self.server_generation_time is None:
            return None
        return self.server_generation_time / (self.output_tokens - 1)


@dataclass(frozen=True)
class ServingBenchmark:
    """A finite offered cohort and its explicit measurement-window duration."""

    requests: tuple[ServingRequest, ...]
    duration: float
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.requests or not all(isinstance(row, ServingRequest) for row in self.requests):
            raise ValueError("a benchmark requires a nonempty request cohort")
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("duration must be finite and positive")


def _read_arrays(source, schema):
    data = json.loads(Path(source).read_text(encoding="utf-8")) if isinstance(source, (str, Path)) else source
    if not isinstance(data, dict):
        raise ValueError("expected a single benchmark JSON object")
    required = ["input_lens", "output_lens", "ttfts", "errors"]
    required += ["latencies"] if schema == "vllm-bench-serve" else ["itls"]
    if any(not isinstance(data.get(key), list) for key in required):
        raise ValueError("detailed per-request arrays are required; aggregate percentiles are insufficient")
    count = len(data["input_lens"])
    if not count or any(len(data[key]) != count for key in required):
        raise ValueError("per-request arrays must be nonempty and aligned")
    optional = ["queue_times", "server_queue_times", "server_ttfts"]
    for key in optional:
        if key in data and (not isinstance(data[key], list) or len(data[key]) != count):
            raise ValueError(f"{key} must align with the request arrays")
    rows = []
    for index in range(count):
        error = data["errors"][index]
        if not isinstance(error, str):
            raise ValueError("each error must be a string")
        completed = error == "" and data["output_lens"][index] > 0
        latency = None
        if completed:
            if schema == "vllm-bench-serve":
                latency = data["latencies"][index]
            else:
                gaps = np.asarray(data["itls"][index], dtype=float)
                if gaps.ndim != 1 or not np.all(np.isfinite(gaps)) or np.any(gaps < 0):
                    raise ValueError("stream gaps must be finite nonnegative vectors")
                latency = float(data["ttfts"][index] + gaps.sum())
        timings = {key: data[key][index] if completed and key in data else None for key in optional}
        rows.append(
            ServingRequest(
                input_tokens=data["input_lens"][index],
                output_tokens=data["output_lens"][index],
                status="completed" if completed else "failed",
                ttft=data["ttfts"][index] if completed else None,
                latency=latency,
                client_queue_time=timings["queue_times"],
                server_queue_time=timings["server_queue_times"],
                server_ttft=timings["server_ttfts"],
            )
        )
    successes = sum(row.status == "completed" for row in rows)
    if data.get("completed") != successes or ("failed" in data and data["failed"] != count - successes):
        raise ValueError("aggregate completion/failure counts disagree with detailed outcomes")
    if "num_prompts" in data and data["num_prompts"] != count:
        raise ValueError("num_prompts disagrees with offered cohort; do not silently drop requests")
    metadata = {
        key: data[key]
        for key in ["model_id", "backend", "max_concurrency", "request_rate", "dataset_name"]
        if key in data
    }
    metadata.update(schema=schema, latency_basis="client_latency" if schema == "vllm-bench-serve" else "stream_span")
    if metadata.get("request_rate") == math.inf:
        metadata["request_rate"] = "inf"
    return ServingBenchmark(tuple(rows), data["duration"], metadata)


def read_vllm_benchmark(source, *, measurement_profile=None):
    """Read vLLM detailed JSON; preserve absent server metrics as None.

    The caller must retain the producer version and hardware provenance. This
    is a schema adapter, not detection of the engine/version from latency data.
    """
    result = _read_arrays(source, "vllm-bench-serve")
    result.metadata["measurement_profile"] = measurement_profile
    return result


def read_sglang_stream_benchmark(source, *, measurement_profile=None):
    """Read detailed streaming arrays with TTFT + sum(ITL) as observed span.

    The span ends at the last output event, not necessarily HTTP completion.
    ITL entries can contain multiple tokens; TPOT uses actual output_tokens.
    """
    result = _read_arrays(source, "sglang-stream-arrays")
    result.metadata["measurement_profile"] = measurement_profile
    return result


def _quantiles(values):
    if not values:
        return None
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        **dict(zip(["p50", "p90", "p99"], np.percentile(values, [50, 90, 99]).tolist())),
    }


def summarize_serving_slo(benchmark, *, ttft_limit, tpot_limit):
    """Joint request SLO with all offered outcomes and pending-outcome bounds.

    Completed one-token replies pass the TPOT condition by the vLLM goodput
    convention but are excluded from TPOT means/quantiles. Pending requests
    are unknown unless their already-observed TTFT violates its limit.
    Counts/quantiles describe this cohort, without an iid confidence claim.
    """
    if any(not math.isfinite(value) or value < 0 for value in (ttft_limit, tpot_limit)):
        raise ValueError("SLO limits must be finite nonnegative seconds")
    counts = {status: 0 for status in ["completed", "failed", "cancelled", "pending"]}
    passed, unknown, single_token = 0, 0, 0
    ttfts, tpots = [], []
    for row in benchmark.requests:
        counts[row.status] += 1
        if row.status == "completed":
            ttfts.append(row.ttft)
            if row.tpot is None:
                single_token += 1
            else:
                tpots.append(row.tpot)
            passed += row.ttft <= ttft_limit and (row.tpot is None or row.tpot <= tpot_limit)
        elif row.status == "pending" and (row.ttft is None or row.ttft <= ttft_limit):
            unknown += 1
    offered = len(benchmark.requests)
    return {
        "offered": offered,
        **counts,
        "single_token_completed": single_token,
        "joint_passed": passed,
        "joint_unknown": unknown,
        "joint_fraction_bounds": [passed / offered, (passed + unknown) / offered],
        "observed_goodput": passed / benchmark.duration,
        "duration": benchmark.duration,
        "ttft_limit": ttft_limit,
        "tpot_limit": tpot_limit,
        "completed_ttft_seconds": _quantiles(ttfts),
        "completed_multitoken_tpot_seconds": _quantiles(tpots),
        "server_queue_observed": sum(row.server_queue_time is not None for row in benchmark.requests),
        "server_ttft_observed": sum(row.server_ttft is not None for row in benchmark.requests),
        "latency_basis": benchmark.metadata.get("latency_basis", "explicit_request_records"),
        "uncertainty": "pending-outcome bounds; no iid confidence interval",
    }


def from_prefill_decode(result, arrivals, inputs, outputs, *, horizon=None):
    """Convert token DES timestamps; optionally censor at a fixed horizon.

    Future requests are outside the offered cohort. Future completions and
    tokens are hidden rather than counted as successful outcomes at the cut.
    No steady-state or stationarity assertion is made for the drained cohort.
    """
    arrivals = np.asarray(arrivals, dtype=float)
    if arrivals.ndim != 1 or not arrivals.size or not np.all(np.isfinite(arrivals)) or np.any(np.diff(arrivals) < 0):
        raise ValueError("arrivals must be nonempty finite chronological times")
    if any(len(values) != len(arrivals) for values in [inputs, outputs, result.output_times, result.completion_times]):
        raise ValueError("DES arrays must align")
    for lengths in (inputs, outputs):
        values = np.asarray(lengths, dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0) or np.any(values != np.floor(values)):
            raise ValueError("DES token lengths must be positive integers")
    for arrival, count, stamps, completion in zip(arrivals, outputs, result.output_times, result.completion_times):
        stamps = np.asarray(stamps, dtype=float)
        if stamps.ndim != 1 or stamps.size != count or not np.all(np.isfinite(stamps)):
            raise ValueError("DES token timestamps must align with output lengths and be finite")
        if (
            np.any(np.diff(stamps) < 0)
            or stamps[0] < arrival
            or not math.isfinite(completion)
            or stamps[-1] != completion
        ):
            raise ValueError("DES token timestamps, output lengths and completion must agree")
    end = max(result.completion_times) if horizon is None else horizon
    if not math.isfinite(end) or end <= arrivals[0]:
        raise ValueError("observation horizon must follow the first arrival")
    rows = []
    for index, arrival in enumerate(arrivals):
        if arrival > end:
            break
        tokens = result.output_times[index]
        observed = [stamp for stamp in tokens if stamp <= end]
        completed = result.completion_times[index] <= end
        rows.append(
            ServingRequest(
                input_tokens=int(inputs[index]),
                output_tokens=int(outputs[index]) if completed else len(observed),
                status="completed" if completed else "pending",
                ttft=tokens[0] - arrival if observed else None,
                latency=result.completion_times[index] - arrival if completed else None,
            )
        )
    return ServingBenchmark(tuple(rows), float(end - arrivals[0]), {"latency_basis": "token_DES"})
