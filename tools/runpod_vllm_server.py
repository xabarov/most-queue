#!/usr/bin/env python3
"""Run pinned vLLM with an observational, final-output engine timestamp trace."""

import argparse
import functools
import json
import math
import os
import sys
import threading
from pathlib import Path

TIMESTAMP_FIELDS = ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts")


def validate_timestamps(record):
    """Reject missing, nonfinite, nonpositive or reordered monotonic timestamps."""
    values = [record.get(field) for field in TIMESTAMP_FIELDS]
    if not all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in values):
        raise ValueError("Engine timestamps must be finite and positive")
    if values != sorted(values):
        raise ValueError("Engine timestamps are not ordered")


def install_trace(request_state_class, trace_path):
    """Wrap final output creation without changing the returned object or statistics.

    Duplicate final IDs raise a contract error. Each complete record is appended
    with one O_APPEND write; no per-token I/O is performed.
    """
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents mixing traces from separate server lifetimes.
    with path.open("x", encoding="utf-8"):
        pass
    original = request_state_class.make_request_output
    seen = set()
    lock = threading.Lock()

    @functools.wraps(original)
    def traced(state, *args, **kwargs):
        result = original(state, *args, **kwargs)
        if result is None or not result.finished:
            return result
        metrics = result.metrics
        if metrics is None:
            raise ValueError("Final output has no engine metrics; enable request statistics")
        record = {field: getattr(metrics, field) for field in TIMESTAMP_FIELDS}
        record.update(
            request_id=result.request_id,
            arrival_time=metrics.arrival_time,
            arrival_time_clock="wall",
            timestamp_clock="monotonic",
            num_generation_tokens=metrics.num_generation_tokens,
            num_preemptions=metrics.num_preemptions,
        )
        validate_timestamps(record)
        encoded = (json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n").encode()
        with lock:
            if result.request_id in seen:
                raise ValueError(f"Duplicate final request ID: {result.request_id}")
            descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
            try:
                if os.write(descriptor, encoded) != len(encoded):
                    raise OSError("Incomplete engine trace write")
            finally:
                os.close(descriptor)
            seen.add(result.request_id)
        return result

    request_state_class.make_request_output = traced
    return original


def main():
    """Import GPU dependencies only at server startup, then invoke the real CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-path", required=True, type=Path)
    parser.add_argument("serve_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    serve_args = args.serve_args
    if serve_args[:1] == ["--"]:
        serve_args = serve_args[1:]
    if not serve_args:
        parser.error("Provide vllm serve arguments after --")
    # pylint: disable=import-outside-toplevel,import-error
    import vllm
    from vllm.entrypoints.cli.main import main as vllm_main
    from vllm.v1.engine.output_processor import RequestState

    if vllm.__version__ != "0.32.0":
        raise RuntimeError(f"Expected vLLM 0.32.0, got {vllm.__version__}")
    install_trace(RequestState, args.trace_path)
    sys.argv = ["vllm", "serve", *serve_args]
    vllm_main()


if __name__ == "__main__":
    main()
