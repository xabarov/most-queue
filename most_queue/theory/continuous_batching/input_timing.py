"""Development-only input timing fitted from primary monotonic timestamps.

Ready includes idle IPC up to engine queue entry; delivery is separate. These
are effective baseline delays, not an exact physical decomposition. No records
are implicitly filtered and no files or per-series fits are used.
"""

from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.optimize import nnls


def _inputs(values):
    raw = np.asarray(values, dtype=object)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in raw.flat):
        raise ValueError("inputs must be finite positive integers, not booleans")
    result = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(result)) or np.any(result <= 0) or np.any(result != np.floor(result)):
        raise ValueError("inputs must be finite positive integers")
    return result


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
        raise ValueError("finite nonnegative numeric clocks and durations required")
    return float(value)


@dataclass(frozen=True)
class InputTimingFit:
    """Development-only common nonnegative affine baseline delay coefficients."""

    ready_intercept: float
    ready_per_input_token: float
    delivery_intercept: float
    delivery_per_input_token: float
    scaled_condition: float
    training_requests: int
    input_range: tuple[int, int]

    def ready_delay(self, inputs):
        """Predict effective ready delay using only input token counts."""
        return np.asarray(self.ready_intercept + self.ready_per_input_token * _inputs(inputs))

    def delivery_delay(self, inputs):
        """Predict first-output delivery delay using only input token counts."""
        return np.asarray(self.delivery_intercept + self.delivery_per_input_token * _inputs(inputs))

    def total_delay(self, inputs):
        """Predict the sum of the separate baseline delays."""
        return self.ready_delay(inputs) + self.delivery_delay(inputs)


def _targets(row):
    """Validate a complete instrumented idle probe without selecting rows."""
    expected = {"phase": "measured", "split": "diagnostic", "condition": "idle", "role": "probe"}
    if any(row[key] != value for key, value in expected.items()) or row["trace_enabled"] is not True:
        raise ValueError("only measured diagnostic traced idle probes are accepted")
    if row["success"] is not True or row["done"] is not True or row.get("error"):
        raise ValueError("incomplete or unsuccessful request")
    if row.get("status", "succeeded") != "succeeded" or row.get("offer_status", "OFFERED") != "OFFERED":
        raise ValueError("unsuccessful or unoffered request")
    n, m = _inputs([row["input_tokens"], row["output_tokens"]])
    usage, trace, engine = row["usage"], row["frontend_trace"], row["server_timestamps"]
    actual = _inputs([usage["prompt_tokens"], usage["completion_tokens"], engine["num_generation_tokens"]])
    if tuple(actual) != (n, m, m):
        raise ValueError("token counts differ")
    if engine["num_preemptions"] != 0 or isinstance(engine["num_preemptions"], bool):
        raise ValueError("idle probe was preempted")
    if engine["request_id"] != "cmpl-" + row["request_id"] + "-0" or trace["request_id"] != row["request_id"]:
        raise ValueError("trace identities differ")
    if (
        isinstance(trace["schema_version"], bool)
        or not isinstance(trace["schema_version"], int)
        or trace["schema_version"] != 1
    ):
        raise ValueError("unsupported frontend schema")
    if trace["timestamp_clock"] != "monotonic" or engine["timestamp_clock"] != "monotonic":
        raise ValueError("primary timestamps must share the monotonic clock")
    if engine["arrival_time_clock"] != "wall":
        raise ValueError("arrival clock label differs")
    _number(engine["arrival_time"])
    if (
        isinstance(trace["response_status"], bool)
        or not isinstance(trace["response_status"], int)
        or trace["response_status"] != 200
    ):
        raise ValueError("frontend response was unsuccessful")
    if (
        trace["completed_response_body"] is not True
        or trace["render_success"] is not True
        or any(trace.get(key) is not None for key in ("exception", "instrumentation_error", "render_exception"))
    ):
        raise ValueError("frontend trace failed or incomplete")
    start = _number(row["client_started_monotonic_s"])
    marks = [start] + [
        _number(trace[key])
        for key in ("http_enter_ts", "body_received_ts", "handler_enter_ts", "render_enter_ts", "render_exit_ts")
    ]
    stamps = [_number(engine[key]) for key in ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts")]
    marks += stamps
    if marks != sorted(marks):
        raise ValueError("inconsistent primary clock order")
    first = start + _number(row["client_ttft_s"])
    done = start + _number(row["latency_to_done_s"])
    ready, headers, body = (
        _number(row["client_json_ready_ts"]),
        _number(row["client_headers_received_ts"]),
        _number(trace["first_response_body_ts"]),
    )
    if not (
        start <= ready <= marks[1] <= headers <= first <= done
        and stamps[2] <= first
        and stamps[3] <= done
        and marks[5] <= body <= done
    ):
        raise ValueError("inconsistent client transport clocks")
    for name, left, right in zip(("queue_time_ms", "time_to_first_token_ms", "generation_time_ms"), stamps, stamps[1:]):
        if not np.isclose(_number(row["metrics"][name]), (right - left) * 1000, rtol=0, atol=1e-5):
            raise ValueError("engine metric disagrees with primary timestamps")
    targets = np.array([stamps[0] - start, first - stamps[2]])
    if not np.all(np.isfinite(targets)) or np.any(targets < 0):
        raise ValueError("invalid timing targets")
    return n, targets


def fit_input_timing(records):
    """Fit two common NNLS affine delays from validated development-only traces.

    Primary stamps define ready=queued-start and delivery=start+TTFT-first_token.
    Every supplied row must satisfy the idle diagnostic profile; none are dropped.
    """
    try:
        rows = [_targets(row) for row in records]
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("incomplete or invalid timing record") from exc
    if not rows:
        raise ValueError("nonempty timing records required")
    inputs = np.array([row[0] for row in rows])
    design = np.column_stack((np.ones(len(rows)), inputs))
    with np.errstate(over="ignore"):
        norms = np.linalg.norm(design, axis=0)
    if not np.all(np.isfinite(norms)):
        raise ValueError("timing design exceeds numeric range")
    scaled = design / norms
    condition = float(np.linalg.cond(scaled))
    if np.linalg.matrix_rank(scaled) != 2 or not np.isfinite(condition) or condition > 1e8:
        raise ValueError("timing design must have full rank and scaled condition <= 1e8")
    targets = np.array([row[1] for row in rows])
    ready = nnls(scaled, targets[:, 0])[0] / norms
    delivery = nnls(scaled, targets[:, 1])[0] / norms
    return InputTimingFit(*map(float, (*ready, *delivery)), condition, len(rows), (int(min(inputs)), int(max(inputs))))
