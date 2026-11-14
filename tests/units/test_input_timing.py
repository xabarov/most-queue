"""Synthetic primary-clock evidence for the development timing fit."""

import copy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from most_queue.theory.continuous_batching.input_timing import fit_input_timing


def record(n, ready=None, delivery=None):
    """Build a clock chain directly, independently of the fitted implementation."""
    ready = 0.125 + 0.015625 * n if ready is None else ready
    delivery = 0.0625 + 0.0078125 * n if delivery is None else delivery
    start = 100.0
    queued = start + ready
    first = queued + 0.5
    return {
        "phase": "measured",
        "split": "diagnostic",
        "condition": "idle",
        "role": "probe",
        "trace_enabled": True,
        "success": True,
        "done": True,
        "error": None,
        "request_id": "synthetic",
        "input_tokens": n,
        "output_tokens": 2,
        "usage": {"prompt_tokens": n, "completion_tokens": 2},
        "client_started_monotonic_s": start,
        "client_json_ready_ts": start,
        "client_headers_received_ts": start,
        "client_ttft_s": first + delivery - start,
        "latency_to_done_s": first + 1 + delivery - start,
        "frontend_trace": {
            "request_id": "synthetic",
            "schema_version": 1,
            "timestamp_clock": "monotonic",
            "http_enter_ts": start,
            "body_received_ts": start,
            "handler_enter_ts": start,
            "render_enter_ts": start,
            "render_exit_ts": start,
            "first_response_body_ts": first,
            "response_status": 200,
            "completed_response_body": True,
            "render_success": True,
            "exception": None,
        },
        "server_timestamps": {
            "request_id": "cmpl-synthetic-0",
            "timestamp_clock": "monotonic",
            "arrival_time_clock": "wall",
            "arrival_time": 1700000000.0,
            "queued_ts": queued,
            "scheduled_ts": queued + 0.25,
            "first_token_ts": first,
            "last_token_ts": first + 1,
            "num_generation_tokens": 2,
            "num_preemptions": 0,
        },
        "metrics": {"queue_time_ms": 250.0, "time_to_first_token_ms": 250.0, "generation_time_ms": 1000.0},
    }


def test_recovery_and_predictions_do_not_read_outcomes():
    """Recover coefficients and preserve an immutable outcome-independent predictor."""
    rows = [record(n) for n in (4, 8, 16, 32)]
    original = copy.deepcopy(rows)
    fit = fit_input_timing(rows)
    assert rows == original
    assert (fit.ready_intercept, fit.ready_per_input_token) == pytest.approx((0.125, 0.015625))
    assert (fit.delivery_intercept, fit.delivery_per_input_token) == pytest.approx((0.0625, 0.0078125))
    assert fit.training_requests == 4
    assert fit.input_range == (4, 32)
    assert 1 <= fit.scaled_condition < 1e8
    rows.clear()
    np.testing.assert_allclose(fit.ready_delay([12, 24]), [0.3125, 0.5])
    np.testing.assert_allclose(fit.delivery_delay([12, 24]), [0.15625, 0.25])
    np.testing.assert_allclose(fit.total_delay([12, 24]), [0.46875, 0.75])
    assert isinstance(fit.ready_delay(12), np.ndarray)
    with pytest.raises(FrozenInstanceError):
        fit.ready_intercept = 4


@pytest.mark.parametrize(
    "key,value",
    [
        ("condition", "loaded"),
        ("trace_enabled", False),
        ("phase", "warmup"),
        ("split", "training"),
        ("role", "background"),
        ("done", False),
        ("success", False),
        ("input_tokens", True),
        ("input_tokens", 1.5),
        ("output_tokens", 0),
        ("client_ttft_s", float("nan")),
        ("client_started_monotonic_s", 200),
    ],
)
def test_profile_and_invalid_values_are_rejected_without_filtering(key, value):
    """A single invalid row rejects the entire supplied profile."""
    rows = [record(4), record(8), record(16)]
    rows[-1][key] = value
    with pytest.raises(ValueError):
        fit_input_timing(rows)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("frontend_trace", "timestamp_clock", "wall"),
        ("frontend_trace", "schema_version", True),
        ("frontend_trace", "schema_version", 2),
        ("frontend_trace", "completed_response_body", False),
        ("frontend_trace", "render_success", False),
        ("frontend_trace", "render_exit_ts", 200),
        ("server_timestamps", "timestamp_clock", "wall"),
        ("server_timestamps", "num_generation_tokens", True),
        ("server_timestamps", "num_generation_tokens", 3),
        ("usage", "prompt_tokens", 30),
        ("metrics", "queue_time_ms", 99),
    ],
)
def test_invalid_schema_tokens_and_clocks(section, key, value):
    """Reject inconsistent trace contracts and primary evidence."""
    rows = [record(4), record(8)]
    rows[0][section][key] = value
    with pytest.raises(ValueError):
        fit_input_timing(rows)


@pytest.mark.parametrize("key", ["frontend_trace", "server_timestamps", "usage", "client_ttft_s", "done"])
def test_missing_record_fields(key):
    """Incomplete evidence cannot silently drop out of the fit."""
    rows = [record(4), record(8)]
    del rows[0][key]
    with pytest.raises(ValueError):
        fit_input_timing(rows)


def test_rank_and_condition_rejected():
    """Reject unidentifiable or numerically unstable designs."""
    for rows in ([], [record(4)], [record(4), record(4)], [record(10**10), record(10**10 + 1)]):
        with pytest.raises(ValueError):
            fit_input_timing(rows)


def test_negative_slope_is_constrained_and_zero_delays_allowed():
    """NNLS handles a decreasing target and permits exact zero delays."""
    fit = fit_input_timing([record(n, ready=1 - n / 64, delivery=0) for n in (4, 8, 16, 32)])
    assert fit.ready_per_input_token == 0
    assert fit.ready_intercept == pytest.approx(np.mean([1 - n / 64 for n in (4, 8, 16, 32)]))
    np.testing.assert_array_equal(fit.delivery_delay([4, 8]), [0, 0])
    fit = fit_input_timing([record(n, ready=0, delivery=0) for n in (4, 8)])
    np.testing.assert_array_equal(fit.total_delay([4, 8]), [0, 0])


@pytest.mark.parametrize("inputs", [[True, 2], [0], [-1], [1.5], [float("inf")], [float("nan")], ["2"]])
def test_prediction_inputs(inputs):
    """Prediction enforces finite positive integer token counts."""
    fit = fit_input_timing([record(4), record(8)])
    for predict in (fit.ready_delay, fit.delivery_delay, fit.total_delay):
        with pytest.raises(ValueError):
            predict(inputs)
