"""Held-out replay checks with hand-calculated FCFS batch-cap-one timings."""

from dataclasses import replace

import numpy as np
import pytest

from most_queue.io.serving import ServingBenchmark, ServingRequest
from most_queue.theory.continuous_batching.serving_calibration import IsolatedServingFit
from most_queue.theory.continuous_batching.serving_validation import replay_serving_cohort


@pytest.fixture
def fit():
    return IsolatedServingFit(0.2, 0.1, 0.01, 0.0, 2.0, 27, (1, 10), (2, 5), "test-profile", "client_to_done")


@pytest.fixture
def cohort():
    # Each job: prefill .12, then two decodes .11 each => service .34.
    # Arrival0 admitted0, first.23, last.34; arrival.1 waits.24,
    # admitted.34, first.57, last.68. Deliberately reverse client order.
    rows = tuple(
        ServingRequest(
            2,
            2,
            "completed",
            ttft=ttft,
            latency=ttft + 0.11,
            server_queue_time=queue,
            server_ttft=ttft - 0.2,
            server_generation_time=0.11,
        )
        for ttft, queue in [(0.67, 0.24), (0.43, 0.0)]
    )
    return ServingBenchmark(
        rows,
        0.88,
        {
            "measurement_profile": "test-profile",
            "latency_basis": "client_to_done",
            "server_ttft_basis": "queue_entry_to_first_token",
            "server_arrival_times": [10.1, 10.0],
            "request_indices": [8, 7],
            "arrival_times": [0.0, 0.0],
        },
    )


def test_hand_calculated_fcfs_replay_keeps_original_order(fit, cohort):
    original = dict(fit.__dict__)
    report = replay_serving_cohort(fit, cohort, batch_cap=1)
    for metric in report["metrics"].values():
        np.testing.assert_allclose(metric["predicted"], metric["observed"], atol=1e-12)
        assert metric["absolute_mean_error_seconds"] < 1e-12
    assert report["request_indices"] == [8, 7]
    assert report["metrics"]["server_queue_wait"]["predicted"] == pytest.approx([0.24, 0])
    assert fit.__dict__ == original


def test_queue_floor_and_no_use_of_client_arrivals(fit, cohort):
    cohort.metadata["server_arrival_times"] = [10.0, 11.0]
    report = replay_serving_cohort(fit, cohort, batch_cap=1)
    metric = report["metrics"]["server_queue_wait"]
    assert metric["predicted_mean_seconds"] == 0
    assert metric["normalized_mean_error"] == pytest.approx(1)
    rows = tuple(replace(row, server_queue_time=0.001) for row in cohort.requests)
    metric = replay_serving_cohort(fit, replace(cohort, requests=rows), batch_cap=1)["metrics"]["server_queue_wait"]
    assert metric["normalized_mean_error"] == pytest.approx(0.02)


@pytest.mark.parametrize("arrivals", [None, [], [0], [0, None], [0, float("nan")], [0, -1], [0, True]])
def test_missing_or_invalid_engine_arrivals_refused(fit, cohort, arrivals):
    cohort.metadata["server_arrival_times"] = arrivals
    with pytest.raises(ValueError, match="arrival"):
        replay_serving_cohort(fit, cohort)


@pytest.mark.parametrize(
    "field,value",
    [
        ("measurement_profile", "other"),
        ("latency_basis", "stream_span"),
        ("server_ttft_basis", "scheduled_to_first_token"),
    ],
)
def test_profile_and_timing_boundaries_checked(fit, cohort, field, value):
    cohort.metadata[field] = value
    with pytest.raises(ValueError):
        replay_serving_cohort(fit, cohort)


def test_failed_or_missing_timing_not_dropped(fit, cohort):
    for row in [
        replace(cohort.requests[0], status="failed", latency=None),
        replace(cohort.requests[0], server_generation_time=None),
        replace(cohort.requests[0], output_tokens=1),
    ]:
        with pytest.raises(ValueError):
            replay_serving_cohort(fit, replace(cohort, requests=(row, cohort.requests[1])))


@pytest.mark.parametrize(
    "field,value", [("alpha", 0), ("alpha", -1), ("beta", -1), ("gamma", float("nan")), ("client_offset", -1)]
)
def test_invalid_fitted_coefficients_refused(fit, cohort, field, value):
    with pytest.raises(ValueError, match="alpha"):
        replay_serving_cohort(replace(fit, **{field: value}), cohort)


def test_zero_queue_relative_error_is_undefined(fit, cohort):
    cohort.metadata["server_arrival_times"] = [10.0, 11.0]
    rows = tuple(replace(row, server_queue_time=0) for row in cohort.requests)
    metric = replay_serving_cohort(fit, replace(cohort, requests=rows))["metrics"]["server_queue_wait"]
    assert metric["relative_mean_error"] is None
    assert metric["normalized_mean_error"] == 0
