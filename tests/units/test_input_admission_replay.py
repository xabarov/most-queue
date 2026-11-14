"""Finite input replay contracts and separation of observations from predictions."""

# pylint: disable=missing-function-docstring,redefined-outer-name

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from most_queue.io.serving import ServingBenchmark, ServingRequest
from most_queue.theory.continuous_batching.advance_reservation_calibration import AdvanceReservationFit
from most_queue.theory.continuous_batching.input_admission_replay import replay_input_admission
from most_queue.theory.continuous_batching.input_timing import InputTimingFit


@pytest.fixture
def setup():
    """A hand-sized, intentionally nonchronological cohort."""
    server = AdvanceReservationFit(0.02, 0.001, 0.0001, 0.03, 0, 1, 1, 10, (1, 100), (2, 20), "test", "done")
    timing = InputTimingFit(0.01, 0.001, 0.02, 0, 1, 10, (1, 100))
    rows = tuple(
        ServingRequest(
            n,
            3,
            "completed",
            0.2,
            0.26,
            server_queue_time=0.01,
            server_ttft=0.1,
            server_scheduled_to_first_token=0.09,
            server_generation_time=0.06,
        )
        for n in (4, 2, 8)
    )
    benchmark = ServingBenchmark(
        rows,
        1,
        {
            "measurement_profile": "test",
            "latency_basis": "done",
            "server_ttft_basis": "queue_entry_to_first_token",
            "arrival_times": [10.12, 10, 10.01],
            "server_arrival_times": [10.14, 10.03, 10.04],
            "request_indices": [9, 2, 7],
        },
    )
    return server, timing, benchmark


def test_hand_cohort_alignment_and_no_mutation(setup):
    server, timing, benchmark = setup
    before = deepcopy(benchmark)
    report = replay_input_admission(server, timing, benchmark)
    assert benchmark == before
    assert report["request_indices"] == [9, 2, 7]
    assert len(report["variants"]) == 5
    for name, variant in report["variants"].items():
        assert len(variant["joint_client_slo"]) == 9
        if name == "baseline_v2":
            assert "prequeue" not in variant["metrics"]
            continue
        clocks = variant["timing"]
        assert clocks["ready"] == pytest.approx([0.134, 0.012, 0.028])
        assert variant["metrics"]["client_ttft"]["predicted"] == pytest.approx(
            np.asarray(clocks["first_output"]) + 0.02 - [0.12, 0, 0.01]
        )
        assert variant["request_indices"] == [9, 2, 7]
    continuous = report["variants"]["continuous_reservation"]
    # First arriving request is alone: prefill emits its first output.
    assert continuous["timing"]["first_output"][1] == pytest.approx(0.012 + 0.02 + 0.0011 * 2)


def test_new_predictions_ignore_observed_outcomes(setup):
    server, timing, benchmark = setup
    changed = replace(
        benchmark,
        requests=tuple(
            replace(
                r,
                ttft=r.ttft + 0.1,
                latency=r.latency + 0.2,
                server_ttft=0.12,
                server_queue_time=0.02,
                server_scheduled_to_first_token=0.1,
                server_generation_time=0.08,
            )
            for r in benchmark.requests
        ),
        metadata={**benchmark.metadata, "server_arrival_times": [10.2, 10.1, 10.11]},
    )
    before = replay_input_admission(server, timing, benchmark)["variants"]
    after = replay_input_admission(server, timing, changed)["variants"]
    for name, before_variant in before.items():
        if name != "baseline_v2":
            assert before_variant["timing"] == after[name]["timing"]
            for metric in before_variant["metrics"]:
                assert before_variant["metrics"][metric]["predicted"] == after[name]["metrics"][metric]["predicted"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("arrival_times", [0, 1]),
        ("arrival_times", [True, 0, 0]),
        ("arrival_times", [float("nan"), 0, 0]),
        ("arrival_times", [[1], [2], [3]]),
        ("server_arrival_times", [None, 11, 12]),
        ("server_arrival_times", [0, 0, 0]),
        ("measurement_profile", "wrong"),
        ("latency_basis", "wrong"),
    ],
)
def test_invalid_metadata_rejected(setup, key, value):
    server, timing, benchmark = setup
    with pytest.raises(ValueError):
        replay_input_admission(server, timing, replace(benchmark, metadata={**benchmark.metadata, key: value}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("output_tokens", 1),
        ("server_queue_time", None),
        ("server_scheduled_to_first_token", None),
        ("server_generation_time", None),
    ],
)
def test_incomplete_measurements_rejected(setup, field, value):
    server, timing, benchmark = setup
    changed = replace(benchmark, requests=(replace(benchmark.requests[0], **{field: value}),) + benchmark.requests[1:])
    with pytest.raises(ValueError):
        replay_input_admission(server, timing, changed)


def test_input_poll_gate_only_partitions_delay(setup):
    server, timing, benchmark = setup
    variants = replay_input_admission(server, timing, benchmark)["variants"]
    gated = variants["input_two_batches"]
    continuous = variants["two_batches_continuous_input"]
    for key in ("ready", "reservation", "first_compute", "first_output", "last_output"):
        assert gated["timing"][key] == pytest.approx(continuous["timing"][key])
    for key in ("client_ttft", "client_tpot", "server_scheduled_to_first_token"):
        assert gated["metrics"][key]["predicted"] == pytest.approx(continuous["metrics"][key]["predicted"])
    for variant in (gated, continuous):
        ttft = np.asarray(variant["metrics"]["server_ttft"]["predicted"])
        outside = np.asarray(variant["metrics"]["outside_engine"]["predicted"])
        assert ttft + outside == pytest.approx(variant["metrics"]["client_ttft"]["predicted"])
