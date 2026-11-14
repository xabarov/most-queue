"""Clock separation, identifiable calibration, and paired finite-cohort checks."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from most_queue.io.runpod import read_runpod_trace
from most_queue.io.serving import ServingBenchmark, ServingRequest, summarize_serving_slo
from most_queue.theory.continuous_batching.advance_reservation_calibration import fit_advance_reservation
from most_queue.theory.continuous_batching.advance_reservation_validation import (
    compare_joint_slo,
    replay_advance_reservation,
)


def cohort(lengths=None):
    """Independent isolated oracle with separate server and client clocks."""
    lengths = lengths or [(n, m) for n in (128, 1024, 4096) for m in (32, 128, 512)]
    rows = []
    for n, m in lengths:
        first = 0.02 + 0.0001002 * n
        tpot = 0.0201 + 0.0000002 * (n + m / 2)
        ttft = first + 0.03 + 3e-6 * n
        rows.append(
            ServingRequest(
                n,
                m,
                "completed",
                ttft,
                ttft + (m - 1) * tpot,
                server_queue_time=0,
                server_ttft=first,
                server_scheduled_to_first_token=first,
                server_generation_time=(m - 1) * tpot,
            )
        )
    return ServingBenchmark(
        tuple(rows),
        1000,
        dict(
            max_concurrency=1,
            measurement_profile="oracle",
            latency_basis="client_to_done",
            server_ttft_basis="queue_entry_to_first_token",
            server_arrival_times=[20 * i for i in range(len(rows))],
        ),
    )


@pytest.fixture
def training():
    return cohort()


def test_recovers_identifiable_costs_and_predicts_unseen_lengths(training):
    fit = fit_advance_reservation([training])
    assert [fit.alpha, fit.beta, fit.gamma] == pytest.approx([0.02, 0.0001, 0.0000002], rel=1e-7)
    assert [fit.client_intercept, fit.client_per_input_token] == pytest.approx([0.03, 3e-6])
    pred = fit.predict([777], [73])
    assert pred["server_ttft"][0] == pytest.approx(0.02 + 0.0001002 * 777)
    assert pred["server_tpot"][0] == pytest.approx(0.0201 + 0.0000002 * (777 + 73 / 2))
    assert pred["client_ttft"][0] - pred["server_ttft"][0] == pytest.approx(0.03 + 3e-6 * 777)


def test_first_token_is_prefill_and_single_token_does_not_supply_decode(training):
    singles = cohort([(128, 1), (1024, 1), (4096, 1)])
    fit = fit_advance_reservation([training, singles])
    pred = fit.predict([128, 1024], [1, 1])
    assert np.isnan(pred["server_tpot"]).all()
    assert pred["server_ttft"] == pytest.approx([0.02 + 0.0001002 * n for n in (128, 1024)])
    with pytest.raises(ValueError, match="multitoken"):
        fit_advance_reservation([singles])


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("max_concurrency", 2, "isolated"),
        ("measurement_profile", "", "explicit"),
        ("latency_basis", "", "explicit"),
        ("server_ttft_basis", "client", "queue entry"),
    ],
)
def test_invalid_measurement_contract(training, key, value, match):
    with pytest.raises(ValueError, match=match):
        fit_advance_reservation([replace(training, metadata={**training.metadata, key: value})])


@pytest.mark.parametrize("key", ["measurement_profile", "latency_basis"])
def test_mixed_clocks_or_profiles_rejected(training, key):
    other = replace(training, metadata={**training.metadata, key: "different"})
    with pytest.raises(ValueError, match="mix"):
        fit_advance_reservation([training, other])


@pytest.mark.parametrize("field", ["server_ttft", "server_scheduled_to_first_token", "server_generation_time"])
def test_missing_fit_timings_rejected(training, field):
    rows = (replace(training.requests[0], **{field: None}),) + training.requests[1:]
    with pytest.raises(ValueError, match="timings|generation"):
        fit_advance_reservation([replace(training, requests=rows)])


def test_failed_training_and_unidentifiable_design_rejected(training):
    failed = ServingRequest(128, 0, "failed")
    with pytest.raises(ValueError, match="complete successful"):
        fit_advance_reservation([replace(training, requests=(failed,) + training.requests)])
    with pytest.raises(ValueError, match="rank-deficient"):
        fit_advance_reservation([cohort([(128, 32)] * 3)])


@pytest.mark.parametrize(
    "inputs,outputs", [([], []), ([0], [2]), ([1.5], [2]), ([np.nan], [2]), ([2], [0]), ([2], [2, 3])]
)
def test_prediction_rejects_invalid_lengths(training, inputs, outputs):
    with pytest.raises(ValueError, match="length|integer"):
        fit_advance_reservation([training]).predict(inputs, outputs)


def test_replay_restores_request_order_and_uses_only_engine_arrivals(training):
    fit = fit_advance_reservation([training])
    base = cohort([(128, 32), (1024, 128), (4096, 512)])
    expected = replay_advance_reservation(fit, base)
    order = [2, 0, 1]
    shuffled = replace(
        base,
        requests=tuple(base.requests[i] for i in order),
        metadata={
            **base.metadata,
            "server_arrival_times": [base.metadata["server_arrival_times"][i] for i in order],
            "arrival_times": [0, 0, 0],
        },
    )
    actual = replay_advance_reservation(fit, shuffled)
    for metric in expected["metrics"]:
        assert actual["metrics"][metric]["predicted"] == pytest.approx(
            np.asarray(expected["metrics"][metric]["predicted"])[order]
        )
    absent = replace(base, metadata={**base.metadata, "server_arrival_times": None, "arrival_times": [0, 20, 40]})
    with pytest.raises(ValueError, match="server_arrival_times"):
        replay_advance_reservation(fit, absent)


@pytest.mark.parametrize(
    "field", ["server_ttft", "server_queue_time", "server_generation_time", "server_scheduled_to_first_token"]
)
def test_replay_never_zero_fills_missing_server_metrics(training, field):
    broken = replace(training, requests=(replace(training.requests[0], **{field: None}),) + training.requests[1:])
    with pytest.raises(ValueError, match="complete server timings"):
        replay_advance_reservation(fit_advance_reservation([training]), broken)


@pytest.mark.parametrize("field,value", [("alpha", 0), ("beta", -1), ("gamma", np.nan), ("client_intercept", np.inf)])
def test_replay_rejects_malformed_costs(training, field, value):
    fit = replace(fit_advance_reservation([training]), **{field: value})
    with pytest.raises(ValueError, match="fitted costs"):
        replay_advance_reservation(fit, training)


def test_slo_uses_paired_client_metrics_and_rejects_misalignment():
    rows = tuple(
        ServingRequest(10, 2, "completed", t, t + p, server_ttft=0.01, server_generation_time=0.01)
        for t, p in [(1, 3), (3, 1), (1, 1), (3, 3)]
    )
    benchmark = ServingBenchmark(rows, 10)
    replay = {
        "metrics": {
            "client_ttft": {"observed": [r.ttft for r in rows], "predicted": [1, 3, 1, 3]},
            "client_tpot": {"observed": [r.tpot for r in rows], "predicted": [3, 1, 1, 3]},
        }
    }
    report = compare_joint_slo(benchmark, replay, ttft_limit=2, tpot_limit=2)
    assert report["observed_fraction"] == report["predicted_fraction"] == 0.25
    replay["metrics"]["client_tpot"]["predicted"] = [1, 3, 1, 3]
    assert compare_joint_slo(benchmark, replay, ttft_limit=2, tpot_limit=2)["predicted_fraction"] == 0.5
    replay["metrics"]["client_ttft"]["observed"].reverse()
    with pytest.raises(ValueError, match="align"):
        compare_joint_slo(benchmark, replay, ttft_limit=2, tpot_limit=2)
    failed = replace(benchmark, requests=rows + (ServingRequest(10, 0, "failed"),))
    summary = summarize_serving_slo(failed, ttft_limit=2, tpot_limit=2)
    assert summary["offered"] == 5 and summary["joint_passed"] == 1
    with pytest.raises(ValueError, match="complete successful"):
        compare_joint_slo(failed, replay, ttft_limit=2, tpot_limit=2)


def test_frozen_campaign_training_only_and_sparse_clock_structure():
    root = (
        Path(__file__).resolve().parents[2]
        / "tests/fixtures/serving_measurements/runpod_qwen3_8b_a40_calibration_v1_20261114"
    )
    cohorts = read_runpod_trace(root / "requests.jsonl.gz", measurement_profile="frozen", include_warmup=False)
    train = [b for b in cohorts if b.metadata["split"] == "train" and b.metadata["mode"] == "isolated"]
    fit = fit_advance_reservation(train)
    assert fit.training_requests == sum(len(b.requests) for b in train) == 27
    sparse = cohort([(128, 32), (4096, 512)])
    sparse = replace(
        sparse, metadata={**sparse.metadata, "measurement_profile": "frozen", "server_arrival_times": [0, 1000]}
    )
    report = replay_advance_reservation(fit, sparse)
    assert report["metrics"]["server_queue_wait"]["predicted"] == pytest.approx([0, 0], abs=1e-10)
    assert report["metrics"]["server_ttft"]["predicted"] == pytest.approx(
        fit.predict([128, 4096], [32, 512])["server_ttft"]
    )


@pytest.mark.parametrize("queue,server_ttft", [(None, 0.1), (0.2, 0.1)])
def test_fit_requires_consistent_server_ttft_clock_identity(training, queue, server_ttft):
    broken = replace(training.requests[0], server_queue_time=queue, server_ttft=server_ttft)
    with pytest.raises(ValueError, match="complete server timings|queue"):
        fit_advance_reservation([replace(training, requests=(broken,) + training.requests[1:])])
