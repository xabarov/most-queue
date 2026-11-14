"""Request-level serving adapters, joint SLOs, and isolated calibration."""

from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from most_queue.io.serving import (
    ServingBenchmark,
    ServingRequest,
    from_prefill_decode,
    read_sglang_stream_benchmark,
    read_vllm_benchmark,
    summarize_serving_slo,
)
from most_queue.sim.prefill_decode import PrefillDecodeSim
from most_queue.theory.continuous_batching.serving_calibration import (
    evaluate_isolated_serving,
    fit_isolated_serving,
)


def detailed_vllm():
    return {
        "duration": 2.0,
        "num_prompts": 3,
        "completed": 2,
        "failed": 1,
        "input_lens": [10, 20, 30],
        "output_lens": [3, 1, 0],
        "ttfts": [0.1, 0.2, 0],
        "latencies": [0.3, 0.2, 0],
        "errors": ["", "", "connection failed"],
        "queue_times": [0.01, 0.02, None],
        "server_queue_times": [0.07, None, None],
        "server_ttfts": [None, 0.15, None],
    }


def test_vllm_detailed_mixed_cohort_and_distinct_queue_measurements():
    benchmark = read_vllm_benchmark(detailed_vllm(), measurement_profile="fixture")
    first, second, failed = benchmark.requests
    assert [row.status for row in benchmark.requests] == ["completed", "completed", "failed"]
    assert first.client_queue_time == 0.01
    assert first.server_queue_time == 0.07
    assert first.server_ttft is None
    assert second.server_queue_time is None
    assert second.server_ttft == 0.15
    assert failed.latency is None
    assert benchmark.metadata["measurement_profile"] == "fixture"
    data = detailed_vllm()
    del data["queue_times"]
    assert read_vllm_benchmark(data).requests[0].client_queue_time is None


@pytest.mark.parametrize(
    "changes",
    [
        {"completed": 3},
        {"failed": 0},
        {"num_prompts": 2},
        {"latencies": [0.3]},
        {"queue_times": []},
        {"server_ttfts": [None]},
        {"errors": ["", ""]},
    ],
)
def test_vllm_rejects_misaligned_arrays_and_declared_counts(changes):
    data = detailed_vllm()
    data.update(changes)
    with pytest.raises(ValueError):
        read_vllm_benchmark(data)


def test_vllm_rejects_aggregate_only_results():
    with pytest.raises(ValueError, match="per-request"):
        read_vllm_benchmark({"duration": 2, "completed": 100, "mean_ttft_ms": 20})


def test_sglang_stream_span_uses_actual_tokens_for_tpot():
    data = detailed_vllm()
    del data["latencies"]
    data["output_lens"] = [5, 1, 0]
    data["itls"] = [[0.04, 0.08], [], []]
    benchmark = read_sglang_stream_benchmark(data)
    assert benchmark.requests[0].latency == pytest.approx(0.22)
    assert benchmark.requests[0].tpot == pytest.approx(0.03)
    assert benchmark.requests[1].tpot is None
    assert benchmark.metadata["latency_basis"] == "stream_span"


def test_joint_slo_uses_request_intersection_and_all_offered_outcomes():
    rows = (
        ServingRequest(4, 3, "completed", 0.1, 0.3),
        ServingRequest(4, 2, "completed", 0.3, 0.4),
        ServingRequest(4, 2, "completed", 0.1, 0.5),
        ServingRequest(4, 1, "completed", 0.1, 0.1),
        ServingRequest(4, 0, "failed"),
        ServingRequest(4, 0, "pending"),
    )
    report = summarize_serving_slo(ServingBenchmark(rows, 2), ttft_limit=0.2, tpot_limit=0.15)
    assert report["offered"] == 6
    assert report["joint_passed"] == 2
    assert report["joint_unknown"] == 1
    assert report["joint_fraction_bounds"] == pytest.approx([2 / 6, 3 / 6])
    assert report["observed_goodput"] == 1
    assert report["single_token_completed"] == 1
    assert report["completed_ttft_seconds"]["count"] == 4
    assert report["completed_multitoken_tpot_seconds"]["count"] == 3


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "unknown"},
        {"input_tokens": -1},
        {"output_tokens": 1.5},
        {"output_tokens": True},
        {"ttft": None},
        {"latency": None},
        {"ttft": -0.1},
        {"latency": float("nan")},
        {"client_queue_time": float("inf")},
        {"server_queue_time": -1},
        {"latency": 0.05},
        {"output_tokens": 0},
        {"status": "pending"},
    ],
)
def test_invalid_request_measurements(changes):
    values = dict(input_tokens=4, output_tokens=3, status="completed", ttft=0.1, latency=0.3)
    values.update(changes)
    with pytest.raises(ValueError):
        ServingRequest(**values)


def simulator():
    sim = PrefillDecodeSim(batch_cap=1, token_budget=None)
    sim.set_servers(1, 1, alpha=0.003, beta=0.0002, gamma=0.00001)
    return sim


def test_des_conversion_preserves_timestamps_and_censors_without_lookahead():
    arrivals, inputs, outputs = [0, 0.001, 1], [8, 5, 2], [4, 3, 2]
    result = simulator().replay(arrivals, inputs, outputs)
    full = from_prefill_decode(result, arrivals, inputs, outputs)
    for index, row in enumerate(full.requests):
        assert row.ttft == pytest.approx(result.output_times[index][0] - arrivals[index])
        assert row.latency == pytest.approx(result.completion_times[index] - arrivals[index])
    horizon = (result.output_times[0][0] + result.output_times[0][1]) / 2
    censored = from_prefill_decode(result, arrivals, inputs, outputs, horizon=horizon)
    assert len(censored.requests) == 2
    first, queued = censored.requests
    assert first.status == queued.status == "pending"
    assert first.output_tokens == 1
    assert first.ttft == pytest.approx(result.output_times[0][0])
    assert first.latency is None
    assert queued.output_tokens == 0
    assert queued.ttft is queued.latency is None
    assert censored.duration == pytest.approx(horizon)


def oracle_benchmark(lengths):
    """Generate measurements through the independent token simulator."""
    rows = []
    for n, m in lengths:
        result = simulator().replay([0], [n], [m])
        rows.append(
            ServingRequest(n, m, "completed", result.output_times[0][0] + 0.02, result.completion_times[0] + 0.02)
        )
    return ServingBenchmark(
        tuple(rows),
        1,
        {"max_concurrency": 1, "latency_basis": "token_DES_plus_offset", "measurement_profile": "synthetic-oracle"},
    )


@pytest.fixture
def training():
    return oracle_benchmark([(2, 2), (5, 8), (13, 3), (21, 15), (8, 1)])


def test_calibration_recovers_independent_token_des_costs_and_retains_provenance(training):
    fit = fit_isolated_serving([training])
    assert [fit.client_offset, fit.alpha, fit.beta, fit.gamma] == pytest.approx(
        [0.02, 0.003, 0.0002, 0.00001], abs=1e-10
    )
    assert fit.measurement_profile == "synthetic-oracle"
    assert fit.latency_basis == "token_DES_plus_offset"


def test_calibration_rejects_rank_deficiency():
    with pytest.raises(ValueError, match="rank"):
        fit_isolated_serving([oracle_benchmark([(5, 3)] * 5)])


@pytest.mark.parametrize(
    "key,value", [("max_concurrency", 2), ("measurement_profile", "other"), ("latency_basis", "other")]
)
def test_calibration_rejects_mixed_or_nonisolated_measurements(training, key, value):
    other = replace(training, metadata={**training.metadata, key: value})
    with pytest.raises(ValueError):
        fit_isolated_serving([training, other])


@pytest.mark.parametrize("key", ["measurement_profile", "latency_basis"])
def test_calibration_requires_explicit_measurement_provenance(training, key):
    metadata = dict(training.metadata)
    del metadata[key]
    with pytest.raises(ValueError):
        fit_isolated_serving([replace(training, metadata=metadata)])


def test_calibration_rejects_failed_cohorts(training):
    failed = replace(training, requests=training.requests + (ServingRequest(5, 0, "failed"),))
    with pytest.raises(ValueError):
        fit_isolated_serving([failed])


def test_heldout_evaluation_does_not_refit_or_mutate_coefficients(training, monkeypatch):
    fit = fit_isolated_serving([training])
    snapshot = asdict(fit)
    heldout = oracle_benchmark([(3, 4), (12, 6), (30, 10)])
    heldout = replace(
        heldout,
        requests=tuple(replace(row, ttft=row.ttft + 0.01, latency=row.latency + 0.01) for row in heldout.requests),
    )

    def forbidden_refit(*args, **kwargs):
        pytest.fail("Held-out evaluation must not fit training coefficients")

    monkeypatch.setattr(
        "most_queue.theory.continuous_batching.serving_calibration.fit_isolated_serving", forbidden_refit
    )
    report = evaluate_isolated_serving(fit, [heldout])
    assert report["ttft"]["mae_seconds"] == pytest.approx(0.01)
    assert report["tpot"]["mae_seconds"] == pytest.approx(0, abs=1e-10)
    assert asdict(fit) == snapshot
    with pytest.raises(FrozenInstanceError):
        fit.alpha = 1


def test_public_measurement_fixture_hashes_and_request_import():
    import hashlib
    import json
    from pathlib import Path

    folder = Path(__file__).parents[1] / "fixtures" / "serving_measurements" / "flash_next"
    manifest = json.loads((folder / "manifest.json").read_text())
    assert len(manifest["files"]) == 9
    total = 0
    for entry in manifest["files"]:
        path = folder / entry["filename"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        benchmark = read_sglang_stream_benchmark(path, measurement_profile=entry["measurement_profile"])
        assert len(benchmark.requests) == entry["request_count"]
        assert benchmark.metadata["latency_basis"] == "stream_span"
        assert benchmark.metadata["request_rate"] == "inf"
        assert all(row.server_queue_time is None for row in benchmark.requests)
        total += len(benchmark.requests)
    assert total == 54


@pytest.mark.parametrize("duration", [0, -1, float("inf"), float("nan")])
def test_benchmark_requires_positive_finite_duration(duration):
    with pytest.raises(ValueError):
        ServingBenchmark((ServingRequest(2, 0, "pending"),), duration)


@pytest.mark.parametrize("key", ["measurement_profile", "latency_basis"])
def test_heldout_evaluation_rejects_changed_measurement_provenance(training, key):
    fit = fit_isolated_serving([training])
    heldout = oracle_benchmark([(3, 4), (12, 6)])
    heldout = replace(heldout, metadata={**heldout.metadata, key: "different"})
    with pytest.raises(ValueError):
        evaluate_isolated_serving(fit, [heldout])


@pytest.mark.parametrize("lengths", [([2.5], [3]), ([0], [3]), ([2], [1.5]), ([2], [0])])
def test_des_conversion_rejects_nonpositive_or_fractional_lengths(lengths):
    result = simulator().replay([0], [2], [3])
    with pytest.raises(ValueError):
        from_prefill_decode(result, [0], *lengths)


@pytest.mark.parametrize("defect", ["token_count", "completion", "token_order", "before_arrival"])
def test_des_conversion_rejects_inconsistent_timestamp_records(defect):
    from copy import deepcopy

    result = deepcopy(simulator().replay([0.1], [2], [3]))
    if defect == "token_count":
        result.output_times[0].pop(0)
    elif defect == "completion":
        result.completion_times[0] += 0.01
    elif defect == "token_order":
        result.output_times[0][0], result.output_times[0][1] = result.output_times[0][1], result.output_times[0][0]
    else:
        result.output_times[0][0] = 0.09
    with pytest.raises(ValueError):
        from_prefill_decode(result, [0.1], [2], [3])
