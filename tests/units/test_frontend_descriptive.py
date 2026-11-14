"""Synthetic arithmetic checks; never read an experiment or contact a provider."""

import copy
import math

import pytest

from tools.frontend_descriptive import pair_comparisons, request_metrics, summarize, summarize_pairs


def record(trace=False, pair_id="pair-a", block_id="block-a", condition="idle", input_tokens=128):
    """Return a hand-calculated synthetic request on one clock."""
    return {
        "success": True,
        "output_tokens": 3,
        "input_tokens": input_tokens,
        "client_ttft_s": 0.5 + (0.125 if trace else 0),
        "latency_to_done_s": 1.5,
        "scheduled_arrival_monotonic_s": 9.5,
        "dispatch_started_monotonic_s": 9.75,
        "client_started_monotonic_s": 10.0,
        "server_timestamps": {
            "queued_ts": 10.125,
            "scheduled_ts": 10.25,
            "first_token_ts": 10.375,
            "last_token_ts": 11.375,
        },
        "trace_enabled": trace,
        "role": "probe",
        "phase": "measured",
        "pair_id": pair_id,
        "block_id": block_id,
        "condition": condition,
        "prompt_sha256": "a" * 64,
        "request_id": pair_id + ("-on" if trace else "-off"),
    }


def test_manual_arithmetic():
    """Timing components match arithmetic independent of the implementation."""
    assert request_metrics(record()) == {
        "client_ttft": 0.5,
        "server_ttft": 0.25,
        "client_tpot": 0.5,
        "server_tpot": 0.5,
        "queue": 0.125,
        "outside_engine": 0.25,
        "pre_engine": 0.125,
        "first_output_delivery": 0.125,
        "dispatch_to_worker_start": 0.25,
        "dispatch_lateness": 0.25,
    }


def test_summary_quartiles_and_singleton():
    """Quartiles use linear sample interpolation and preserve all observations."""
    assert summarize([4, 1, 3, 2]) == {
        "count": 4,
        "minimum": 1.0,
        "q25": 1.75,
        "median": 2.5,
        "q75": 3.25,
        "maximum": 4.0,
        "mean": 2.5,
    }
    assert summarize([2])["q25"] == summarize([2])["q75"] == 2.0
    assert summarize([-2, 2])["mean"] == 0.0


@pytest.mark.parametrize("values", [[], [math.nan], [math.inf], [-math.inf], [True], ["1"]])
def test_summary_rejects_invalid_values(values):
    """No missing/nonfinite observations are silently excluded."""
    with pytest.raises(ValueError):
        summarize(values)


@pytest.mark.parametrize(
    "field,value",
    [
        ("success", False),
        ("output_tokens", 1),
        ("output_tokens", 2.5),
        ("client_ttft_s", math.nan),
        ("latency_to_done_s", -1),
        ("latency_to_done_s", 0.25),
        ("client_started_monotonic_s", 11),
    ],
)
def test_request_rejects_invalid_values(field, value):
    """Invalid success, counts, durations and clock ordering stop arithmetic."""
    row = record()
    row[field] = value
    with pytest.raises(ValueError):
        request_metrics(row)


def test_clock_tolerance():
    """Only tiny clock inversions are accepted and clamped to zero."""
    row = record()
    row["dispatch_started_monotonic_s"] = 10 + 5e-9
    assert request_metrics(row)["dispatch_to_worker_start"] == 0.0
    row["dispatch_started_monotonic_s"] = 10 + 2e-8
    with pytest.raises(ValueError):
        request_metrics(row)


def test_pair_sign_and_exclusions():
    """Pairs preserve controls and use on-minus-off even with reversed input order."""
    ignored = [dict(record(), role=role) for role in ("warmup", "background", "arrival")]
    ignored.append(dict(record(), phase="warmup"))
    pairs = pair_comparisons([record(True), *ignored, record(False)])
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["request_id_on"] == "pair-a-on"
    assert pair["request_id_off"] == "pair-a-off"
    assert pair["difference"] == {
        "client_ttft": 0.125,
        "server_ttft": 0,
        "client_tpot": -0.0625,
        "server_tpot": 0,
        "queue": 0,
        "outside_engine": 0.125,
    }


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "same_control",
        "prompt_sha256",
        "input_tokens",
        "output_tokens",
        "block_id",
        "condition",
        "failed",
        "nonboolean",
    ],
)
def test_bad_pairs(change):
    """Incomplete, unequal, failed and ambiguous controls remain errors."""
    rows = [record(False), record(True)]
    if change == "missing":
        rows.pop()
    elif change == "extra":
        rows.append(copy.deepcopy(rows[0]))
    elif change == "same_control":
        rows[1]["trace_enabled"] = False
    elif change == "failed":
        rows[1]["success"] = False
    elif change == "nonboolean":
        rows[1]["trace_enabled"] = 1
    else:
        rows[1][change] = "different"
    with pytest.raises(ValueError):
        pair_comparisons(rows)


def test_groups_retain_blocks_lengths_and_conditions():
    """Pooling retains both blocks and supports explicit per-block filtering."""
    rows = []
    for block in ("a", "b"):
        for condition in ("idle", "loaded"):
            for length in (128, 1024):
                for trace in (False, True):
                    rows.append(record(trace, f"{block}-{condition}-{length}", block, condition, length))
    pairs = pair_comparisons(rows)
    groups = summarize_pairs(pairs)
    assert len(groups) == 4
    assert all(group["blocks"] == ["a", "b"] for group in groups)
    assert all(group["pair_count"] == 2 and group["expected_request_count"] == 4 for group in groups)
    assert sum(group["pair_count"] for group in groups) == len(pairs)
    assert all(group["difference"]["client_ttft"]["count"] == 2 for group in groups)
    assert all(group["warning"] for group in groups)
    block_a = summarize_pairs(pair for pair in pairs if pair["block_id"] == "a")
    assert all(group["blocks"] == ["a"] and group["pair_count"] == 1 for group in block_a)


@pytest.mark.parametrize("mean_off,threshold", [(0.01, 0.005), (1.0, 0.05)])
def test_warning_uses_strict_threshold(mean_off, threshold):
    """Equality is accepted; either sign beyond the fixed threshold warns."""
    pair = pair_comparisons([record(False), record(True)])[0]
    pair["metrics_off"]["client_ttft"] = mean_off
    for difference, warning in (
        (threshold, False),
        (-threshold, False),
        (threshold + 1e-9, True),
        (-threshold - 1e-9, True),
    ):
        pair["difference"]["client_ttft"] = difference
        result = summarize_pairs([pair])[0]
        assert result["warning_threshold"] == threshold
        assert result["warning"] is warning
