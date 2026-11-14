"""Independent interval arithmetic and saved-evidence checks for SLO diagnostics."""

import copy
import gzip
import json

import pytest

from examples.serving_slo_diagnostics import (
    COMPONENTS,
    FIXTURE,
    FOCUS,
    REPORT,
    ROOT,
    arrival_occupancy,
    build_diagnostics,
    decompose_request,
    outcomes,
)


@pytest.fixture
def timing_pair():
    """A common-clock trace with independently chosen client and engine intervals."""
    record = {
        "request_index": 7,
        "input_tokens": 12,
        "output_tokens": 5,
        "success": True,
        "client_started_monotonic_s": 100.0,
        "client_ttft_s": 0.22,
        "latency_to_done_s": 0.65,
        "dispatch_started_monotonic_s": 99.8,
        "scheduled_arrival_monotonic_s": 99.7,
        "server_timestamps": {
            "timestamp_clock": "monotonic",
            "queued_ts": 100.04,
            "scheduled_ts": 100.05,
            "first_token_ts": 100.2,
            "last_token_ts": 100.6,
        },
    }
    predicted = {
        "server_queue_wait": 0.005,
        "server_scheduled_to_first_token": 0.14,
        "server_ttft": 0.145,
        "client_ttft": 0.185,
        "client_tpot": 0.1,
        "server_tpot": 0.1,
    }
    return record, predicted


def test_hand_computed_ttft_and_tpot_identities_exclude_dispatch(timing_pair):
    record, predicted = timing_pair
    row = decompose_request(record, predicted)
    # Queue .01, scheduled .15, outside .06: client first output is at 100.22.
    assert row["components"]["queue"] == pytest.approx({"observed": 0.01, "predicted": 0.005, "error": 0.005})
    assert row["components"]["scheduled_to_first"] == pytest.approx(
        {"observed": 0.15, "predicted": 0.14, "error": 0.01}
    )
    assert row["components"]["outside_engine"] == pytest.approx({"observed": 0.06, "predicted": 0.04, "error": 0.02})
    assert row["ttft_error"] == pytest.approx(0.035, abs=1e-12)
    assert sum(component["error"] for component in row["components"].values()) == pytest.approx(
        row["ttft_error"], abs=1e-12
    )
    assert row["pre_engine"] == pytest.approx(0.04)
    assert row["first_output_delivery"] == pytest.approx(0.02)
    assert row["after_last_engine_token_to_done"] == pytest.approx(0.05)
    assert row["observed_tpot"] == pytest.approx(0.1075)
    assert row["client_minus_server_tpot"] == pytest.approx(0.0075)
    assert row["observed_tpot"] - 0.1 == pytest.approx(row["client_minus_server_tpot"], abs=1e-12)
    assert row["dispatch_to_worker_start"] == pytest.approx(0.2)
    assert row["dispatch_lateness"] == pytest.approx(0.1)
    record["dispatch_started_monotonic_s"] = 90.0
    changed = decompose_request(record, predicted)
    assert changed["components"] == row["components"]
    assert changed["observed_ttft"] == row["observed_ttft"]
    assert changed["dispatch_to_worker_start"] == 10.0


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("server", "scheduled_ts", 100.03),
        ("server", "last_token_ts", 100.1),
        ("server", "first_token_ts", float("nan")),
        ("server", "timestamp_clock", "wall"),
        ("record", "client_ttft_s", float("inf")),
        ("record", "latency_to_done_s", 0.2),
        ("record", "success", False),
        ("record", "output_tokens", 1),
        ("record", "client_started_monotonic_s", 100.1),
        ("record", "client_ttft_s", 0.15),
        ("prediction", "client_tpot", float("nan")),
        ("prediction", "server_ttft", 0.15),
        ("prediction", "client_ttft", 0.1),
    ],
)
def test_reject_invalid_timing_contract(timing_pair, section, key, value):
    record, predicted = timing_pair
    target = {"server": record["server_timestamps"], "record": record, "prediction": predicted}[section]
    target[key] = value
    with pytest.raises(ValueError):
        decompose_request(record, predicted)


def test_reject_missing_clock(timing_pair):
    record, predicted = timing_pair
    del record["server_timestamps"]["timestamp_clock"]
    with pytest.raises(ValueError, match="monotonic"):
        decompose_request(record, predicted)


def test_arrival_occupancy_uses_half_open_intervals_and_excludes_self():
    # Target is scheduled immediately at time 5, so counting it would add one.
    records = [
        {"server_timestamps": dict(zip(("queued_ts", "scheduled_ts", "last_token_ts"), stamps))}
        for stamps in [(5, 5, 8), (1, 2, 5), (2, 5, 9), (5, 6, 10), (6, 7, 11), (1, 3, 8)]
    ]
    assert arrival_occupancy(records, 0) == {
        "scheduled_unfinished_at_arrival": 2,
        "queued_unscheduled_at_arrival": 1,
    }
    assert arrival_occupancy(records[:1], 0) == {
        "scheduled_unfinished_at_arrival": 0,
        "queued_unscheduled_at_arrival": 0,
    }


def outcome_row(index, observed_ttft, observed_tpot, actual, predicted, predicted_tpot=0.08):
    """Create accounting rows with explicit interval triples in seconds."""
    return {
        "request_index": index,
        "observed_ttft": observed_ttft,
        "observed_tpot": observed_tpot,
        "predicted_tpot": predicted_tpot,
        "components": {name: {"observed": actual[i], "predicted": predicted[i]} for i, name in enumerate(COMPONENTS)},
    }


@pytest.fixture
def outcome_rows():
    return [
        outcome_row(0, 1.0, 0.1, (0.25, 0.5, 0.25), (0.25, 0.5, 0.25), 0.1),
        outcome_row(1, 1.2, 0.08, (0.4, 0.6, 0.2), (0.2, 0.4, 0.2)),
        outcome_row(2, 0.8, 0.12, (0.2, 0.4, 0.2), (0.2, 0.4, 0.2)),
        outcome_row(3, 1.2, 0.12, (0.4, 0.6, 0.2), (0.2, 0.4, 0.2)),
        outcome_row(4, 0.8, 0.08, (0.2, 0.4, 0.2), (0.4, 0.6, 0.2)),
    ]


def test_outcomes_inclusive_boundaries_and_false_pass_causes(outcome_rows):
    result = outcomes(outcome_rows, 1.0, 0.1)
    assert result == {
        "offered": 5,
        "observed_passed": 2,
        "predicted_passed": 4,
        "false_passes": [1, 2, 3],
        "false_failures": [4],
        "false_pass_causes": {"ttft_only": 1, "tpot_only": 1, "both": 1},
        "signed_fraction_error": 0.4,
    }


@pytest.mark.parametrize(
    "components,gap,false_passes,false_failures,passed",
    [
        (("queue",), False, [1, 2, 3], [], 5),
        (("scheduled_to_first",), False, [1, 2, 3], [], 5),
        (("queue", "scheduled_to_first"), False, [2], [], 3),
        ((), True, [1], [4], 2),
        (COMPONENTS, True, [], [], 2),
    ],
)
def test_observed_component_substitutions_have_known_hand_counts(
    outcome_rows, components, gap, false_passes, false_failures, passed
):
    result = outcomes(outcome_rows, 1.0, 0.1, observed_components=components, observed_tpot=gap)
    assert result["false_passes"] == false_passes
    assert result["false_failures"] == false_failures
    assert result["predicted_passed"] == passed
    assert result["observed_passed"] == 2
    assert result["offered"] == 5


def test_unknown_component_rejected(outcome_rows):
    with pytest.raises(ValueError, match="unknown TTFT component"):
        outcomes(outcome_rows, 1.0, 0.1, observed_components=("dispatch",))


def test_saved_raw_join_preserves_every_evaluation_request_and_focus_id():
    report = json.loads(REPORT.read_text())
    with gzip.open(FIXTURE / "requests.jsonl.gz", "rt") as stream:
        records = [json.loads(line) for line in stream]
    assert len(records) == 690
    diagnostic = build_diagnostics(report, records)
    saved = json.loads((ROOT / "docs/research/serving-slo-v2-diagnostics.json").read_text())
    assert diagnostic == {key: value for key, value in saved.items() if key != "provenance"}
    expected_ids = {record["request_index"] for record in records if record["split"] == "validation"}
    ids = [row["request_index"] for row in diagnostic["requests"]]
    assert len(ids) == len(set(ids)) == len(expected_ids) == 669
    assert set(ids) == expected_ids
    focus = diagnostic["focus"]
    assert focus["series_id"] == FOCUS
    assert focus["false_passes"] == [
        292,
        293,
        296,
        297,
        299,
        301,
        302,
        306,
        314,
        315,
        316,
        327,
        332,
        333,
        334,
        336,
        338,
        339,
        340,
        343,
        349,
        354,
        357,
        359,
        361,
        362,
        366,
        367,
        372,
    ]
    assert focus["false_pass_causes"] == {"ttft_only": 29, "tpot_only": 0, "both": 0}
    focus_rows = [row for row in diagnostic["requests"] if row["series_id"] == FOCUS]
    assert focus["offered"] == len(focus_rows) == 91
    for row in focus_rows:
        if row["request_index"] in focus["false_passes"]:
            assert row["observed_ttft"] > 0.25
            assert row["observed_tpot"] <= 0.035
    full_observations = outcomes(focus_rows, 0.25, 0.035, observed_components=COMPONENTS, observed_tpot=True)
    assert full_observations["predicted_passed"] == full_observations["observed_passed"]
    assert full_observations["false_passes"] == full_observations["false_failures"] == []


def test_duplicate_raw_identity_rejected():
    with gzip.open(FIXTURE / "requests.jsonl.gz", "rt") as stream:
        record = json.loads(next(stream))
    with pytest.raises(ValueError, match="duplicate raw request identity"):
        build_diagnostics({"cohorts": []}, [record, copy.deepcopy(record)])
