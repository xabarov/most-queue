"""Synthetic checks for the prespecified development report and engineering gate."""

# pylint: disable=missing-function-docstring

from copy import deepcopy

import pytest

from examples.serving_input_admission import PRIMARY, VARIANTS, aggregate, brief_cohort, evaluate_gate


def _variant(tight=300, wide=50, mae=1.05):
    metric = {
        "observed_mean_seconds": 1.0,
        "predicted_mean_seconds": 1.1,
        "relative_mean_error": 0.1,
        "mae_seconds": mae,
        "observed": [1.0],
        "predicted": [1.1],
        "observed_quantiles_seconds": [1, 1, 1],
    }
    return {
        "metrics": {"client_ttft": metric},
        "timing": {"ready": [0], "first_output": [1]},
        "joint_client_slo": [
            {
                "ttft_limit_seconds": ttft,
                "tpot_limit_seconds": tpot,
                "offered": 1000,
                "observed_passed": 0,
                "predicted_passed": wide if ttft == 1.0 else tight,
            }
            for ttft in (0.25, 0.5, 1.0)
            for tpot in (0.03, 0.035, 0.04)
        ],
    }


def _cohorts():
    return [
        {
            "dataset": dataset,
            "mode": "open_loop",
            "series_id": f"{dataset}-{index}",
            "requests": 1000,
            "variants": {
                name: _variant(tight=400, mae=1.0) if name == "baseline_v2" else _variant() for name in VARIANTS
            },
        }
        for dataset, count in (("calibration_v1", 6), ("reservation_v2", 6), ("frontend_v1", 4))
        for index in range(count)
    ]


def _set_tight(variant, passed):
    for item in variant["joint_client_slo"]:
        if item["ttft_limit_seconds"] == 0.25:
            item["predicted_passed"] = passed


def test_gate_accepts_exact_maximum_wide_slo_and_mae_boundaries():
    cohorts = _cohorts()
    decision = evaluate_gate(cohorts)
    assert decision["primary_variant"] == PRIMARY == "input_two_batches"
    assert decision["numerical_gate_passed"]
    checks = {item["name"]: item for item in decision["checks"]}
    for name in (
        "tight_slo_max_reduced_by_25_percent",
        "one_second_slo_max_at_most_5_points",
        "calibration_v1_client_ttft_mae_within_5_percent",
        "reservation_v2_client_ttft_mae_within_5_percent",
        "frontend_v1_client_ttft_mae_within_5_percent",
    ):
        assert checks[name]["actual"] == pytest.approx(checks[name]["maximum"])
        assert checks[name]["passed"]


def test_gate_accepts_exact_median_reduction_boundary():
    cohorts = _cohorts()
    for row in cohorts:
        _set_tight(row["variants"][PRIMARY], 320)
    _set_tight(cohorts[0]["variants"]["baseline_v2"], 800)
    decision = evaluate_gate(cohorts)
    assert decision["numerical_gate_passed"]
    check = next(item for item in decision["checks"] if item["name"] == "tight_slo_median_reduced_by_20_percent")
    assert check["actual"] == pytest.approx(check["maximum"])


@pytest.mark.parametrize("change", ["missing", "extra", "wrong_dataset", "wrong_mode"])
def test_gate_requires_complete_six_six_four_series(change):
    cohorts = _cohorts()
    if change == "missing":
        cohorts.pop()
    elif change == "extra":
        cohorts.append(deepcopy(cohorts[0]))
    elif change == "wrong_dataset":
        cohorts[0]["dataset"] = "frontend_v1"
    else:
        cohorts[0]["mode"] = "burst"
    with pytest.raises(ValueError, match="16 open-loop"):
        evaluate_gate(cohorts)


def test_gate_uses_fixed_primary_even_when_ablation_is_perfect():
    cohorts = _cohorts()
    for row in cohorts:
        for name in VARIANTS:
            if name not in ("baseline_v2", PRIMARY):
                row["variants"][name] = _variant(tight=0, wide=0, mae=0)
        _set_tight(row["variants"][PRIMARY], 401)
    decision = evaluate_gate(cohorts)
    assert decision["primary_variant"] == PRIMARY
    assert not decision["numerical_gate_passed"]


def test_gate_fails_immediately_above_wide_slo_limit():
    cohorts = _cohorts()
    for point in cohorts[0]["variants"][PRIMARY]["joint_client_slo"]:
        if point["ttft_limit_seconds"] == 1:
            point["predicted_passed"] = 51
    decision = evaluate_gate(cohorts)
    assert not decision["numerical_gate_passed"]
    failed = [check["name"] for check in decision["checks"] if not check["passed"]]
    assert failed == ["one_second_slo_max_at_most_5_points"]


def test_aggregate_request_weights_mae_and_keeps_cohort_errors_unweighted():
    cohorts = _cohorts()[:2]
    cohorts[0]["requests"], cohorts[1]["requests"] = 1, 3
    for name in VARIANTS:
        first = cohorts[0]["variants"][name]["metrics"]["client_ttft"]
        second = cohorts[1]["variants"][name]["metrics"]["client_ttft"]
        first.update(mae_seconds=2, relative_mean_error=0.2)
        second.update(mae_seconds=6, relative_mean_error=0.6)
    report = aggregate(cohorts)
    for dataset in ("calibration_v1", "all"):
        group = report[dataset]["open_loop"]
        assert (group["series"], group["requests"]) == (2, 4)
        for variant in group["variants"].values():
            metric = variant["metrics"]["client_ttft"]
            assert metric["request_mae_seconds"] == 5
            assert metric["median_cohort_relative_mean_error"] == pytest.approx(0.4)
            assert metric["maximum_cohort_relative_mean_error"] == 0.6
    assert report["frontend_v1"] == {}


def test_gate_mae_is_request_weighted_within_each_dataset():
    cohorts = _cohorts()
    rows = [row for row in cohorts if row["dataset"] == "calibration_v1"]
    for row in rows:
        row["requests"] = 1
        row["variants"][PRIMARY]["metrics"]["client_ttft"]["mae_seconds"] = 0
    rows[0]["requests"] = 100
    rows[0]["variants"][PRIMARY]["metrics"]["client_ttft"]["mae_seconds"] = 2
    decision = evaluate_gate(cohorts)
    check = next(
        item for item in decision["checks"] if item["name"] == "calibration_v1_client_ttft_mae_within_5_percent"
    )
    assert check["actual"] == pytest.approx(200 / 105)
    assert not check["passed"]


def test_brief_removes_raw_arrays_and_preserves_full_grid_without_mutation():
    cohort = _cohorts()[0]
    before = deepcopy(cohort)
    brief = brief_cohort(cohort)
    assert cohort == before
    assert brief["series_id"] == cohort["series_id"]
    for name, variant in brief["variants"].items():
        assert "timing" not in variant
        assert variant["joint_client_slo"] == cohort["variants"][name]["joint_client_slo"]
        assert len(variant["joint_client_slo"]) == 9
        metric = variant["metrics"]["client_ttft"]
        assert set(metric) == {"observed_mean_seconds", "predicted_mean_seconds", "relative_mean_error", "mae_seconds"}
