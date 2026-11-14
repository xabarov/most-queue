#!/usr/bin/env python3
"""Audit a local Runpod trace and evaluate the frozen serving calibration plan.

The default pilot supports an exploratory isolated fit without held-out
validation. A new complete campaign needs its saved plan and summary.
No GPU, network access, or coefficient retuning is performed by this example.
"""

# Exact integer checks also reject bool, which isinstance accepts as int.
# pylint: disable=unidiomatic-typecheck

import argparse
import gzip
import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from most_queue.io.runpod import read_runpod_trace
from most_queue.theory.continuous_batching.serving_calibration import evaluate_isolated_serving, fit_isolated_serving
from most_queue.theory.continuous_batching.serving_validation import replay_serving_cohort

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE = ROOT / "tests/fixtures/serving_measurements/runpod_qwen3_8b_a40_20261114/requests.jsonl.gz"
DEFAULT_PROFILE = "runpod-qwen3-8b-a40-smoke-20261114"
CAMPAIGN_SEED = 2026111402
OPEN_LOOP_COUNTS = {
    (0.2, 54101): 35,
    (0.2, 54102): 41,
    (0.5, 54101): 78,
    (0.5, 54102): 89,
    (0.8, 54101): 147,
    (0.8, 54102): 166,
}
SPEC_FIELDS = (
    "series_id",
    "request_id",
    "phase",
    "split",
    "mode",
    "repetition",
    "concurrency",
    "request_rate",
    "input_tokens",
    "output_tokens",
    "seed",
    "arrival_offset_s",
)
DESIGN_FIELDS = ("split", "mode", "phase", "input_tokens", "output_tokens", "repetition", "concurrency", "request_rate")
SCOPE = (
    "Effective isolated-request coefficients and finite cohorts from an empty server; "
    "specified PrefillDecodeSim policy, not exact vLLM scheduling. "
    "No steady-state, p99, SLO, or confidence-interval guarantees. Held-out outcomes never retune coefficients."
)


def _read_records(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _provenance(path):
    path = path.resolve()
    public_path = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name
    return {"path": public_path, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _observed(values):
    values = [value for value in values if value is not None]
    return {
        "count": len(values),
        "mean_seconds": sum(value / len(values) for value in values) if values else None,
        "minimum_seconds": min(values) if values else None,
        "maximum_seconds": max(values) if values else None,
    }


def _cohort_report(benchmark):
    rows, metadata = benchmark.requests, benchmark.metadata
    return {
        "series_id": metadata["series_id"],
        "phase": metadata["phase"],
        "split": metadata["split"],
        "mode": metadata["mode"],
        "repetition": metadata["repetition"],
        "max_concurrency": metadata["max_concurrency"],
        "request_rate": metadata["request_rate"],
        "requests": len(rows),
        "statuses": dict(Counter(row.status for row in rows)),
        "duration_seconds": benchmark.duration,
        "request_indices": metadata["request_indices"],
        "requested_inputs": metadata["requested_inputs"],
        "requested_outputs": metadata["requested_outputs"],
        "client_arrival_times": metadata["arrival_times"],
        "server_arrival_times": metadata["server_arrival_times"],
        "dispatch_lateness_s": metadata["dispatch_lateness_s"],
        "errors": metadata["errors"],
        "observed": {
            "client_ttft": _observed(row.ttft for row in rows),
            "server_ttft": _observed(row.server_ttft for row in rows),
            "server_tpot": _observed(row.server_tpot for row in rows),
            "server_queue_wait": _observed(row.server_queue_time for row in rows),
        },
    }


def _design_counts():
    result = Counter()

    def add(
        split, mode, n, m, repetition, concurrency, count, rate=None
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        phase = "warmup" if split == "warmup" else "measured"
        result[(split, mode, phase, n, m, repetition, concurrency, rate)] += count

    for n in (128, 1024, 4096):
        for m in (32, 128, 512):
            add("warmup", "isolated", n, m, 0, 1, 1)
            for repetition in (1, 2, 3):
                add("train", "isolated", n, m, repetition, 1, 1)
            add("validation", "isolated", n, m, 4, 1, 1)
            for repetition in (1, 2):
                for concurrency in (2, 4):
                    add("validation", "burst", n, m, repetition, concurrency, concurrency)
        add("warmup", "burst", n, 32, 0, 4, 4)
    for (rate, repetition), count in OPEN_LOOP_COUNTS.items():
        add("validation", "open_loop", 1024, 128, repetition, None, count, rate)
    return result


def _clock(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _arrival_plan_errors(series):
    specs = series["requests"]
    first = specs[0]
    series_id = series["series_id"]
    if not _clock(series.get("duration_s")) or any(not _clock(spec.get("arrival_offset_s")) for spec in specs):
        return [f"planned series {series_id} requires finite nonnegative duration and arrival offsets"]
    if first.get("mode") != "open_loop":
        if series.get("duration_s") != 0 or any(spec.get("arrival_offset_s") != 0 for spec in specs):
            return [f"planned series {series_id} must have zero duration and arrival offsets"]
        return []
    rate, arrival_seed = first.get("request_rate"), series.get("arrival_seed")
    if (
        type(arrival_seed) is not int
        or arrival_seed != first.get("repetition")
        or rate not in (0.2, 0.5, 0.8)
        or series.get("duration_s") != 180
    ):
        return [f"planned series {series_id} has an invalid frozen open-loop arrival seed, rate or duration"]
    rate_index = (0.2, 0.5, 0.8).index(rate)
    generator = random.Random(CAMPAIGN_SEED + rate_index * 1000 + arrival_seed)
    expected_offsets = []
    offset = generator.expovariate(rate)
    while offset < 180:
        expected_offsets.append(offset)
        offset += generator.expovariate(rate)
    if [spec.get("arrival_offset_s") for spec in specs] != expected_offsets:
        return [f"planned series {series_id} differs from its frozen exponential arrival schedule"]
    return []


def _request_clock_errors(record, spec, series_summary):
    index = record["request_index"]
    start = series_summary.get("scheduled_start_monotonic_s")
    scheduled = record.get("scheduled_arrival_monotonic_s")
    dispatch = record.get("dispatch_started_monotonic_s")
    if not all(_clock(value) for value in (start, scheduled, dispatch, record.get("arrival_offset_s"))):
        return [f"request {index} requires finite nonnegative scheduled-start, arrival and dispatch clocks"]
    errors = []
    if (
        spec is None
        or not _clock(spec.get("arrival_offset_s"))
        or not math.isclose(scheduled, start + spec["arrival_offset_s"], rel_tol=0, abs_tol=1e-7)
    ):
        errors.append(f"request {index} scheduled arrival disagrees with its saved series start and offset")
    if dispatch < scheduled - 1e-7:
        errors.append(f"request {index} was dispatched before its scheduled arrival")
    return errors


def _campaign_audit(records, plan, summary):
    errors = []
    if type(plan.get("seed")) is not int or plan["seed"] != CAMPAIGN_SEED:
        errors.append(f"plan seed must equal the frozen campaign seed {CAMPAIGN_SEED}")
    series = plan.get("series", [])
    if not isinstance(series, list) or any(not isinstance(item, dict) for item in series):
        return ["plan.series must be a list of series objects"]
    specifications = []
    series_sizes = {}
    for item in series:
        specs = item.get("requests")
        series_id = item.get("series_id")
        if not isinstance(specs, list) or not specs or any(not isinstance(spec, dict) for spec in specs):
            errors.append("plan contains an empty or malformed request series")
            continue
        if not isinstance(series_id, str) or series_id in series_sizes:
            errors.append("plan series IDs must be unique strings")
        series_sizes[series_id] = len(specs)
        if any(spec.get("series_id") != series_id for spec in specs):
            errors.append("planned request series_id disagrees with its enclosing series")
        first = specs[0]
        expected_size = (
            OPEN_LOOP_COUNTS.get((first.get("request_rate"), first.get("repetition")))
            if first.get("mode") == "open_loop"
            else first.get("concurrency")
        )
        if len(specs) != expected_size or any(
            tuple(spec.get(key) for key in DESIGN_FIELDS) != tuple(first.get(key) for key in DESIGN_FIELDS)
            for spec in specs
        ):
            errors.append(f"planned series {series_id} does not match its frozen cohort design")
        errors.extend(_arrival_plan_errors(item))
        specifications.extend(specs)
    planned_indices = [spec.get("request_index") for spec in specifications]
    actual_indices = [record.get("request_index") for record in records]
    if any(type(index) is not int for index in planned_indices + actual_indices):
        errors.append("request indices must be integers")
        return errors
    if len(set(planned_indices)) != len(planned_indices) or set(planned_indices) != set(range(1, 722)):
        errors.append("plan must contain exactly the 721 unique planned request indices")
    if len(actual_indices) != len(set(actual_indices)):
        errors.append("trace contains duplicate request indices")
    missing = sorted(set(planned_indices) - set(actual_indices))
    extra = sorted(set(actual_indices) - set(planned_indices))
    if missing:
        errors.append(f"missing planned outcomes: {missing}")
    if extra:
        errors.append(f"unplanned outcomes: {extra}")
    if plan.get("expected") != 721 or len(specifications) != 721 or len(records) != 721 or len(series) != 90:
        errors.append("campaign must contain all 721 outcomes in the frozen 90 series")
    if Counter(tuple(spec.get(key) for key in DESIGN_FIELDS) for spec in specifications) != _design_counts():
        errors.append("plan differs from the frozen length cells, repetitions, or six open-loop request counts")
    by_index = {spec["request_index"]: spec for spec in specifications}
    for spec in specifications:
        index = spec["request_index"]
        if type(spec.get("seed")) is not int or spec["seed"] != CAMPAIGN_SEED + index:
            errors.append(f"planned request {index} seed differs from campaign seed + request index")
        if spec.get("request_id") != f"mqcal-{index}":
            errors.append(f"planned request {index} ID differs from mqcal-<request index>")
    results = summary.get("series")
    summary_by_series = (
        {result.get("series_id"): result for result in results if isinstance(result, dict)}
        if isinstance(results, list)
        else {}
    )
    for record in records:
        spec = by_index.get(record["request_index"])
        if spec is not None and any(record.get(key) != spec.get(key) for key in SPEC_FIELDS):
            errors.append(f"request {record['request_index']} differs from its planned specification")
        if record.get("offer_status") != "OFFERED" or (spec is not None and spec.get("offer_status") != "OFFERED"):
            errors.append(f"request {record['request_index']} is not recorded as OFFERED in both plan and trace")
        expected_status = "succeeded" if record.get("success") is True else "failed"
        if record.get("status") != expected_status:
            errors.append(f"request {record['request_index']} status contradicts its success outcome")
        if record.get("success") is True and not isinstance(record.get("server_timestamps"), dict):
            errors.append(f"request {record['request_index']} has no exact engine timestamps")
        errors.extend(_request_clock_errors(record, spec, summary_by_series.get(record.get("series_id"), {})))
    successes = sum(record.get("success") is True for record in records)
    measured = [record for record in records if record.get("phase") == "measured"]
    expected_summary = {
        "expected": 721,
        "completed": len(records),
        "offered": len(records),
        "successful": successes,
        "not_offered": 721 - len(records),
        "measured_completed": len(measured),
        "measured_successful": sum(record.get("success") is True for record in measured),
    }
    if any(type(summary.get(key)) is not int or summary[key] != value for key, value in expected_summary.items()):
        errors.append("summary counts disagree with the saved plan and request outcomes")
    if successes != 721 or summary.get("experiment_valid") is not True:
        errors.append("failed or partial campaign: all planned outcomes and experiment_valid=true are required")
    if summary.get("deadline_exhausted") is not False or summary.get("abort_reason") is not None:
        errors.append("campaign ended with a deadline or abort diagnostic")
    if not isinstance(results, list) or len(results) != len(series):
        errors.append("summary must retain all planned series outcomes")
    else:
        seen = set()
        actual_by_series = Counter(record.get("series_id") for record in records)
        successful_by_series = Counter(record.get("series_id") for record in records if record.get("success") is True)
        for result in results:
            if not isinstance(result, dict):
                errors.append("summary series outcomes must be objects")
                continue
            series_id = result.get("series_id")
            if series_id in seen or series_id not in series_sizes:
                errors.append("summary series IDs do not match the saved plan")
            seen.add(series_id)
            expected = {
                "planned": series_sizes.get(series_id),
                "offered": actual_by_series[series_id],
                "successful": successful_by_series[series_id],
                "failed": actual_by_series[series_id] - successful_by_series[series_id],
            }
            if any(type(result.get(key)) is not int or result[key] != value for key, value in expected.items()):
                errors.append(f"summary series {series_id} counts contradict request outcomes")
            if result.get("not_offered_request_indices") != [] or result.get("deadline_exhausted") is not False:
                errors.append(f"summary series {series_id} is incomplete or exhausted its deadline")
            if not all(
                _clock(result.get(key))
                for key in ("scheduled_start_monotonic_s", "scheduled_end_monotonic_s", "completed_monotonic_s")
            ):
                errors.append(f"summary series {series_id} requires finite nonnegative clocks")
    return errors


def _finite(value):
    # Also catches nonfinite model outputs before they enter strict JSON.
    json.dumps(value, allow_nan=False)
    return value


def build_report(source, *, measurement_profile=DEFAULT_PROFILE, plan_path=None, summary_path=None):
    """Audit all saved outcomes, fit training only, and apply the frozen gates."""
    source = Path(source)
    records = _read_records(source)
    if not records or any(not isinstance(record, dict) for record in records):
        raise ValueError("trace must contain request objects")
    new_campaign = any("series_id" in record for record in records)
    report = {
        "schema": "runpod-serving-calibration-report-v1",
        "scope": SCOPE,
        "measurement_profile": measurement_profile,
        "provenance": {"trace": _provenance(source), "frozen_plan": "docs/research/serving-calibration-v1-plan.md"},
        "campaign_kind": "calibration_v1" if new_campaign else "legacy_pilot",
        "experiment_valid": False,
        "validation_passed": None,
        "requests": len(records),
        "statuses": dict(Counter("completed" if row.get("success") is True else "failed" for row in records)),
        "outcomes": [
            {
                "request_index": row.get("request_index"),
                "series_id": row.get("series_id"),
                "status": row.get("status", "completed" if row.get("success") is True else "failed"),
                "success": row.get("success"),
                "error": row.get("error"),
            }
            for row in records
        ],
        "diagnostics": [],
        "cohorts": [],
        "fit": None,
        "isolated_validation": None,
        "burst_validation": [],
        "open_loop_validation": [],
    }
    diagnostics = report["diagnostics"]
    try:
        cohorts = read_runpod_trace(records, measurement_profile=measurement_profile)
        report["cohorts"] = [_cohort_report(cohort) for cohort in cohorts]
    except ValueError as exc:
        diagnostics.append(f"trace import failed; all raw outcomes retained: {exc}")
        cohorts = []
    if not new_campaign:
        report["calibration_status"] = "exploratory_only_no_holdout"
        diagnostics.append(
            "exploratory legacy fit only: no prespecified held-out split; "
            "held-out validation and DES replay are not reported"
        )
        if any(value is None for cohort in cohorts for value in cohort.metadata["server_arrival_times"]):
            diagnostics.append("exact server arrivals are absent; client arrivals are not substituted")
        legacy_isolated = [
            cohort
            for cohort in cohorts
            if cohort.metadata["phase"] == "measured"
            and cohort.metadata["mode"] == "isolated"
            and cohort.metadata["max_concurrency"] == 1
        ]
        try:
            if not cohorts or any(row.status != "completed" for cohort in cohorts for row in cohort.requests):
                raise ValueError("legacy exploratory fit requires a completely successful imported trace")
            report["fit"] = _finite(asdict(fit_isolated_serving(legacy_isolated)))
        except ValueError as exc:
            report["calibration_status"] = "exploratory_fit_unavailable"
            diagnostics.append(f"legacy exploratory fit unavailable: {exc}")
        return _finite(report)
    report["calibration_status"] = "invalid_campaign"
    if measurement_profile == DEFAULT_PROFILE:
        diagnostics.append("supply an explicit measurement profile for the timestamp-instrumented campaign")
    if plan_path is None or summary_path is None:
        diagnostics.append("new calibration campaigns require both --plan and --summary")
        return _finite(report)
    plan_path, summary_path = Path(plan_path), Path(summary_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report["provenance"].update(plan=_provenance(plan_path), workload_summary=_provenance(summary_path))
    if not isinstance(plan, dict) or not isinstance(summary, dict):
        diagnostics.append("plan and summary must be JSON objects")
    else:
        report["workload_summary"] = {
            key: summary.get(key)
            for key in (
                "expected",
                "completed",
                "offered",
                "not_offered",
                "successful",
                "measured_completed",
                "measured_successful",
                "experiment_valid",
                "abort_reason",
                "deadline_exhausted",
            )
        }
        try:
            diagnostics.extend(_campaign_audit(records, plan, summary))
        except (TypeError, KeyError, ValueError) as exc:
            diagnostics.append(f"malformed campaign plan or summary: {exc}")
    if diagnostics:
        return _finite(report)
    report["experiment_valid"] = True
    training = [cohort for cohort in cohorts if cohort.metadata["split"] == "train"]
    isolated = [
        cohort
        for cohort in cohorts
        if cohort.metadata["split"] == "validation" and cohort.metadata["mode"] == "isolated"
    ]
    try:
        fit = fit_isolated_serving(training)
        report["fit"] = _finite(asdict(fit))
        if fit.alpha <= 0:
            report["calibration_status"] = "nonpositive_alpha"
            diagnostics.append("nonpositive alpha: fitted coefficients are diagnostic only; DES replay is refused")
            return report
        isolated_report = _finite(evaluate_isolated_serving(fit, isolated))
        report["isolated_validation"] = isolated_report
        for cohort in cohorts:
            if cohort.metadata["split"] != "validation" or cohort.metadata["mode"] not in {"burst", "open_loop"}:
                continue
            result = _finite(replay_serving_cohort(fit, cohort, batch_cap=4, token_budget=18432))
            result.update(
                series_id=cohort.metadata["series_id"],
                request_rate=cohort.metadata["request_rate"],
                repetition=cohort.metadata["repetition"],
            )
            report[cohort.metadata["mode"] + "_validation"].append(result)
    except ValueError as exc:
        diagnostics.append(f"calibration/evaluation unavailable: {exc}")
        report["calibration_status"] = "evaluation_unavailable"
        return _finite(report)
    solo_gates = {name: isolated_report[name]["median_absolute_relative_error"] <= 0.15 for name in ("ttft", "tpot")}
    load_gates = []
    for result in report["open_loop_validation"]:
        metrics = result["metrics"]
        checks = {
            "client_ttft": metrics["client_ttft"]["relative_mean_error"] is not None
            and metrics["client_ttft"]["relative_mean_error"] <= 0.20,
            "server_tpot": metrics["server_tpot"]["relative_mean_error"] is not None
            and metrics["server_tpot"]["relative_mean_error"] <= 0.20,
            "server_queue_wait": metrics["server_queue_wait"]["normalized_mean_error"] is not None
            and metrics["server_queue_wait"]["normalized_mean_error"] <= 0.25,
        }
        load_gates.append({"series_id": result["series_id"], "checks": checks, "passed": all(checks.values())})
    report["gates"] = {
        "isolated_median_absolute_relative_error_limit": 0.15,
        "open_loop_relative_mean_error_limit": 0.20,
        "queue_normalized_mean_error_limit": 0.25,
        "queue_denominator_floor_seconds": 0.05,
        "isolated": solo_gates,
        "open_loop": load_gates,
        "burst": "reported in full; no fitting or acceptance threshold",
    }
    report["validation_passed"] = (
        all(solo_gates.values()) and len(load_gates) == 6 and all(result["passed"] for result in load_gates)
    )
    report["calibration_status"] = "validated" if report["validation_passed"] else "failed_frozen_gates"
    return _finite(report)


def main(argv=None):
    """Print strict finite JSON, optionally saving the identical report to disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_TRACE)
    parser.add_argument("--measurement-profile", default=DEFAULT_PROFILE)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = build_report(
        args.input, measurement_profile=args.measurement_profile, plan_path=args.plan, summary_path=args.summary
    )
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if report["campaign_kind"] == "legacy_pilot":
        return 0 if report["cohorts"] else 1
    return 0 if report["validation_passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
