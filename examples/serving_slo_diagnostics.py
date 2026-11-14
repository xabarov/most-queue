#!/usr/bin/env python3
"""Attribute saved SLO errors to measured timing intervals, without fitting.

Observed-component substitutions are retrospective accounting diagnostics.
They use outcomes unavailable to a predictor and are never model validation.
"""

import argparse
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from tools.audit_reservation_validation import audit_fixture

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/serving_measurements/runpod_qwen3_8b_a40_reservation_v2_20261114"
REPORT = ROOT / "docs/research/serving-reservation-v2-results.json"
COMPONENTS = ("queue", "scheduled_to_first", "outside_engine")
FOCUS = "s0061-validation-open_loop"


def digest(path):
    """Return the identity of exact input bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(left, right):
    """Check interval identities with a tolerance for monotonic-clock subtraction."""
    return math.isclose(left, right, rel_tol=0, abs_tol=1e-8)


def decompose_request(record, predicted):
    """Split one client TTFT into queue, post-scheduling and outside intervals.

    The pre-engine and first-output-delivery split assumes the recorded common
    host monotonic clock. The total outside residual does not require this
    cross-process clock assumption. Dispatch-to-worker startup is excluded from
    the recorded client TTFT and is reported separately.
    """
    stamps = record["server_timestamps"]
    clocks = (stamps["queued_ts"], stamps["scheduled_ts"], stamps["first_token_ts"], stamps["last_token_ts"])
    client_start, ttft = record["client_started_monotonic_s"], record["client_ttft_s"]
    done = record["latency_to_done_s"]
    outputs = record["output_tokens"]
    values = (*clocks, client_start, ttft, done, *predicted.values())
    if record.get("success") is not True or stamps.get("timestamp_clock") != "monotonic" or outputs <= 1:
        raise ValueError("complete monotonic multitoken timings required")
    if (
        any(isinstance(v, bool) or not math.isfinite(v) for v in values)
        or any(b < a for a, b in zip(clocks, clocks[1:]))
        or done < ttft
        or any(v < 0 for v in predicted.values())
    ):
        raise ValueError("complete, ordered, finite monotonic multitoken timings required")
    queue, scheduled = clocks[1] - clocks[0], clocks[2] - clocks[1]
    server = clocks[2] - clocks[0]
    if not close(
        predicted["server_ttft"], predicted["server_queue_wait"] + predicted["server_scheduled_to_first_token"]
    ):
        raise ValueError("predicted server TTFT must equal queue plus scheduled interval")
    observed = (queue, scheduled, ttft - server)
    prediction = (
        predicted["server_queue_wait"],
        predicted["server_scheduled_to_first_token"],
        predicted["client_ttft"] - predicted["server_ttft"],
    )
    components = {
        key: {"observed": actual, "predicted": estimate, "error": actual - estimate}
        for key, actual, estimate in zip(COMPONENTS, observed, prediction)
    }
    pre_engine = clocks[0] - client_start
    delivery = client_start + ttft - clocks[2]
    after_last = client_start + done - clocks[3]
    if min(pre_engine, delivery, after_last, *observed, *prediction) < -1e-8:
        raise ValueError("client and engine intervals violate the common-clock contract")
    client_tpot, server_tpot = (done - ttft) / (outputs - 1), (clocks[3] - clocks[2]) / (outputs - 1)
    tpot_residual = (after_last - delivery) / (outputs - 1)
    if not close(client_tpot - server_tpot, tpot_residual):
        raise ValueError("client/server TPOT interval identity failed")
    error = ttft - predicted["client_ttft"]
    if not close(error, sum(c["error"] for c in components.values())):
        raise ValueError("TTFT error decomposition failed")
    return {
        "request_index": record["request_index"],
        "input_tokens": record["input_tokens"],
        "output_tokens": outputs,
        "observed_ttft": ttft,
        "predicted_ttft": predicted["client_ttft"],
        "observed_tpot": client_tpot,
        "predicted_tpot": predicted["client_tpot"],
        "client_tpot_error": client_tpot - predicted["client_tpot"],
        "observed_server_tpot": server_tpot,
        "predicted_server_tpot": predicted["server_tpot"],
        "server_tpot_error": server_tpot - predicted["server_tpot"],
        "generation_error": (outputs - 1) * (server_tpot - predicted["server_tpot"]),
        "ttft_error": error,
        "components": components,
        "pre_engine": pre_engine,
        "first_output_delivery": delivery,
        "after_last_engine_token_to_done": after_last,
        "client_minus_server_tpot": tpot_residual,
        "dispatch_to_worker_start": client_start - record["dispatch_started_monotonic_s"],
        "dispatch_lateness": record["dispatch_started_monotonic_s"] - record["scheduled_arrival_monotonic_s"],
    }


def arrival_occupancy(records, index):
    """Count other scheduled/uncompleted and queued/unscheduled traces at arrival.

    Use half-open intervals: a completion at arrival frees its slot, and a
    scheduling at arrival is included. This is a recorded scheduling proxy;
    it does not observe GPU execution or reproduce the simulator's state.
    """
    target = records[index]
    time = target["server_timestamps"]["queued_ts"]
    others = [row["server_timestamps"] for i, row in enumerate(records) if i != index]
    return {
        "scheduled_unfinished_at_arrival": sum(s["scheduled_ts"] <= time < s["last_token_ts"] for s in others),
        "queued_unscheduled_at_arrival": sum(s["queued_ts"] <= time < s["scheduled_ts"] for s in others),
    }


def outcomes(rows, ttft_limit, tpot_limit, *, observed_components=(), observed_tpot=False):
    """Count paired outcomes after optional observation-dependent substitutions."""
    if not set(observed_components).issubset(COMPONENTS):
        raise ValueError("unknown TTFT component")
    false_passes, false_failures = [], []
    causes = {"ttft_only": 0, "tpot_only": 0, "both": 0}
    observed_passed, predicted_passed = 0, 0
    for row in rows:
        # Summing the selected component values also recovers exact observations
        # at the inclusive boundary when every observed component is selected.
        first = sum(row["components"][c]["observed" if c in observed_components else "predicted"] for c in COMPONENTS)
        if len(set(observed_components)) == len(COMPONENTS):
            first = row["observed_ttft"]
        gap = row["observed_tpot"] if observed_tpot else row["predicted_tpot"]
        actual = row["observed_ttft"] <= ttft_limit and row["observed_tpot"] <= tpot_limit
        predicted = first <= ttft_limit and gap <= tpot_limit
        observed_passed += actual
        predicted_passed += predicted
        if predicted and not actual:
            false_passes.append(row["request_index"])
            bad_first, bad_gap = row["observed_ttft"] > ttft_limit, row["observed_tpot"] > tpot_limit
            causes["both" if bad_first and bad_gap else "ttft_only" if bad_first else "tpot_only"] += 1
        if actual and not predicted:
            false_failures.append(row["request_index"])
    return {
        "offered": len(rows),
        "observed_passed": int(observed_passed),
        "predicted_passed": int(predicted_passed),
        "false_passes": false_passes,
        "false_failures": false_failures,
        "false_pass_causes": causes,
        "signed_fraction_error": (predicted_passed - observed_passed) / len(rows),
    }


def statistics(values):
    """Summarize recorded seconds; signed means remain signed."""
    values = np.asarray(values, dtype=float)
    return {
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "maximum": float(values.max()),
        "mean_absolute": float(np.abs(values).mean()),
    }


def summarize(rows):
    """Keep sample size and signed component errors alongside absolute errors."""
    return {
        "requests": len(rows),
        "components": {
            component: {
                kind: statistics([r["components"][component][kind] for r in rows])
                for kind in ("observed", "predicted", "error")
            }
            for component in COMPONENTS
        },
        "timings": {
            key: statistics([r[key] for r in rows])
            for key in (
                "observed_ttft",
                "predicted_ttft",
                "ttft_error",
                "pre_engine",
                "first_output_delivery",
                "after_last_engine_token_to_done",
                "client_minus_server_tpot",
                "dispatch_to_worker_start",
                "dispatch_lateness",
                "client_tpot_error",
                "server_tpot_error",
                "generation_error",
            )
        },
    }


def build_diagnostics(report, records):
    """Join by exact identities and preserve every evaluation request and SLO."""
    indices = [r["request_index"] for r in records]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate raw request identity")
    raw = {r["request_index"]: r for r in records}
    rows, grid, summaries = [], [], []
    for cohort in report["cohorts"]:
        replay = cohort["advance_reservation"]
        source = [raw[i] for i in replay["request_indices"]]
        group = []
        for i, record in enumerate(source):
            if record["series_id"] != cohort["series_id"]:
                raise ValueError("raw and replay series identities differ")
            predicted = {k: v["predicted"][i] for k, v in replay["metrics"].items()}
            row = decompose_request(record, predicted)
            observed = {
                "client_ttft": row["observed_ttft"],
                "client_tpot": row["observed_tpot"],
                "server_ttft": row["components"]["queue"]["observed"]
                + row["components"]["scheduled_to_first"]["observed"],
                "server_tpot": row["observed_tpot"] - row["client_minus_server_tpot"],
                "server_queue_wait": row["components"]["queue"]["observed"],
                "server_scheduled_to_first_token": row["components"]["scheduled_to_first"]["observed"],
            }
            if any(not close(value, replay["metrics"][key]["observed"][i]) for key, value in observed.items()):
                raise ValueError("raw and replay observations differ")
            row.update({k: cohort[k] for k in ("series_id", "mode", "request_rate", "repetition")})
            row.update(arrival_occupancy(source, i))
            group.append(row)
        rows.extend(group)
        summaries.append({"series_id": cohort["series_id"], "mode": cohort["mode"], **summarize(group)})
        for slo in cohort["joint_client_slo"]:
            result = outcomes(group, slo["ttft_limit_seconds"], slo["tpot_limit_seconds"])
            for name in ("observed_passed", "predicted_passed"):
                if result[name] != slo[name]:
                    raise ValueError("raw decomposition and saved SLO counts differ")
            if (
                len(result["false_passes"]) != slo["false_passes"]
                or len(result["false_failures"]) != slo["false_failures"]
            ):
                raise ValueError("raw decomposition and saved SLO confusion differ")
            grid.append(
                {
                    "series_id": cohort["series_id"],
                    "mode": cohort["mode"],
                    "ttft_limit": slo["ttft_limit_seconds"],
                    "tpot_limit": slo["tpot_limit_seconds"],
                    **result,
                }
            )
    if len(rows) != len(set(r["request_index"] for r in rows)) or {r["request_index"] for r in rows} != {
        r["request_index"] for r in records if r["split"] == "validation"
    }:
        raise ValueError("every validation request must appear exactly once")
    focus = [r for r in rows if r["series_id"] == FOCUS]
    result = outcomes(focus, 0.25, 0.035)
    ids = set(result["false_passes"])
    false = [r for r in focus if r["request_index"] in ids]
    substitutions = []
    for count in range(4):
        for selected in itertools.combinations(COMPONENTS, count):
            for gap in (False, True):
                substitutions.append(
                    {
                        "observed_components": list(selected),
                        "observed_tpot": gap,
                        **outcomes(focus, 0.25, 0.035, observed_components=selected, observed_tpot=gap),
                    }
                )
    return {
        "schema_version": 1,
        "data_role": "retrospective diagnosis of frozen v2 predictions; no fitting or new validation",
        "units": "seconds, except counts, lengths and fractions",
        "error_sign": "observed minus predicted",
        "clock_scope": "same-host monotonic clock; pre/post split is conditional on that contract",
        "substitution_scope": (
            "uses observed outcomes; accounting diagnostic, never a usable predictor or new validation"
        ),
        "requests": rows,
        "series_summaries": summaries,
        "slo_grid": grid,
        "focus": {
            "series_id": FOCUS,
            "ttft_limit": 0.25,
            "tpot_limit": 0.035,
            **result,
            "false_pass_summary": summarize(false),
            "substitutions": substitutions,
        },
        "open_loop_by_recorded_occupancy": [
            {"scheduled_unfinished_at_arrival": n, **summarize(group)}
            for n in range(5)
            if (group := [r for r in rows if r["mode"] == "open_loop" and r["scheduled_unfinished_at_arrival"] == n])
        ],
    }


def main():
    """Verify saved evidence and write deterministic offline diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    config = json.loads((ROOT / "configs/runpod/reservation-v2-workload.json").read_text())
    audit = audit_fixture(args.fixture, config, 2026111403)
    if audit["status"] != "complete_success" or report["audit"] != audit:
        raise ValueError("complete fixture and matching saved audit are required")
    if report["trace_sha256"] != digest(args.fixture / "requests.jsonl.gz"):
        raise ValueError("report trace identity differs")
    if report["fit"] != json.loads((args.fixture / "frozen-fit.json").read_text()):
        raise ValueError("report fit differs from frozen fit")
    with gzip.open(args.fixture / "requests.jsonl.gz", "rt") as stream:
        result = build_diagnostics(report, [json.loads(line) for line in stream])
    result["provenance"] = {
        "report_sha256": digest(args.report),
        "trace_sha256": report["trace_sha256"],
        "frozen_fit_sha256": digest(args.fixture / "frozen-fit.json"),
        "diagnostic_code_sha256": digest(__file__),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "evaluation_requests": len(result["requests"]),
                "focus_false_passes": len(result["focus"]["false_passes"]),
                "fitting_performed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
