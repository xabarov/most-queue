#!/usr/bin/env python3
"""Summarize the frontend study and replay fixed v2 costs without fitting."""

import argparse
import gzip
import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from most_queue.io.runpod import read_runpod_trace
from most_queue.theory.continuous_batching.advance_reservation_calibration import AdvanceReservationFit
from most_queue.theory.continuous_batching.advance_reservation_validation import (
    compare_joint_slo,
    replay_advance_reservation,
)
from tools.audit_frontend_measurements import audit
from tools.frontend_descriptive import pair_comparisons, request_metrics, summarize, summarize_pairs

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests/fixtures/serving_measurements/runpod_qwen3_8b_a40_frontend_v1_20261114"
RAW_FILES = ("requests", "server_timestamps", "frontend_timestamps")


def digest(path):
    """Hash exact saved bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unpack_fixture(fixture, target):
    """Verify public artifacts and restore the producer layout in temporary space."""
    manifest = json.loads((fixture / "manifest.json").read_text())
    for name, sha in manifest["sha256"].items():
        path = fixture / name
        if path.parent != fixture or path.is_symlink() or digest(path) != sha:
            raise ValueError("fixture file identity differs: " + name)
    measurements = target / "measurements"
    measurements.mkdir()
    for name in ("plan.json", "series.json", "summary.json", "progress.json"):
        shutil.copy2(fixture / name, measurements / name)
    shutil.copy2(fixture / "producer-manifest.json", measurements / "manifest.json")
    for name in RAW_FILES:
        raw = gzip.decompress((fixture / (name + ".jsonl.gz")).read_bytes())
        destination = measurements if name == "requests" else target
        (destination / (name + ".jsonl")).write_bytes(raw)
    return measurements


def summarize_groups(rows):
    """Keep block, role, length and trace treatment separate in every summary."""
    fields = ("block_id", "condition", "role", "input_tokens", "output_tokens", "trace_enabled")
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in fields)].append(row)
    result = []
    for key, selected in sorted(grouped.items()):
        metrics = selected[0]["metrics"]
        item = {
            **dict(zip(fields, key)),
            "requests": len(selected),
            "metrics": {name: summarize([row["metrics"][name] for row in selected]) for name in metrics},
        }
        if selected[0]["intervals"] is not None:
            item["intervals"] = {
                name: summarize([row["intervals"][name] for row in selected]) for name in selected[0]["intervals"]
            }
        result.append(item)
    return result


def replay_series(records, fit):
    """Adapt diagnostic rows in memory; preserve their stored split and provenance.

    The original importer intentionally rejects split=diagnostic. Its temporary
    validation label here supplies its parsing contract, not a scientific claim
    of model-v3 validation. Only arrivals and lengths enter the fixed v2 replay.
    """
    adapted = [{**row, "split": "validation"} for row in records]
    benchmarks = read_runpod_trace(adapted, measurement_profile=fit.measurement_profile, include_warmup=False)
    if len(benchmarks) != 1:
        raise ValueError("exactly one complete open-loop series required")
    benchmark = benchmarks[0]
    replay = replay_advance_reservation(fit, benchmark)
    raw = {row["request_index"]: row for row in records}
    metrics = replay["metrics"]
    chronological = []
    for i, index in enumerate(replay["request_indices"]):
        row = raw[index]
        stamps = row["server_timestamps"]
        queue_error = metrics["server_queue_wait"]["observed"][i] - metrics["server_queue_wait"]["predicted"][i]
        stage_error = (
            metrics["server_scheduled_to_first_token"]["observed"][i]
            - metrics["server_scheduled_to_first_token"]["predicted"][i]
        )
        outside_error = metrics["client_ttft"]["observed"][i] - metrics["server_ttft"]["observed"][i]
        outside_error -= metrics["client_ttft"]["predicted"][i] - metrics["server_ttft"]["predicted"][i]
        chronological.append(
            {
                "request_index": index,
                "engine_arrival": stamps["queued_ts"],
                "engine_last_token": stamps["last_token_ts"],
                "queue_error": queue_error,
                "scheduled_to_first_error": stage_error,
                "outside_engine_error": outside_error,
                "generation_span_error": (row["output_tokens"] - 1)
                * (metrics["server_tpot"]["observed"][i] - metrics["server_tpot"]["predicted"][i]),
            }
        )
    chronological.sort(key=lambda row: (row["engine_arrival"], row["request_index"]))
    busy_end, busy_index = float("-inf"), 0
    for row in chronological:
        if row["engine_arrival"] >= busy_end:
            busy_index += 1
        row["observed_engine_busy_period"] = busy_index
        busy_end = max(busy_end, row["engine_last_token"])
    return {
        "series_id": records[0]["series_id"],
        "block_id": records[0]["block_id"],
        "request_rate": records[0]["request_rate"],
        "source_split": "diagnostic",
        "replay": replay,
        "joint_client_slo": [
            compare_joint_slo(benchmark, replay, ttft_limit=t, tpot_limit=p)
            for t in (0.25, 0.5, 1.0)
            for p in (0.030, 0.035, 0.040)
        ],
        "chronological_errors": chronological,
        "busy_period_scope": "posthoc union of observed [engine_queue, last_token) intervals; never a predictor",
    }


def build_report(fixture):
    """Audit all data before aggregating; keep condition failures and all requests."""
    fixture = Path(fixture)
    config = json.loads((fixture / "workload-config.json").read_text())
    profile = json.loads((fixture / "profile.json").read_text())
    frozen = json.loads((fixture / "frozen-v2-fit.json").read_text())
    for key in ("image", "model", "revision", "server_args"):
        if profile[key] != frozen["profile"][key]:
            raise ValueError("common engine profile differs from frozen v2 fit")
    runtime = json.loads((fixture / "runtime.json").read_text())
    reference = json.loads(
        (
            ROOT / "tests/fixtures/serving_measurements/runpod_qwen3_8b_a40_calibration_v1_20261114/runtime.json"
        ).read_text()
    )
    if any(runtime[key] != reference[key] for key in ("gpu", "vllm", "torch", "cuda")):
        raise ValueError("runtime differs from frozen v2 comparison profile")
    with tempfile.TemporaryDirectory(prefix="mq-frontend-audit-") as temporary:
        directory = unpack_fixture(fixture, Path(temporary))
        checked = audit(directory, config, profile["seed"])
        records = [json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines()]
    if checked["errors"]:
        return {"data_role": "incomplete or failed diagnostic study; analysis refused", "audit": checked}
    derived = {row["request_index"]: row["intervals"] for row in checked["intervals"]}
    measured = [row for row in records if row["phase"] == "measured"]
    rows = []
    for row in measured:
        item = {
            key: row[key]
            for key in (
                "request_index",
                "series_id",
                "block_id",
                "condition",
                "role",
                "input_tokens",
                "output_tokens",
                "trace_enabled",
            )
        }
        item.update(metrics=request_metrics(row), intervals=derived.get(row["request_index"]))
        rows.append(item)
    pairs = pair_comparisons(measured)
    groups = summarize_pairs(pairs)
    by_block = {
        block["block_id"]: summarize_pairs([pair for pair in pairs if pair["block_id"] == block["block_id"]])
        for block in config["blocks"]
    }
    open_loop = defaultdict(list)
    for row in measured:
        if row["mode"] == "open_loop":
            open_loop[row["series_id"]].append(row)
    fit = AdvanceReservationFit(**frozen["advance_reservation"])
    audit_summary = {key: value for key, value in checked.items() if key != "intervals"}
    return {
        "schema_version": 1,
        "data_role": (
            "diagnostic repeatability and instrumentation study; fixed v2 comparison; no fitting or v3 validation"
        ),
        "units": "seconds except counts, lengths and fractions",
        "audit": audit_summary,
        "counts": {
            "all": len(records),
            "warmup": len(records) - len(measured),
            "diagnostic": len(measured),
            "by_condition": dict(Counter(row["condition"] for row in measured)),
        },
        "pairs": pairs,
        "pair_summaries": groups,
        "pair_summaries_by_block": by_block,
        "group_summaries": summarize_groups(rows),
        "requests": rows,
        "open_loop_v2_comparisons": [replay_series(selected, fit) for _, selected in sorted(open_loop.items())],
        "frozen_v2_fit": frozen,
        "provenance": {
            "fixture_manifest_sha256": digest(fixture / "manifest.json"),
            "requests_gzip_sha256": digest(fixture / "requests.jsonl.gz"),
            "frozen_v2_fit_sha256": digest(fixture / "frozen-v2-fit.json"),
            "analysis_code_sha256": digest(__file__),
            "descriptive_code_sha256": digest(ROOT / "tools/frontend_descriptive.py"),
        },
        "limitations": [
            "outside_engine names a residual outside QUEUED-to-first-token; "
            "it may include EngineCore input-queue waiting.",
            "On/off controls share wrapper/engine tracing; they measure only added frontend tracing work and header.",
            "Repeated prompts, one pod, two blocks; no iid confidence interval, p99 or capacity claim.",
            "Condition failures remain in summaries and are listed by audit; no favorable subset is selected.",
            "Frontend intervals follow the recorded common monotonic-clock contract.",
        ],
    }


def main():
    """Produce a saved report locally, without a GPU or network access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.fixture)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "data_role": report["data_role"],
                "audit_status": report["audit"]["status"],
                "counts": report.get("counts"),
            },
            indent=2,
        )
    )
    return 0 if report["audit"]["status"] == "complete_success" else 1


if __name__ == "__main__":
    main()
