"""Plot all development cohorts and prespecified traced frontend probe groups."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

DEFAULT_REPORT = Path(__file__).resolve().parents[1] / "research/serving-input-admission-development.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "serving_input_admission_development.png"
VARIANTS = {
    "baseline_v2",
    "continuous_reservation",
    "input_one_batch",
    "input_two_batches",
    "two_batches_continuous_input",
}
PRIMARY = "input_two_batches"
GROUPS = (("idle", 128), ("loaded", 128), ("idle", 1024), ("loaded", 1024))
ABBREVIATIONS = {"calibration_v1": "c1", "reservation_v2": "r2", "frontend_v1": "f1"}
COLORS = ("#a6abb1", "#397cb5")


def load_report(path):
    """Verify both archive hashes and require the complete development inventory."""
    report = json.loads(path.read_text(encoding="utf-8"))
    details = report["request_details"]
    compressed = (path.parent / details["path"]).read_bytes()
    if len(compressed) != details["bytes"] or hashlib.sha256(compressed).hexdigest() != details["sha256"]:
        raise ValueError("Request archive size or SHA256 mismatch")
    raw = gzip.decompress(compressed)
    if hashlib.sha256(raw).hexdigest() != details["uncompressed_sha256"]:
        raise ValueError("Uncompressed request archive SHA256 mismatch")
    archive = json.loads(raw)
    counts = report["counts"]
    if (counts["requests"], counts["series"], counts["open_loop_series"]) != (1978, 233, 16):
        raise ValueError("Require the full 1978-request, 233-series, 16-open-loop study")
    cohorts = archive["cohorts"]
    if len(cohorts) != 233 or sum(c["requests"] for c in cohorts) != 1978:
        raise ValueError("Archive inventory does not match the complete study")
    if len(report["series_inventory"]) != 233 or len(report["open_loop_cohorts"]) != 16:
        raise ValueError("Incomplete summary inventory")
    if any(set(c["variants"]) != VARIANTS for c in cohorts + report["open_loop_cohorts"]):
        raise ValueError("Require all five variants for every cohort")
    if report["decision"]["numerical_gate_passed"] is not False:
        raise ValueError("This figure describes the failed development gate")
    full_open = [(c["dataset"], c["series_id"]) for c in cohorts if c["mode"] == "open_loop"]
    if full_open != [(c["dataset"], c["series_id"]) for c in report["open_loop_cohorts"]]:
        raise ValueError("Open-loop summary must preserve every archive cohort in original order")
    return report, cohorts


def probe_groups(cohorts):
    """Keep every traced frontend probe and pair per-request prequeue timings."""
    groups = {group: ([], []) for group in GROUPS}
    for cohort in cohorts:
        if cohort["dataset"] != "frontend_v1":
            continue
        variant = cohort["variants"][PRIMARY]
        indices = variant["request_indices"]
        metric = variant["metrics"]["prequeue"]
        for source in cohort["source_requests"]:
            if source["trace_enabled"] is True and source["role"] == "probe":
                group = source["condition"], source["input_tokens"]
                position = indices.index(source["request_index"])
                groups[group][0].append(metric["observed"][position] * 1000)
                groups[group][1].append(metric["predicted"][position] * 1000)
    if [len(groups[g][0]) for g in GROUPS] != [20, 10, 20, 10]:
        raise ValueError("Require all 20/10/20/10 traced frontend probes")
    return groups


def plot_report(report, cohorts, output):
    """Render the prescribed comparisons without filtering failed cohorts."""
    open_loop = report["open_loop_cohorts"]
    groups = probe_groups(cohorts)
    labels = [f"{ABBREVIATIONS[c['dataset']]} {c['request_rate']:g}" for c in open_loop]
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        try:
            for variant, label, color in zip(("baseline_v2", PRIMARY), ("Baseline v2", "Input: two batches"), COLORS):
                errors = []
                for cohort in open_loop:
                    matches = [
                        item
                        for item in cohort["variants"][variant]["joint_client_slo"]
                        if item["ttft_limit_seconds"] == 0.25 and item["tpot_limit_seconds"] == 0.035
                    ]
                    if len(matches) != 1:
                        raise ValueError("Require exactly one 250/35 ms SLO per cohort")
                    errors.append(matches[0]["absolute_fraction_error"] * 100)
                mae = [c["variants"][variant]["metrics"]["client_ttft"]["mae_seconds"] * 1000 for c in open_loop]
                axes[0].plot(range(16), errors, "o-", color=color, label=label, markersize=4)
                axes[1].plot(range(16), mae, "o-", color=color, label=label, markersize=4)
            for axis in axes[:2]:
                axis.set_xticks(range(16), labels, rotation=65, ha="right")
                axis.set_xlabel("Dataset / offered requests per second (all 16 cohorts)")
                for index in range(1, 16):
                    if open_loop[index]["dataset"] != open_loop[index - 1]["dataset"]:
                        axis.axvline(index - 0.5, color="#cccccc", linestyle=":", linewidth=1)
                axis.grid(axis="y", alpha=0.2)
                axis.legend(fontsize=8)
            axes[0].set(
                title="Joint client SLO: TTFT 250 / TPOT 35 ms", ylabel="Absolute fraction error (percentage points)"
            )
            axes[0].set_ylim(bottom=0)
            axes[1].set(title="Per-request client TTFT error", ylabel="MAE (ms)", yscale="log")
            axes[1].set_ylim(8, 1400)
            for offset, series, label, color in zip((-0.18, 0.18), (0, 1), ("Observed", "Input: two batches"), COLORS):
                axes[2].bar(
                    [i + offset for i in range(4)],
                    [fmean(groups[g][series]) for g in GROUPS],
                    width=0.36,
                    color=color,
                    label=label,
                )
            axes[2].set_xticks(
                range(4),
                [f"{condition} {length}\nn={len(groups[(condition, length)][0])}" for condition, length in GROUPS],
            )
            axes[2].set(
                title="Frontend traced probes: prequeue",
                ylabel="Mean prequeue time (ms)",
                xlabel="All traced probes; idle probes used for fit",
            )
            axes[2].legend(fontsize=8)
            axes[2].grid(axis="y", alpha=0.2)
            figure.suptitle("Development only · fixed server costs · numerical gate FAILED", fontsize=13)
            output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output, dpi=180, metadata={"Software": "Most-Queue"})
        finally:
            plt.close(figure)
    return [len(groups[g][0]) for g in GROUPS]


def main():
    """Read the verified report and save the deterministic development figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report, cohorts = load_report(args.report)
    group_counts = plot_report(report, cohorts, args.output)
    print(f"Verified: 1978 requests, 233 series, 16 open-loop cohorts, 5 variants; probes {group_counts}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
