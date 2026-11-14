"""Render the complete frontend diagnostic report without fitting or filtering rows."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
from matplotlib.patches import Patch  # pylint: disable=wrong-import-position

DEFAULT_REPORT = Path(__file__).resolve().parents[1] / "research/serving-frontend-v1-results.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "serving_frontend_v1.png"
GROUPS = (("idle", 128), ("loaded", 128), ("idle", 1024), ("loaded", 1024))
STAGES = (
    ("To handler", ("client_to_http", "receive_body", "parse_validate_route"), "#8ab7d8"),
    ("Preparation", ("handler_to_render", "render"), "#d7ae71"),
    ("To QUEUED", ("render_to_engine_queue",), "#87b99b"),
    ("Queue", ("queue",), "#a0a5ac"),
    ("Scheduled to first", ("scheduled_to_first",), "#426b9c"),
    ("Delivery", ("first_output_delivery",), "#b68bad"),
)
ON_COLOR, OFF_COLOR = "#397cb5", "#a6abb1"


def finite(value, nonnegative=False):
    """Require finite plotted values, rejecting booleans and missing values."""
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Every plotted value must be finite")
    if nonnegative and value < 0:
        raise ValueError("A timing duration is negative")
    return value


def select_rows(report):
    """Require the complete diagnostic study, including all mismatched conditions."""
    audit = report["audit"]
    conditions = audit["condition_failures"]
    allowed = audit["status"] == "complete_success" or (
        audit["status"] == "incomplete_failed_or_condition_mismatch" and bool(conditions)
    )
    required = {
        "planned": 613,
        "completed": 613,
        "successful": 613,
        "instrumented_expected": 533,
        "instrumented_validated": 533,
    }
    if audit["errors"] or not allowed or any(audit.get(key) != count for key, count in required.items()):
        raise ValueError("Refuse an incomplete or failed measurement audit")
    counts = report["counts"]
    if any(counts.get(key) != count for key, count in {"all": 613, "warmup": 4, "diagnostic": 609}.items()):
        raise ValueError("Missing or inconsistent report counts")
    if counts.get("by_condition") != {"idle": 80, "loaded": 60, "open_loop": 469}:
        raise ValueError("Condition counts differ from the complete study")
    rows = report["requests"]
    if len(rows) != 609 or len({row["request_index"] for row in rows}) != 609:
        raise ValueError("Missing or duplicate diagnostic request rows")
    if Counter(row["condition"] for row in rows) != counts["by_condition"]:
        raise ValueError("Request rows differ from condition counts")
    probes = [row for row in rows if row["role"] == "probe"]
    groups = []
    for condition, length in GROUPS:
        for traced in (False, True):
            selected = [
                row
                for row in probes
                if row["condition"] == condition and row["input_tokens"] == length and row["trace_enabled"] is traced
            ]
            if len(selected) != (20 if condition == "idle" else 10):
                raise ValueError("Missing or inconsistent on/off probe group")
            for row in selected:
                finite(row["metrics"]["client_ttft"], nonnegative=True)
                if traced:
                    for key in (key for _, keys, _ in STAGES for key in keys):
                        finite(row["intervals"][key], nonnegative=True)
            groups.append(selected)
    if len(probes) != 120:
        raise ValueError("Require all 120 measured probes")
    comparisons = report["open_loop_v2_comparisons"]
    expected = {("a", 0.5): 91, ("a", 0.8): 143, ("b", 0.8): 148, ("b", 0.5): 87}
    if len(comparisons) != 4 or Counter((c["block_id"], c["request_rate"]) for c in comparisons) != Counter(
        expected.keys()
    ):
        raise ValueError("Require all four open-loop comparisons")
    series_ids = set()
    for comparison in comparisons:
        identity = comparison["block_id"], comparison["request_rate"]
        series_ids.add(comparison["series_id"])
        selected = [row for row in rows if row["series_id"] == comparison["series_id"]]
        errors = comparison["chronological_errors"]
        if (
            len(errors) != expected[identity]
            or len(selected) != len(errors)
            or Counter(row["request_index"] for row in errors) != Counter(row["request_index"] for row in selected)
            or any(row["condition"] != "open_loop" or row["block_id"] != identity[0] for row in selected)
        ):
            raise ValueError("Missing or duplicate chronological queue-error points")
        for row in errors:
            finite(row["engine_arrival"], nonnegative=True)
            finite(row["queue_error"])
    if len(series_ids) != 4 or sum(len(c["chronological_errors"]) for c in comparisons) != 469:
        raise ValueError("Require all 469 open-loop queue-error points")
    return groups, comparisons, bool(conditions)


def plot_report(report, output):
    """Save three panels using every prespecified probe and queue-error point."""
    groups, comparisons, condition_mismatches = select_rows(report)
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        try:
            samples = [[row["metrics"]["client_ttft"] * 1000 for row in selected] for selected in groups]
            boxes = axes[0].boxplot(
                samples,
                patch_artist=True,
                showfliers=True,
                medianprops={"color": "#202020"},
                flierprops={"markersize": 3},
            )
            for index, box in enumerate(boxes["boxes"]):
                box.set_facecolor(ON_COLOR if index % 2 else OFF_COLOR)
            labels = [
                f"{condition} {length}\n{'on' if traced else 'off'} · n={len(groups[index])}"
                for index, (condition, length, traced) in enumerate(
                    (condition, length, traced) for condition, length in GROUPS for traced in (False, True)
                )
            ]
            axes[0].set_xticks(range(1, 9), labels, rotation=45, ha="right", fontsize=8)
            axes[0].set(title="All measured probes · n=120", ylabel="Client TTFT, ms")
            axes[0].legend(
                handles=[
                    Patch(facecolor=OFF_COLOR, label="Tracing off"),
                    Patch(facecolor=ON_COLOR, label="Tracing on"),
                ],
                frameon=False,
                fontsize=8,
            )
            bottom = [0.0] * 4
            for label, keys, color in STAGES:
                values = [
                    1000 * fmean(sum(row["intervals"][key] for key in keys) for row in groups[index])
                    for index in (1, 3, 5, 7)
                ]
                axes[1].bar(range(4), values, bottom=bottom, label=label, color=color)
                bottom = [left + right for left, right in zip(bottom, values)]
            axes[1].set_xticks(
                range(4),
                [f"{condition} {length}\nn={len(groups[2 * i + 1])}" for i, (condition, length) in enumerate(GROUPS)],
            )
            axes[1].set(title="Mean stages · all instrumented probes", ylabel="Mean client TTFT, ms")
            axes[1].set_ylim(0, max(bottom) * 1.35 if max(bottom) else 1)
            axes[1].legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
            colors = ("#397cb5", "#d68a32", "#65a078", "#9b6caa")
            for comparison, color in zip(comparisons, colors):
                points = sorted(
                    comparison["chronological_errors"], key=lambda row: (row["engine_arrival"], row["request_index"])
                )
                first = points[0]["engine_arrival"]
                axes[2].plot(
                    [row["engine_arrival"] - first for row in points],
                    [row["queue_error"] * 1000 for row in points],
                    color=color,
                    linewidth=0.9,
                    marker=".",
                    markersize=2.5,
                    label=f"{comparison['block_id']} · {comparison['request_rate']:g}/s · n={len(points)}",
                )
            axes[2].axhline(0, color="#555555", linewidth=0.8, linestyle="--")
            axes[2].set(
                title="All four open-loop series · n=469",
                xlabel="Engine arrival since series first arrival, s",
                ylabel="Queue error: observed − predicted, ms",
            )
            axes[2].legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=8)
            for axis in axes:
                axis.grid(axis="y", alpha=0.2)
                axis.set_axisbelow(True)
            title = "Diagnostic A40 study · fixed v2 costs · no fitting"
            if condition_mismatches:
                title += " · condition mismatches retained"
            if report.get("synthetic") is True:
                title += " · SYNTHETIC"
            figure.suptitle(title, fontsize=12)
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output, dpi=180)
        finally:
            plt.close(figure)
    return output


def main():
    """Plot a complete saved report without accessing measurement fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot_report(json.loads(args.report.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
