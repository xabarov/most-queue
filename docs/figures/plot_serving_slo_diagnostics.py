"""Render retrospective diagnostics from the frozen v2 diagnostic JSON only."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position

DEFAULT_REPORT = Path(__file__).resolve().parents[1] / "research/serving-slo-v2-diagnostics.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "serving_slo_v2_diagnostics.png"


def select_rows(report):
    """Require the complete recorded focus and open-loop samples."""
    focus = report["focus"]
    if focus["ttft_limit"] != 0.25 or focus["tpot_limit"] != 0.035:
        raise ValueError("Require the recorded 250 ms TTFT / 35 ms TPOT focus")
    rows = report["requests"]
    selected = [row for row in rows if row["series_id"] == focus["series_id"]]
    loaded = [row for row in rows if row["mode"] == "open_loop"]
    if len(selected) != focus["offered"] or len(selected) != 91 or len(loaded) != 552:
        raise ValueError("Require all 91 focus requests and all 552 open-loop requests")
    ids = focus["false_passes"]
    if len(ids) != 29 or len(set(ids)) != len(ids):
        raise ValueError("Require exactly 29 distinct recorded focus false-pass IDs")
    false_passes = [row for row in selected if row["request_index"] in ids]
    if len(false_passes) != len(ids):
        raise ValueError("Missing or duplicate false-pass request rows")
    groups = [[row for row in loaded if row["scheduled_unfinished_at_arrival"] == value] for value in range(5)]
    counts = {
        row["scheduled_unfinished_at_arrival"]: row["requests"] for row in report["open_loop_by_recorded_occupancy"]
    }
    if sum(map(len, groups)) != len(loaded) or any(
        not group or len(group) != counts[i] for i, group in enumerate(groups)
    ):
        raise ValueError("Missing or inconsistent recorded occupancy groups 0..4")
    return selected, sorted(false_passes, key=lambda row: row["ttft_error"]), groups


def plot_joint_pass(axis, rows, focus):
    """Plot an empirical joint pass fraction with every offered request retained."""
    denominator = len(rows)
    for source, label, color, offset in (
        ("observed", "Observed", "#727b84", -20),
        ("predicted", "Predicted", "#2266a5", 12),
    ):
        ttft = np.array([row[f"{source}_ttft"] for row in rows])
        tpot = np.array([row[f"{source}_tpot"] for row in rows])
        if not np.all(np.isfinite([ttft, tpot])):
            raise ValueError("Joint-pass timings must be finite")
        thresholds = np.unique(np.r_[0.0, ttft[(ttft >= 0) & (ttft <= 0.5)], 0.5])
        passed = [np.count_nonzero((ttft <= value) & (tpot <= focus["tpot_limit"])) for value in thresholds]
        axis.step(thresholds * 1000, np.array(passed) / denominator, where="post", label=label, color=color)
        count = np.count_nonzero((ttft <= focus["ttft_limit"]) & (tpot <= focus["tpot_limit"]))
        if count != focus[f"{source}_passed"]:
            raise ValueError("Focus pass count differs from request rows")
        axis.scatter([focus["ttft_limit"] * 1000], [count / denominator], color=color, s=20, zorder=4)
        axis.annotate(
            f"{count}/{denominator}",
            (focus["ttft_limit"] * 1000, count / denominator),
            xytext=(10, offset),
            textcoords="offset points",
            color=color,
        )
    axis.axvline(focus["ttft_limit"] * 1000, color="0.5", linestyle="--", linewidth=1)
    axis.set_xticks([0, 100, 200, focus["ttft_limit"] * 1000, 300, 400, 500])
    axis.set(
        title=f"Selected focus series · n={denominator}\nJoint pass: TTFT ≤ t and TPOT ≤ 35 ms",
        xlabel="TTFT threshold t, ms · view 0–500 ms",
        ylabel=f"Joint pass fraction · denominator all {denominator}",
        xlim=(0, 500),
        ylim=(0, 1),
    )
    axis.legend(frameon=False, loc="upper left")


def plot_stage_errors(axis, rows):
    """Stack positive and negative stage errors separately for selected false passes."""
    positions = np.arange(len(rows))
    positive, negative = np.zeros(len(rows)), np.zeros(len(rows))
    for stage, label, color in (
        ("queue", "Queue", "#727b84"),
        ("scheduled_to_first", "Scheduled to first", "#cf7027"),
        ("outside_engine", "Outside engine", "#2266a5"),
    ):
        values = np.array([row["components"][stage]["error"] for row in rows]) * 1000
        if not np.all(np.isfinite(values)):
            raise ValueError("Stage errors must be finite")
        axis.bar(positions, values, bottom=np.where(values >= 0, positive, negative), color=color, label=label)
        positive += np.maximum(values, 0)
        negative += np.minimum(values, 0)
    axis.axhline(0, color="0.3", linewidth=0.8)
    axis.set_xticks(positions, [str(row["request_index"]) for row in rows], rotation=90, fontsize=6)
    axis.set(
        title=f"Selected focus false passes · n={len(rows)}",
        xlabel="Request ID · sorted by total TTFT error",
        ylabel="Stage error (observed − predicted), ms",
    )
    axis.legend(frameon=False, fontsize=8, loc="upper left")


def plot_occupancy(axis, groups):
    """Show all pre-engine observations by the recorded half-open occupancy proxy."""
    values = [np.array([row["pre_engine"] for row in group]) * 1000 for group in groups]
    if any(not np.all(np.isfinite(group)) for group in values):
        raise ValueError("Pre-engine timings must be finite")
    axis.boxplot(values, showfliers=True, patch_artist=True, boxprops={"facecolor": "#c8d9e8"})
    axis.set_xticks(range(1, 6), [f"{i}\nn={len(group)}" for i, group in enumerate(groups)])
    axis.set(
        title=f"Full open-loop sample · n={sum(map(len, groups))}\nRecorded half-open observational proxy",
        xlabel="Scheduled unfinished at arrival (count)",
        ylabel="Pre-engine time, ms",
    )


def plot_report(report, output):
    """Save three diagnostic panels without changing the input report or predictions."""
    selected, false_passes, groups = select_rows(report)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    plot_joint_pass(axes[0], selected, report["focus"])
    plot_stage_errors(axes[1], false_passes)
    plot_occupancy(axes[2], groups)
    for axis in axes:
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    figure.suptitle("Retrospective diagnostics · frozen v2 predictions · no fitting", fontsize=13)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    """Read a diagnostic report and write the requested PNG artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with args.report.open(encoding="utf-8") as source:
        report = json.load(source)
    plot_report(report, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
