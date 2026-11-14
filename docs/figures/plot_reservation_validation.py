"""Plot all burst and open-loop cohorts from a fixed-fit reservation report."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position

DEFAULT_REPORT = Path(__file__).resolve().parents[1] / "research/serving-v2-development-results.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "serving_reservation_v2_development.png"


def plot_report(report, output):
    """Save all 36 burst means and six paired client-SLO comparisons unchanged."""
    # Each panel keeps its observed/predicted arrays explicit for review.
    # pylint: disable=too-many-locals
    bursts = [row for row in report["cohorts"] if row["mode"] == "burst"]
    loaded = sorted(
        (row for row in report["cohorts"] if row["mode"] == "open_loop"),
        key=lambda row: (row["request_rate"], row["repetition"]),
    )
    if len(bursts) != 36 or len(loaded) != 6:
        raise ValueError("A complete report with 36 burst and six open-loop cohorts is required")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    baseline_color, reservation_color = "#cf7027", "#2266a5"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for axis, metric, title in zip(axes[:2], ("server_ttft", "server_tpot"), ("Server TTFT", "Server TPOT")):
        limit = 0.0
        for model, color, marker, label in (
            ("baseline", baseline_color, "x", "Old baseline"),
            ("advance_reservation", reservation_color, "o", "Advance reservation"),
        ):
            values = [row[model]["metrics"][metric] for row in bursts]
            observed = np.array([value["observed_mean_seconds"] for value in values]) * 1000
            predicted = np.array([value["predicted_mean_seconds"] for value in values]) * 1000
            if not np.all(np.isfinite([observed, predicted])) or np.any(np.array([observed, predicted]) < 0):
                raise ValueError("Burst cohort means must be finite and nonnegative")
            axis.scatter(observed, predicted, color=color, marker=marker, s=32, alpha=0.8, label=label, zorder=3)
            limit = max(limit, float(observed.max()), float(predicted.max()))
        limit *= 1.08
        axis.plot([0, limit], [0, limit], linestyle="--", color="0.6", linewidth=1)
        axis.set(
            xlim=(0, limit), ylim=(0, limit), xlabel="Observed cohort mean, ms", ylabel="Predicted cohort mean, ms"
        )
        axis.set_title(f"{title} · all 36 burst cohorts")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.15)
    axes[0].legend(frameon=False, fontsize=9)
    slo_rows = []
    for row in loaded:
        matches = [
            item
            for item in row["joint_client_slo"]
            if item["ttft_limit_seconds"] == 1.0 and item["tpot_limit_seconds"] == 0.035
        ]
        if len(matches) != 1:
            raise ValueError("Every open-loop cohort must have exactly one TTFT<=1s, TPOT<=35ms SLO")
        slo_rows.append(matches[0])
    observed = np.array([row["observed_fraction"] for row in slo_rows])
    predicted = np.array([row["predicted_fraction"] for row in slo_rows])
    fractions = np.array([observed, predicted])
    if not np.all(np.isfinite(fractions)) or np.any((fractions < 0) | (fractions > 1)):
        raise ValueError("SLO fractions must lie in [0,1]")
    positions = np.arange(len(loaded))
    axes[2].bar(positions - 0.19, observed, width=0.38, label="Observed", color="#727b84")
    axes[2].bar(positions + 0.19, predicted, width=0.38, label="Advance reservation", color=reservation_color)
    axes[2].set_xticks(positions, [f"{row['request_rate']:g}\n{row['repetition']}" for row in loaded])
    axes[2].set(
        title="Joint client SLO · all six open-loop series\nTTFT ≤ 1 s and TPOT ≤ 35 ms",
        ylim=(0, 1),
        ylabel="Fraction meeting both limits",
        xlabel="Offered rate, requests/s · seed",
    )
    axes[2].grid(axis="y", alpha=0.15)
    axes[2].set_axisbelow(True)
    axes[2].legend(frameon=False, fontsize=9, loc="upper right")
    fresh = "fresh" in str(report.get("data_role", "development")).lower()
    title = (
        "Fresh validation data · fixed isolated-training fits"
        if fresh
        else "Development data · fixed isolated-training fits · fresh validation pending"
    )
    fig.suptitle(title, fontsize=13)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main():
    """Render a saved report without fitting, filtering cohorts or changing SLOs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(plot_report(json.loads(args.report.read_text(encoding="utf-8")), args.output))


if __name__ == "__main__":
    main()
