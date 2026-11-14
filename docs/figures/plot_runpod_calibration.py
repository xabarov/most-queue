"""Plot an exported calibration report without fitting or changing its gates."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position


def plot_report(report, output_dir):
    """Save isolated scatter plots and all six finite loaded-series comparisons."""
    if report.get("isolated_validation") is None or len(report.get("open_loop_validation", [])) != 6:
        raise ValueError("A complete evaluated report with six loaded series is required")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    observed_color, predicted_color = "#2266a5", "#cf7027"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for axis, name in zip(axes, ("ttft", "tpot")):
        metric = report["isolated_validation"][name]
        observed, predicted = np.asarray(metric["observed"]) * 1000, np.asarray(metric["predicted"]) * 1000
        limit = max(observed.max(), predicted.max()) * 1.08
        axis.plot([0, limit], [0, limit], linestyle="--", color="0.6", linewidth=1)
        axis.scatter(observed, predicted, color=observed_color, s=40, zorder=3)
        axis.set(xlim=(0, limit), ylim=(0, limit), xlabel="Observed, ms", ylabel="Predicted, ms")
        axis.set_title(
            f"Client {name.upper()}\nMedian absolute relative error: " f"{metric['median_absolute_relative_error']:.1%}"
        )
        axis.grid(alpha=0.15)
    fig.suptitle("Held-out isolated requests · training coefficients fixed", fontsize=13)
    isolated_path = output_dir / "runpod_calibration_v1_isolated.png"
    fig.savefig(isolated_path, dpi=180)
    plt.close(fig)

    series = sorted(report["open_loop_validation"], key=lambda row: (row["request_rate"], row["repetition"]))
    labels = [f"{row['request_rate']:g}\n{row['repetition']}" for row in series]
    positions = np.arange(len(series))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, name, title in zip(
        axes,
        ("client_ttft", "server_tpot", "server_queue_wait"),
        ("Client TTFT", "Server TPOT", "Server queue wait"),
    ):
        observed = [row["metrics"][name]["observed_mean_seconds"] * 1000 for row in series]
        predicted = [row["metrics"][name]["predicted_mean_seconds"] * 1000 for row in series]
        axis.bar(positions - 0.19, observed, width=0.38, label="Observed", color=observed_color)
        axis.bar(positions + 0.19, predicted, width=0.38, label="Predicted", color=predicted_color)
        axis.set_xticks(positions, labels)
        axis.set(title=title, ylabel="Mean, ms", xlabel="Offered rate, requests/s · seed")
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.suptitle("Six held-out finite series · all runs shown · no refitting", fontsize=13)
    loaded_path = output_dir / "runpod_calibration_v1_loaded.png"
    fig.savefig(loaded_path, dpi=180)
    plt.close(fig)
    return [isolated_path, loaded_path]


def main():
    """Read a strict exported report and write standalone scientific figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    for path in plot_report(json.loads(args.report.read_text(encoding="utf-8")), args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
