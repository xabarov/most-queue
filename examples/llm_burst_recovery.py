"""A serving-pool burst ends before its queue recovers; no GPU is required.

Run from the repository root:
    python -m examples.llm_burst_recovery --output-dir /tmp/llm-burst

Produces a PNG, a vector PDF and JSON with every plotted point. Rates are
synthetic requests/second. This occupancy-cap exponential-completion model
predicts queue-to-slot delay, not end-to-end TTFT or a real GPU's performance.
"""

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib

# Backend selection must precede pyplot import on headless systems.
# pylint: disable=wrong-import-position
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from most_queue.theory.continuous_batching import TransientOccupancyQueueCalc


def service_rate(occupancy):
    """Synthetic per-request completion rate, decreasing with active load."""
    return 3.0 / (1 + 0.15 * (occupancy - 1))


def stationary_tail(rate, cap, deadline):
    """Infinite product-form anchor, including the full geometric tail.

    Returns None in an overloaded regime, where no stationary anchor exists.
    Erlang mixing simplifies the waiting tail to C*exp(-(capacity-rate)*d).
    This comparison deliberately avoids finite-buffer normalization.
    """
    capacity = cap * service_rate(cap)
    rho = rate / capacity
    if rho >= 1:
        return None
    if rate == 0:
        return 0.0
    deaths = np.array([j * service_rate(j) for j in range(1, cap + 1)])
    weights = np.r_[0.0, np.cumsum(np.log(rate) - np.log(deaths))]
    tail_weight = weights[-1] - np.log1p(-rho)
    normalizer = logsumexp(np.r_[weights[:-1], tail_weight])
    return float(np.exp(tail_weight - normalizer - (capacity - rate) * deadline))


def experiment():
    """Compute two caps on the same fixed burst and initial empty state."""
    schedule = [(0, 3.0), (2, 13.0), (5, 3.0)]
    times = np.linspace(0, 14, 281)
    deadline = 0.5
    payload = {
        "model": "piecewise-Poisson FCFS occupancy-dependent exponential completion",
        "units": {"time": "seconds", "arrival_rate": "requests/second"},
        "calibration": "synthetic; no GPU measurement or TTFT claim",
        "schedule": schedule,
        "initial_population": 0,
        "queue_to_slot_deadline": deadline,
        "roundoff_certified": False,
        "cases": [],
    }
    for cap in [4, 8]:
        calc = TransientOccupancyQueueCalc(k=cap, queue_truncation=64)
        calc.set_sources(schedule)
        calc.set_servers(service_rate)
        result = calc.run(times)
        bounds = result.waiting_tail_bounds(deadline)
        refined = TransientOccupancyQueueCalc(k=cap, queue_truncation=96)
        refined.set_sources(schedule)
        refined.set_servers(service_rate)
        check = refined.run(times).waiting_tail_bounds(deadline)
        stationary = [stationary_tail(rate, cap, deadline) for rate in result.arrival_rates]
        after = np.flatnonzero(times >= 5)
        recovered = after[bounds[after, 1] <= 0.05]
        # Recovery is reported only once every later sampled point also passes.
        lasting = [i for i in recovered if np.all(bounds[i:, 1] <= 0.05)]
        payload["cases"].append(
            {
                "cap": cap,
                "truncation": 64,
                "refinement_truncation": 96,
                "times": times.tolist(),
                "arrival_rates": result.arrival_rates.tolist(),
                "saturated_departure_rate": result.saturated_departure_rate,
                "waiting_tail_bounds": bounds.tolist(),
                "stationary_waiting_tail": stationary,
                "escape_probability": result.escape_probability.tolist(),
                "max_escape_probability": float(result.escape_probability.max()),
                "mass_residual": result.mass_residual,
                "max_refinement_difference": float(np.max(np.abs(bounds - check))),
                "violation_at_burst_end": bounds[np.searchsorted(times, 5)].tolist(),
                "recovery_time_on_grid": float(times[lasting[0]]) if lasting else None,
                "recovery_grid_step": float(times[1] - times[0]),
            }
        )
    return payload


def plot(payload, output):
    """Plot rates and queue-delay risk, including failed stationary domains."""
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout="constrained")
    colors = ["#235789", "#B64B32"]
    first = payload["cases"][0]
    axes[0].step(first["times"], first["arrival_rates"], where="post", color="#333333", label="Arrival rate")
    for case, color in zip(payload["cases"], colors):
        times = np.array(case["times"])
        bounds = np.array(case["waiting_tail_bounds"])
        label = f'{case["cap"]} active slots'
        axes[0].axhline(case["saturated_departure_rate"], color=color, linestyle=":", label=label + " capacity")
        axes[1].plot(times, bounds[:, 0], color=color, linewidth=2, label=label + ": transient")
        axes[1].fill_between(times, bounds[:, 0], bounds[:, 1], color=color, alpha=0.2)
        stationary = [np.nan if value is None else value for value in case["stationary_waiting_tail"]]
        axes[1].step(
            times, stationary, where="post", color=color, linestyle="--", alpha=0.65, label=label + ": stationary"
        )
    for ax in axes:
        ax.axvspan(2, 5, color="#E6E6E6", alpha=0.6)
        ax.grid(axis="y", alpha=0.18)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set(ylabel="Requests / second", title="The burst ends. The queue does not.")
    axes[1].axhline(0.05, color="#666666", linewidth=1, linestyle=":")
    axes[1].set(xlabel="Time (seconds)", ylabel="P(queue-to-slot wait > 0.5 s)", ylim=(-0.02, 1.02))
    fig.suptitle("Synthetic shared serving pool · infinite FIFO queue · empty initial state", fontsize=10)
    fig.savefig(output / "llm-burst-recovery.png", dpi=170)
    fig.savefig(output / "llm-burst-recovery.pdf")
    plt.close(fig)


def main():
    """Write reproducible figure and numeric evidence without cloud services."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(tempfile.gettempdir()) / "most-queue-llm-burst")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = experiment()
    (args.output_dir / "llm-burst-recovery.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    plot(payload, args.output_dir)
    for case in payload["cases"]:
        print(
            f'cap={case["cap"]}: burst-end risk={case["violation_at_burst_end"]}, '
            f'recovery={case["recovery_time_on_grid"]}, '
            f'max escaped mass={case["max_escape_probability"]:.3g}'
        )
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
