"""Frozen GL2 protocol; length-only replay, never a GPU benchmark.

python -m examples.llm_mark_order_experiment --trace /path/BurstGPT_1.csv
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from most_queue.random.trace_resampling import lag_correlation
from most_queue.sim.prefill_decode import PrefillDecodeSim
from most_queue.sim.utils.llm_trace import permute_joint_marks, read_burstgpt

SOURCE_SHA = "4bb3783693d0a435686fbfc885615d2349bd067239079fa4b749f2e679e12122"
COSTS = (0.00668, 0.0000201, 0.0000000552)
CONFIGS = ((4, 256), (16, 2048))
DAYS = (32, 36, 40, 44, 48, 52, 56, 60)
MODES = ("observed", "global", "local", "blocks", "shift")


def controlled_marks(times, marks, mode, seed, warmup=128):
    """Keep history fixed AND preserve the measured cohort's joint law."""
    pairs = marks.copy()
    pairs[warmup:] = permute_joint_marks(times[warmup:], marks[warmup:], mode, seed)
    assert sorted(map(tuple, pairs[warmup:])) == sorted(map(tuple, marks[warmup:]))
    return pairs


def exposure_diagnostic(times, marks, config, warmup=128):
    """Serialized isolated work per slot/exposure, NOT batch stability rho."""
    cap, budget = config
    n, m = marks[warmup:].T
    alpha, beta, gamma = COSTS
    chunks = np.ceil(n / budget)
    pre = chunks * alpha + beta * n + gamma * (budget * (chunks - 1) * chunks / 2 + n)
    full = pre + m * (alpha + beta) + gamma * (m * n + m * (m + 1) / 2)
    return float(full.sum() / (cap * (times[-1] - times[warmup])))


def replay_metrics(times, marks, config, warmup=128):
    """Measure all tagged first tokens, and occupancy BEFORE the final drain."""
    sim = PrefillDecodeSim(*config)
    sim.set_servers(1, 1, *COSTS)
    result = sim.replay(times, *marks.T, warmup_fraction=warmup / len(times))
    first = np.array([row[0] for row in result.output_times])
    ttft = first[warmup:] - times[warmup:]
    done = np.asarray(result.completion_times)
    return {
        "slo_fraction": float(np.mean(ttft <= 1)),
        "ttft_mean": float(ttft.mean()),
        "ttft_p99": float(np.quantile(ttft, 0.99)),
        "backlog_at_last_arrival": int(np.sum(done > times[-1])),
        "busy_exposure": result.busy_probability,
        "observation_time": result.observation_time,
        "requests": len(ttft),
        "unobserved_first_tokens": 0,
    }


def paired_interval(values):
    """Equal-weight day bootstrap, not a request-level iid confidence interval."""
    values = np.asarray(values, dtype=float)
    draws = np.random.default_rng(92999).integers(len(values), size=(10000, len(values)))
    return {"mean": float(values.mean()), "ci95": np.quantile(values[draws].mean(axis=1), [0.025, 0.975]).tolist()}


def run(trace_path):
    """Execute the fixed pair without test-based parameter selection."""
    digest = hashlib.sha256(Path(trace_path).read_bytes()).hexdigest()
    if digest != SOURCE_SHA:
        raise ValueError("unexpected trace SHA-256")
    trace = read_burstgpt(trace_path)
    rows, day_differences = [], {mode: [] for mode in MODES}
    cohorts = []
    for day in DAYS:
        mask = (trace.arrivals // 86400) == day
        times, marks = trace.arrivals[mask][:1152], trace.marks[mask][:1152]
        if len(times) != 1152 or times[-1] <= times[128]:
            raise ValueError(f"day {day} has no full fixed cohort")
        times = times - times[0]
        cohorts.append(
            {
                "day": day,
                "serialized_work_per_slot_exposure": [exposure_diagnostic(times, marks, cfg) for cfg in CONFIGS],
                "measured_span": float(times[-1] - times[128]),
            }
        )
        for mode in MODES:
            differences = []
            for seed in (None,) if mode == "observed" else range(92000, 92003):
                pairs = controlled_marks(times, marks, mode, seed)
                # An exact multiset assertion, not just a moments check.
                assert sorted(map(tuple, pairs)) == sorted(map(tuple, marks))
                metrics = [replay_metrics(times, pairs, cfg) for cfg in CONFIGS]
                differences.append(metrics[1]["slo_fraction"] - metrics[0]["slo_fraction"])
                rows.append(
                    {
                        "day": day,
                        "mode": mode,
                        "seed": seed,
                        "configs": metrics,
                        "log_output_lag1": lag_correlation(np.log1p(pairs[:, 1])),
                        "joint_mean": pairs.mean(axis=0).tolist(),
                        "mean_so": float(np.mean(pairs[:, 0] * pairs[:, 1])),
                        "arrival_span": float(times[-1]),
                    }
                )
            day_differences[mode].append(float(np.mean(differences)))
        print(f"GL2 day {day} complete", flush=True)
    summaries = {mode: paired_interval(values) for mode, values in day_differences.items()}
    summaries["global_minus_observed"] = paired_interval(
        np.array(day_differences["global"]) - day_differences["observed"]
    )
    observed, shuffled = summaries["observed"]["ci95"], summaries["global"]["ci95"]
    reversal = (observed[0] > 0 > shuffled[1]) or (shuffled[0] > 0 > observed[1])
    practical = abs(summaries["global_minus_observed"]["mean"]) >= 0.05
    return {
        "schema_version": 1,
        "source_sha256": digest,
        "audit": trace.audit,
        "cohorts": cohorts,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "protocol": {
            "days": DAYS,
            "configs": CONFIGS,
            "costs": COSTS,
            "slo": 1,
            "cohort": 1152,
            "permutation_population": "measured suffix only; warmup fixed",
            "warmup": 128,
            "seeds": [92000, 92001, 92002],
        },
        "rows": rows,
        "day_differences": day_differences,
        "summaries": summaries,
        "rank_reversal_confirmed": bool(reversal and practical),
        "scope": "One BurstGPT successful API model; synthetic costs; no GPU or tail certificate",
    }


def main():
    """Write only aggregated scientific results; raw data remain private."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/research/llm-mark-order-results-2026.json"))
    args = parser.parse_args()
    result = run(args.trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summaries"], indent=2))


if __name__ == "__main__":
    main()
