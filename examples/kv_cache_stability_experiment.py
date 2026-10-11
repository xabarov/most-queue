"""Reproducible KV stability/flow experiment; no production GPU measurements.

Run from repository root with PYTHONPATH=.: analytical bounds for correlated
joint lengths, fixed-horizon slot replications, and independent stage-area
audits. Stored summaries do not claim to prove positive recurrence.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from most_queue.sim.kv_cache_stability import KvCacheStabilitySim
from most_queue.theory.continuous_batching import KvCacheStabilityCalc


def experiment(slots=2000, replications=4):
    """Return 36 synthetic settings; preserve means, covariance and delta."""
    workloads = {
        "fixed": {(4, 2): 1},
        "positive_correlation": {(2, 2): 0.5, (6, 6): 0.5},
        "negative_correlation": {(2, 6): 0.5, (6, 2): 0.5},
    }
    rows = []
    for name, pmf in workloads.items():
        for memory in (40, 200):
            for chunk in (1, 2):
                calc = KvCacheStabilityCalc(memory, chunk)
                calc.set_sources(1)
                calc.set_servers(pmf, slot_time=1)
                bounds = calc.run()
                rates = {
                    "below_lower": 0.65 * bounds.capacity_lower,
                    "gap": (bounds.capacity_lower + bounds.capacity_upper) / 2,
                    "above_upper": 1.2 * bounds.capacity_upper,
                }
                stages = {f"{s},{o}": list(range(chunk, s + 1, chunk)) + list(range(s + 1, s + o + 1)) for s, o in pmf}
                for regime, rate in rates.items():
                    calc.set_sources(rate)
                    analytic = asdict(calc.run())
                    analytic.pop("duration")
                    samples = []
                    for replica in range(replications):
                        seed = 2000 + 10 * len(rows) + replica
                        sim = KvCacheStabilitySim(memory, chunk, seed=seed)
                        sim.set_sources(rate)
                        sim.set_servers(pmf, 1)
                        sample = asdict(sim.run(slots, warmup_slots=slots // 5))
                        assert (
                            sample["input_memory_area"]
                            == sample["fulfilled_memory_area"] + sample["outstanding_memory_area"]
                        )
                        assert sample["arrived"] == sample["completed"] + sample["unfinished"]
                        sample.pop("duration")
                        sample.pop("slot_trace")
                        sample["seed"] = seed
                        samples.append(sample)
                    throughputs = [r["throughput"] for r in samples]
                    rows.append(
                        {
                            "workload": name,
                            "joint_pmf": [[s, o, p] for (s, o), p in pmf.items()],
                            "memory_tokens": memory,
                            "chunk_size": chunk,
                            "slot_time": 1,
                            "arrival_rate": rate,
                            "regime": regime,
                            "stage_footprints": stages,
                            "analytic": analytic,
                            "samples": samples,
                            "throughput_mean": float(np.mean(throughputs)),
                            "throughput_se": float(np.std(throughputs, ddof=1) / np.sqrt(replications)),
                        }
                    )
    return {
        "schema_version": 1,
        "source": "https://arxiv.org/html/2605.04595v1",
        "scope": "Synthetic constant-slot, maximal/free-swap model, not GPU replication or a recurrence proof.",
        "slots": slots,
        "warmup_slots": slots // 5,
        "replications": replications,
        "rows": rows,
    }


def main():
    """Write the generated research summary to a user-selected path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", type=int, default=2000)
    parser.add_argument("--replications", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("docs/research/kv-cache-stability-results-2026.json"))
    args = parser.parse_args()
    if args.slots < 5 or args.replications < 2:
        parser.error("need at least five slots and two replications")
    result = experiment(args.slots, args.replications)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(result['rows'])} settings to {args.output}")


if __name__ == "__main__":
    main()
