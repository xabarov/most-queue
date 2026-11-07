"""Scan every integer batch cap and return the smallest individually feasible one.

No monotonicity is assumed; a batch cap is not itself a GPU-memory estimate.
The legacy numerical example uses one fixed stationary truncation, without a
certified infinite-queue error. See examples.bulk_sla_revision for refinement,
all-arrival outcomes, competing moment fits and independent cohort simulation.
"""

import json
from pathlib import Path

from examples.batch_service_sla_exact_tail_comparison import (
    ALPHA,
    ERLANG_K,
    H2_CV,
    LAM,
    QUEUE_TRUNCATION,
    TAU0,
    erlang_rate_fn,
    h2_params_fn,
)
from most_queue.theory.batch.bulk_service_erlang import BulkServiceErlangCalc
from most_queue.theory.batch.bulk_service_h2 import BulkServiceH2Calc

B_MAX_VALUES = tuple(range(2, 129))
DEADLINES = (5.0, 10.0, 20.0, 30.0, 50.0)
SLA_TARGETS = (1e-2, 1e-3, 1e-4)
OUTPUT_PATH = Path("works/batch_service_sla_exact_tail/optimal_bmax_corrected.json")


def tails_by_b_max(regime: str) -> dict[int, dict[float, float]]:
    """Exact P(W>D) for every (b_max, D) in the grid, for one variance regime."""
    out = {}
    for b_max in B_MAX_VALUES:
        if regime == "erlang_low_variance":
            calc = BulkServiceErlangCalc(a=1, b=b_max, k=ERLANG_K, queue_truncation=QUEUE_TRUNCATION)
            calc.set_sources(LAM)
            calc.set_servers(erlang_rate_fn)
        else:
            calc = BulkServiceH2Calc(a=1, b=b_max, queue_truncation=QUEUE_TRUNCATION)
            calc.set_sources(LAM)
            calc.set_servers(
                p1=lambda s: h2_params_fn(s)[0],
                mu1=lambda s: h2_params_fn(s)[1],
                mu2=lambda s: h2_params_fn(s)[2],
            )
        out[b_max] = {d: calc.get_tail(d) for d in DEADLINES}
    return out


def minimal_b_max_for_sla(tails: dict[int, dict[float, float]], deadline: float, target: float) -> int | None:
    """Smallest supplied candidate satisfying the target, without a monotonicity assumption."""
    b_maxes = sorted(tails)
    for b_max in b_maxes:
        if tails[b_max][deadline] <= target:
            return b_max
    return None


def is_monotone(tails: dict[int, dict[float, float]]) -> bool:
    """Describe, but do not assume, monotonicity on the tested grid."""
    b_maxes = sorted(tails)
    return all(tails[b_maxes[i]][d] >= tails[b_maxes[i + 1]][d] for d in DEADLINES for i in range(len(b_maxes) - 1))


def main():
    """Save the corrected integer-cap scan separately from historical results."""
    result = {
        "parameters": {
            "lambda": LAM,
            "alpha": ALPHA,
            "tau0": TAU0,
            "erlang_k": ERLANG_K,
            "h2_cv": H2_CV,
            "b_max_values": list(B_MAX_VALUES),
            "deadlines": list(DEADLINES),
            "sla_targets": list(SLA_TARGETS),
        },
        "regimes": {},
    }
    for regime in ("erlang_low_variance", "h2_high_variance"):
        tails = tails_by_b_max(regime)
        result["regimes"][regime] = {
            "monotone_in_grid": is_monotone(tails),
            "exact_tail_by_b_max": {str(b): {str(d): values[d] for d in DEADLINES} for b, values in tails.items()},
            "minimal_b_max_by_deadline": {
                str(d): {str(eps): minimal_b_max_for_sla(tails, d, eps) for eps in SLA_TARGETS} for d in DEADLINES
            },
        }

    print(json.dumps(result, indent=2))
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
