"""EPIC-066 follow-up: minimal b_max meeting a target SLA, from the exact tail.

    python -m examples.batch_service_optimal_bmax

Turns the exact tail into a prescriptive answer: given a deadline D and an SLA target
eps (P(W>D) <= eps), what is the SMALLEST b_max -- hence the smallest per-batch GPU
memory footprint -- that meets it? No new theory: a straight scan of the exact
`get_tail` over the finite b_max grid used throughout EPIC-066
(batch_service_sla_exact_tail_comparison.py), extended with a few larger deadlines so
the high-variance (H2) regime's SLA-crossing point is actually visible (at the
original D in {5,10,20} it never crosses the targets tested here -- see
"never_in_grid" entries below and the "Численные эксперименты" section of the
article).

Monotonicity of P(W>D) in b_max is NOT proven in general (see
docs/roadmaps/batch_service_sla_exact_tail_part2_roadmap.md, direction 1/C2 --
reduces to a monotonicity conjecture about the mean batch size, confirmed numerically
but not proven). We therefore do not assume it here: "minimal b_max meeting the SLA"
means the smallest b_max in the grid such that the SLA holds at that b_max AND at
every larger b_max tested -- the smallest point after which the whole remaining grid
is SLA-compliant.
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

B_MAX_VALUES = (2, 4, 8, 16, 32, 64, 128)
DEADLINES = (5.0, 10.0, 20.0, 30.0, 50.0)
SLA_TARGETS = (1e-2, 1e-3, 1e-4)
OUTPUT_PATH = Path("works/batch_service_sla_exact_tail/optimal_bmax.json")


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
    """Smallest b_max after which tail[deadline] <= target holds for the rest of the grid."""
    b_maxes = sorted(tails)
    for idx, b_max in enumerate(b_maxes):
        if all(tails[b][deadline] <= target for b in b_maxes[idx:]):
            return b_max
    return None


def is_monotone(tails: dict[int, dict[float, float]]) -> bool:
    b_maxes = sorted(tails)
    return all(tails[b_maxes[i]][d] >= tails[b_maxes[i + 1]][d] for d in DEADLINES for i in range(len(b_maxes) - 1))


def main():
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
            "exact_tail_by_b_max": {str(b): {str(d): tails[b][d] for d in DEADLINES} for b in tails},
            "minimal_b_max_by_deadline": {
                str(d): {str(eps): minimal_b_max_for_sla(tails, d, eps) for eps in SLA_TARGETS} for d in DEADLINES
            },
        }

    print(json.dumps(result, indent=2))
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
