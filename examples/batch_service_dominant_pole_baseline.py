"""EPIC-066 follow-up: single-dominant-pole tail approximation as a quantitative baseline.

    python -m examples.batch_service_dominant_pole_baseline

Claeys et al. (2012, Computers & OR 39(11):2733-2741 -- open-access precursor of the
2013 batch-size-dependent paper we cite as [13]/claeys2013) approximate the delay tail
via the DOMINANT SINGULARITY of the delay's generating function: P(delay>n) ~ C*eta^-n
for discrete n, i.e. a single-exponential/saddle-point approximation governed by the
pole closest to the origin. We cannot reproduce their exact batch-size-dependent
formula (the 2013 extension is paywalled), so this is NOT a numerical reproduction of
their published figures -- it is an honest, from-scratch implementation of the SAME
general technique (single-dominant-pole tail approximation), applied fairly to our
continuous-time, a=1 model, using the exact decay rate we derived and verified in
docs/roadmaps/batch_service_sla_exact_tail_part2_roadmap.md (direction 1/C1):

    eta(b_max) = rate(b_max)                       for the E_k case
    eta(b_max) = min(mu1(b_max), mu2(b_max))        for the H2 case

calibrated at t=0 the same way the exact M/M/1 tail is (P(W>0) = P(busy) = rho), giving
the single-exponential approximation

    P_approx(W>t) = rho * exp(-eta * t)

This is compared against the EXACT tail already computed in
batch_service_sla_exact_tail_comparison.py. We already showed (direction 1/C1 of the
roadmap) that eta is the TRUE t->inf asymptotic rate but the crossover point before it
governs the curve is far beyond realistic SLA deadlines for small b_max -- so this
experiment is also a direct, quantitative demonstration of exactly how bad a
dominant-pole approximation is in the practically relevant regime, not just a
qualitative claim.
"""

import json
from pathlib import Path

from examples.batch_service_sla_exact_tail_comparison import (
    ALPHA,
    DEADLINES,
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

COMPARISON_PATH = Path("works/batch_service_sla_exact_tail/comparison.json")
OUTPUT_PATH = Path("works/batch_service_sla_exact_tail/dominant_pole_baseline.json")
B_MAX_VALUES = (2, 4, 8, 16, 32, 64, 128)


def run_erlang(b_max):
    calc = BulkServiceErlangCalc(a=1, b=b_max, k=ERLANG_K, queue_truncation=QUEUE_TRUNCATION)
    calc.set_sources(LAM)
    calc.set_servers(erlang_rate_fn)
    rho = calc.run().utilization
    eta = erlang_rate_fn(b_max)
    return rho, eta


def run_h2(b_max):
    calc = BulkServiceH2Calc(a=1, b=b_max, queue_truncation=QUEUE_TRUNCATION)
    calc.set_sources(LAM)
    calc.set_servers(
        p1=lambda s: h2_params_fn(s)[0],
        mu1=lambda s: h2_params_fn(s)[1],
        mu2=lambda s: h2_params_fn(s)[2],
    )
    rho = calc.run().utilization
    _, mu1, mu2 = h2_params_fn(b_max)
    eta = min(mu1, mu2)
    return rho, eta


def main():
    comparison = json.loads(COMPARISON_PATH.read_text())
    result = {"b_max_values": list(B_MAX_VALUES), "deadlines": list(DEADLINES), "regimes": {}}

    for regime, runner in (("erlang_low_variance", run_erlang), ("h2_high_variance", run_h2)):
        exact_by_b_max = {row["b_max"]: row["exact_tail"] for row in comparison[regime]}
        rows = []
        for b_max in B_MAX_VALUES:
            rho, eta = runner(b_max)
            approx_tail = {str(d): rho * pow(2.718281828459045, -eta * d) for d in DEADLINES}
            exact_tail = exact_by_b_max[b_max]
            rows.append(
                {
                    "b_max": b_max,
                    "rho": rho,
                    "eta": eta,
                    "approx_tail": approx_tail,
                    "exact_tail": exact_tail,
                    "relative_error_of_approx": {
                        str(d): approx_tail[str(d)] / exact_tail[str(d)] - 1 for d in DEADLINES
                    },
                }
            )
        result["regimes"][regime] = rows

    print(json.dumps(result, indent=2))
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
