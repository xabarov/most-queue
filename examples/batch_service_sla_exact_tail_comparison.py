"""EPIC-066: exact batch-service SLA tail vs Inoue's (2021) mean bound and vs
fit-based SLA approximation.

    python -m examples.batch_service_sla_exact_tail_comparison

Pure synthetic numerical study (no external data, no randomness): deterministic
given the prescribed parameters below, so no hashing/manifest machinery is
needed for reproducibility (identical code -> identical output).

Reuses Inoue Y., "Queueing Analysis of GPU-Based Inference Servers with
Dynamic Batching: A Closed-Form Characterization", Performance Evaluation 147
(2021), doi:10.1016/j.peva.2020.102183 (open, arXiv:1912.06322) -- Theorem 2's
closed-form upper bound phi(lambda, alpha, tau0) for E[W], derived under an
UNBOUNDED max batch size and a DETERMINISTIC, linear-in-batch-size processing
time H^[b] = alpha*b + tau0. We keep the SAME mean function (so the comparison
isolates the effect of a bounded [1, b_max] window and of nonzero service
variance, not a different mean model) but use our exact, phase-type
(Erlang-fitted) CTMC to get the TRUE E[W] and the EXACT SLA tail P(W>D) under
a REALISTIC bounded window -- something Inoue's own closed form cannot do (his
Fig. 8 shows it is only a good approximation for large b_max).
"""

import json
from pathlib import Path

from most_queue.random.distributions import H2Distribution
from most_queue.theory.batch.bulk_service_erlang import BulkServiceErlangCalc
from most_queue.theory.batch.bulk_service_h2 import BulkServiceH2Calc
from most_queue.theory.utils.sla import deadline_violation_prob

# Prescribed parameters (fixed before looking at any result below).
ALPHA, TAU0 = 0.1438, 1.8874  # Inoue (2021) Table 1(a)/Fig. 9 least-squares fit, Tesla V100 ResNet50
LAM = 0.8  # stable even at the smallest b_max tested (max throughput there is ~0.92)
B_MAX_VALUES = (2, 4, 8, 16, 32, 64, 128)  # 128 stands in for Inoue's own Fig. 8/9 "large b_max"
DEADLINES = (5.0, 10.0, 20.0)
ERLANG_K = 10  # CV = 1/sqrt(10) = 0.316, a low-variance stand-in for "deterministic"
H2_CV = 2.0  # a genuinely high-variance case Inoue's model cannot express at all
QUEUE_TRUNCATION = 400


def inoue_phi(lam, alpha, tau0):
    """Theorem 2 (Inoue 2021): closed-form upper bound for E[W], b_max = infinity."""
    rho = lam * alpha
    if rho >= 1:
        raise ValueError("unstable: lambda * alpha must be < 1")
    phi0 = (alpha + tau0) / (2 * (1 - rho)) * (1 + 2 * lam * tau0 + (1 - lam * tau0) / (1 + lam * tau0))
    phi1 = 1.5 * tau0 / (1 - rho) + 0.5 * alpha * (rho + 2) / (1 - rho**2)
    return min(phi0, phi1)


def erlang_rate_fn(size):
    """Per-phase Erlang rate giving mean batch time alpha*size + tau0 at k=ERLANG_K phases."""
    return ERLANG_K / (ALPHA * size + TAU0)


def h2_params_fn(size):
    """H2(p1, mu1, mu2) matched to mean alpha*size+tau0 and the prescribed CV (Aliev's method)."""
    mean = ALPHA * size + TAU0
    params = H2Distribution.get_params_by_mean_and_cv(mean, H2_CV)
    return params.p1, params.mu1, params.mu2


def run_erlang(b_max):
    """Exact mean/tail for the low-variance (Erlang) case at one b_max, plus the fit comparison."""
    calc = BulkServiceErlangCalc(a=1, b=b_max, k=ERLANG_K, queue_truncation=QUEUE_TRUNCATION)
    calc.set_sources(LAM)
    calc.set_servers(erlang_rate_fn)
    w_moments = calc.get_w(num=2)
    exact_tail = {d: calc.get_tail(d) for d in DEADLINES}
    fitted_tail = {d: deadline_violation_prob(w_moments, d) for d in DEADLINES}
    return {
        "b_max": b_max,
        "exact_mean_w": w_moments[0],
        "exact_tail": exact_tail,
        "fitted_tail": fitted_tail,
        "relative_error_of_fit": {
            d: fitted_tail[d] / exact_tail[d] - 1 if exact_tail[d] > 0 else None for d in DEADLINES
        },
    }


def run_h2(b_max):
    """Exact mean/tail for the high-variance (H2, CV=H2_CV) case at one b_max."""
    calc = BulkServiceH2Calc(a=1, b=b_max, queue_truncation=QUEUE_TRUNCATION)
    calc.set_sources(LAM)
    calc.set_servers(p1=h2_params_fn_p1, mu1=h2_params_fn_mu1, mu2=h2_params_fn_mu2)
    exact_tail = {d: calc.get_tail(d) for d in DEADLINES}
    mean_w = calc.run().w[0]
    return {"b_max": b_max, "exact_mean_w": mean_w, "exact_tail": exact_tail}


def h2_params_fn_p1(size):
    """p1 component of h2_params_fn, as a standalone callable for set_servers."""
    return h2_params_fn(size)[0]


def h2_params_fn_mu1(size):
    """mu1 component of h2_params_fn, as a standalone callable for set_servers."""
    return h2_params_fn(size)[1]


def h2_params_fn_mu2(size):
    """mu2 component of h2_params_fn, as a standalone callable for set_servers."""
    return h2_params_fn(size)[2]


def main():
    """Run both variance regimes across the prescribed b_max sweep and write the comparison."""
    inoue_bound = inoue_phi(LAM, ALPHA, TAU0)
    erlang_rows = [run_erlang(b) for b in B_MAX_VALUES]
    h2_rows = [run_h2(b) for b in B_MAX_VALUES]

    result = {
        "parameters": {
            "lambda": LAM,
            "alpha": ALPHA,
            "tau0": TAU0,
            "rho": LAM * ALPHA,
            "erlang_k": ERLANG_K,
            "h2_cv": H2_CV,
            "b_max_values": list(B_MAX_VALUES),
            "deadlines": list(DEADLINES),
        },
        "inoue_2021_bound_e_w": inoue_bound,
        "erlang_low_variance": erlang_rows,
        "h2_high_variance": h2_rows,
        "interpretation": "Same mean function (alpha*b+tau0) as Inoue (2021) Theorem 2; bounded "
        "[1,b_max] window and phase-type variance are the only things this adds relative to his "
        "unbounded, deterministic assumption. Not a production GPU/LLM latency claim -- a controlled "
        "comparison of two closed-form treatments of the same mean model.",
    }
    print(json.dumps(result, indent=2))

    output_dir = Path("works/batch_service_sla_exact_tail")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
