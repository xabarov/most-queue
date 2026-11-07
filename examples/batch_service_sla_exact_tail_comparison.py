"""Historical EPIC-066 scenario, corrected metric and Inoue formula.

Inoue's W denotes SOJOURN, not queue waiting. Its upper bound applies only to
unbounded deterministic affine batch service. Finite-cap PH waiting values
below are different quantities, NOT tests or violations of that bound.
A finite stationary truncation is not an exact infinite-queue calculation.
Use examples.bulk_sla_revision for matched metrics and convergence diagnostics.
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
    """Theorem 2 (Inoue 2021): upper bound for mean SOJOURN, unbounded deterministic affine service."""
    rho = lam * alpha
    if rho >= 1:
        raise ValueError("unstable: lambda * alpha must be < 1")
    phi0 = (alpha + tau0) / (2 * (1 - rho)) * (1 + 2 * lam * tau0 + (1 - lam * tau0) / (1 + lam * alpha))
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
        "inoue_2021_bound_mean_sojourn": inoue_bound,
        "erlang_low_variance": erlang_rows,
        "h2_high_variance": h2_rows,
        "interpretation": "Inoue bound is for unbounded deterministic SOJOURN. PH finite-cap "
        "WAITING results have a different metric and model; do not compare as an inequality.",
    }
    print(json.dumps(result, indent=2))

    output_dir = Path("works/batch_service_sla_exact_tail")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_corrected_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
