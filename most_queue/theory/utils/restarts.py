"""
Utilities for queueing models with Poisson restarts.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from most_queue.theory.utils.derivative import derivative


def beff_pls_from_beta(beta_pls: Callable[[float], complex], s: float, r: float) -> complex:
    """
    Effective-service LST for the 'interrupt-repeat' (Poisson restart) model.

    If the original service time B has LST β(s) and restarts happen during service
    with rate r, then the completion-time LST is:

        B_eff^*(s) = β(s+r) * (s+r) / (s + r * β(s+r)).
    """
    if r == 0:
        return beta_pls(s)
    sr = s + r
    beta_sr = beta_pls(sr)
    return beta_sr * sr / (s + r * beta_sr)


def beta_pls_from_moments(mean: float, var: float) -> Callable[[float], float]:
    """
    Build a simple β(s) approximation from the first two moments.

    - If variance is available: Gamma approximation with mean/variance match.
    - Zero variance: deterministic service with the same mean.
    """
    mean_b = float(mean)
    var_b = float(var)

    if var_b > 0.0 and mean_b > 0.0:
        k = mean_b * mean_b / var_b
        theta = var_b / mean_b

        def beta(s: float) -> float:
            return float((1.0 + theta * s) ** (-k))

        return beta

    def beta_exp(s: float) -> float:
        return float(np.exp(-mean_b * s))

    return beta_exp


def raw_moments_from_pls(
    pls: Callable[[float], complex | float],
    num_of_moments: int,
    *,
    dx: float,
    order: int = 9,
) -> list[float]:
    """
    Compute raw moments from an LST by numerical differentiation at s=0:
        E[T^n] = (-1)^n d^n/ds^n (T^*(s))|_{s=0}.
    """
    n_mom = int(num_of_moments)
    if n_mom <= 0:
        return []

    out: list[float] = [0.0] * n_mom
    for i in range(n_mom):
        val = derivative(pls, 0, dx=dx, n=i + 1, order=order)
        if i % 2 == 0:
            val = -val
        out[i] = float(np.real(val))
    return out


def beff_moments_repeat_without_resampling_from_moments(
    mean: float,
    var: float,
    r: float,
    *,
    num_of_moments: int,
) -> list[float]:
    """
    Effective completion-time moments for *preemptive-repeat without resampling*.

    Semantics:
    - Service requirement B is sampled ONCE for the job.
    - During service, Poisson interruptions with rate r cause a restart.
    - Restart loses all progress, but DOES NOT resample B (the same initial B repeats).

    For a fixed B=b the completion time T has:
      E[T|b]  = (e^{r b} - 1) / r,
      E[T^2|b]= (2 e^{r b} / r^2) (e^{r b} - 1 - r b).

    To obtain a practical approximation from moments only, we approximate B by a Gamma law
    matching the given mean/variance, compute E[T] and E[T^2] in closed form, then fit a
    Gamma law for T and return its raw moments.

    Notes:
    - Moments can be infinite if the MGF of B at r (or 2r) does not exist. For Gamma(k,θ),
      this happens when r >= 1/θ (mean infinite) or 2r >= 1/θ (second moment infinite).
    """
    n_mom = int(num_of_moments)
    if n_mom <= 0:
        return []

    mean_b = float(mean)
    var_b = float(var)
    r = float(r)

    if r < 0:
        raise ValueError("Restart intensity must be nonnegative.")
    if r == 0.0:
        # No interruptions: Gamma moment fit, or deterministic at zero variance.
        if var_b > 0.0 and mean_b > 0.0:
            k_b = mean_b * mean_b / var_b
            theta_b = var_b / mean_b
            out = []
            prod = 1.0
            for m in range(1, n_mom + 1):
                prod *= k_b + (m - 1)
                out.append(float((theta_b**m) * prod))
            return out
        return [mean_b**m for m in range(1, n_mom + 1)]

    if var_b == 0:
        t1 = math.expm1(r * mean_b) / r
        t2 = 2 * math.exp(r * mean_b) * (math.expm1(r * mean_b) - r * mean_b) / r**2
        var_t = t2 - t1 * t1
        shape, scale = t1 * t1 / var_t, var_t / t1
        out, prod = [], 1.0
        for m in range(1, n_mom + 1):
            prod *= shape + m - 1
            out.append(scale**m * prod)
        return out

    # Gamma approximation for B: shape k, scale theta
    if var_b > 0.0 and mean_b > 0.0:
        k = mean_b * mean_b / var_b
        theta = var_b / mean_b
    else:
        # Exponential fallback: k=1, theta=mean
        k = 1.0
        theta = max(mean_b, 0.0)

    # Check existence of MGF at r and 2r for Gamma(k,theta).
    if theta <= 0.0:
        return [0.0] * n_mom

    if theta * r >= 1.0 - 1e-15:
        return [float("inf")] * n_mom

    m1 = (1.0 - theta * r) ** (-k)  # E[e^{rB}]
    t1 = (m1 - 1.0) / r

    # E[B e^{rB}] = d/dr E[e^{rB}] = k*theta*(1-theta*r)^(-k-1)
    ebe = k * theta * (1.0 - theta * r) ** (-(k + 1.0))

    if 2.0 * theta * r >= 1.0 - 1e-15:
        # Second moment infinite -> variance infinite
        return [float(t1)] + [float("inf")] * (n_mom - 1)

    m2 = (1.0 - 2.0 * theta * r) ** (-k)  # E[e^{2rB}]
    t2 = (2.0 / (r * r)) * (m2 - m1 - r * ebe)  # E[T^2]

    var_t = t2 - t1 * t1
    if not np.isfinite(var_t) or var_t <= 0.0 or not np.isfinite(t1) or t1 <= 0.0:
        # Degenerate fallback (should be rare)
        var_t = max(0.0, float(var_t) if np.isfinite(var_t) else 0.0)
        if var_t <= 0.0:
            return [float(t1**m) for m in range(1, n_mom + 1)]

    # Gamma fit for T
    k_t = (t1 * t1) / var_t
    theta_t = var_t / t1

    out: list[float] = []
    prod = 1.0
    for m in range(1, n_mom + 1):
        prod *= k_t + (m - 1)
        out.append(float((theta_t**m) * prod))
    return out


def beff_moments_repeat_without_resampling_from_h2(
    y: list[complex] | tuple[complex, ...],
    mu: list[complex] | tuple[complex, ...],
    r: float,
    *,
    num_of_moments: int,
) -> list[float]:
    """
    Same as `beff_moments_repeat_without_resampling_from_moments`, but uses an explicit H2
    (mixture of exponentials) representation of B to compute MGF terms robustly:

      B ~ H2(y, mu),  beta(s)=Σ y_i * mu_i/(mu_i+s).
      M_B(t)=E[e^{tB}]=Σ y_i * mu_i/(mu_i - t).

    Requires a real probability mixture for positive r, with moments evaluated
    only in their convergence domain. Moments 1 and 2 are computed in closed
    form; the Gamma fit supplies approximate moments of higher order.
    """
    n_mom = int(num_of_moments)
    if n_mom <= 0:
        return []

    r = float(r)
    if r < 0:
        raise ValueError("Restart intensity must be nonnegative.")
    if r == 0:
        return [
            float(np.real(math.factorial(k) * sum(yi / mui**k for yi, mui in zip(y, mu)))) for k in range(1, n_mom + 1)
        ]

    if not is_real_h2(y, mu):
        raise ValueError("Moment existence under fixed-service restart requires a real H2 distribution.")
    active_rates = [float(np.real(mui)) for yi, mui in zip(y, mu) if float(np.real(yi)) > 0]
    if r >= min(active_rates):
        return [float("inf")] * n_mom
    if 2 * r >= min(active_rates):
        t1 = float(np.real(sum(yi * mui / (mui - r) for yi, mui in zip(y, mu)) - 1)) / r
        return [t1] + [float("inf")] * (n_mom - 1)

    # Compute MGF terms for H2 mixture.
    m1 = 0.0 + 0.0j  # E[e^{rB}]
    m2 = 0.0 + 0.0j  # E[e^{2rB}]
    ebe = 0.0 + 0.0j  # E[B e^{rB}]
    for yi, mui in zip(y, mu):
        den1 = mui - r
        den2 = mui - 2.0 * r
        # If den is ~0, MGF diverges -> treat as inf.
        if abs(den1) <= 1e-15 or abs(den2) <= 1e-15:
            return [float("inf")] * n_mom
        m1 += yi * (mui / den1)
        m2 += yi * (mui / den2)
        ebe += yi * (mui / (den1 * den1))

    t1 = (m1 - 1.0) / r
    t2 = (2.0 / (r * r)) * (m2 - m1 - r * ebe)

    t1r = float(np.real(t1))
    t2r = float(np.real(t2))
    if not np.isfinite(t1r) or not np.isfinite(t2r) or t1r <= 0.0:
        return [float("inf")] * n_mom

    var_t = t2r - t1r * t1r
    if not np.isfinite(var_t) or var_t <= 0.0:
        # Degenerate fallback
        return [float(t1r**m) for m in range(1, n_mom + 1)]

    k_t = (t1r * t1r) / var_t
    theta_t = var_t / t1r

    out: list[float] = []
    prod = 1.0
    for m in range(1, n_mom + 1):
        prod *= k_t + (m - 1)
        out.append(float((theta_t**m) * prod))
    return out


def is_real_h2(y, mu):
    """Whether an algebraic two-phase fit is a probability mixture."""
    return (
        all(abs(complex(x).imag) < 1e-10 for x in [*y, *mu])
        and all(0 <= float(np.real(x)) <= 1 for x in y)
        and all(float(np.real(x)) > 0 for x in mu)
        and abs(sum(y) - 1) < 1e-10
    )


def effective_service_moments(b, y, mu, r, *, fixed=False, num_of_moments=4):
    """Real H2 when available, otherwise Gamma matching mean/variance.

    Resampling moments use analytic series division. Fixed-service moments 1 and 2
    are exact for the selected input law; higher moments are a Gamma closure.
    """
    order = num_of_moments
    if r == 0:
        return [float(x) for x in b[:order]]
    if r < 0:
        raise ValueError("Restart intensity must be nonnegative.")
    real_h2 = is_real_h2(y, mu)
    mean, var = float(b[0]), float(b[1] - b[0] ** 2)
    if fixed:
        if real_h2:
            return beff_moments_repeat_without_resampling_from_h2(y, mu, r, num_of_moments=order)
        return beff_moments_repeat_without_resampling_from_moments(mean, var, r, num_of_moments=order)
    r = np.longdouble(r)
    if real_h2:
        real_y = [np.longdouble(np.real(x)) for x in y]
        real_mu = [np.longdouble(np.real(x)) for x in mu]
        c = [
            sum(yi * mui * (-1) ** k / (mui + r) ** (k + 1) for yi, mui in zip(real_y, real_mu))
            for k in range(order + 1)
        ]
    elif var > 0:
        shape, scale = np.longdouble(mean * mean / var), np.longdouble(var / mean)
        c = [(1 + scale * r) ** (-shape)]
        for k in range(1, order + 1):
            c.append(-c[-1] * (shape + k - 1) * scale / (k * (1 + scale * r)))
    else:
        c = [math.exp(-mean * r) * (-mean) ** k / math.factorial(k) for k in range(order + 1)]
    numerator = [r * c[k] + (c[k - 1] if k else 0) for k in range(order + 1)]
    denominator = [r * c[k] + (1 if k == 1 else 0) for k in range(order + 1)]
    if abs(denominator[0]) < 1e-300:
        raise ValueError("Effective-service transform is numerically singular.")
    quotient = []
    for k in range(order + 1):
        quotient.append(
            (numerator[k] - sum(denominator[j] * quotient[k - j] for j in range(1, k + 1))) / denominator[0]
        )
    return [float(np.real((-1) ** k * math.factorial(k) * quotient[k])) for k in range(1, order + 1)]
