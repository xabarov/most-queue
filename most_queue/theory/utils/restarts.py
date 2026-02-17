"""
Utilities for queueing models with Poisson restarts.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.misc import derivative


def beff_pls_from_beta(beta_pls: Callable[[float], complex], s: float, r: float) -> complex:
    """
    Effective-service LST for the 'interrupt-repeat' (Poisson restart) model.

    If the original service time B has LST β(s) and restarts happen during service
    with rate r, then the completion-time LST is:

        B_eff^*(s) = β(s+r) * (s+r) / (s + r * β(s+r)).
    """
    sr = s + r
    beta_sr = beta_pls(sr)
    return beta_sr * sr / (s + r * beta_sr)


def beta_pls_from_moments(mean: float, var: float) -> Callable[[float], float]:
    """
    Build a simple β(s) approximation from the first two moments.

    - If variance is available: Gamma approximation with mean/variance match.
    - Otherwise: exponential (degenerate) fallback with the same mean.
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
