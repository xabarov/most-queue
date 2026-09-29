"""
Generic numeric primitives for working with distribution tails.

``upper_bound`` -- bisection search for the smallest ``x`` such that the tail
``1 - CDF(x)`` drops below a given probability. Used both by the size-based
(SRPT/SJF/...) grid precomputation and by the SLA/deadline-violation layer
(``most_queue.theory.utils.sla``).
"""

from __future__ import annotations

from collections.abc import Callable

CdfFn = Callable[[float], float]


def upper_bound(cdf_fn: CdfFn, p: float = 1e-7, start: float = 1.0, max_iter: int = 100) -> float:
    """
    Find the smallest x such that the tail 1 - CDF(x) < p.

    Uses exponential expansion to bracket the root, then 80 bisection steps.
    """
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    if start <= 0:
        raise ValueError("start must be positive")

    left = 0.0
    right = float(start)

    for _ in range(max_iter):
        if 1.0 - float(cdf_fn(right)) < p:
            break
        left = right
        right *= 2.0
    else:
        raise ValueError("Could not find a finite upper integration bound")

    for _ in range(80):
        mid = 0.5 * (left + right)
        if 1.0 - float(cdf_fn(mid)) < p:
            right = mid
        else:
            left = mid

    return right
