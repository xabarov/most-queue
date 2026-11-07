"""Bounds for truncating the behind-customer count of an impatient tagged queue.

Until dispatch, behind customers form an immigration/death process (lambda,
k*gamma). No batch ahead consumes them. Couple the version suppressing births
at C with the infinite process: tagged outcomes agree until the first attempted
birth from C, H_C. Independent tag patience A~Exp(gamma) gives the uniform
dispatch-event error bound P(H_C<A). The finite killed birth/death chain also
computes E[A**n;H_C<A], bounding unnormalised served-wait moment errors.

These mathematical bounds use ordinary floating evaluation and do not bound the
separate truncation of the stationary queue-length distribution.
"""

import math
from functools import lru_cache

import numpy as np
from scipy.linalg import solve_banded


@lru_cache(maxsize=256)
def behind_error(lam: float, gamma: float, cap: int, num: int = 0) -> tuple[float, ...]:
    """Return (P(H_C<A), E[A;H_C<A], ..., E[A**num;H_C<A])."""
    if not np.isfinite(lam) or not np.isfinite(gamma) or lam <= 0 or gamma <= 0:
        raise ValueError("behind-count bounds require finite positive lambda and gamma")
    if isinstance(cap, bool) or not isinstance(cap, (int, np.integer)) or cap < 0:
        raise ValueError("cap must be a nonnegative integer")
    if isinstance(num, bool) or not isinstance(num, int) or num < 0:
        raise ValueError("num must be a nonnegative integer")
    matrix = np.zeros((3, cap + 1))
    matrix[0, 1:] = -lam
    matrix[1] = lam + gamma * (np.arange(cap + 1) + 1)
    matrix[2, :-1] = -gamma * np.arange(1, cap + 1)
    x = np.zeros(cap + 1)
    x[-1] = lam
    laplace_moments = []
    for _ in range(num + 1):
        x = solve_banded((1, 1), matrix, x)
        laplace_moments.append(float(x[0]))  # E[H_C**n exp(-gamma H_C)] / n!
    values = tuple(
        math.factorial(n) * sum(laplace_moments[k] / gamma ** (n - k) for k in range(n + 1)) for n in range(num + 1)
    )
    if not all(np.isfinite(v) and v >= 0 for v in values) or values[0] > 1 + 1e-12:
        raise ValueError("invalid behind-count truncation bound")
    return values


@lru_cache(maxsize=256)
def behind_cap(a: int, lam: float, gamma: float, tolerance: float = 1e-13) -> int:
    """Smallest C>=a-1 whose dispatch-event error is at most tolerance.

    At a=1 behind customers are irrelevant. Without impatience, saturation at
    a-1 is an exact lumping; this does not hold at a>1, gamma>0.
    """
    if isinstance(a, bool) or not isinstance(a, (int, np.integer)) or a < 1:
        raise ValueError("a must be a positive integer")
    if not np.isfinite(lam) or lam <= 0 or not np.isfinite(gamma) or gamma < 0:
        raise ValueError("require finite positive lambda and nonnegative gamma")
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("tolerance must be finite and in (0, 1)")
    if a == 1:
        return 0
    if gamma == 0:
        return a - 1
    lo, hi = a - 2, max(a - 1, 8)
    while behind_error(lam, gamma, hi)[0] > tolerance:
        lo, hi = hi, 2 * hi
        if hi > 4096:
            raise ValueError("behind-count budget exceeded; use a different solver")
    while hi - lo > 1:
        middle = (lo + hi) // 2
        if behind_error(lam, gamma, middle)[0] <= tolerance:
            hi = middle
        else:
            lo = middle
    return hi
