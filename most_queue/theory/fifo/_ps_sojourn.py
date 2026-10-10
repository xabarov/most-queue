"""
Exact moments of the CONDITIONAL sojourn time in the M/G/1 egalitarian
processor-sharing queue -- the Yashkov/Ott machinery that
:mod:`most_queue.theory.fifo.mg1_ps` had been documenting as a known gap.

Implementation of
    Yashkov S.F., "Explicit formulas for the moments of the sojourn time in
    the M/G/1 processor sharing queue with permanent jobs", arXiv:math/0512281
    (2005), building on Yashkov S.F., Probl. of Contr. and Info. Theory
    12(2):133-148, 1983, and Yashkov S.F., Queueing Systems 2(1):1-17, 1987.

THE RESULT IS HIS, not this library's. Equation numbers below refer to the
arXiv note.

Why this is not a one-liner. The MEAN conditional sojourn time of a job of size
``u`` is the famous insensitive ``u / (1 - rho)`` -- it does not depend on the
service-time distribution at all beyond its mean. Every higher moment does. The
reason is that a PS job's sojourn depends not only on what it found on arrival
but also on every shorter job that overtakes it later, which is what makes PS
harder to analyse than FCFS and why the exact distribution stayed open for
fifteen years after Kleinrock posed it.

The route used here (Theorems 3.1 and 3.2 of the note):

1. The RECIPROCAL of the conditional sojourn-time LST has a clean power series,

       1 / v(r, u) = sum_n (r^n / n!) xi_n(u),
       xi_n(u) = n / (1-rho)^n * int_0^u (u-y)^(n-1) W^((n-1)*)(y) dy,   (3.2)

   where ``W`` is the ordinary M/G/1-FCFS waiting-time distribution -- the
   Pollaczek-Khinchine one -- and ``W^(m*)`` is its m-fold convolution. The
   equivalent form used in the code,

       xi_n(u) = 1 / (1-rho)^n * E[(u - S_{n-1})^n ; S_{n-1} <= u],

   with ``S_m`` a sum of m i.i.d. FCFS waiting times, follows by Fubini and is
   what makes the phase-type evaluation below possible.

2. Matching the two series against ``v(r, u) * (1/v(r, u)) = 1`` gives the
   moment recursion

       v_n(u) = sum_{i=1..n} C(n,i) v_{n-i}(u) xi_i(u) (-1)^(i+1).         (3.7)

   (Re-derived here rather than taken on trust; see the epic.)

Making it computable. The note itself concludes that the exact expressions
"involve an integration term, making an exact computation difficult from a
practical point of view", and the literature responded with bounds and
approximations. That difficulty disappears once the service time is phase-type,
because then every object in the chain stays phase-type in closed form:

- the equilibrium (excess) distribution of ``PH(tau, T)`` is ``PH(pi, T)`` with
  ``pi = tau (-T)^-1 / mean`` -- same generator, re-weighted entry;
- the M/G/1 waiting time is a geometric sum of those, hence an atom of mass
  ``1-rho`` at zero plus the defective ``PH(rho*pi, T + rho*t*pi)``;
- its m-fold convolution is the usual block-bidiagonal phase-type;
- and ``int_0^u (u-y)^k e^{Ay} dy`` is a single block of one matrix exponential
  (the Van Loan augmented-matrix identity), with no inverse to go singular.

So for phase-type service -- which is how this library represents a general
service time fitted from raw moments -- the moments come out exact, with matrix
exponentials of a handful of small matrices and no quadrature at all.

Permanent jobs. The note also covers ``K`` permanent jobs of infinite size
sharing the processor (Theorem 3.3). Its transform identity
``v_K(r,u) = v(r,u)^(K+1)`` says something stronger and more useful than the two
moment formulas it is used for: the sojourn time with ``K`` permanent jobs is
distributed exactly as a sum of ``K+1`` independent copies of the sojourn time
without them. Here that is realised by raising the ``1/v`` series to the power
``K+1`` before applying the same recursion, which gives every moment rather
than just the first two.
"""

import math

import numpy as np
import scipy.linalg as sla

from most_queue.random.utils.params import ErlangParams, GammaParams, H2Params


def ph_representation(params) -> tuple[np.ndarray, np.ndarray]:
    """
    ``(alpha, T)`` of a phase-type service time.

    :param params: :class:`H2Params` (hyperexponential, CV >= 1),
        :class:`ErlangParams` (CV <= 1) or :class:`GammaParams` with an integer
        shape (then it IS an Erlang).
    """
    if isinstance(params, H2Params):
        alpha = np.array([params.p1, 1.0 - params.p1], dtype=float)
        t_mat = np.diag([-params.mu1, -params.mu2]).astype(float)
        return alpha, t_mat
    if isinstance(params, GammaParams):
        shape = params.alpha
        if abs(shape - round(shape)) > 1e-9:
            raise ValueError(
                f"Gamma with non-integer shape {shape} is not phase-type; pass ErlangParams "
                "or H2Params, or let set_servers_from_moments fit one"
            )
        params = ErlangParams(r=int(round(shape)), mu=params.mu)
    if isinstance(params, ErlangParams):
        r = int(params.r)
        t_mat = np.zeros((r, r))
        for i in range(r):
            t_mat[i, i] = -params.mu
            if i + 1 < r:
                t_mat[i, i + 1] = params.mu
        alpha = np.zeros(r)
        alpha[0] = 1.0
        return alpha, t_mat
    raise TypeError(f"unsupported service-time params for a phase-type representation: {type(params).__name__}")


def ph_mean(alpha: np.ndarray, t_mat: np.ndarray) -> float:
    """Mean of ``PH(alpha, T)``."""
    return float(alpha @ np.linalg.solve(-t_mat, np.ones(len(alpha))))


def equilibrium_ph(alpha: np.ndarray, t_mat: np.ndarray) -> np.ndarray:
    """
    Entry vector of the equilibrium (excess) distribution of ``PH(alpha, T)``.

    The generator is unchanged; only the entry vector is re-weighted, since the
    excess density ``(1 - B(x)) / mean`` equals ``pi e^{Tx} t`` for
    ``pi = alpha (-T)^-1 / mean``.
    """
    weights = alpha @ np.linalg.inv(-t_mat)
    return weights / weights.sum()


def waiting_time_ph(pi_eq: np.ndarray, t_mat: np.ndarray, rho: float) -> np.ndarray:
    """
    Sub-generator of the M/G/1-FCFS waiting time: ``S = T + rho * t * pi_eq``.

    The waiting time is an atom of mass ``1 - rho`` at zero plus the defective
    ``PH(rho * pi_eq, S)``; conditioned on being positive it is ``PH(pi_eq, S)``.
    This is the matrix form of the Pollaczek-Khinchine geometric sum
    ``W = (1-rho) sum_k rho^k F^(k*)``.
    """
    exit_rates = -t_mat @ np.ones(len(pi_eq))
    return t_mat + rho * np.outer(exit_rates, pi_eq)


def _integral_poly_exp(a_mat: np.ndarray, u: float, power: int) -> np.ndarray:
    """
    ``int_0^u (u-y)^power e^{A y} dy`` as one matrix-exponential block.

    Van Loan's identity: for the block-bidiagonal ``M`` with ``A`` in the top
    corner, ones on the super-diagonal and zeros elsewhere,
    ``expm(M u)[0, p]`` is ``int_0^u (u-y)^(p-1) / (p-1)! e^{Ay} dy``. Used
    instead of the factored closed form because no inverse of ``A`` appears.
    """
    dim = a_mat.shape[0]
    blocks = power + 2
    big = np.zeros((blocks * dim, blocks * dim))
    big[:dim, :dim] = a_mat
    for b in range(blocks - 1):
        big[b * dim : (b + 1) * dim, (b + 1) * dim : (b + 2) * dim] = np.eye(dim)
    return math.factorial(power) * sla.expm(big * u)[:dim, (blocks - 1) * dim :]


def _convolve_ph(pi_vec: np.ndarray, s_mat: np.ndarray, times: int) -> tuple[np.ndarray, np.ndarray]:
    """``times``-fold convolution of ``PH(pi, S)``, as the usual block-bidiagonal phase-type."""
    dim = len(pi_vec)
    exit_rates = -s_mat @ np.ones(dim)
    big = np.zeros((times * dim, times * dim))
    init = np.zeros(times * dim)
    init[:dim] = pi_vec
    for b in range(times):
        big[b * dim : (b + 1) * dim, b * dim : (b + 1) * dim] = s_mat
        if b + 1 < times:
            big[b * dim : (b + 1) * dim, (b + 1) * dim : (b + 2) * dim] = np.outer(exit_rates, pi_vec)
    return init, big


def _xi(order: int, u: float, pi_vec: np.ndarray, s_mat: np.ndarray, rho: float) -> float:
    """
    ``xi_n(u) = (1-rho)^-n E[(u - S_{n-1})^n ; S_{n-1} <= u]``, equation (3.2).

    ``S_{n-1}`` is a sum of ``n-1`` i.i.d. FCFS waiting times, each zero with
    probability ``1-rho``, so the sum is conditioned on how many of them are
    positive -- a binomial mixture of phase-type convolutions.
    """
    summands = order - 1
    total = 0.0
    for positive in range(summands + 1):
        weight = math.comb(summands, positive) * (1.0 - rho) ** (summands - positive) * rho**positive
        if weight == 0.0:
            continue
        if positive == 0:
            partial = u**order  # all of them are zero: S = 0
        else:
            init, big = _convolve_ph(pi_vec, s_mat, positive)
            tail_integral = init @ _integral_poly_exp(big, u, order - 1) @ np.ones(big.shape[0])
            partial = u**order - order * float(tail_integral)
        total += weight * partial
    return total / (1.0 - rho) ** order


def _series_power(coeffs: list[float], power: int) -> list[float]:
    """Truncated power-series exponentiation, used for the ``K`` permanent jobs case."""
    result = [1.0] + [0.0] * (len(coeffs) - 1)
    for _ in range(power):
        result = [sum(result[j] * coeffs[n - j] for j in range(n + 1)) for n in range(len(coeffs))]
    return result


def conditional_sojourn_moments(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    u: float,
    num: int,
    pi_vec: np.ndarray,
    s_mat: np.ndarray,
    rho: float,
    permanent_jobs: int = 0,
) -> list[float]:
    """
    Raw moments ``E[V(u)^1..n]`` of the sojourn time of a job of size ``u``.

    :param u: job size (service requirement).
    :param num: how many raw moments.
    :param pi_vec: entry vector of the positive part of the FCFS waiting time.
    :param s_mat: its sub-generator, from :func:`waiting_time_ph`.
    :param rho: utilization.
    :param permanent_jobs: ``K`` permanent jobs of infinite size also sharing the
        processor; the sojourn time is then a sum of ``K+1`` independent copies.
    """
    if u < 0:
        raise ValueError(f"job size must be non-negative, got {u}")
    if num < 1:
        raise ValueError(f"num must be at least 1, got {num}")
    if u == 0.0:
        return [0.0] * num

    # Coefficients of 1/v(r,u) = sum_n c_n r^n, i.e. c_n = xi_n / n!.
    coeffs = [1.0] + [_xi(n, u, pi_vec, s_mat, rho) / math.factorial(n) for n in range(1, num + 1)]
    if permanent_jobs:
        coeffs = _series_power(coeffs, permanent_jobs + 1)
    xis = [coeffs[n] * math.factorial(n) for n in range(num + 1)]

    moments = [1.0]
    for n in range(1, num + 1):
        moments.append(sum(math.comb(n, i) * moments[n - i] * xis[i] * (-1) ** (i + 1) for i in range(1, n + 1)))
    return moments[1:]
