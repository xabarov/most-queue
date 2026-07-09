"""
Fitting a Markovian Arrival Process (MAP) to data.

The workhorse is the two-state MMPP (MMPP-2): a Poisson process whose rate
switches between two levels driven by a two-state Markov chain. It is fitted
by matching three interarrival statistics — the rate, the squared coefficient
of variation (SCV) and the lag-1 autocorrelation — for which the MAP has
closed-form expressions (see ``most_queue.random.map_ph``).

An MMPP is always overdispersed and positively correlated: SCV >= 1 and
lag-1 correlation >= 0. Targets outside this region cannot be represented by
an MMPP-2 and raise a clear error.

References:
    Fischer W., Meier-Hellstern K. The Markov-modulated Poisson process (MMPP)
        cookbook. Performance Evaluation, 18(2), 1993.
        doi:10.1016/0166-5316(93)90035-s.
    Heffes H., Lucantoni D. A Markov modulated characterization of packetized
        voice and data traffic. IEEE JSAC, 4(6), 1986. doi:10.1109/JSAC.1986.1146428.
"""

import numpy as np
from scipy.optimize import least_squares

from most_queue.random.map_ph import MAP, MAPParams


def map_statistics(params: MAPParams) -> tuple[float, float, float]:
    """
    Return (rate, scv, lag1) of a MAP: fundamental arrival rate, squared
    coefficient of variation and lag-1 autocorrelation of interarrival times.
    """
    rate = MAP.arrival_rate(params)
    moments = MAP.calc_theory_moments(params, 2)
    m1, m2 = moments[0], moments[1]
    scv = m2 / (m1 * m1) - 1.0
    lag1 = MAP.lag_correlation(params, 1)
    return rate, scv, lag1


def _mmpp2_from_vector(x: np.ndarray) -> MAPParams:
    """Build an MMPP-2 from log-parameters x = log([l1, l2, q12, q21])."""
    l1, l2, q12, q21 = np.exp(x)
    return MAP.mmpp([l1, l2], np.array([[-q12, q12], [q21, -q21]]))


def fit_mmpp2(
    rate: float,
    scv: float,
    lag1: float,
    n_restarts: int = 6,
    tol: float = 1e-9,
) -> MAPParams:
    """
    Fit an MMPP-2 matching the given interarrival rate, SCV and lag-1
    autocorrelation.

    :param rate: target fundamental arrival rate (> 0)
    :param scv: target squared coefficient of variation (>= 1 for an MMPP)
    :param lag1: target lag-1 autocorrelation (>= 0 for an MMPP)
    :param n_restarts: number of random restarts for the local optimizer
    :param tol: acceptable residual norm to stop early
    :return: MAPParams of the fitted MMPP-2
    :raises ValueError: if the target is outside the MMPP-representable region
    """
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")
    if scv < 1.0 - 1e-6:
        raise ValueError(f"an MMPP-2 requires SCV >= 1 (overdispersed), got {scv:.4f}")
    if lag1 < -1e-6:
        raise ValueError(f"an MMPP-2 requires non-negative lag-1 correlation, got {lag1:.4f}")

    target = np.array([rate, scv, lag1])
    weights = np.array([1.0 / max(rate, 1e-9), 1.0, 1.0])

    def residuals(x: np.ndarray) -> np.ndarray:
        try:
            stats = np.array(map_statistics(_mmpp2_from_vector(x)))
        except (np.linalg.LinAlgError, ValueError):
            return np.full(3, 1e3)
        return (stats - target) * weights

    rng = np.random.default_rng(12345)
    best = np.log([rate, rate, 0.1 * rate + 1e-3, 0.1 * rate + 1e-3])
    best_cost = np.inf
    for k in range(max(1, n_restarts)):
        if k == 0:
            # deterministic heuristic start: rates spread by the overdispersion,
            # slow switching to create the correlation
            spread = 1.0 + np.sqrt(max(scv - 1.0, 0.0))
            x0 = np.log([rate * spread, rate / spread, 0.1 * rate + 1e-3, 0.1 * rate + 1e-3])
        else:
            x0 = np.log(rate) + rng.uniform(-2.0, 2.0, size=4)
        # trf (not lm): the 4-parameter MMPP-2 is under-determined by the
        # 3 target statistics, so residuals < variables is expected
        sol = least_squares(residuals, x0, method="trf", max_nfev=2000)
        if sol.cost < best_cost:
            best, best_cost = sol.x, sol.cost
            if np.linalg.norm(sol.fun) < tol:
                break

    fitted = _mmpp2_from_vector(best)
    achieved = map_statistics(fitted)
    err = np.abs(np.array(achieved) - target) * weights
    if np.max(err) > 1e-2:
        raise ValueError(
            f"failed to fit MMPP-2 to (rate={rate:.4f}, scv={scv:.4f}, lag1={lag1:.4f}); "
            f"achieved (rate={achieved[0]:.4f}, scv={achieved[1]:.4f}, lag1={achieved[2]:.4f}). "
            "The target may be near the boundary of the MMPP-representable region."
        )
    return fitted


def trace_statistics(interarrivals) -> tuple[float, float, float]:
    """
    Empirical (rate, scv, lag1) of a sequence of interarrival times.
    """
    x = np.asarray(interarrivals, dtype=float)
    if x.size < 3:
        raise ValueError("need at least 3 interarrival samples")
    mean = x.mean()
    scv = x.var() / (mean * mean)
    lag1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    return 1.0 / mean, scv, lag1


def fit_map_from_trace(interarrivals, **kwargs) -> MAPParams:
    """
    Fit an MMPP-2 to a trace of interarrival times by matching its empirical
    rate, SCV and lag-1 autocorrelation.

    :param interarrivals: sequence of interarrival times
    :param kwargs: forwarded to :func:`fit_mmpp2`
    :return: MAPParams of the fitted MMPP-2
    """
    rate, scv, lag1 = trace_statistics(interarrivals)
    return fit_mmpp2(rate, max(scv, 1.0), max(lag1, 0.0), **kwargs)
