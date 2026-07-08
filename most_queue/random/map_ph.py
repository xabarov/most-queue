"""
Phase-type (PH) distributions and Markovian Arrival Processes (MAP).

A PH distribution is the time to absorption of a CTMC with initial vector
alpha and sub-generator T (absorption rates t0 = -T @ 1). A MAP is defined by
matrices (D0, D1): D0 holds phase transitions without arrivals, D1 —
transitions that generate an arrival; D0 + D1 is the generator of the phase
process. MAPs model correlated (bursty) traffic; a renewal PH source and the
MMPP are special cases.

References:
    Neuts M.F. Matrix-Geometric Solutions in Stochastic Models. Johns Hopkins
        University Press, 1981.
    Latouche G., Ramaswami V. Introduction to Matrix Analytic Methods in
        Stochastic Modeling. SIAM, 1999. doi:10.1137/1.9780898719734.
    Fischer W., Meier-Hellstern K. The Markov-modulated Poisson process (MMPP)
        cookbook. Performance Evaluation, 18(2), 1993.
        doi:10.1016/0166-5316(93)90035-s.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from most_queue.random.utils.params import Cox2Params, ErlangParams, H2Params


@dataclass
class PHParams:
    """Phase-type distribution parameters: initial vector and sub-generator."""

    alpha: np.ndarray  # initial probability row vector, shape (m,)
    T: np.ndarray  # sub-generator matrix, shape (m, m)


@dataclass
class MAPParams:
    """Markovian Arrival Process parameters."""

    D0: np.ndarray  # transitions without arrivals (sub-generator), shape (m, m)
    D1: np.ndarray  # transitions generating an arrival, shape (m, m)


def _validate_ph(alpha: np.ndarray, T: np.ndarray) -> None:
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("T must be a square matrix")
    if alpha.shape != (T.shape[0],):
        raise ValueError(f"alpha shape {alpha.shape} does not match T {T.shape}")
    if not math.isclose(float(np.sum(alpha)), 1.0, rel_tol=1e-9):
        raise ValueError("alpha must sum to 1 (mass at zero is not supported)")
    if np.any(np.diag(T) >= 0):
        raise ValueError("diagonal of T must be negative")
    exit_rates = -T @ np.ones(T.shape[0])
    if np.any(exit_rates < -1e-12):
        raise ValueError("row sums of T must be <= 0 (t0 = -T@1 must be non-negative)")


class PHDistribution:
    """
    Phase-type distribution PH(alpha, T).

    Follows the library's distribution interface: ``generate()`` draws a
    sample by simulating the absorbing CTMC, static ``calc_theory_moments``
    returns raw moments m_k = k! * alpha @ (-T)^{-k} @ 1.
    """

    def __init__(self, params: PHParams, generator=None):
        self.params = params
        self.alpha = np.asarray(params.alpha, dtype=float)
        self.T = np.asarray(params.T, dtype=float)
        _validate_ph(self.alpha, self.T)
        self.t0 = -self.T @ np.ones(self.T.shape[0])
        self.type = "PH"
        self.generator = generator if generator is not None else np.random.default_rng()

    # ------------------------------------------------------------- theory
    @staticmethod
    def calc_theory_moments(params: PHParams, num: int = 4) -> list[float]:
        """
        Raw moments of PH(alpha, T): m_k = k! * alpha @ (-T)^{-k} @ 1.
        """
        alpha = np.asarray(params.alpha, dtype=float)
        T = np.asarray(params.T, dtype=float)
        m_inv = np.linalg.inv(-T)
        ones = np.ones(T.shape[0])
        moments, vec = [], alpha.copy()
        for k in range(1, num + 1):
            vec = vec @ m_inv
            moments.append(math.factorial(k) * float(vec @ ones))
        return moments

    @staticmethod
    def get_pdf(params: PHParams, x: float) -> float:
        """Probability density f(x) = alpha @ expm(T x) @ t0."""
        alpha = np.asarray(params.alpha, dtype=float)
        T = np.asarray(params.T, dtype=float)
        t0 = -T @ np.ones(T.shape[0])
        return float(alpha @ expm(T * x) @ t0) if x >= 0 else 0.0

    @staticmethod
    def get_cdf(params: PHParams, x: float) -> float:
        """Cumulative distribution F(x) = 1 - alpha @ expm(T x) @ 1."""
        alpha = np.asarray(params.alpha, dtype=float)
        T = np.asarray(params.T, dtype=float)
        return 1.0 - float(alpha @ expm(T * x) @ np.ones(T.shape[0])) if x >= 0 else 0.0

    # ---------------------------------------------------------- converters
    @staticmethod
    def from_exp(mu: float) -> PHParams:
        """Exponential(mu) as a single-phase PH."""
        return PHParams(alpha=np.array([1.0]), T=np.array([[-mu]]))

    @staticmethod
    def from_erlang(params: ErlangParams) -> PHParams:
        """Erlang-r as a chain of r exponential phases."""
        r, mu = params.r, params.mu
        T = -mu * np.eye(r) + mu * np.eye(r, k=1)
        alpha = np.zeros(r)
        alpha[0] = 1.0
        return PHParams(alpha=alpha, T=T)

    @staticmethod
    def from_h2(params: H2Params) -> PHParams:
        """Hyperexponential H2 (real parameters only) as a two-phase PH."""
        p1, mu1, mu2 = float(params.p1.real), float(params.mu1.real), float(params.mu2.real)
        return PHParams(alpha=np.array([p1, 1.0 - p1]), T=np.array([[-mu1, 0.0], [0.0, -mu2]]))

    @staticmethod
    def from_cox(params: Cox2Params) -> PHParams:
        """Second-order Coxian as a two-phase PH."""
        mu1, mu2, p1 = float(params.mu1.real), float(params.mu2.real), float(params.p1.real)
        return PHParams(alpha=np.array([1.0, 0.0]), T=np.array([[-mu1, p1 * mu1], [0.0, -mu2]]))

    # ------------------------------------------------------------ sampling
    def generate(self) -> float:
        """Draw one sample by simulating the absorbing phase process."""
        rng = self.generator
        m = self.T.shape[0]
        phase = int(rng.choice(m, p=self.alpha))
        elapsed = 0.0
        while True:
            rate = -self.T[phase, phase]
            elapsed += rng.exponential(1.0 / rate)
            # transition probabilities from `phase`: to other phases or absorb
            probs = np.append(self.T[phase].copy(), self.t0[phase])
            probs[phase] = 0.0
            probs /= rate
            nxt = int(rng.choice(m + 1, p=probs))
            if nxt == m:
                return elapsed
            phase = nxt


class MAP:
    """
    Markovian Arrival Process MAP(D0, D1).

    ``generate()`` returns successive interarrival times (the phase is kept
    between calls, starting from the arrival-stationary distribution).
    """

    def __init__(self, params: MAPParams, generator=None):
        self.params = params
        self.D0 = np.asarray(params.D0, dtype=float)
        self.D1 = np.asarray(params.D1, dtype=float)
        if self.D0.shape != self.D1.shape or self.D0.ndim != 2 or self.D0.shape[0] != self.D0.shape[1]:
            raise ValueError("D0 and D1 must be square matrices of the same shape")
        gen_row_sums = (self.D0 + self.D1) @ np.ones(self.D0.shape[0])
        if not np.allclose(gen_row_sums, 0.0, atol=1e-9):
            raise ValueError("D0 + D1 must be a generator (zero row sums)")
        if np.any(self.D1 < -1e-12):
            raise ValueError("D1 must be non-negative")
        self.type = "MAP"
        self.generator = generator if generator is not None else np.random.default_rng()
        self._phase = int(self.generator.choice(self.D0.shape[0], p=MAP.arrival_stationary_phase(params)))

    # ------------------------------------------------------------- theory
    @staticmethod
    def phase_stationary(params: MAPParams) -> np.ndarray:
        """Stationary distribution pi of the phase generator D = D0 + D1."""
        d = np.asarray(params.D0, dtype=float) + np.asarray(params.D1, dtype=float)
        m = d.shape[0]
        a = np.vstack([d.T, np.ones(m)])
        b = np.zeros(m + 1)
        b[-1] = 1.0
        pi, *_ = np.linalg.lstsq(a, b, rcond=None)
        return pi

    @staticmethod
    def arrival_rate(params: MAPParams) -> float:
        """Fundamental arrival rate lambda = pi @ D1 @ 1."""
        pi = MAP.phase_stationary(params)
        return float(pi @ np.asarray(params.D1, dtype=float) @ np.ones(len(pi)))

    @staticmethod
    def arrival_stationary_phase(params: MAPParams) -> np.ndarray:
        """Phase distribution just after an arrival: pi @ D1 / lambda."""
        pi = MAP.phase_stationary(params)
        d1 = np.asarray(params.D1, dtype=float)
        vec = pi @ d1
        return vec / vec.sum()

    @staticmethod
    def calc_theory_moments(params: MAPParams, num: int = 4) -> list[float]:
        """
        Raw moments of the stationary interarrival time:
        m_k = k! * pi_a @ (-D0)^{-k} @ 1.
        """
        pi_a = MAP.arrival_stationary_phase(params)
        m_inv = np.linalg.inv(-np.asarray(params.D0, dtype=float))
        ones = np.ones(len(pi_a))
        moments, vec = [], pi_a.copy()
        for k in range(1, num + 1):
            vec = vec @ m_inv
            moments.append(math.factorial(k) * float(vec @ ones))
        return moments

    @staticmethod
    def lag_correlation(params: MAPParams, k: int = 1) -> float:
        """
        Lag-k autocorrelation of interarrival times:
        corr(X_0, X_k) with E[X_0 X_k] = pi_a @ M @ P^k @ M @ 1, M = (-D0)^{-1},
        P = M @ D1 (phase transition matrix over one arrival).
        """
        if k < 1:
            raise ValueError("lag k must be >= 1")
        pi_a = MAP.arrival_stationary_phase(params)
        m_inv = np.linalg.inv(-np.asarray(params.D0, dtype=float))
        p = m_inv @ np.asarray(params.D1, dtype=float)
        ones = np.ones(len(pi_a))
        m1 = float(pi_a @ m_inv @ ones)
        m2 = 2.0 * float(pi_a @ m_inv @ m_inv @ ones)
        joint = float(pi_a @ m_inv @ np.linalg.matrix_power(p, k) @ m_inv @ ones)
        var = m2 - m1 * m1
        return (joint - m1 * m1) / var

    # ---------------------------------------------------------- factories
    @staticmethod
    def poisson(rate: float) -> MAPParams:
        """Poisson process as a trivial one-phase MAP."""
        return MAPParams(D0=np.array([[-rate]]), D1=np.array([[rate]]))

    @staticmethod
    def from_ph_renewal(ph: PHParams) -> MAPParams:
        """Renewal process with PH interarrival times: D0 = T, D1 = t0 @ alpha."""
        alpha = np.asarray(ph.alpha, dtype=float)
        T = np.asarray(ph.T, dtype=float)
        t0 = -T @ np.ones(T.shape[0])
        return MAPParams(D0=T, D1=np.outer(t0, alpha))

    @staticmethod
    def mmpp(rates: list[float], q: np.ndarray) -> MAPParams:
        """
        Markov-modulated Poisson process: Poisson rate rates[i] while the
        modulating chain (generator q) is in state i.
        """
        rates_arr = np.asarray(rates, dtype=float)
        q_arr = np.asarray(q, dtype=float)
        if not np.allclose(q_arr @ np.ones(q_arr.shape[0]), 0.0, atol=1e-9):
            raise ValueError("q must be a generator (zero row sums)")
        return MAPParams(D0=q_arr - np.diag(rates_arr), D1=np.diag(rates_arr))

    # ------------------------------------------------------------ sampling
    def generate(self) -> float:
        """Time until the next arrival (phase state persists between calls)."""
        rng = self.generator
        m = self.D0.shape[0]
        elapsed = 0.0
        while True:
            rate = -self.D0[self._phase, self._phase]
            elapsed += rng.exponential(1.0 / rate)
            # competing transitions: D0 off-diagonal (no arrival) vs any D1 entry
            probs = np.concatenate([self.D0[self._phase], self.D1[self._phase]])
            probs[self._phase] = 0.0
            probs = probs / rate
            nxt = int(rng.choice(2 * m, p=probs))
            if nxt >= m:
                self._phase = nxt - m
                return elapsed
            self._phase = nxt
