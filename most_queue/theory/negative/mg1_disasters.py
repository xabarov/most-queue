"""
Class for calculating M/G/1 queue with disasters.
Use results from the paper:
    Jain, Gautam, and Karl Sigman. "A Pollaczek–Khintchine formula for M/G/1 queues with disasters."
    Journal of Applied Probability 33.4 (1996): 1191-1200.
"""

import numpy as np
from scipy.misc import derivative
from scipy.optimize import brentq

from most_queue.random.distributions import GammaDistribution, H2Distribution
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.utils.transforms import lst_gamma, lst_h2


class MG1Disasters(BaseQueue):
    """
    Class for calculating M/G/1 queue with disasters.
    """

    def __init__(
        self,
        calc_params: CalcParams | None = None,
    ):
        """
        Initialize the MG1Disasters class.
        :param calc_params: Calculation parameters. If None, default parameters are used.
        """

        super().__init__(n=1, calc_params=calc_params)
        self.l_pos = None
        self.l_neg = None
        self.b = None
        self.approximation = self.calc_params.approx_distr

        self.lst_function = None
        self.params = None

        # Cache for workload/waiting-time LST parameters
        self._workload_root: float | None = None
        self._workload_phi_r: float | None = None
        self._workload_beta_r: float | None = None

    def set_sources(self, l_pos: float, l_neg: float):  # pylint: disable=arguments-differ
        """
        Set the arrival rates of positive and negative jobs
        :param l_pos: arrival rate of positive jobs
        :param l_neg: arrival rate of negative jobs
        """
        self.l_pos = l_pos
        self.l_neg = l_neg
        self._workload_root = None
        self._workload_phi_r = None
        self._workload_beta_r = None
        self.is_sources_set = True

    def set_servers(self, b: list[float]):  # pylint: disable=arguments-differ
        """
        Set the raw moments of service time distribution
        :param b: raw moments of service time distribution
        """
        self.b = b

        if self.approximation == "h2":
            self.lst_function = lst_h2
            self.params = H2Distribution.get_params(b)
        elif self.approximation == "gamma":
            self.lst_function = lst_gamma
            self.params = GammaDistribution.get_params(b)
        else:
            raise ValueError("Approximation must be 'h2' or 'gamma'.")
        self.is_servers_set = True
        self._workload_root = None
        self._workload_phi_r = None
        self._workload_beta_r = None

    def run(self, num_of_moments: int = 4) -> QueueResults:
        """
        Run calculation
        """
        start = self._measure_time()
        v = self.get_v(num_of_moments)
        utilization = self.get_utilization()

        result = QueueResults(v=v, utilization=utilization)
        self._set_duration(result, start)
        return result

    def get_utilization(self) -> float:
        """
        Calculate utilization factor.

        Note: This is a simplified implementation. A more sophisticated
        calculation would better account for the impact of disaster events
        on system utilization.
        """
        return self.l_pos * self.b[0]

    def get_v(self, num_of_moments: int = 4) -> list[float]:
        """
        Calculate first three moments of sojourn time in the system.
        """
        self._check_if_servers_and_sources_set()

        if not self.v is None:
            return self.v

        v = [0] * num_of_moments
        for i in range(num_of_moments):
            v[i] = derivative(self._v_lst, 0, dx=1e-3 / self.b[0], n=i + 1, order=9)
            if i % 2 == 0:
                v[i] = -v[i]
        v = np.array(v)

        self.v = v
        return v

    def _beta(self, s: float) -> float:
        """
        Service-time Laplace-Stieltjes transform β(s) = E[e^{-sB}].
        """
        return float(self.lst_function(self.params, s))

    def _get_workload_root(self) -> float:
        """
        Find the unique positive root s0 of:
            f(s) = s - (λ + δ) + λ β(s) = 0
        where λ = l_pos, δ = l_neg.

        This root is used to cancel the pole in the workload LST.
        """
        if self._workload_root is not None:
            return self._workload_root

        if self.l_neg <= 0:
            self._workload_root = 0.0
            return self._workload_root

        lam = float(self.l_pos)
        delta = float(self.l_neg)
        r = lam + delta

        def f(x: float) -> float:
            return x - r + lam * self._beta(x)

        # f(0) = -delta < 0; f(x) -> +inf as x -> +inf, so a bracket exists.
        lo = 0.0
        hi = max(1.0 / self.b[0], r, 1.0)
        while f(hi) <= 0.0:
            hi *= 2.0
            if hi > 1e6 / self.b[0]:
                raise RuntimeError("Failed to bracket the positive root for workload LST.")

        self._workload_root = float(brentq(f, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=500))
        return self._workload_root

    def _ensure_workload_constants(self) -> None:
        """
        Precompute constants needed for the workload/waiting-time LST.
        """
        if self.l_neg <= 0:
            return
        if self._workload_phi_r is not None and self._workload_beta_r is not None:
            return

        lam = float(self.l_pos)
        delta = float(self.l_neg)
        r = lam + delta
        s0 = self._get_workload_root()

        beta_r = self._beta(r)
        if beta_r <= 0:
            raise RuntimeError("Service-time LST β(r) must be positive.")

        # Cancellation condition at the pole s = s0:
        #   delta (s0 - r) + lam * s0 * phi(r) * beta(r) = 0
        # => phi(r) = delta (r - s0) / (lam * s0 * beta(r))
        phi_r = delta * (r - s0) / (lam * s0 * beta_r)

        self._workload_beta_r = float(beta_r)
        self._workload_phi_r = float(phi_r)

    def _w_lst(self, s: float) -> float:
        """
        Workload / waiting-time LST at arrival epochs (PASTA):
            φ(s) = E[e^{-s W}]
        for an M/G/1 queue with Poisson disasters (rate δ) that clear the system.
        """
        lam = float(self.l_pos)
        delta = float(self.l_neg)

        if delta <= 0:
            # Classic M/G/1 waiting-time LST (stable case only).
            ro = lam * self.b[0]
            if ro >= 1:
                raise ValueError("M/G/1 without disasters requires utilization < 1.")
            beta_s = self._beta(s)
            return (1.0 - ro) * s / (s - lam + lam * beta_s)

        r = lam + delta
        self._ensure_workload_constants()
        beta_s = self._beta(s)
        denom = r * (s - r + lam * beta_s)
        numer = delta * (s - r) + lam * s * self._workload_phi_r * self._workload_beta_r
        return float(numer / denom)

    def _v_lst(self, s):
        """
        Laplace-Stieltjes transform of the sojourn time V in the system:
            V = min(W + B, Y),  Y ~ Exp(δ) (next disaster after arrival),
        where W is the stationary workload seen by a positive arrival.
        """
        lam = float(self.l_pos)
        delta = float(self.l_neg)

        if delta <= 0:
            # Classic M/G/1 sojourn-time LST (stable case only).
            ro = lam * self.b[0]
            if ro >= 1:
                raise ValueError("M/G/1 without disasters requires utilization < 1.")
            beta_s = self._beta(s)
            return (1.0 - ro) * s * beta_s / (s - lam + lam * beta_s)

        sp = s + delta
        return float(delta / (s + delta) + (s / (s + delta)) * self._w_lst(sp) * self._beta(sp))
