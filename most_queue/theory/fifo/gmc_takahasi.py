"""
Calculate H2/M/n queue using the Takahashi-Takami method.

H2 arrival (hyperexponential interarrival), M service (exponential).
Uses the simplified algorithm from §7.6.1 of the textbook.

Key formulas (CH7 eqs 7.6.1–7.6.3):
- z_j = μ_j / s, where s = Σ λ_i t_{j-1,i} (cut-balance)
- x_j from (7.6.3), t_{j,i} from (7.6.2)
- Level 0: x_0 = 1/(μ Σ t_{1,i}/λ_i), t_{0,i} ∝ t_{1,i}/λ_i
"""

import math
import time

import numpy as np

from most_queue.random.distributions import H2Distribution
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import TakahashiTakamiParams
from most_queue.theory.utils.conv import conv_moments


class H2MnCalc(BaseQueue):
    """
    Calculate H2/M/n queue using the Takahashi-Takami method.

    H2 (hyperexponential) interarrival, M (exponential) service.
    Uses the simplified component-wise recurrences from §7.6.1.
    """

    def __init__(
        self,
        n: int,
        buffer: int | None = None,
        calc_params: TakahashiTakamiParams | None = None,
    ):
        """
        Args:
            n: number of servers
            buffer: size of the buffer (optional, for limited queue)
            calc_params: parameters for the Takahashi-Takami method
        """
        super().__init__(n=n, calc_params=calc_params)

        self.calc_params = calc_params or TakahashiTakamiParams()
        self.e1 = self.calc_params.tolerance
        self.n = n
        self.verbose = self.calc_params.verbose

        if buffer is not None:
            self.R = buffer + n
            self.N = self.R + 1
        else:
            self.R = None
            self.N = self.calc_params.N

        # H2 arrival: u_i = branch probabilities, lam_i = branch intensities
        self.u: list[float] = []
        self.lam: list[float] = []
        self.a: list[float] = []  # raw moments of interarrival

        # M service
        self.mu: float = 0.0  # service rate per channel
        self.b: float = 0.0  # mean service time

        # State: t[j,i] = conditional prob of phase i at level j (i=0,1 for k=2)
        self.t: list[np.ndarray] = []
        self.x = np.zeros(self.N)
        self.z = np.zeros(self.N)
        self.p = np.zeros(self.N)

        self.num_of_iter_ = 0
        self.w: list[float] | None = None
        self.v: list[float] | None = None

    def set_sources(self, a: list[float]):  # pylint: disable=arguments-differ
        """
        Set interarrival distribution via raw moments.

        Args:
            a: raw moments of interarrival time (a[0]=mean, a[1]=2nd moment, ...)
        """
        self.a = a
        h2_params = H2Distribution.get_params(a)
        self.u = [float(h2_params.p1), 1.0 - float(h2_params.p1)]
        self.lam = [float(h2_params.mu1), float(h2_params.mu2)]
        self.is_sources_set = True

    def set_servers(self, b: float | list[float]):  # pylint: disable=arguments-differ
        """
        Set exponential service.

        Args:
            b: mean service time (float) or raw moments (list); b[0] used as mean.
        """
        if isinstance(b, (int, float)):
            self.b = float(b)
        else:
            self.b = float(b[0])
        self.mu = 1.0 / self.b
        self.is_servers_set = True

    def get_utilization(self) -> float:
        """Utilization = (1/a1) * b / n."""
        return (1.0 / self.a[0]) * self.b / self.n

    def _mu_j(self, j: int) -> float:
        """Service intensity when j customers in system."""
        return min(j, self.n) * self.mu

    def _x_inf(self) -> float:
        """Asymptotic x = rho^(2/(v_A^2 + v_B^2)). For M service v_B^2=1."""
        rho = self.get_utilization()
        v_a_sq = self.a[1] / (self.a[0] ** 2) - 1.0
        v_a_sq = max(v_a_sq, 0.01)
        return float(rho ** (2.0 / (v_a_sq + 1.0)))

    def _t_inf(self, x: float) -> np.ndarray:
        """Limit vector t_inf from the linear system for k=2."""
        n, mu = self.n, self.mu
        u1, u2 = self.u[0], self.u[1]
        l1, l2 = self.lam[0], self.lam[1]

        # (u_i/x)*S + [n*mu*(x-1) - lambda_i]*t_i = 0, t_1 + t_2 = 1
        # S = l1*t1 + l2*t2. From sum: S/x + n*mu*(x-1) = S => S = n*mu*x
        # So (u1/x)*n*mu*x + [n*mu*(x-1)-l1]*t1 = 0 => u1*n*mu + [n*mu*(x-1)-l1]*t1 = 0
        denom1 = n * mu * (1 - x) + l1
        denom2 = n * mu * (1 - x) + l2
        if abs(denom1) < 1e-12:
            t1 = 0.5
        else:
            t1 = self.u[0] * n * mu / denom1
        if abs(denom2) < 1e-12:
            t2 = 0.5
        else:
            t2 = self.u[1] * n * mu / denom2
        s = t1 + t2
        t1, t2 = t1 / s, t2 / s
        return np.array([t1, t2])

    def run(self) -> QueueResults:
        """Run the Takahashi-Takami algorithm for H2/M/n."""
        start = time.process_time()
        self._check_if_servers_and_sources_set()

        x_inf = self._x_inf()
        self.t = [np.zeros(2) for _ in range(self.N)]
        self.x = np.zeros(self.N)
        self.z = np.zeros(self.N)

        # Textbook §7.6.1: for j < n use u (phase distr); for j >= n use t_inf
        t_lim = self._t_inf(x_inf)
        u_arr = np.array(self.u)
        for j in range(self.N):
            if j < self.n:
                self.t[j] = u_arr.copy()
            else:
                self.t[j] = t_lim.copy()
        for j in range(self.N - 1):
            self.x[j] = x_inf
        self.x[self.N - 1] = x_inf

        # Boundary for limited queue
        if self.R is not None:
            for j in range(self.N):
                self.t[j] = np.array(self.u)

        self.num_of_iter_ = 0
        # Textbook §7.5: iterate until max |x_j refinement| < eps (change between iters)
        x_delta = float("inf")

        while x_delta >= self.e1:
            x_old = self.x.copy()
            self.num_of_iter_ += 1

            # Sweep j=N-1 down to 1 then level 0 (textbook "снизу вверх")
            for j in range(self.N - 1, 0, -1):
                self._update_level_j(j)

            self._update_level_0()

            x_delta = float(np.max(np.abs(self.x - x_old)))
            if self.num_of_iter_ >= self.calc_params.max_iter:
                break
            if self.verbose:
                print(f"Iter #{self.num_of_iter_}, max|x_delta|={x_delta:.2e}")

        self._calculate_p()
        results = self.get_results()
        results.duration = time.process_time() - start
        return results

    def _update_level_j(self, j: int) -> None:
        """Update t[j], x[j], z[j] from t[j+1] (and t[j-1] for s).
        Textbook §7.6.1: s = Σ λ_m t_{j-1,m}, cut-balance z_j s = μ_j ⇒ z_j = μ_j/s."""
        u = self.u
        lam = self.lam
        mu_j = self._mu_j(j)
        mu_jp1 = self._mu_j(j + 1)

        if j == 0:
            return

        s = lam[0] * self.t[j - 1][0] + lam[1] * self.t[j - 1][1]

        # z_j = μ_j / s per cut-balance (7.6.1)
        if s > 1e-20:
            self.z[j] = mu_j / s
        else:
            self.z[j] = 1.0 / self.x[j]  # fallback if s≈0

        # x_j = (1 - mu_j * sum_i u_i/(lam_i+mu_j)) / (mu_{j+1} * sum_i t_{j+1,i}/(lam_i+mu_j))
        sum_denom = 0.0
        for i in range(2):
            sum_denom += u[i] / (lam[i] + mu_j)
        num = 1.0 - mu_j * sum_denom

        t_next_for_denom = self.t[j + 1] if j + 1 < self.N else self.t[j - 1]
        sum_denom2 = sum(t_next_for_denom[i] / (lam[i] + mu_j) for i in range(2))

        denom = mu_jp1 * sum_denom2
        if abs(denom) < 1e-14:
            pass  # keep x[j]
        else:
            self.x[j] = num / denom

        # t_{j,i} = mu_j * u_i/(lam_i+mu_j) + x_j * mu_{j+1} * t_{j+1,i}/(lam_i+mu_j)
        # Closure at top: t_{j+1} ≈ t_{j-1} when j+1 >= N
        t_next = self.t[j + 1] if j + 1 < self.N else self.t[j - 1]
        for i in range(2):
            self.t[j][i] = mu_j * u[i] / (lam[i] + mu_j) + self.x[j] * mu_jp1 * t_next[i] / (lam[i] + mu_j)
        # Normalize
        s_t = self.t[j].sum()
        if s_t > 0:
            self.t[j] /= s_t

    def _update_level_0(self) -> None:
        """Level 0 per §7.6.1: t_{0,i} λ_i = x_0 μ t_{1,i} ⇒ t_{0,i} ∝ t_{1,i}/λ_i,
        x_0 = 1/(μ Σ t_{1,i}/λ_i)."""
        lam = self.lam
        mu_1 = self._mu_j(1)
        sum_t1_over_lam = sum(self.t[1][i] / lam[i] for i in range(2))
        if sum_t1_over_lam > 1e-20:
            self.x[0] = 1.0 / (mu_1 * sum_t1_over_lam)
        # t[0,i] ∝ t[1,i]/λ_i, normalize
        raw = np.array([self.t[1][i] / lam[i] for i in range(2)])
        s = raw.sum()
        if s > 0:
            self.t[0] = raw / s
        else:
            self.t[0] = np.array(self.u)
        self.z[0] = 1.0 / self.x[0]
        if self.verbose:
            for i in range(2):
                lhs = self.t[0][i] * lam[i]
                rhs = self.x[0] * mu_1 * self.t[1][i]
                print(f"  level0 check i={i}: t0*λ={lhs:.6e}, x0*μ*t1={rhs:.6e}")

    def _calculate_p(self) -> None:
        """Compute state probabilities from x."""
        a1 = self.a[0]
        lam_eff = 1.0 / a1

        znam = self.n + sum((self.n - j) * np.prod(self.x[:j]) for j in range(1, self.n))

        if self.R is not None:
            prod_r = np.prod(self.x[: self.N])
            znam -= lam_eff * self.b * prod_r

        self.p[0] = (self.n - lam_eff * self.b) / znam
        for j in range(self.N - 1):
            self.p[j + 1] = self.p[j] * self.x[j]

        total = self.p.sum()
        if total > 0:
            self.p /= total
        assert np.isclose(self.p.sum(), 1.0, atol=1e-10, rtol=1e-9), f"sum(p) = {self.p.sum():.12f}, expected 1.0"

    def get_results(self, num_of_moments: int = 4) -> QueueResults:
        """Return QueueResults with p, w, v, utilization."""
        self.p = self.get_p()
        self.w = self.get_w(num_of_moments=num_of_moments)
        self.v = self.get_v(num_of_moments)
        return QueueResults(
            v=self.v,
            w=self.w,
            p=list(self.p),
            utilization=self.get_utilization(),
        )

    def get_p(self) -> np.ndarray:
        """Return state probabilities (real)."""
        return np.asarray(self.p, dtype=float)

    def get_w(self, num_of_moments: int = 4) -> list[float]:
        """Waiting time moments via queue-length distribution and Little."""
        w = [0.0] * 4
        for j in range(1, min(len(self.p) - self.n, 500)):
            w[0] += j * self.p[self.n + j]
        for j in range(2, min(len(self.p) - self.n, 500)):
            w[1] += j * (j - 1) * self.p[self.n + j]
        for j in range(3, min(len(self.p) - self.n, 500)):
            w[2] += j * (j - 1) * (j - 2) * self.p[self.n + j]
        for j in range(4, min(len(self.p) - self.n, 500)):
            w[3] += j * (j - 1) * (j - 2) * (j - 3) * self.p[self.n + j]

        lam_eff = 1.0 / self.a[0]
        for i in range(4):
            w[i] /= lam_eff ** (i + 1)
        return w[:num_of_moments]

    def get_v(self, num_of_moments: int = 4) -> list[float]:
        """Sojourn time moments = convolution of waiting and service."""
        if self.v is not None:
            return self.v
        self.w = self.w or self.get_w(num_of_moments=num_of_moments)
        # Exponential service raw moments: E[S^k] = k!/mu^k
        b_moments = [math.factorial(k) / (self.mu**k) for k in range(1, num_of_moments + 1)]
        self.v = conv_moments(b_moments, self.w, num=num_of_moments)
        return list(self.v)
