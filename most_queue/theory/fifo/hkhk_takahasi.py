"""
Calculate H_k/H_k/n queue using the Takahashi-Takami method (CH7 §7.6.2).

H_k arrival (hyperexponential interarrival), H_k service (hyperexponential).
Supports k=2 (H2) with moment-based or explicit params; k>2 with explicit params only.
"""

import math
import time

import numpy as np

from most_queue.random.distributions import H2Distribution
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import TakahashiTakamiParams
from most_queue.theory.utils.conv import conv_moments


def _compositions(busy: int, k: int) -> list[tuple[int, ...]]:
    """All compositions of `busy` into `k` non-negative parts (sum = busy).
    Returns list of tuples (m_1, ..., m_k) in lex order."""
    if busy == 0:
        return [(0,) * k]
    result: list[tuple[int, ...]] = []

    def _gen(rem: int, parts: list[int], idx: int) -> None:
        if idx == k - 1:
            parts[idx] = rem
            result.append(tuple(parts))
            return
        for x in range(rem + 1):
            parts[idx] = x
            _gen(rem - x, parts, idx + 1)

    _gen(busy, [0] * k, 0)
    return result


def _n_compositions(busy: int, k: int) -> int:
    """Number of compositions of busy into k parts = C(busy+k-1, k-1)."""
    if busy == 0:
        return 1
    n = busy + k - 1
    r = k - 1
    return math.comb(n, r)


class HkHkNCalc(BaseQueue):
    """
    Calculate H_k/H_k/n queue using the Takahashi-Takami method (§7.6.2).

    H_k arrival and H_k service. For k=2 supports moment-based params and get_params_clx.
    """

    def __init__(
        self,
        n: int,
        k: int = 2,
        buffer: int | None = None,
        calc_params: TakahashiTakamiParams | None = None,
    ):
        super().__init__(n=n, calc_params=calc_params)
        self.calc_params = calc_params or TakahashiTakamiParams()
        self.e1 = self.calc_params.tolerance
        self.n = n
        self.k = k
        self.verbose = self.calc_params.verbose

        if buffer is not None:
            self.R = buffer + n
            self.N = self.R + 1
        else:
            self.R = None
            self.N = self.calc_params.N

        # Arrival: u_i, lam_i (may be complex for CV<1 "complex-fit")
        self.u: list[complex] = []
        self.lam: list[complex] = []
        self.a: list[complex] = []

        # Service: y_i, mu_i (may be complex for CV<1 "complex-fit")
        self.y: list[complex] = []
        self.mu: list[complex] = []
        self.b: list[complex] = []

        # Structure
        self.cols: list[int] = []
        self._compositions_cache: dict[int, list[tuple[int, ...]]] = {}

        # State
        self.t: list[list[np.ndarray]] = []  # t[j][i] row vector for tier i, level j
        self.x = np.zeros(self.N, dtype=complex)
        self.z = np.zeros(self.N, dtype=complex)
        self.p = np.zeros(self.N)

        self.A: list[np.ndarray] = []
        self.B: list[np.ndarray] = []
        self.D: list[list[np.ndarray]] = []  # D[j][i] diagonal for level j, tier i

        self.num_of_iter_ = 0
        self.w: list[float] | None = None
        self.v: list[float] | None = None

    def set_sources(
        self,
        a: list[float] | None = None,
        u: list[float] | None = None,
        lam: list[float] | None = None,
        use_clx: bool = False,
    ):
        """Set arrival distribution.
        Either (a) moments a for H2 fit, or (u, lam) explicit params."""
        if a is not None:
            if self.k != 2:
                raise ValueError("Moment-based fit only for k=2")
            self.a = a
            if use_clx:
                h2 = H2Distribution.get_params_clx(a)
            else:
                h2 = H2Distribution.get_params(a)
            # Keep complex params (CV<1) as-is; for real params, complex(...) is harmless.
            self.u = [complex(h2.p1), 1.0 - complex(h2.p1)]
            self.lam = [complex(h2.mu1), complex(h2.mu2)]
        elif u is not None and lam is not None:
            if len(u) != self.k or len(lam) != self.k:
                raise ValueError(f"u and lam must have length {self.k}")
            if not np.isclose(sum(u), 1.0):
                raise ValueError("sum(u) must be 1")
            self.u = [complex(x) for x in u]
            self.lam = [complex(x) for x in lam]
            if not self.a:
                self.a = self._moments_from_ph(self.u, self.lam)
        else:
            raise ValueError("Provide either a (moments) or (u, lam)")
        self.is_sources_set = True

    def set_servers(
        self,
        b: list[float] | None = None,
        y: list[float] | None = None,
        mu: list[float] | None = None,
        use_clx: bool = False,
    ):
        """Set service distribution. Either moments b for H2 fit, or (y, mu) explicit."""
        if b is not None:
            if self.k != 2:
                raise ValueError("Moment-based fit only for k=2")
            self.b = b
            if use_clx:
                h2 = H2Distribution.get_params_clx(b)
            else:
                h2 = H2Distribution.get_params(b)
            self.y = [complex(h2.p1), 1.0 - complex(h2.p1)]
            self.mu = [complex(h2.mu1), complex(h2.mu2)]
        elif y is not None and mu is not None:
            if len(y) != self.k or len(mu) != self.k:
                raise ValueError(f"y and mu must have length {self.k}")
            if not np.isclose(sum(y), 1.0):
                raise ValueError("sum(y) must be 1")
            self.y = [complex(x) for x in y]
            self.mu = [complex(x) for x in mu]
            if not self.b:
                self.b = self._moments_from_ph(self.y, self.mu)
        else:
            raise ValueError("Provide either b (moments) or (y, mu)")
        self.is_servers_set = True

    def _moments_from_ph(self, probs: list[float], rates: list[float]) -> list[float]:
        """First 4 raw moments of phase-type distribution (mixture of exponentials)."""
        moments = [0.0] * 4
        for p, r in zip(probs, rates):
            mean = 1.0 / r
            for kk in range(4):
                moments[kk] += p * math.factorial(kk + 1) / (r ** (kk + 1))
        return moments

    def get_utilization(self) -> float:
        lam_eff = 1.0 / self.a[0] if self.a else sum(self.u[i] * self.lam[i] for i in range(self.k))
        util = lam_eff * self.b[0] / self.n
        return float(np.real(util))

    def _get_compositions(self, busy: int) -> list[tuple[int, ...]]:
        if busy not in self._compositions_cache:
            self._compositions_cache[busy] = _compositions(busy, self.k)
        return self._compositions_cache[busy]

    def _fill_cols(self) -> None:
        """Number of microstates per level (service keys)."""
        self.cols = []
        for j in range(self.N):
            busy = min(j, self.n)
            self.cols.append(_n_compositions(busy, self.k))
        self.cols.append(self.cols[-1])  # for B[N-1] closure

    def _build_a_matrix(self, j: int) -> np.ndarray:
        """A_{j}: arrivals, level j -> j+1. Unit intensity (textbook §7.6.2)."""
        busy_lo = min(j, self.n)
        busy_hi = min(j + 1, self.n)
        r, c = self.cols[j], self.cols[j + 1]
        out = np.zeros((r, c), dtype=complex)

        comps_lo = self._get_compositions(busy_lo)
        comps_hi = self._get_compositions(busy_hi)

        if j < self.n:
            # One more server: (m1..mk) -> (m1+1,m2..) with y1, (m1,m2+1,..) with y2, ...
            idx_hi = {m: i for i, m in enumerate(comps_hi)}
            for i_lo, m_lo in enumerate(comps_lo):
                for phase in range(self.k):
                    m_hi = list(m_lo)
                    m_hi[phase] += 1
                    m_hi = tuple(m_hi)
                    if m_hi in idx_hi:
                        out[i_lo, idx_hi[m_hi]] += self.y[phase]
        else:
            out = np.eye(r, dtype=complex)

        return out

    def _build_b_matrix(self, j: int) -> np.ndarray:
        """B_{j}: departures from level j to j-1. Shape (cols[j], cols[j-1])."""
        if j == 0:
            return np.zeros((1, 1))
        busy_hi = min(j, self.n)
        busy_lo = min(j - 1, self.n)
        r, c = self.cols[j], self.cols[j - 1]
        out = np.zeros((r, c), dtype=complex)

        comps_hi = self._get_compositions(busy_hi)
        comps_lo = self._get_compositions(busy_lo)
        idx_lo = {m: i for i, m in enumerate(comps_lo)}

        if j <= self.n:
            for i_hi, m_hi in enumerate(comps_hi):
                for phase in range(self.k):
                    if m_hi[phase] == 0:
                        continue
                    m_lo = list(m_hi)
                    m_lo[phase] -= 1
                    m_lo = tuple(m_lo)
                    if m_lo in idx_lo:
                        rate = m_hi[phase] * self.mu[phase]
                        out[i_hi, idx_lo[m_lo]] += rate
        else:
            for i_hi, m_hi in enumerate(comps_hi):
                for phase_out in range(self.k):
                    if m_hi[phase_out] == 0:
                        continue
                    rate_dep = m_hi[phase_out] * self.mu[phase_out]
                    for phase_in in range(self.k):
                        m_lo = list(m_hi)
                        m_lo[phase_out] -= 1
                        m_lo[phase_in] += 1
                        m_lo = tuple(m_lo)
                        if m_lo in idx_lo:
                            out[i_hi, idx_lo[m_lo]] += rate_dep * self.y[phase_in]

        return out.astype(complex)

    def _build_d_diag(self, j: int, i_tier: int) -> np.ndarray:
        """Diagonal of D_{j,i}: lam_i + service outflow per microstate."""
        busy = min(j, self.n)
        comps = self._get_compositions(busy)
        diag = np.zeros(len(comps), dtype=complex)
        for idx, m in enumerate(comps):
            outflow = sum(m[ph] * self.mu[ph] for ph in range(self.k))
            diag[idx] = self.lam[i_tier] + outflow
        return diag

    def run(self) -> QueueResults:
        start = time.process_time()
        self._check_if_servers_and_sources_set()

        self._fill_cols()
        self.A = [self._build_a_matrix(j) for j in range(self.N - 1)]
        self.B = [self._build_b_matrix(j) for j in range(self.N + 1)]
        self.D = []
        for j in range(self.N):
            self.D.append([self._build_d_diag(j, i) for i in range(self.k)])

        self._initial_probabilities()
        x_delta = float("inf")
        self.num_of_iter_ = 0

        while x_delta >= self.e1:
            x_old = self.x.copy()
            self.num_of_iter_ += 1

            for j in range(self.N - 1, 0, -1):
                self._update_level_j(j)

            self._update_level_0()
            x_delta = float(np.max(np.abs(self.x - x_old)))
            if self.num_of_iter_ >= self.calc_params.max_iter:
                break
            if self.verbose:
                print(f"Iter #{self.num_of_iter_}, max|x_delta|={x_delta:.2e}")

        self._calculate_p()
        if self.verbose:
            self._run_diagnostics()
        results = self.get_results()
        results.duration = time.process_time() - start
        return results

    def _initial_probabilities(self) -> None:
        for j in range(self.N):
            self.t.append([])
            c = self.cols[j]
            for i in range(self.k):
                self.t[j].append(np.ones(c, dtype=complex) / (c * self.k))
        for j in range(self.N):
            self.x[j] = 0.5 + 0.0j
            self.z[j] = 0.5 + 0.0j

    def _update_level_j(self, j: int) -> None:
        if j == 0:
            return

        V = sum(self.lam[i] * self.t[j - 1][i] for i in range(self.k))
        beta_prime = []
        beta_dbl = []

        for i in range(self.k):
            d_inv = 1.0 / (self.D[j][i] + 1e-20)
            va = V @ self.A[j - 1]
            bp = self.u[i] * va * d_inv
            beta_prime.append(bp)

            t_next = self.t[j + 1][i] if j + 1 < self.N else self.t[j - 1][i]
            tb = t_next @ self.B[j + 1]
            bd = tb * d_inv
            beta_dbl.append(bd)

        Bp = sum(beta_prime)
        Bd = sum(beta_dbl)

        one_hi = np.ones(self.cols[j])
        one_lo = np.ones(self.cols[j - 1])
        num = np.dot(Bd @ self.B[j], one_lo)
        den = np.dot(V, one_lo) - np.dot(Bp @ self.B[j], one_lo)
        if abs(den) < 1e-20:
            c = 1.0
        else:
            c = num / den

        den_x = np.dot(c * Bp + Bd, one_hi)
        if abs(den_x) < 1e-20:
            pass
        else:
            self.x[j] = 1.0 / den_x
        self.z[j] = c * self.x[j]

        for i in range(self.k):
            self.t[j][i] = self.z[j] * beta_prime[i] + self.x[j] * beta_dbl[i]

        s = sum(self.t[j][i].sum() for i in range(self.k))
        if s > 0:
            for i in range(self.k):
                self.t[j][i] /= s

    def _update_level_0(self) -> None:
        lam = self.lam
        # Level 1 microstates correspond to compositions of busy=1; each microstate has
        # its own service outflow rate Σ m_l * μ_l. Do NOT assume microstate order equals phase order.
        mu = np.array(self.mu, dtype=complex)
        comps1 = self._get_compositions(1)
        outflow_1 = np.array(
            [sum(m[ph] * mu[ph] for ph in range(self.k)) for m in comps1],
            dtype=complex,
        )
        raw = np.array([(self.t[1][i] * outflow_1).sum() / lam[i] for i in range(self.k)], dtype=complex)
        total = raw.sum()
        if total > 1e-20:
            self.x[0] = 1.0 / total
            for i in range(self.k):
                self.t[0][i] = np.array([raw[i] / total], dtype=complex)
        self.z[0] = 1.0 / self.x[0]

    def _calculate_p(self) -> None:
        lam_eff = 1.0 / self.a[0] if self.a else sum(self.u[i] * self.lam[i] for i in range(self.k))
        b_mean = self.b[0] if self.b else sum(self.y[i] / self.mu[i] for i in range(self.k))
        lam_eff_r = float(np.real(lam_eff))
        b_mean_r = float(np.real(b_mean))
        znam = self.n + sum((self.n - j) * float(np.real(np.prod(self.x[:j]))) for j in range(1, self.n))
        if self.R is not None:
            prod_r = float(np.real(np.prod(self.x[: self.N])))
            znam -= lam_eff_r * b_mean_r * prod_r
        self.p[0] = (self.n - lam_eff_r * b_mean_r) / znam
        for j in range(self.N - 1):
            self.p[j + 1] = self.p[j] * float(np.real(self.x[j]))
        total = self.p.sum()
        if total > 0:
            self.p /= total
        assert np.isclose(self.p.sum(), 1.0, atol=1e-9, rtol=1e-8)

    def _run_diagnostics(self) -> None:
        """Diagnostic checks under verbose: (7.6.13) level 0, (7.6.5) balance residual, x vs p."""
        lam = np.array(self.lam, dtype=complex)
        mu = np.array(self.mu, dtype=complex)
        comps1 = self._get_compositions(1)
        outflow_1 = np.array(
            [sum(m[ph] * mu[ph] for ph in range(self.k)) for m in comps1],
            dtype=complex,
        )
        print("  [diag] Level 0 (7.6.13): t0*lam vs x0*sum(t1*outflow)")
        for i in range(self.k):
            lhs = float(np.real(self.t[0][i][0] * lam[i]))
            rhs = float(np.real(self.x[0] * (self.t[1][i] * outflow_1).sum()))
            print(f"    i={i}: LHS={lhs:.6e}, RHS={rhs:.6e}, diff={abs(lhs-rhs):.2e}")

        for j_check in [1, min(2, self.N - 2)] if self.N > 2 else [1]:
            if j_check >= self.N - 1:
                continue
            V = sum(self.lam[i] * self.t[j_check - 1][i] for i in range(self.k))
            residual_max = 0.0
            for i in range(self.k):
                lhs = self.t[j_check][i] * self.D[j_check][i]
                term2 = self.u[i] * self.z[j_check] * (V @ self.A[j_check - 1])
                t_next = self.t[j_check + 1][i] if j_check + 1 < self.N else self.t[j_check - 1][i]
                term3 = self.x[j_check] * (t_next @ self.B[j_check + 1])
                res = np.abs(lhs - term2 - term3)
                residual_max = max(residual_max, float(np.max(res)))
            print(f"  [diag] Balance (7.6.5) j={j_check}: max|residual|={residual_max:.2e}")

        print("  [diag] x_j vs p[j+1]/p[j]:")
        for j in range(min(5, self.N - 1)):
            if self.p[j] > 1e-15:
                ratio = self.p[j + 1] / self.p[j]
                print(f"    j={j}: x_j={float(np.real(self.x[j])):.6f}, p[j+1]/p[j]={ratio:.6f}")

    def get_results(self, num_of_moments: int = 4) -> QueueResults:
        self.p = self.get_p()
        self.w = self.get_w(num_of_moments)
        self.v = self.get_v(num_of_moments)
        return QueueResults(
            v=self.v,
            w=self.w,
            p=list(self.p),
            utilization=self.get_utilization(),
        )

    def get_p(self) -> np.ndarray:
        return np.asarray(self.p, dtype=float)

    def get_w(self, num_of_moments: int = 4) -> list[float]:
        w = [0.0] * 4
        for j in range(1, min(len(self.p) - self.n, 500)):
            w[0] += j * self.p[self.n + j]
        for j in range(2, min(len(self.p) - self.n, 500)):
            w[1] += j * (j - 1) * self.p[self.n + j]
        for j in range(3, min(len(self.p) - self.n, 500)):
            w[2] += j * (j - 1) * (j - 2) * self.p[self.n + j]
        for j in range(4, min(len(self.p) - self.n, 500)):
            w[3] += j * (j - 1) * (j - 2) * (j - 3) * self.p[self.n + j]

        lam_eff = 1.0 / self.a[0] if self.a else sum(self.u[i] * self.lam[i] for i in range(self.k))
        for i in range(4):
            w[i] /= lam_eff ** (i + 1)
        return w[:num_of_moments]

    def get_v(self, num_of_moments: int = 4) -> list[float]:
        if self.v is not None:
            return self.v
        self.w = self.w or self.get_w(num_of_moments)
        b_moments = [
            sum(self.y[i] * math.factorial(kk) / (self.mu[i] ** kk) for i in range(self.k))
            for kk in range(1, num_of_moments + 1)
        ]
        self.v = conv_moments(b_moments, self.w, num=num_of_moments)
        return list(self.v)
