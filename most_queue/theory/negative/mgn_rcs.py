"""
Calculate M/H2/n queue with negative jobs with RCS discipline,
(remove customer from service)
"""

import math
import time

import numpy as np
from scipy.misc import derivative

from most_queue.random.distributions import H2Distribution
from most_queue.structs import NegativeArrivalsResults
from most_queue.theory.fifo.mgn_takahasi import MGnCalc, TakahashiTakamiParams
from most_queue.theory.utils.restarts import beff_moments_repeat_without_resampling_from_h2
from most_queue.theory.utils.transforms import lst_exp


class MGnNegativeRCSCalc(MGnCalc):
    """
    Calculate M/H2/n queue with negative jobs with RCS discipline,
    (remove customer from service)
    """

    def __init__(
        self,
        n: int,
        buffer: int | None = None,
        calc_params: TakahashiTakamiParams | None = None,
        requeue_on_disaster: bool = False,
        resume_on_negative: bool = False,
        repeat_without_resampling: bool = False,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """
        n: number of servers
        buffer: size of the buffer (optional)
        calc_params: TakahashiTakamiParams object with parameters for calculation
        """

        super().__init__(n=n, buffer=buffer, calc_params=calc_params)

        self.l_pos = None
        self.l_neg = None
        self.base_mgn = None
        self.w = None
        self._w_def_pls_cache: dict[float, complex] = {}
        self.requeue_on_disaster = bool(requeue_on_disaster)
        self.resume_on_negative = bool(resume_on_negative)
        self.repeat_without_resampling = bool(repeat_without_resampling)
        self._requeue_results: NegativeArrivalsResults | None = None
        self._resume_results: NegativeArrivalsResults | None = None

    def set_sources(self, l_pos: float, l_neg: float):  # pylint: disable=arguments-differ
        """
        Set the arrival rates of positive and negative jobs
        :param l_pos: arrival rate of positive jobs
        :param l_neg: arrival rate of negative jobs
        """
        self.l_pos = l_pos
        self.l = l_pos
        self.l_neg = l_neg
        self._w_def_pls_cache = {}
        self._requeue_results = None
        self._resume_results = None
        self.is_sources_set = True

    def set_servers(self, b: list[float]):  # pylint: disable=arguments-differ
        """
        Set the raw moments of service time distribution
        :param b: raw moments of service time distribution
        """
        self.b = b

        h2_params = H2Distribution.get_params_clx(b)
        # params of H2-distribution:
        self.y = [h2_params.p1, 1.0 - h2_params.p1]
        self.mu = [h2_params.mu1, h2_params.mu2]

        self.is_servers_set = True
        self._w_def_pls_cache = {}
        self._requeue_results = None
        self._resume_results = None

    def run(self):  # pylint: disable=arguments-differ
        """
        Run calculation.

        If requeue_on_disaster=True, treat negative arrivals as service restarts
        (no removals): every job is eventually served.
        """
        if self.resume_on_negative:
            start = time.process_time()
            res = self._ensure_resume_results(num_of_moments=4)
            res.duration = time.process_time() - start
            return res
        if self.requeue_on_disaster:
            start = time.process_time()
            res = self._ensure_requeue_results(num_of_moments=4)
            res.duration = time.process_time() - start
            return res
        return super().run()  # type: ignore[return-value]

    def _ensure_resume_results(self, num_of_moments: int = 4) -> NegativeArrivalsResults:
        """
        RESUME scenario for RCS: negative arrivals interrupt service and requeue the task,
        but service continues with the remaining work (preemptive-resume; sample once).

        Important: in the current simulator implementation, an RCS "RESUME" negative arrival
        preempts a job and immediately starts service again (same instant), preserving the
        remaining service requirement. There is no additional setup / repair / delay time.

        Under these semantics, negative arrivals do not change the total required service
        per job and do not introduce downtime, so (to a very good approximation) the system
        behaves like the base M/G/n without negative jobs.

        Therefore we compute RESUME results by running a standard MGnCalc with the original
        service-time moments and positive arrival rate, and then wrap them into
        NegativeArrivalsResults with q=1 and v_broken=0.
        """
        if self._resume_results is not None and len(self._resume_results.v) >= num_of_moments:
            return self._resume_results

        self._check_if_servers_and_sources_set()

        lam = float(self.l_pos)

        eff = MGnCalc(n=self.n, buffer=self.buffer, calc_params=self.calc_params)
        eff.set_sources(l=lam)
        eff.set_servers(b=[float(x) for x in self.b[:num_of_moments]])
        eff_res = eff.run()

        v = [float(x) for x in eff_res.v[:num_of_moments]]
        w = [float(x) for x in eff_res.w[:num_of_moments]]
        p = [float(x) for x in eff_res.p]

        self._resume_results = NegativeArrivalsResults(
            v=v,
            w=w,
            p=p,
            utilization=float(eff_res.utilization),
            v_broken=[0.0] * num_of_moments,
            v_served=v,
            q=1.0,
            duration=0.0,
        )
        return self._resume_results

    def _ensure_requeue_results(self, num_of_moments: int = 4) -> NegativeArrivalsResults:
        """
        REQUEUE scenario for RCS: negative arrivals interrupt service and force restart,
        but the customer remains in the system (no "broken" jobs).

        Approximation: model it as a standard M/G/n (no negative departures) with an
        effective service time B_eff that accounts for Poisson restarts.

        For service time B with LST β(s) and restart rate r (during service):

            B_eff^*(s) = β(s+r) * (s+r) / (s + r * β(s+r)).

        If repeat_without_resampling=True, we use *repeat without resampling* semantics:
        B is sampled once per job and reused on every restart, so the above LST does not
        apply; instead we approximate moments of B_eff using a Gamma fit (see restarts.py).

        Here the restart intensity depends on the number of busy servers m:
        r(m)=l_neg/m (uniform selection among busy servers). To better match
        finite-load regimes (where m can be noticeably smaller than n), we use a
        simple self-consistent closure:

          m ≈ n * utilization,
          r = l_neg / max(1, m),

        where utilization is taken from the auxiliary M/G/n run with B_eff, and
        the fixed point is refined by a few iterations.
        """
        if self._requeue_results is not None and len(self._requeue_results.v) >= num_of_moments:
            return self._requeue_results

        self._check_if_servers_and_sources_set()

        lam = float(self.l_pos)
        delta = float(self.l_neg)
        r = 0.0 if self.n <= 0 else delta / float(self.n)

        mean_b = float(self.b[0])
        var_b = float(self.b[1] - self.b[0] ** 2)

        # Gamma approximation parameters for β(s) by mean/variance
        if var_b > 0 and mean_b > 0:
            k = mean_b * mean_b / var_b
            theta = var_b / mean_b

        def get_b_eff(service_restart_rate: float) -> list[float]:
            """
            Compute raw moments of the effective service time under Poisson restarts.

            Instead of numerically differentiating B_eff^*(s) at s=0, compute the Taylor
            series of

              B_eff^*(s) = β(s+r) * (s+r) / (s + r * β(s+r))

            using β-derivatives at s=r. For Gamma/Exp β(s) used here, these derivatives
            are analytic and this approach is much more stable for large CV.
            """
            if service_restart_rate <= 0:
                return [float(x) for x in self.b[:num_of_moments]]

            r_loc = float(service_restart_rate)
            order = int(num_of_moments)

            if self.repeat_without_resampling:
                return beff_moments_repeat_without_resampling_from_h2(
                    [complex(self.y[0]), complex(self.y[1])],
                    [complex(self.mu[0]), complex(self.mu[1])],
                    r_loc,
                    num_of_moments=order,
                )

            # β^(n)(s0) for n=0..order at s0=r_loc
            s0 = r_loc
            beta_derivs: list[float] = [0.0] * (order + 1)

            if var_b > 0 and mean_b > 0:
                # β(s)=(1+θ s)^(-k)
                base = 1.0 + theta * s0

                # rising factorial (k)_n = k (k+1) ... (k+n-1)
                rf = 1.0
                for n in range(order + 1):
                    if n == 0:
                        rf = 1.0
                    elif n == 1:
                        rf = float(k)
                    else:
                        rf *= float(k + (n - 1))
                    beta_derivs[n] = float(((-1.0) ** n) * (theta**n) * rf * (base ** (-(k + n))))
            else:
                # β(s)=exp(-mean_b * s)
                e0 = float(np.exp(-mean_b * s0))
                for n in range(order + 1):
                    beta_derivs[n] = float(((-mean_b) ** n) * e0)

            # Convert to ordinary power-series coefficients for f(s)=β(s+r_loc):
            # f(s)=Σ c_f[n] s^n, where c_f[n]=β^(n)(r_loc)/n!
            c_f = [beta_derivs[n] / float(math.factorial(n)) for n in range(order + 1)]

            # Numerator N(s)=(r+s)f(s) = r f(s) + s f(s)
            c_N = [0.0] * (order + 1)
            for n in range(order + 1):
                c_N[n] += r_loc * c_f[n]
                if n > 0:
                    c_N[n] += c_f[n - 1]

            # Denominator D(s)=s + r f(s)
            c_D = [0.0] * (order + 1)
            for n in range(order + 1):
                c_D[n] += r_loc * c_f[n]
            if order >= 1:
                c_D[1] += 1.0

            # Series division T(s)=N(s)/D(s) up to s^order
            c_T = [0.0] * (order + 1)
            d0 = float(c_D[0])
            if not np.isfinite(d0) or abs(d0) <= 1e-300:
                # Extremely small/invalid denominator -> treat as no restarts.
                return [float(x) for x in self.b[:num_of_moments]]

            c_T[0] = float(c_N[0]) / d0
            for n in range(1, order + 1):
                acc = float(c_N[n])
                for k2 in range(1, n + 1):
                    acc -= float(c_D[k2]) * float(c_T[n - k2])
                c_T[n] = acc / d0

            # Moments from Laplace series: T(s)=Σ (-1)^n E[T^n] s^n / n!
            moments: list[float] = []
            for n in range(1, order + 1):
                mom = ((-1.0) ** n) * float(math.factorial(n)) * float(c_T[n])
                moments.append(float(mom))
            return moments

        # Self-consistent refinement of r using the auxiliary M/G/n run.
        #
        # Important: for some parameter combinations, a too aggressive r-update may lead
        # to an unstable effective model (ρ_eff >= 1) and numerical overflow inside MGnCalc.
        # We therefore use a damped fixed-point iteration and fall back to the last
        # successful auxiliary run.
        eff_res = None
        last_good_eff_res = None
        max_iter = 8
        tol = 1e-6
        damping = 0.5
        for _ in range(max_iter):
            b_eff = get_b_eff(r)
            # Basic sanity checks to avoid passing unstable/invalid moments to MGnCalc.
            if (not np.all(np.isfinite(b_eff))) or b_eff[0] <= 0:
                r *= 0.5
                continue
            # If the auxiliary M/G/n would be unstable (ρ_eff >= 1), back off on r.
            if lam * float(b_eff[0]) >= 0.999 * float(self.n):
                r *= 0.5
                continue

            eff = MGnCalc(n=self.n, buffer=self.buffer, calc_params=self.calc_params)
            eff.set_sources(l=lam)
            eff.set_servers(b=b_eff)
            try:
                eff_res = eff.run()
            except Exception:  # pylint: disable=broad-exception-caught
                # Back off: keep the last good result, and reduce r to regain stability.
                if last_good_eff_res is not None:
                    eff_res = last_good_eff_res
                    break
                r *= 0.5
                continue

            last_good_eff_res = eff_res

            # Approximate restart intensity for a job in service.
            #
            # A naive closure uses r ≈ δ / E[M | M>0] (M = number of busy servers).
            # However, for high-variance service times, restart phenomena can reduce the
            # *effective* completion time (stochastic restart effect), and empirical
            # agreement improves if we bias the estimate towards larger M values.
            #
            # Heuristic: r ≈ δ * E[M] / E[M^2], where moments are taken under the auxiliary
            # M/G/n stationary distribution (with M = min(n, level)).
            p = [float(x) for x in getattr(eff_res, "p", [])]
            if p:
                m1 = 0.0
                m2 = 0.0
                for k, pk in enumerate(p):
                    m = min(float(self.n), float(k))
                    w = max(0.0, pk)
                    m1 += m * w
                    m2 += (m * m) * w
                if m2 <= 1e-12:
                    target_r = delta / float(self.n)
                else:
                    target_r = delta * (m1 / m2)
            else:
                target_r = delta / float(self.n)
            if abs(target_r - r) <= tol * max(1.0, abs(r)):
                r = target_r
                break

            r = (1.0 - damping) * r + damping * target_r
            r = max(0.0, min(delta, r))

        assert last_good_eff_res is not None
        eff_res = last_good_eff_res

        v = [float(x) for x in eff_res.v[:num_of_moments]]
        w = [float(x) for x in eff_res.w[:num_of_moments]]
        p = [float(x) for x in eff_res.p]

        self._requeue_results = NegativeArrivalsResults(
            v=v,
            w=w,
            p=p,
            utilization=float(eff_res.utilization),
            v_broken=[0.0] * num_of_moments,
            v_served=v,
            q=1.0,
            duration=0.0,
        )
        return self._requeue_results

    def _calculate_p(self):
        """
        Calculate level probabilities.

        Base MGnCalc uses a closed-form p[0] formula that assumes the standard
        M/H2/n structure without additional horizontal transitions.

        Note:
        In the REQUEUE scenario (`requeue_on_disaster=True`) this calculator returns
        results via the B_eff approximation and does not run the TT iterations.
        The normalization-by-x branch below is kept as a fallback if someone runs
        the matrix iteration path manually.
        """
        if self.requeue_on_disaster:
            self.p[0] = 1.0 + 0.0j
            for j in range(self.N - 1):
                self.p[j + 1] = self.p[j] * self.x[j]
            total = sum(self.p)
            self.p = np.array([val / total for val in self.p], dtype=self.dt)
            return None

        super()._calculate_p()
        return None

    def _update_level_0(self):
        """
        Update level 0 - skip t[0] update for RCS discipline.
        t[0] remains [1.0] always for this model.
        """
        self.x[0] = (1.0 + 0.0j) / self.z[1]
        # Note: t[0] update is skipped - it remains [1.0] for this model

    def get_results(self, num_of_moments: int = 4, derivate=False) -> NegativeArrivalsResults:
        """
        Get all results - override to return NegativeArrivalsResults instead of QueueResults.
        """
        _ = derivate
        return self.collect_results(num_of_moments)

    def collect_results(self, num_of_moments: int = 4) -> NegativeArrivalsResults:
        """
        Get all results
        """
        if self.resume_on_negative:
            return self._ensure_resume_results(num_of_moments=num_of_moments)
        if self.requeue_on_disaster:
            return self._ensure_requeue_results(num_of_moments=num_of_moments)

        self.p = self.get_p()
        self.w = self.get_w(num_of_moments)
        self.v = self.get_v(num_of_moments)
        v_served = self.get_v_served(num_of_moments)
        v_broken = self.get_v_broken(num_of_moments)

        utilization = self.get_utilization()

        return NegativeArrivalsResults(
            v=self.v,
            w=self.w,
            p=self.p,
            utilization=utilization,
            v_broken=v_broken,
            v_served=v_served,
            q=float(self.get_q()),
        )

    def get_utilization(self):
        """
        Calculate utilization of the queue.

        Note: This is a simplified version that does not fully account for
        the effect of disasters on utilization. A more accurate calculation
        would consider the impact of disaster events on system utilization.
        """
        if self.requeue_on_disaster:
            return float(self._ensure_requeue_results(num_of_moments=1).utilization)
        return self.l_pos * self.b[0] / self.n

    def get_p(self) -> list[float]:
        """
        Level probabilities p[k].
        """
        if self.requeue_on_disaster:
            return list(self._ensure_requeue_results(num_of_moments=1).p)
        return super().get_p()

    def get_q(self) -> float:
        """
        Calculation of the conditional probability of successful service completion at a node
        """
        if self.requeue_on_disaster:
            return 1.0
        return 1.0 - (self.l_neg / self.l) * (1.0 - self.p[0].real)

    def _beta_pls(self, s: float) -> complex:
        """
        LST of service time B ~ H2(y, mu): β(s) = E[e^{-sB}]
        """
        return self.y[0] * lst_exp(self.mu[0], s) + self.y[1] * lst_exp(self.mu[1], s)

    @staticmethod
    def _min_service_pls(s: float, r: float, beta_at: complex) -> complex:
        """
        LST of min(B, Y) where Y ~ Exp(r), independent of B:
            E[e^{-s min(B,Y)}] = r/(s+r) + s/(s+r) * β(s+r)
        Here β(s+r) is passed as beta_at for convenience.
        """
        return r / (s + r) + (s / (s + r)) * beta_at

    def _calc_w_def_pls(self, s: float) -> complex:
        """
        Defective LST contribution for waiting time W in RCS model:
        computes E[e^{-sW}; arrival sees level >= n] (i.e., W>0 part).

        Compared to the base MGnCalc waiting LST, we must account that a
        "departure" freeing a server can happen either by service completion
        or by negative job removing someone in service. At levels >= n during
        the waiting period, all servers are busy, so the inter-departure time
        in microstate j is Exp(service_rate(j) + l_neg).
        """
        s_key = float(s)
        cached = self._w_def_pls_cache.get(s_key)
        if cached is not None:
            return cached

        w_def = 0.0 + 0.0j

        key_numbers = self._get_key_numbers(self.n)
        a = np.array(
            [
                lst_exp(key_numbers[j][0] * self.mu[0] + key_numbers[j][1] * self.mu[1] + self.l_neg, s)
                for j in range(self.n + 1)
            ],
            dtype=self.dt,
        )

        up_transition_mrx = self.calc_up_probs(self.n + 1)  # (n+1 x n+1)

        for k in range(self.n, self.N):
            pa = np.linalg.matrix_power(up_transition_mrx * a, k - self.n)  # (n+1 x n+1)
            ys = np.array([self.Y[k][0, i] for i in range(self.n + 1)], dtype=self.dt)
            a_pa = np.dot(pa, a)
            w_def += ys.dot(a_pa)

        self._w_def_pls_cache[s_key] = w_def
        return w_def

    def _w_pls(self, s: float) -> complex:
        """
        Full LST of waiting time W for RCS semantics:
        customers in queue are NOT removed by negative jobs, so W is simply
        the time until service begins (including the effect of negative removals
        of other customers that free servers).
        """
        p_immediate = sum(float(self.p[k]) for k in range(min(self.n, len(self.p))))
        return p_immediate + self._calc_w_def_pls(s)

    def get_w(self, num_of_moments: int = 4) -> list[float]:
        """
        Waiting time moments for RCS model (time in queue until service begins).
        Computed via derivatives of W*(s) at s=0.
        """
        if self.requeue_on_disaster:
            return list(self._ensure_requeue_results(num_of_moments=num_of_moments).w[:num_of_moments])
        if self.w is not None:
            return self.w[:num_of_moments]

        w = [0.0] * num_of_moments
        for i in range(num_of_moments):
            w[i] = derivative(self._w_pls, 0, dx=1e-3 / self.b[0], n=i + 1, order=9)
            if i % 2 == 0:
                w[i] = -w[i]

        self.w = [w_m.real if isinstance(w_m, complex) else float(w_m) for w_m in w]
        return self.w

    def _v_pls(self, s: float) -> complex:
        """
        LST of sojourn time V in RCS model under the approximation:

        - while waiting: customer is never removed by negatives;
        - during service: removal hazard is approximately δ/m where m is the number
          of busy servers when the customer starts service (kept constant).

        This matches the existing model intent (using l_neg/i) but fixes the
        previous inconsistency by combining it with a correct waiting-time LST.
        """
        delta = float(self.l_neg)

        # Immediate service cases (level k < n at arrival -> busy servers m = k+1).
        v = 0.0 + 0.0j
        for k in range(min(self.n, len(self.p))):
            m = k + 1
            r = delta / m
            v += float(self.p[k]) * self._min_service_pls(s, r, self._beta_pls(s + r))

        # Waiting cases (level >= n at arrival -> assume m = n during service).
        r_wait = delta / self.n
        v += self._calc_w_def_pls(s) * self._min_service_pls(s, r_wait, self._beta_pls(s + r_wait))
        return v

    def get_v(self, num_of_moments: int = 4) -> list[float]:
        """
        Get the sojourn time moments
        """
        if self.requeue_on_disaster:
            return list(self._ensure_requeue_results(num_of_moments=num_of_moments).v[:num_of_moments])
        if self.v is not None:
            return self.v[:num_of_moments]

        v = [0.0] * num_of_moments
        for i in range(num_of_moments):
            v[i] = derivative(self._v_pls, 0, dx=1e-3 / self.b[0], n=i + 1, order=9)
            if i % 2 == 0:
                v[i] = -v[i]

        self.v = [v_m.real if isinstance(v_m, complex) else float(v_m) for v_m in v]
        return self.v

    def get_v_served(self, num_of_moments: int = 4) -> list[float]:
        """
        Sojourn time moments conditional on being served (service completion
        occurs before negative removal).
        """
        if self.requeue_on_disaster:
            return self.get_v(num_of_moments=num_of_moments)
        delta = float(self.l_neg)

        def served_num_pls(s: float) -> complex:
            # E[e^{-sV}; served] = sum_{k<n} p[k] * β(s+r_k) + W_def*(s) * β(s+r_wait)
            num = 0.0 + 0.0j
            for k in range(min(self.n, len(self.p))):
                r = delta / (k + 1)
                num += float(self.p[k]) * self._beta_pls(s + r)
            r_wait = delta / self.n
            num += self._calc_w_def_pls(s) * self._beta_pls(s + r_wait)
            return num

        p_served = float(served_num_pls(0.0).real)
        if p_served <= 0:
            return [0.0] * num_of_moments

        def v_served_pls(s: float) -> complex:
            return served_num_pls(s) / p_served

        moments = [0.0] * num_of_moments
        for i in range(num_of_moments):
            moments[i] = derivative(v_served_pls, 0, dx=1e-3 / self.b[0], n=i + 1, order=9)
            if i % 2 == 0:
                moments[i] = -moments[i]

        return [m.real if isinstance(m, complex) else float(m) for m in moments]

    def get_v_broken(self, num_of_moments: int = 4) -> list[float]:
        """
        Sojourn time moments conditional on being broken by negative removal
        while in service.
        """
        if self.requeue_on_disaster:
            return [0.0] * num_of_moments
        delta = float(self.l_neg)

        def served_prob() -> float:
            p = 0.0
            for k in range(min(self.n, len(self.p))):
                r = delta / (k + 1)
                p += float(self.p[k]) * float(self._beta_pls(r).real)
            r_wait = delta / self.n
            p_wait = float(self._calc_w_def_pls(0.0).real)
            p += p_wait * float(self._beta_pls(r_wait).real)
            return p

        p_served = served_prob()
        p_broken = 1.0 - p_served
        if p_broken <= 0:
            return [0.0] * num_of_moments

        def broken_num_pls(s: float) -> complex:
            # E[e^{-sV}; broken] = sum_{k<n} p[k] * r_k/(s+r_k)*(1-β(s+r_k))
            #                 + W_def*(s) * r_wait/(s+r_wait)*(1-β(s+r_wait))
            num = 0.0 + 0.0j
            for k in range(min(self.n, len(self.p))):
                r = delta / (k + 1)
                num += float(self.p[k]) * (r / (s + r)) * (1.0 - self._beta_pls(s + r))
            r_wait = delta / self.n
            num += self._calc_w_def_pls(s) * (r_wait / (s + r_wait)) * (1.0 - self._beta_pls(s + r_wait))
            return num

        def v_broken_pls(s: float) -> complex:
            return broken_num_pls(s) / p_broken

        moments = [0.0] * num_of_moments
        for i in range(num_of_moments):
            moments[i] = derivative(v_broken_pls, 0, dx=1e-3 / self.b[0], n=i + 1, order=9)
            if i % 2 == 0:
                moments[i] = -moments[i]

        return [m.real if isinstance(m, complex) else float(m) for m in moments]

    def _calc_service_probs(self) -> list[float]:
        """
        Returns the conditional probabilities of loaded states.
        """
        ps = np.array([self.p[i] for i in range(1, self.n + 1)])
        ps /= np.sum(ps)

        return ps

    def _build_big_b_matrix(self, num):
        """
        Create matrix B by the given level number.
        """
        if self.requeue_on_disaster:
            # In REQUEUE mode there are no negative removals (no downward jumps).
            return super()._build_big_b_matrix(num)
        if num == 0:
            return np.zeros((1, 1), dtype=self.dt)

        if num <= self.n:
            col = self.cols[num - 1]
            row = self.cols[num]
        else:
            col = self.cols[self.n + 1]
            row = self.cols[self.n + 1]

        output = np.zeros((row, col), dtype=self.dt)

        if num > self.n + 1:
            output = self.B[self.n + 1]
            return output

        for i in range(col):
            if num <= self.n:
                # key has two parts: left and right, like 21
                # fill B matrix by pattern of 2 elements:
                # up arrow, arrow from next to current key
                #  x 0
                #  x y
                #  0 y
                first_neg, second_neg = (
                    self.l_neg * (num - i) / num,
                    self.l_neg * (i + 1) / num,
                )
                output[i, i] = (num - i) * self.mu[0] + first_neg
                output[i + 1, i] = (i + 1) * self.mu[1] + second_neg
            else:
                # key has two parts: left and right, like 21
                # fill B matrix by pattern of 3 elements:
                # up arrow, right arrow, arrow from next to current key
                #  x x 0
                #  x y y
                #  0 y 0
                left = self.l_neg * (num - i - 1) / (num - 1)
                right = self.l_neg * i / (num - 1)
                left_from_next = self.l_neg * (i + 1) / (num - 1)

                output[i, i] = ((num - i - 1) * self.mu[0] + left) * self.y[0] + (i * self.mu[1] + right) * self.y[1]

                if i != num - 1:
                    output[i, i + 1] = ((num - i - 1) * self.mu[0] + left) * self.y[1]

                    output[i + 1, i] = ((i + 1) * self.mu[1] + left_from_next) * self.y[0]
        return output

    def _build_big_d_matrix(self, num):
        """
        Create matrix D by the given level number.
        """
        if self.requeue_on_disaster:
            return super()._build_big_d_matrix(num)

        if num < self.n:
            col = self.cols[num]
            row = col
        else:
            col = self.cols[self.n]
            row = col

        output = np.zeros((row, col), dtype=self.dt)

        if num > self.n:
            output = self.D[self.n]
            return output

        for i in range(row):
            output[i, i] = self.l + self.l_neg + (num - i) * self.mu[0] + i * self.mu[1]

        return output
