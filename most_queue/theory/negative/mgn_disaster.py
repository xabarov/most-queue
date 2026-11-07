"""
Calculate M/H2/n queue with negative jobs with disasters,
"""

import numpy as np

from most_queue.random.distributions import H2Distribution
from most_queue.structs import NegativeArrivalsResults
from most_queue.theory.fifo.mgn_takahasi import MGnCalc, TakahashiTakamiParams
from most_queue.theory.utils.derivative import derivative
from most_queue.theory.utils.restarts import effective_service_moments
from most_queue.theory.utils.transforms import lst_exp


class MGnNegativeDisasterCalc(MGnCalc):
    """
    Calculate M/H2/n queue with negative jobs (DISASTER-type).

    Three scenarios are supported:

    - `False` (default): negative arrival clears the system (removes all positive jobs),
      matching the original "DISASTER" semantics.
    - `requeue_on_disaster=True`: negative arrival interrupts service and returns the
      positive jobs in service back to the head of the queue (service restarts with resampling).
      In this mode the calculation is approximated via an equivalent M/G/n with an effective
      service time that accounts for *restart with resampling*.
    - `resume_on_disaster=True`: negative arrival interrupts service and returns the positive
      jobs in service back to the head of the queue, but service continues with the remaining
      work (preemptive-resume; service time is sampled once). In this mode we use an
      approximation via an equivalent M/G/n with an effective completion-time distribution.
    """

    def __init__(
        self,
        n: int,
        buffer: int | None = None,
        calc_params: TakahashiTakamiParams | None = None,
        requeue_on_disaster: bool = False,
        resume_on_disaster: bool = False,
        repeat_without_resampling: bool = False,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """
        n: number of servers
        buffer: size of the buffer (optional)
        calc_params: TakahashiTakamiParams object with parameters for calculation
        requeue_on_disaster: if True, use "restart to queue" scenario instead of clearing
        """

        super().__init__(n=n, buffer=buffer, calc_params=calc_params)

        self.l_pos = None
        self.l_neg = None
        self.gamma = None
        self.base_mgn = None
        self.w = None
        self._w0_pls_cache: dict[float, complex] = {}
        self._z0_pls_cache: dict[float, complex] = {}
        self.requeue_on_disaster = bool(requeue_on_disaster)
        self.resume_on_disaster = bool(resume_on_disaster)
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
        self._w0_pls_cache = {}
        self._z0_pls_cache = {}
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
        self.gamma = 1.0 / b[0]  # Auxiliary rate; artificial holding times are censored.
        self.is_servers_set = True
        self._w0_pls_cache = {}
        self._z0_pls_cache = {}
        self._requeue_results = None
        self._resume_results = None

    def run(self) -> NegativeArrivalsResults:  # pylint: disable=arguments-differ
        """
        Run calculation.

        In REQUEUE mode the calculation uses an equivalent M/G/n approximation with
        effective service moments B_eff; dependence between servers is neglected.
        """
        if self.resume_on_disaster:
            return self._ensure_resume_results(num_of_moments=4)
        if self.requeue_on_disaster:
            return self._ensure_requeue_results(num_of_moments=4)
        return super().run()  # type: ignore[return-value]

    def _pre_run_setup(self):
        """
        Setup before main iteration loop - initialize base_mgn for matrix references.
        """
        if not self.requeue_on_disaster:
            # Initialize base MGnCalc for matrix reference (needed for CLEAR_SYSTEM mode).
            self.base_mgn = MGnCalc(n=self.n, buffer=self.buffer, calc_params=self.calc_params)
            self.base_mgn.set_sources(l=self.l_pos)
            self.base_mgn.set_servers(b=self.b)
            self.base_mgn.fill_cols()
            self.base_mgn.build_matrices()

        # Now do standard setup
        super()._pre_run_setup()

    def _ensure_requeue_results(self, num_of_moments: int = 4) -> NegativeArrivalsResults:
        """
        REQUEUE scenario (service restarts on each disaster):

        We approximate the system by a standard M/G/n (no negative departures),
        but with an *effective* service time B_eff that accounts for restart.

        For a renewal service time B with LST β(s) and Poisson restarts with rate δ,
        the completion time LST is:

            B_eff^*(s) = β(s+δ) * (s+δ) / (s + δ * β(s+δ)).

        Here β(s) uses a real H2 fit when available, otherwise a Gamma moment fit.
        Shared disasters correlate servers; the iid M/G/n reduction is approximate.
        """
        if self._requeue_results is not None and len(self._requeue_results.v) >= num_of_moments:
            return self._requeue_results

        self._check_if_servers_and_sources_set()

        lam = float(self.l_pos)
        delta = float(self.l_neg)

        b_eff = effective_service_moments(
            self.b,
            self.y,
            self.mu,
            delta,
            fixed=self.repeat_without_resampling,
            num_of_moments=max(3, num_of_moments),
        )
        if not np.all(np.isfinite(b_eff)) or b_eff[0] <= 0:
            raise ValueError("Effective-service moments do not exist at the specified restart rate.")
        if self.buffer is None and lam * b_eff[0] >= self.n:
            raise ValueError("Auxiliary restart approximation is unstable (lambda E[B_eff] >= n).")

        # Run a standard MGnCalc with effective service moments
        eff = MGnCalc(n=self.n, buffer=self.buffer, calc_params=self.calc_params)
        eff.set_sources(l=lam)
        eff.set_servers(b=b_eff)
        eff_res = eff.run()

        v = [float(x) for x in eff_res.v[:num_of_moments]]
        w = [float(x) for x in eff_res.w[:num_of_moments]]
        p = [float(x) for x in eff_res.p]
        v_broken = [0.0] * num_of_moments

        self._requeue_results = NegativeArrivalsResults(
            v=v,
            w=w,
            p=p,
            utilization=float(eff_res.utilization),
            v_broken=v_broken,
            v_served=v,
            q=1.0,
            duration=0.0,
        )
        return self._requeue_results

    def _ensure_resume_results(self, num_of_moments: int = 4) -> NegativeArrivalsResults:
        """
        RESUME scenario (service continues after interruptions with the remaining work):

        A disaster interrupts service of all jobs-in-service; jobs return to the head of the queue.
        Service time is sampled once and then continued (preemptive-resume).

        Important: in the current simulator implementation, a DISASTER "RESUME_ALL" event
        preempts jobs and immediately starts service again (same instant), preserving the
        remaining service requirements. There is no additional setup / repair / delay time.

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

    def _calculate_p(self):
        """
        Calculate level probabilities.

        Base MGnCalc uses a closed-form expression for p[0] that assumes the
        standard M/H2/n structure without additional horizontal transitions.

        For CLEAR_SYSTEM DISASTER we add an auxiliary microstate D at each level,
        so the base closed-form does not apply; we recover probabilities from the
        computed x-ratios and normalize.
        """
        # For the CLEAR_SYSTEM DISASTER model the state-space differs from the base
        # MGnCalc assumptions, so we compute level probabilities by normalizing x-ratios.
        if not self.requeue_on_disaster:
            self.p[0] = 1.0 + 0.0j
            for j in range(self.N - 1):
                self.p[j + 1] = self.p[j] * self.x[j]
            total = sum(self.p)
            self.p = np.array([val / total for val in self.p], dtype=self.dt)
            return None

        super()._calculate_p()
        return None

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
        if self.resume_on_disaster:
            return self._ensure_resume_results(num_of_moments=num_of_moments)
        if self.requeue_on_disaster:
            return self._ensure_requeue_results(num_of_moments=num_of_moments)

        self.p = self.get_p()
        self.w = self.get_w(num_of_moments)
        self.v = self.get_v(num_of_moments)
        v_served = self.get_v_served()
        v_broken = self.get_v_broken()

        utilization = self.get_utilization()

        q = 1.0
        if not self.requeue_on_disaster:
            # Probability of being served before the first disaster after arrival.
            q = float(self._z0_pls(float(self.l_neg)).real)
        return NegativeArrivalsResults(
            v=self.v,
            w=self.w,
            p=self.p,
            utilization=utilization,
            v_broken=v_broken,
            v_served=v_served,
            q=float(q),
        )

    def get_utilization(self):
        """
        Stationary fraction of busy servers, with artificial holding times removed.
        """
        if self.requeue_on_disaster:
            return float(self._ensure_requeue_results(num_of_moments=1).utilization)
        return sum(min(k, self.n) * pk for k, pk in enumerate(self.get_p())) / self.n

    def get_p(self) -> list[float]:
        if self.requeue_on_disaster:
            return self._ensure_requeue_results(num_of_moments=4).p

        # The trace on ordinary states is the instantaneous-clear process.
        # Delete artificial holding times rather than adding their mass to idle.
        ordinary = [self.Y[0][0, 0]] + [np.sum(self.Y[i][0, 1:]) for i in range(1, self.N)]
        mass = sum(ordinary)
        return [float((prob / mass).real) for prob in ordinary]

    def _ordinary_mass(self):
        return self.Y[0][0, 0] + sum(np.sum(self.Y[i][0, 1:]) for i in range(1, self.N))

    def get_w(self, num_of_moments: int = 4) -> list[float]:
        """
        Get the waiting time moments.

        Here "waiting time" matches the simulator semantics: time spent in queue
        until either service begins OR a disaster clears the system.
        """

        if self.requeue_on_disaster:
            return self._ensure_requeue_results(num_of_moments=num_of_moments).w

        if not self.w is None:
            return self.w

        w = [0.0] * num_of_moments

        for i in range(num_of_moments):
            w[i] = derivative(self._w_pls, 0, dx=1e-3 / self.b[0], n=i + 1, order=9)
            if i % 2 == 0:
                w[i] = -w[i]

        self.w = [w_m.real if isinstance(w_m, complex) else float(w_m) for w_m in w]

        return w

    def get_v(self, num_of_moments: int = 4) -> list[float]:
        """
        Get the sojourn time moments.

        A positive customer leaves either by completing service OR by the first
        disaster after its arrival (which clears the system). If we define Z0 as
        the completion time that would occur *if no disasters happened after
        the customer's arrival* (given the system state at arrival), then:

            V = min(Z0, Y),  Y ~ Exp(l_neg), independent of Z0.

        This gives an LST-based computation that matches the simulator semantics.
        """
        if self.requeue_on_disaster:
            return self._ensure_requeue_results(num_of_moments=num_of_moments).v

        if not self.v is None:
            return self.v

        v = [0.0] * num_of_moments
        for i in range(num_of_moments):
            v[i] = derivative(self._v_pls, 0, dx=1e-3 / self.b[0], n=i + 1, order=9)
            if i % 2 == 0:
                v[i] = -v[i]

        self.v = [v_m.real if isinstance(v_m, complex) else float(v_m) for v_m in v]
        return self.v

    def get_v_served(self) -> list[float]:
        """
        Sojourn time moments conditional on being served (service completion
        occurs before the first disaster after arrival).
        """
        if self.requeue_on_disaster:
            return self._ensure_requeue_results(num_of_moments=4).v

        delta = float(self.l_neg)
        if delta <= 0:
            return self.get_v(4)

        p_served = float(self._z0_pls(delta).real)
        if p_served <= 0:
            return [0.0, 0.0, 0.0, 0.0]

        def v_served_pls(s: float) -> complex:
            return self._z0_pls(s + delta) / p_served

        moments = [0.0] * 4
        for i in range(4):
            moments[i] = derivative(v_served_pls, 0, dx=1e-3 / self.b[0], n=i + 1, order=9)
            if i % 2 == 0:
                moments[i] = -moments[i]

        return [m.real if isinstance(m, complex) else float(m) for m in moments]

    def get_v_broken(self) -> list[float]:
        """
        Sojourn time moments conditional on being broken by a disaster
        (first disaster after arrival occurs before service completion).
        """
        if self.requeue_on_disaster:
            return self._ensure_requeue_results(num_of_moments=4).v_broken

        delta = float(self.l_neg)
        if delta <= 0:
            return [0.0] * 4

        p_served = float(self._z0_pls(delta).real)
        p_broken = 1.0 - p_served
        if p_broken <= 0:
            return [0.0, 0.0, 0.0, 0.0]

        def v_broken_pls(s: float) -> complex:
            # E[e^{-sY}; Y<Z0] = delta/(s+delta) * (1 - Z0*(s+delta))
            num = delta / (s + delta) * (1.0 - self._z0_pls(s + delta))
            return num / p_broken

        moments = [0.0] * 4
        for i in range(4):
            moments[i] = derivative(v_broken_pls, 0, dx=1e-3 / self.b[0], n=i + 1, order=9)
            if i % 2 == 0:
                moments[i] = -moments[i]

        return [m.real if isinstance(m, complex) else float(m) for m in moments]

    def fill_cols(self):
        """
        Add disasters states.
        For n=3 example (D - disaster state)
          00
        D 10 01
        D 20 11 02
        D 30 21 12 03

        Catastrophes enter D on the next lower level; D states descend to 00.
        Censor their holding times to recover instantaneous system clearing.
        """
        if self.requeue_on_disaster:
            super().fill_cols()
            return None
        for i in range(self.N):
            if i < self.n + 1:
                if i == 0:
                    self.cols.append(1)
                else:

                    self.cols.append(i + 2)
            else:
                self.cols.append(self.n + 2)
        return None

    def _build_big_a_matrix(self, num):
        """
        Create matrix A by the given level number.
        """
        if self.requeue_on_disaster:
            return super()._build_big_a_matrix(num)
        if num < self.n:
            col = self.cols[num + 1]
            row = self.cols[num]
        else:
            col = self.cols[self.n]
            row = self.cols[self.n]

        output = np.zeros((row, col), dtype=self.dt)

        if num > self.n:
            output = self.A[self.n]
            return output

        if num == 0:
            output[0, 1] = self.l * self.y[0]
            output[0, 2] = self.l * self.y[1]
            return output

        for i in range(1, row):
            if num < self.n:
                output[i, i] = self.l * self.y[0]
                output[i, i + 1] = self.l * self.y[1]
            else:
                output[i, i] = self.l

        return output

    def _build_big_b_matrix(self, num):
        """
        Create matrix B by the given level number.
        """
        if self.requeue_on_disaster:
            return super()._build_big_b_matrix(num)

        base_matrix = self.base_mgn.B[num]
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

        output[0, 0] = self.gamma
        if num == 1:
            output[1, 0] = self.mu[0] + self.l_neg
            output[2, 0] = self.mu[1] + self.l_neg
            return output

        # fill first col
        if num <= self.n:
            for j in range(1, num + 2):
                output[j, 0] = self.l_neg
        else:
            for j in range(1, self.n + 2):
                output[j, 0] = self.l_neg

        # copy base matrix on position 1,1
        output[1:, 1:] = base_matrix
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

        if num == 0:
            output[0, 0] = self.l
            return output

        output[0, 0] = self.gamma

        for i in range(1, row):
            output[i, i] = self.l + self.l_neg + (num - i + 1) * self.mu[0] + (i - 1) * self.mu[1]

        return output

    def _get_key_numbers(self, level):
        key_numbers = []
        if level >= self.n:
            for i in range(level + 1):
                key_numbers.append((self.n - i, i))
        else:
            for i in range(level + 1):
                key_numbers.append((level - i, i))
        return np.array(key_numbers, dtype=self.dt)

    def _beta_pls(self, s: float) -> complex:
        """
        LST of tagged customer's service time B ~ H2(y, mu).
        """
        return self.y[0] * lst_exp(self.mu[0], s) + self.y[1] * lst_exp(self.mu[1], s)

    def _calc_w0_def_pls(self, s: float) -> complex:
        """
        Defective LST of W0: waiting time to start service in a *no-disaster*
        evolution after arrival, given the stationary (disaster) state at arrival.

        This only covers the part where the customer finds the system with
        at least n jobs (so W0>0); the mass at W0=0 is added in `_w0_pls`.
        """
        w_def = 0.0 + 0.0j

        # Use no-disaster embedded chain (base MGn, without the added disaster-state).
        key_numbers = self._get_key_numbers(self.n)

        # a[0] corresponds to the artificial disaster-state; it should not contribute here.
        a = [1.0 + 0.0j]
        for j in range(self.n + 1):
            rate = key_numbers[j][0] * self.mu[0] + key_numbers[j][1] * self.mu[1]
            a.append(lst_exp(rate, s))
        a = np.array(a, dtype=self.dt)  # (n+2,)

        # Build extended (n+2)x(n+2) transition matrix using base (n+1)x(n+1) probabilities.
        p_base = self.base_mgn.calc_up_probs(self.n + 1)  # (n+1)x(n+1)
        p_ext = np.zeros((self.n + 2, self.n + 2), dtype=self.dt)
        p_ext[0, 0] = 1.0
        p_ext[1:, 1:] = p_base

        step = a[:, None] * p_ext  # holding time belongs to the departure's source state
        waiting = a.copy()

        for k in range(self.n, self.N):
            ys = np.array([self.Y[k][0, i] for i in range(self.n + 2)], dtype=self.dt)
            ys[0] = 0.0
            w_def += ys.dot(waiting)
            waiting = step @ waiting

        return w_def / self._ordinary_mass()

    def _w0_pls(self, s: float) -> complex:
        """
        Full LST of W0 (waiting time to *start service* if disasters are disabled
        after the customer's arrival), under the stationary distribution at arrival.
        """
        s_key = float(s)
        cached = self._w0_pls_cache.get(s_key)
        if cached is not None:
            return cached

        # Arrival sees the physical distribution with artificial holding times removed.
        p_immediate = sum(float(self.p[k]) for k in range(min(self.n, len(self.p))))
        w0 = p_immediate + self._calc_w0_def_pls(s)
        self._w0_pls_cache[s_key] = w0
        return w0

    def _z0_pls(self, s: float) -> complex:
        """
        LST of Z0 = W0 + B (completion time with disasters disabled after arrival).
        Independence holds because the tagged service time is independent from its waiting time.
        """
        s_key = float(s)
        cached = self._z0_pls_cache.get(s_key)
        if cached is not None:
            return cached
        z0 = self._w0_pls(s) * self._beta_pls(s)
        self._z0_pls_cache[s_key] = z0
        return z0

    def _w_pls(self, s: float) -> complex:
        """
        LST of waiting time in the *disaster* system, matching simulator:
            W = min(W0, Y),  Y ~ Exp(l_neg)
        """
        delta = float(self.l_neg)
        if delta <= 0:
            # Fallback to the no-disaster waiting time (start-service).
            return self._w0_pls(s)

        sp = s + delta
        return delta / (s + delta) + (s / (s + delta)) * self._w0_pls(sp)

    def _v_pls(self, s: float) -> complex:
        """
        LST of sojourn time in the disaster system:
            V = min(Z0, Y),  Y ~ Exp(l_neg)
        """
        delta = float(self.l_neg)
        if delta <= 0:
            return self._z0_pls(s)

        sp = s + delta
        return delta / (s + delta) + (s / (s + delta)) * self._z0_pls(sp)

    def _calc_service_probs(self) -> list[float]:
        """
        Returns the conditional probabilities of loaded states.
        """
        ps = np.array([self.p[i] for i in range(1, self.n + 1)])
        ps /= np.sum(ps)

        return [prob.real for prob in ps]
