"""
Calculate M/H2/n queue with negative jobs with disasters,
"""

import math

import numpy as np
from scipy.misc import derivative

from most_queue.random.distributions import H2Distribution, H2Params
from most_queue.structs import NegativeArrivalsResults
from most_queue.theory.fifo.mgn_takahasi import MGnCalc, TakahashiTakamiParams
from most_queue.theory.utils.transforms import lst_exp


class MGnNegativeDisasterCalc(MGnCalc):
    """
    Calculate M/H2/n queue with negative jobs (DISASTER-type).

    Two scenarios are supported (selected by `requeue_on_disaster`):

    - `False` (default): negative arrival clears the system (removes all positive jobs),
      matching the original "DISASTER" semantics.
    - `True`: negative arrival interrupts service and returns the positive jobs in service
      back to the head of the queue (service restarts). In this mode the calculation
      is approximated via an equivalent M/G/n with an effective service time that
      accounts for restart.
    """

    def __init__(
        self,
        n: int,
        buffer: int | None = None,
        calc_params: TakahashiTakamiParams | None = None,
        requeue_on_disaster: bool = False,
        matrix_requeue: bool = False,
    ):
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
        # If True, use exact TT with modified matrices (C, D) for REQUEUE_ALL.
        # If False (default), use the effective-service approximation (closer to Gamma sim).
        self.matrix_requeue = bool(matrix_requeue)
        self._requeue_results: NegativeArrivalsResults | None = None

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
        self.gamma = 1e3 * b[0]  # disaster artifitial states intensity
        self.is_servers_set = True
        self._w0_pls_cache = {}
        self._z0_pls_cache = {}
        self._requeue_results = None

    def run(self) -> NegativeArrivalsResults:  # pylint: disable=arguments-differ
        """
        Run calculation.

        In REQUEUE mode there are two options:

        - default (`matrix_requeue=False`): approximate via an equivalent M/G/n with
          effective service moments B_eff (stable and matches Gamma-based simulator well);
        - optional (`matrix_requeue=True`): keep the standard TT loop but modify the
          within-level transitions (C, D) to model mass service restarts in the H2 world.
        """
        if self.requeue_on_disaster and not self.matrix_requeue:
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

        Here β(s) is taken from a Gamma approximation based on the first two moments
        (consistent with simulation defaults).
        """
        if self._requeue_results is not None and len(self._requeue_results.v) >= num_of_moments:
            return self._requeue_results

        self._check_if_servers_and_sources_set()

        lam = float(self.l_pos)
        delta = float(self.l_neg)

        mean_b = float(self.b[0])
        var_b = float(self.b[1] - self.b[0] ** 2)

        # Gamma approximation for β(s) by mean/variance
        if var_b > 0 and mean_b > 0:
            k = mean_b * mean_b / var_b
            theta = var_b / mean_b

            def beta(s: float) -> float:
                return float((1.0 + theta * s) ** (-k))

        else:

            def beta(s: float) -> float:
                # Degenerate fallback: no variability information
                return float(np.exp(-mean_b * s))

        if delta <= 0:
            b_eff = [float(x) for x in self.b[:num_of_moments]]
        else:

            def b_eff_pls(s: float) -> float:
                sp = s + delta
                beta_sp = beta(sp)
                return float(beta_sp * (s + delta) / (s + delta * beta_sp))

            b_eff = [0.0] * num_of_moments
            dx = 1e-3 / max(mean_b, 1e-9)
            for i in range(num_of_moments):
                val = derivative(b_eff_pls, 0, dx=dx, n=i + 1, order=9)
                if i % 2 == 0:
                    val = -val
                b_eff[i] = float(val)

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
            duration=0.0,
        )
        return self._requeue_results

    def _calculate_p(self):
        """
        Calculate level probabilities.

        Base MGnCalc uses a closed-form expression for p[0] that assumes the
        standard M/H2/n structure without additional horizontal transitions.
        In REQUEUE mode we add disaster-driven horizontal transitions, so we
        recover probabilities from the computed x-ratios and normalize.
        """
        # For the DISASTER model (both CLEAR_SYSTEM and matrix REQUEUE_ALL),
        # the state-space differs from the base MGnCalc assumptions, so we
        # compute level probabilities by normalizing the x-ratios.
        if (not self.requeue_on_disaster) or (self.requeue_on_disaster and self.matrix_requeue):
            self.p[0] = 1.0 + 0.0j
            for j in range(self.N - 1):
                self.p[j + 1] = self.p[j] * self.x[j]
            total = sum(self.p)
            self.p = np.array([val / total for val in self.p], dtype=self.dt)
            return

        return super()._calculate_p()

    def get_results(self, num_of_moments: int = 4, derivate=False) -> NegativeArrivalsResults:
        """
        Get all results - override to return NegativeArrivalsResults instead of QueueResults.
        """
        return self.collect_results(num_of_moments)

    def collect_results(self, num_of_moments: int = 4) -> NegativeArrivalsResults:
        """
        Get all results
        """
        if self.requeue_on_disaster and self.matrix_requeue:
            # No removals: system is stable QBD with within-level restart transitions.
            # Use Little's law for first moments from stationary level probabilities.
            p = self.get_p()
            lam = float(self.l_pos)
            if lam <= 0:
                w1 = 0.0
                v1 = 0.0
            else:
                en = sum(k * pk for k, pk in enumerate(p))
                eq = sum(max(0, k - self.n) * pk for k, pk in enumerate(p))
                v1 = en / lam
                w1 = eq / lam

            ebusy = sum(min(self.n, k) * pk for k, pk in enumerate(p))
            utilization = float(ebusy) / float(self.n) if self.n > 0 else 0.0

            v = [float(v1)] + [0.0] * max(0, num_of_moments - 1)
            w = [float(w1)] + [0.0] * max(0, num_of_moments - 1)
            return NegativeArrivalsResults(
                v=v,
                w=w,
                p=p,
                utilization=utilization,
                v_broken=[0.0] * num_of_moments,
                v_served=v,
                duration=0.0,
            )

        self.p = self.get_p()
        self.w = self.get_w(num_of_moments)
        self.v = self.get_v(num_of_moments)
        v_served = self.get_v_served()
        v_broken = self.get_v_broken()

        utilization = self.get_utilization()

        return NegativeArrivalsResults(
            v=self.v, w=self.w, p=self.p, utilization=utilization, v_broken=v_broken, v_served=v_served
        )

    def get_utilization(self):
        """
        Calculate utilization of the queue.

        Note: This is a simplified version that does not fully account for
        the effect of disasters on utilization. A more accurate calculation
        would consider the impact of disaster events on system utilization.
        """
        if self.requeue_on_disaster and self.matrix_requeue:
            p = self.get_p()
            ebusy = sum(min(self.n, k) * pk for k, pk in enumerate(p))
            return float(ebusy) / float(self.n) if self.n > 0 else 0.0
        return self.l_pos * self.b[0] / self.n

    def get_p(self) -> list[float]:
        if self.requeue_on_disaster:
            if self.matrix_requeue:
                return [float(val.real) for val in list(self.p)]
            return self._ensure_requeue_results(num_of_moments=4).p

        first_col_sum = 0
        for i in range(1, self.N):
            first_col_sum += self.Y[i][0, 0]
            self.p[i] = np.sum(self.Y[i][0, 1:])
        self.p[0] += first_col_sum

        return [prob.real for prob in self.p]

    def get_w(self, num_of_moments: int = 4) -> list[float]:
        """
        Get the waiting time moments.

        Here "waiting time" matches the simulator semantics: time spent in queue
        until either service begins OR a disaster clears the system.
        """

        if self.requeue_on_disaster:
            if not self.matrix_requeue:
                return self._ensure_requeue_results(num_of_moments=num_of_moments).w
            p = self.get_p()
            lam = float(self.l_pos)
            if lam <= 0:
                w1 = 0.0
            else:
                eq = sum(max(0, k - self.n) * pk for k, pk in enumerate(p))
                w1 = eq / lam
            return [float(w1)] + [0.0] * max(0, num_of_moments - 1)

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
            if not self.matrix_requeue:
                return self._ensure_requeue_results(num_of_moments=num_of_moments).v
            p = self.get_p()
            lam = float(self.l_pos)
            if lam <= 0:
                v1 = 0.0
            else:
                en = sum(k * pk for k, pk in enumerate(p))
                v1 = en / lam
            return [float(v1)] + [0.0] * max(0, num_of_moments - 1)

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
            if not self.matrix_requeue:
                return self._ensure_requeue_results(num_of_moments=4).v
            return self.get_v(num_of_moments=4)

        delta = float(self.l_neg)
        if delta <= 0:
            raise ValueError("get_v_served() requires l_neg > 0 for disaster model.")

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
            if not self.matrix_requeue:
                return self._ensure_requeue_results(num_of_moments=4).v_broken
            return [0.0, 0.0, 0.0, 0.0]

        delta = float(self.l_neg)
        if delta <= 0:
            raise ValueError("get_v_broken() requires l_neg > 0 for disaster model.")

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

        Notice, we can't go from we can't go straight to state 00,
        we need these artificial states on each layer with high intensity
        of transition up to state 00
        """
        if self.requeue_on_disaster:
            return super().fill_cols()
        for i in range(self.N):
            if i < self.n + 1:
                if i == 0:
                    self.cols.append(1)
                else:

                    self.cols.append(i + 2)
            else:
                self.cols.append(self.n + 2)

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

    def _build_big_c_matrix(self, num):
        """
        Create matrix C (horizontal transitions) by the given level number.

        In REQUEUE mode, a disaster does not change the level (number in system),
        but it resets all ongoing services and immediately starts service anew.
        This is modeled as horizontal transitions between microstates at rate l_neg.
        """
        if not (self.requeue_on_disaster and self.matrix_requeue):
            return super()._build_big_c_matrix(num)

        output = super()._build_big_c_matrix(num)
        delta = float(self.l_neg)
        if num <= 0 or delta <= 0:
            return output

        m = min(num, self.n)
        # microstate index = number of phase-2 servers among m busy servers
        # after disaster, the phase-2 count is Bin(m, y2)
        # Note: y can be complex (for H2 complex-fit); keep dt-consistent algebra.
        y1 = self.y[0]
        y2 = self.y[1]
        probs = np.array([math.comb(m, j) * (y2**j) * (y1 ** (m - j)) for j in range(m + 1)], dtype=self.dt)
        # add transitions from any current microstate to the new microstate distribution,
        # excluding diagonal (stay-in-state) part; the corresponding leaving rate is
        # added to D.
        for j_from in range(m + 1):
            for j_to in range(m + 1):
                if j_to == j_from:
                    continue
                output[j_from, j_to] += delta * probs[j_to]
        return output

    def _build_big_d_matrix(self, num):
        """
        Create matrix D by the given level number.
        """
        if self.requeue_on_disaster and self.matrix_requeue:
            output = super()._build_big_d_matrix(num)
            delta = float(self.l_neg)
            if num > 0 and delta > 0:
                m = min(num, self.n)
                y1 = self.y[0]
                y2 = self.y[1]
                probs = np.array(
                    [math.comb(m, j) * (y2**j) * (y1 ** (m - j)) for j in range(m + 1)],
                    dtype=self.dt,
                )
                for i in range(output.shape[0]):
                    # leaving rate due to requeue transitions that change microstate
                    output[i, i] += delta * (1.0 - probs[i])
            return output

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

    def calc_up_probs(self, from_level):
        if from_level == self.n:
            return np.eye(self.n + 2)
        b_matrix = self.B[from_level]
        probs = []
        for i in range(self.cols[from_level - 1]):
            probs_on_level = []
            for j in range(self.cols[from_level - 1]):
                if from_level != 1:
                    probs_on_level.append(b_matrix[i, j] / sum(b_matrix[i, :]))
                else:
                    probs_on_level.append(b_matrix[i, 0] / sum(b_matrix[i, :]))
            probs.append(probs_on_level)
        return np.array(probs, dtype=self.dt)

    def _matrix_pow(self, matrix, k):
        res = np.eye(self.n + 2, dtype=self.dt)
        for _i in range(k):
            res = np.dot(res, matrix)
        return res

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

        t = (p_ext * a).T  # iterate powers of (p_ext * a) transposed
        pa = np.eye(self.n + 2, dtype=self.dt)

        for k in range(self.n, self.N):
            ys = np.array([self.Y[k][0, i] for i in range(self.n + 2)], dtype=self.dt)
            ys[0] = 0.0  # artificial disaster-state is counted into p[0] already
            a_pa = a.dot(pa)
            w_def += ys.dot(a_pa)
            pa = pa.dot(t)

        return w_def

    def _w0_pls(self, s: float) -> complex:
        """
        Full LST of W0 (waiting time to *start service* if disasters are disabled
        after the customer's arrival), under the stationary distribution at arrival.
        """
        s_key = float(s)
        cached = self._w0_pls_cache.get(s_key)
        if cached is not None:
            return cached

        # Immediate service if actual number in system < n (includes artificial disaster-mass mapped into p[0]).
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

    def _calc_w_pls(self, s) -> float:
        """
        Calculate Laplace-Stietjes transform of the waiting time distribution.
        :param s: Laplace variable
        :return: Laplace-Stieltjes transform of the waiting time distribution
        """

        w = 0

        key_numbers = self._get_key_numbers(self.n)
        a = [lst_exp(self.gamma, s)]
        for j in range(self.n + 1):
            a.append(
                lst_exp(
                    key_numbers[j][0] * self.mu[0] + key_numbers[j][1] * self.mu[1] + self.l_neg,
                    s,
                )
            )

        a = np.array(a)

        Pn_plus = self.calc_up_probs(self.n + 1)

        for k in range(self.n, self.N):

            Pa = np.transpose(self._matrix_pow(Pn_plus * a, k - self.n))  # size = (n+2, n+2)

            Ys = np.array([self.Y[k][0, i] for i in range(self.n + 2)])
            aPa = np.dot(a, Pa)
            w += Ys.dot(aPa)

        return w

    def _calc_service_probs(self) -> list[float]:
        """
        Returns the conditional probabilities of loaded states.
        """
        ps = np.array([self.p[i] for i in range(1, self.n + 1)])
        ps /= np.sum(ps)

        return [prob.real for prob in ps]
