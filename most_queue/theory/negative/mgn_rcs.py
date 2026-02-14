"""
Calculate M/H2/n queue with negative jobs with RCS discipline,
(remove customer from service)
"""

import numpy as np
from scipy.misc import derivative

from most_queue.random.distributions import H2Distribution
from most_queue.structs import NegativeArrivalsResults
from most_queue.theory.fifo.mgn_takahasi import MGnCalc, TakahashiTakamiParams
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
    ):
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
        return self.collect_results(num_of_moments)

    def collect_results(self, num_of_moments: int = 4) -> NegativeArrivalsResults:
        """
        Get all results
        """

        self.p = self.get_p()
        self.w = self.get_w(num_of_moments)
        self.v = self.get_v(num_of_moments)
        v_served = self.get_v_served(num_of_moments)
        v_broken = self.get_v_broken(num_of_moments)

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
        return self.l_pos * self.b[0] / self.n

    def get_q(self) -> float:
        """
        Calculation of the conditional probability of successful service completion at a node
        """
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
