"""
Calculate M/H2/n queue with negative jobs with RCS discipline,
(remove customer from service)
"""

import math
import time

import numpy as np

from most_queue.random.distributions import H2Distribution, H2Params
from most_queue.structs import NegativeArrivalsResults
from most_queue.theory.fifo.mgn_takahasi import MGnCalc, TakahashiTakamiParams
from most_queue.theory.utils.conditional import moments_exp_less_than_h2, moments_h2_less_than_exp
from most_queue.theory.utils.conv import conv_moments


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

    def set_sources(self, l_pos: float, l_neg: float):  # pylint: disable=arguments-differ
        """
        Set the arrival rates of positive and negative jobs
        :param l_pos: arrival rate of positive jobs
        :param l_neg: arrival rate of negative jobs
        """
        self.l_pos = l_pos
        self.l = l_pos
        self.l_neg = l_neg
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

    def run(self) -> NegativeArrivalsResults:
        """
        Run the algorithm.
        """

        start = time.process_time()

        self.fill_cols()
        self._fill_t_b()
        self.build_matrices()
        self._initial_probabilities()

        self.b1[0][0, 0] = 0.0 + 0.0j
        self.b2[0][0, 0] = 0.0 + 0.0j
        x_max1 = np.max(self.x)
        x_max2 = 0.0 + 0.0j

        self._calc_support_matrices()

        self.num_of_iter_ = 0  # number of iterations

        while math.fabs(x_max2.real - x_max1.real) >= self.e1:
            x_max2 = x_max1
            self.num_of_iter_ += 1

            for j in range(1, self.N):  # for all levels except the first.

                # b':
                self.b1[j] = np.dot(self.t[j - 1], self.big_ag[j])

                # b":
                if j != (self.N - 1):
                    self.b2[j] = np.dot(self.t[j + 1], self.big_bg[j])
                else:
                    self.b2[j] = np.dot(self.t[j - 1], self.big_bg[j])

                c = self._calculate_c(j)

                x_znam = np.dot(c, self.b1[j]) + self.b2[j]
                self.x[j] = 0.0 + 0.0j
                for k in range(x_znam.shape[1]):
                    self.x[j] += x_znam[0, k]

                self.x[j] = (1.0 + 0.0j) / self.x[j]

                if self.R and j == (self.N - 1):
                    tA = np.dot(self.t[j - 1], self.A[j - 1])
                    tag = np.dot(tA, self.big_g[j])
                    tag_sum = 0
                    for t_i in range(tag.shape[1]):
                        tag_sum += tag[0, t_i]
                    self.z[j] = 1.0 / tag_sum
                    self.t[j] = self.z[j] * tag

                else:
                    self.z[j] = np.dot(c, self.x[j])
                    self.t[j] = np.dot(self.z[j], self.b1[j]) + np.dot(self.x[j], self.b2[j])

            self.x[0] = (1.0 + 0.0j) / self.z[1]

            # Why?, because t[0] = [1.0] always
            # self.t[0] = self.x[0] * \
            #     (np.dot(self.t[1], self.B[1]).dot(self.big_g[0]))

            x_max1 = np.max(self.x)

            if self.verbose:
                print(f"End iter # {self.num_of_iter_}")

        self._calculate_p()
        self._calculate_y()

        results = self.collect_results()

        results.duration = time.process_time() - start

        return results

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
        Calc utilization of the queue.
        """
        # TODO Simplified version, not considering disasters effect
        return self.l_pos * self.b[0] / self.n

    def get_q(self) -> float:
        """
        Calculation of the conditional probability of successful service completion at a node
        """
        return 1.0 - (self.l_neg / self.l) * (1.0 - self.p[0].real)

    def get_v(self, num_of_moments: int = 4) -> list[float]:
        """
        Get the sojourn time moments
        """
        w = self.get_w(num_of_moments)

        # serving = min(H2_b, exp(l_neg)) = H2(y1=y1, mu1 = mu1+l_neg,
        # mu2=mu2+l_neg)

        params = H2Params(p1=self.y[0], mu1=self.mu[0], mu2=self.mu[1])

        service_probs = self._calc_service_probs()
        b_cum = np.array([0.0] * num_of_moments)
        for i in range(1, self.n + 1):
            l_neg = self.l_neg / i

            b = H2Distribution.calc_theory_moments(
                H2Params(p1=params.p1, mu1=l_neg + params.mu1, mu2=l_neg + params.mu2), num=num_of_moments
            )
            b_cum += service_probs[i - 1].real * np.array([mom.real for mom in b])

        return [mom.real for mom in conv_moments(w, b_cum, num=num_of_moments)]

    def get_v_served(self, num_of_moments: int = 4) -> list[float]:
        """
        Get the sojourn time moments
        """
        w = self.get_w(num_of_moments)

        # serving = P(H2 | H2 < exp(l_neg))

        service_probs = self._calc_service_probs()
        b_served = np.array([0.0] * num_of_moments)
        h2_params = H2Params(p1=self.y[0], mu1=self.mu[0], mu2=self.mu[1])
        for i in range(1, self.n + 1):
            l_neg = self.l_neg / i

            b = moments_h2_less_than_exp(l_neg, h2_params)[:num_of_moments]
            b_served += service_probs[i - 1].real * b

        return [mom.real for mom in conv_moments(w, b_served, num=num_of_moments)]

    def get_v_broken(self, num_of_moments: int = 4) -> list[float]:
        """
        Get the sojourn time moments
        """
        w = self.get_w(num_of_moments)

        # serving = P(exp(l_neg) | exp(l_neg) < H2)

        service_probs = self._calc_service_probs()
        b_cum = np.array([0.0] * num_of_moments)
        h2_params = H2Params(p1=self.y[0], mu1=self.mu[0], mu2=self.mu[1])
        for i in range(1, self.n + 1):
            l_neg = self.l_neg / i
            b = moments_exp_less_than_h2(l_neg, h2_params)[:num_of_moments]
            b_cum += service_probs[i - 1].real * b

        return [mom.real for mom in conv_moments(w, b_cum, num_of_moments)]

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
