"""
Calculate M/H2/n queue with negative jobs with RCS discipline,
(remove customer from service)
"""

import numpy as np
from scipy.misc import derivative

from most_queue.general.conditional import (
    moments_exp_less_than_h2,
    moments_h2_less_than_exp,
)
from most_queue.general.conv import conv_moments
from most_queue.rand_distribution import H2Distribution, H2Params
from most_queue.theory.queueing_systems.fifo.mgn_takahasi import MGnCalc
from most_queue.theory.queueing_systems.negative.structs import NegativeArrivalsResults
from most_queue.theory.utils.transforms import (
    laplace_stieltjes_exp_transform as lst_exp,
)


class MGnNegativeRCSCalc(MGnCalc):
    """
    Calculate M/H2/n queue with negative jobs with RCS discipline,
    (remove customer from service)
    """

    def __init__(self, n: int, l_pos: float, l_neg: float, b: list[float],
                 buffer: int | None = None, N: int = 150,
                 accuracy: float = 1e-6, dtype="c16", verbose: bool = False):
        """
        n: number of servers
        l: arrival rate of positive jobs
        l_neg: arrival rate of negative jobs
        b: initial moments of service time distribution
        buffer: size of the buffer (optional)
        N: number of levels in the system (default is 150)
        accuracy: accuracy parameter for stopping the iteration
        dtype: data type for calculations (default is complex double precision)
        verbose: whether to print intermediate results (default is False)
        """

        super().__init__(n=n, l=l_pos, b=b, buffer=buffer, N=N,
                         accuracy=accuracy, dtype=dtype, verbose=verbose)

        self.l_neg = l_neg

    def get_q(self) -> float:
        """
        Calculation of the conditional probability of successful service completion at a node
        """
        return 1.0 - (self.l_neg/self.l)*(1.0-self.p[0].real)

    def _calc_service_probs(self) -> list[float]:
        """
        Returns the conditional probabilities of loaded states.
        """
        ps = np.array([self.p[i] for i in range(1, self.n+1)])
        ps /= np.sum(ps)

        return ps

    def get_v(self) -> list[float]:
        """
        Get the sojourn time moments
        """
        w = self.get_w()

        # serving = min(H2_b, exp(l_neg)) = H2(y1=y1, mu1 = mu1+l_neg, mu2=mu2+l_neg)

        params = H2Params(p1=self.y[0],
                          mu1=self.mu[0],
                          mu2=self.mu[1])

        service_probs = self._calc_service_probs()
        b_cum = np.array([0.0, 0.0, 0.0])
        for i in range(1, self.n+1):
            l_neg = self.l_neg/i

            b = H2Distribution.calc_theory_moments(H2Params(p1=params.p1,
                                                            mu1=l_neg + params.mu1,
                                                            mu2=l_neg + params.mu2))
            b_cum += service_probs[i-1].real*np.array([mom.real for mom in b])

        return [mom.real for mom in conv_moments(w, b_cum)]

    def _get_key_numbers(self, level):
        key_numbers = []
        if level >= self.n:
            for i in range(level + 1):
                key_numbers.append((self.n - i, i))
        else:
            for i in range(level + 1):
                key_numbers.append((level - i, i))
        return np.array(key_numbers, dtype=self.dt)

    def _calc_b_served_pls(self, s):

        b = 0

        for k in range(1, self.n):
            # for k=1, [(2,0), (1,1), (0, 2)]
            keys = self._get_key_numbers(k)

            keys_mu = np.array(
                [keys[j][0] * self.mu[0] + keys[j][1] * self.mu[1] for j in
                 range(k + 1)])

            a = np.array(
                [lst_exp(keys_mu[j], s) for j in
                 range(k + 1)])
            probs = np.array([(keys_mu[j]) / (keys_mu[j] + self.l_neg) for j in
                              range(k + 1)])

            Ys = np.array([self.Y[k][0, i] for i in range(k + 1)])

            b += np.sum(a * probs * Ys)

        keys = self._get_key_numbers(self.n)
        keys_mu = np.array(
            [keys[j][0] * self.mu[0] + keys[j][1] * self.mu[1] for j in
             range(self.n + 1)])

        a = np.array(
            [lst_exp(keys_mu[j], s) for j in
                range(self.n + 1)])
        probs = np.array([(keys_mu[j]) / (keys_mu[j] + self.l_neg) for j in
                          range(self.n + 1)])

        for k in range(self.n, self.N):
            Ys = np.array([self.Y[k][0, i] for i in range(self.n+1)])

            b += np.sum(a * probs * Ys)

        return b

    def _calc_b_served(self) -> list[float]:
        """
        Calculate the expected number of served jobs.
        """

        b = [0.0] * 3

        for i in range(3):
            b[i] = derivative(self._calc_b_served_pls, 0,
                              dx=1e-3 / self.b[0], n=i + 1, order=9)
        return [-b[0], b[1].real, -b[2]]

    def get_v_served(self, derivate=False) -> list[float]:
        """
        Get the sojourn time moments
        """
        w = self.get_w()

        if derivate:
            b_served = self._calc_b_served()
        else:
            # serving = P(H2 | H2 < exp(l_neg))

            service_probs = self._calc_service_probs()
            b_served = np.array([0.0, 0.0, 0.0])
            h2_params = H2Params(p1=self.y[0], mu1=self.mu[0], mu2=self.mu[1])
            for i in range(1, self.n+1):
                l_neg = self.l_neg/i

                b = moments_h2_less_than_exp(l_neg, h2_params)
                b_served += service_probs[i-1].real*b

        return [mom.real for mom in conv_moments(w, b_served)]

    def get_v_broken(self) -> list[float]:
        """
        Get the sojourn time moments
        """
        w = self.get_w()

        # serving = P(exp(l_neg) | exp(l_neg) < H2)

        service_probs = self._calc_service_probs()
        b_cum = np.array([0.0, 0.0, 0.0])
        h2_params = H2Params(p1=self.y[0], mu1=self.mu[0], mu2=self.mu[1])
        for i in range(1, self.n+1):
            l_neg = self.l_neg/i
            b = moments_exp_less_than_h2(l_neg, h2_params)
            b_cum += service_probs[i-1].real*b

        return [mom.real for mom in conv_moments(w, b_cum)]

    def get_results(self, max_p: int = 100) -> NegativeArrivalsResults:
        """
        Get the results of the calculation.
        max_p: Maximum number of probabilities to calculate
        :return: Results object containing calculated values.
        """
        p = self.get_p()[:max_p]
        v = self.get_v()
        v_served = self.get_v_served()
        v_broken = self.get_v_broken()
        w = self.get_w()
        return NegativeArrivalsResults(p=p, v=v, v_served=v_served, v_broken=v_broken, w=w)

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
                first_neg, second_neg = self.l_neg * \
                    (num-i)/num, self.l_neg*(i+1)/num
                output[i, i] = (num - i) * self.mu[0] + first_neg
                output[i + 1, i] = (i + 1) * self.mu[1] + second_neg
            else:
                # key has two parts: left and right, like 21
                # fill B matrix by pattern of 3 elements:
                # up arrow, right arrow, arrow from next to current key
                #  x x 0
                #  x y y
                #  0 y 0
                left = self.l_neg * (num-i-1)/(num-1)
                right = self.l_neg * i/(num-1)
                left_from_next = self.l_neg*(i+1)/(num-1)

                output[i, i] = ((num - i - 1) * self.mu[0] + left) * \
                    self.y[0] + (i * self.mu[1] + right) * self.y[1]

                if i != num - 1:
                    output[i, i + 1] = ((num - i - 1) * self.mu[0] +
                                        left) * self.y[1]

                    output[i + 1, i] = ((i + 1) * self.mu[1] +
                                        left_from_next) * self.y[0]
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
            output[i, i] = self.l + self.l_neg + \
                (num - i) * self.mu[0] + i * self.mu[1]

        return output
