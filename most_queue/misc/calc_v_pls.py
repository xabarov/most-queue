"""
Calculate M/H2/n queue with negative jobs with RCS discipline,
(remove customer from service)
"""
import math
from functools import lru_cache

import numpy as np
from scipy.misc import derivative

from most_queue.rand_distribution import H2Distribution, H2Params
from most_queue.theory.fifo.mgn_takahasi import MGnCalc
from most_queue.theory.utils.conv import conv_moments
from most_queue.theory.utils.transforms import lst_exp


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

        self.l_neg = l_neg

        super().__init__(n=n, l=l_pos, b=b, buffer=buffer, N=N,
                         accuracy=accuracy, dtype=dtype, verbose=verbose)

    def get_q(self) -> float:
        """
        Calculation of the conditional probability of successful service completion at a node
        """
        return 1.0 - (self.l_neg/self.l)*(1.0-self.p[0].real)

    def get_w(self, derivate=False) -> list[float]:
        """
        Get the waiting time moments
        """

        w = [0.0] * 3

        if derivate:
            for i in range(3):
                w[i] = derivative(self._calc_w_pls, 0,
                                  dx=1e-3 / self.b[0], n=i + 1, order=9)
            return np.array([-w[0], w[1].real, -w[2]])

        for j in range(1, len(self.p) - self.n):
            w[0] += j * self.p[self.n + j]
        for j in range(2, len(self.p) - self.n):
            w[1] += j * (j - 1) * self.p[self.n + j]
        for j in range(3, len(self.p) - self.n):
            w[2] += j * (j - 1) * (j - 2) * self.p[self.n + j]

        for j in range(3):
            w[j] /= math.pow(self.l, j + 1)
            w[j] = w[j].real

        return np.array(w)

    def _get_key_numbers(self, level):
        key_numbers = []
        if level >= self.n:
            for i in range(level + 1):
                key_numbers.append((self.n - i, i))
        else:
            for i in range(level + 1):
                key_numbers.append((level - i, i))
        return np.array(key_numbers, dtype=self.dt)

    def _calc_up_probs(self, from_level):
        if from_level == self.n:
            return np.eye(self.n + 1)
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
        res = np.eye(self.n + 1, dtype=self.dt)
        for i in range(k):
            res = np.dot(res, matrix)
        return res

    @lru_cache
    def _build_probs_down(self, num):
        """
        Create matrix with transition down probs by the given level number.
        """
        if num < self.n:
            col = self.cols[num + 1]
            row = self.cols[num]
        else:
            col = self.cols[self.n]
            row = self.cols[self.n]

        output = np.zeros((row, col), dtype=self.dt)

        if num > self.n:
            output = self._build_probs_down(self.n)
            return output

        for i in range(row):
            if num < self.n:
                output[i, i] = self.y[0]
                output[i, i + 1] = self.y[1]
            else:
                output[i, i] = 1.0

        return output

    def _calc_v_pls(self, s) -> float:

        v = 0

        for k in range(self.n):
            # When arrived, job see current state, move to next and need to serve on it
            # so, we work with next levels (for example, k + 1 instead k)

            # for k=1, [(2,0), (1,1), (0, 2)]
            key_numbers = self._get_key_numbers(k+1)

            # LST for serving in next state, for k=1, size=(1, 3)
            a = np.array(
                [lst_exp(key_numbers[j][0] * self.mu[0] + key_numbers[j][1] * self.mu[1] + self.l_neg, s) for j in
                 range(k + 2)])

            # P - down transition matrix, for k=1 size = (2, 3)
            # for 1 state y1 y2 0
            #             0  y1 y2
            P = self._build_probs_down(k)
            # current state, for k=1 size = (1, 2)
            Ys = np.array([self.Y[k][0, i] for i in range(k + 1)])

            v_tek = np.dot(Ys, P.dot(a.T))

            v += v_tek

        key_numbers = self._get_key_numbers(self.n)

        a_n = np.array(
            [lst_exp(key_numbers[j][0] * self.mu[0] + key_numbers[j][1] * self.mu[1] + self.l_neg, s) for j in
             range(self.n + 1)])

        Pn_plus = self._calc_up_probs(self.n + 1)  # size=(4, 4)

        Pa = Pn_plus*a_n  # 4,4

        PdotaT = Pn_plus.dot(a_n.T)  # 4,1

        prod = np.eye(self.n+1)

        for k in range(self.n, self.N):

            Ys = np.array([self.Y[k][0, i]
                          for i in range(self.n + 1)])  # (1, 4)

            ya = Ys*a_n  # 1,4

            v += np.dot(ya, prod.dot(PdotaT))

            prod = prod.dot(Pa)

        return v

    def _calc_w_pls(self, s) -> float:
        """
        Calculate Laplace-Stietjes transform of the waiting time distribution.
        :param s: Laplace variable
        :return: Laplace-Stieltjes transform of the waiting time distribution
        """
        w = 0

        key_numbers = self._get_key_numbers(self.n)

        a = np.array(
            [lst_exp(key_numbers[j][0] * self.mu[0] + key_numbers[j][1] * self.mu[1] + self.l_neg, s) for j in
             range(self.n + 1)])

        Pn_plus = self._calc_up_probs(self.n+1)

        for k in range(self.n, self.N):

            Pa = np.transpose(self._matrix_pow(
                Pn_plus * a, k - self.n))  # n+1 x n+1

            Ys = [self.Y[k][0, i] for i in range(self.n + 1)]
            aPa = np.dot(a, Pa)
            for i in range(self.n + 1):
                w += Ys[i] * aPa[i]

        return w

    def get_v(self) -> list[float]:
        """
        Get the sojourn time moments
        """
        w = self.get_w(derivate=False)

        # serving = min(H2_b, exp(l_neg)) = H2(y1=y1, mu1 = mu1+l_neg, mu2=mu2+l_neg)

        params = H2Params(p1=self.y[0],
                          mu1=self.mu[0],
                          mu2=self.mu[1])
        #
        l_neg = 0
        for i in range(1, self.n):
            l_neg += self.p[i]*self.l_neg/i

        b = H2Distribution.calc_theory_moments(H2Params(p1=params.p1,
                                                        mu1=l_neg + params.mu1,
                                                        mu2=l_neg + params.mu2))

        return conv_moments(w, b)

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
