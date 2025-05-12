"""
Calculate M/H2/n queue with negative jobs with disasters,
"""
import numpy as np
from scipy.misc import derivative

from most_queue.general.conditional import (moments_exp_less_than_H2,
                                            moments_H2_less_than_exp)
from most_queue.general.conv import conv_moments
from most_queue.rand_distribution import H2Distribution, H2Params
from most_queue.theory.queueing_systems.fifo.mgn_takahasi import MGnCalc
from most_queue.theory.utils.transforms import \
    laplace_stieltjes_exp_transform as lst_exp


class MGnNegativeDisasterCalc(MGnCalc):
    """
    Calculate M/H2/n queue with negative jobs with disasters,
    (remove all customer from system)
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
        self.gamma = 1e3*b[0]  # disaster artifitial states intensity
        
        # for calc B matrices
        self.base_mgn = MGnCalc(n=n, l=l_pos, b=b, buffer=buffer, N=N,
                         accuracy=accuracy, dtype=dtype, verbose=verbose)
        self.base_mgn._fill_cols()
        self.base_mgn._build_matrices()

        self.w = None
        
    def _fill_cols(self):
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
            for j in range(1, num+2):
                output[j, 0] = self.l_neg
        else:
            for j in range(1, self.n+2):
                output[j, 0] = self.l_neg
        
        # copy base matrix on position 1,1
        output[1:, 1:] = base_matrix
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
        
        if num == 0:
            output[0, 0] = self.l
            return output
        
        output[0, 0] = self.gamma
        
        for i in range(1, row):
            output[i, i] = self.l + self.l_neg + \
                (num - i + 1) * self.mu[0] + (i-1) * self.mu[1]

        return output

    
    def get_p(self) -> list[float]:
        
        first_col_sum = 0
        for i in range(1, self.N):
            first_col_sum += self.Y[i][0, 0]
            self.p[i] = np.sum(self.Y[i][0, 1:])
        self.p[0] += first_col_sum
        
        return [prob.real for prob in self.p]
    
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
        for i in range(k):
            res = np.dot(res, matrix)
        return res
    
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
            a.append(lst_exp(key_numbers[j][0] * self.mu[0] + key_numbers[j][1] * self.mu[1] + self.l_neg, s))
        
        a = np.array(a)

        Pn_plus = self._calc_up_probs(self.n+1)

        for k in range(self.n, self.N):

            Pa = np.transpose(self._matrix_pow(
                Pn_plus * a, k - self.n))  # size = (n+2, n+2)

            Ys = np.array([self.Y[k][0, i] for i in range(self.n + 2)])
            aPa = np.dot(a, Pa)
            w += Ys.dot(aPa)

        return w
    
    def get_w(self) -> list[float]:
        """
        Get the waiting time moments
        """

        if not self.w is None:
            return self.w
        
        w = [0.0] * 3

        for i in range(3):
            w[i] = derivative(self._calc_w_pls, 0,
                                dx=1e-3 / self.b[0], n=i + 1, order=9)
        w = np.array([-w[0].real, w[1].real, -w[2].real])
        self.w = w

        return w

    def get_v(self) -> list[float]:
        """
        Get the sojourn time moments
        """
        w = self.get_w()

        # serving = min(H2_b, exp(l_neg)) = H2(y1=y1, mu1 = mu1+l_neg, mu2=mu2+l_neg)

        params = H2Params(p1=self.y[0],
                          mu1=self.mu[0],
                          mu2=self.mu[1])
        
        l_neg = self.l_neg

        b = H2Distribution.calc_theory_moments(H2Params(p1=params.p1,
                                                        mu1=l_neg + params.mu1,
                                                        mu2=l_neg + params.mu2))

        return conv_moments(w, b)
    
    def _calc_service_probs(self) -> list[float]:
        """
        Returns the conditional probabilities of loaded states.
        """
        ps = np.array([self.p[i] for i in range(1, self.n+1)])
        ps /= np.sum(ps)

        return ps
    
    def get_v_served(self) -> list[float]:
        """
        Get the sojourn time moments
        """
        w = self.get_w()

        # serving = P(H2 | H2 < exp(l_neg))

        service_probs = self._calc_service_probs()
        b_cum = np.array([0.0, 0.0, 0.0])
        h2_params=H2Params(p1=self.y[0], mu1=self.mu[0], mu2=self.mu[1])
        for i in range(1, self.n+1):
            l_neg = self.l_neg

            b = moments_H2_less_than_exp(l_neg, h2_params)
            b_cum += service_probs[i-1].real*b

        return conv_moments(w, b_cum)

    def get_v_broken(self) -> list[float]:
        """
        Get the sojourn time moments
        """
        w = self.get_w()

        # serving = P(exp(l_neg) | exp(l_neg) < H2)

        service_probs = self._calc_service_probs()
        b_cum = np.array([0.0, 0.0, 0.0])
        h2_params=H2Params(p1=self.y[0], mu1=self.mu[0], mu2=self.mu[1])
        for i in range(1, self.n+1):
            l_neg = self.l_neg
            b = moments_exp_less_than_H2(l_neg, h2_params)
            b_cum += service_probs[i-1].real*b

        return conv_moments(w, b_cum)

    
