"""
Calculate M/H2/n queue with complex parameters using the Takahasi-Takagi method.
"""
import math
from functools import lru_cache

import numpy as np
from scipy.misc import derivative

from most_queue.general.conv import conv_moments, get_residual_distr_moments
from most_queue.rand_distribution import H2Distribution
from most_queue.theory.utils.transforms import (
    laplace_stieltjes_exp_transform as lst_exp,
)


class MGnNegativeRCSCalc:
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
        self.dt = np.dtype(dtype)
        if buffer:
            self.R = buffer + n  # max number of requests in the system - queue + channels
            self.N = self.R + 1  # number of levels in the system
        else:
            self.N = N
            self.R = None

        self.e1 = accuracy
        self.n = n
        self.b = b
        self.verbose = verbose

        h2_params = H2Distribution.get_params_clx(b)
        # params of H2-distribution:
        self.y = [h2_params.p1, 1.0 - h2_params.p1]
        self.l_pos = l_pos
        self.l_neg = l_neg
        self.mu = [h2_params.mu1, h2_params.mu2]
        # Cols massive holds the number of columns for each level, it is more convenient to calculate it once:
        self.cols = [] * N

        # Parameters for Takahashi's method
        self.t = []
        self.b1 = []
        self.b2 = []

        # convert to numpy arrays for faster calculations
        self.x = np.array([0.0 + 0.0j] * N)
        self.z = np.array([0.0 + 0.0j] * N)

        # Probabilities of states to be searched for
        self.p = np.array([0.0 + 0.0j] * N)

        self.num_of_iter_ = 0  # numer of iterations of the algorithm

        # Transition matrices
        self.A = []
        self.B = []
        self.C = []
        self.D = []
        self.Y = []

        for i in range(N):
            if i < n + 1:
                self.cols.append(i + 1)
            else:
                self.cols.append(n + 1)

            self.t.append(np.zeros((1, self.cols[i]), dtype=self.dt))
            self.b1.append(np.zeros((1, self.cols[i]), dtype=self.dt))
            self.b2.append(np.zeros((1, self.cols[i]), dtype=self.dt))

        self._build_matrices()
        self._initial_probabilities()

    def run(self):
        """
        Run the algorithm.
        """
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
                self.b1[j] = np.dot(self.t[j - 1], self.AG[j])

                # b":
                if j != (self.N - 1):
                    self.b2[j] = np.dot(self.t[j + 1], self.BG[j])
                else:
                    self.b2[j] = np.dot(self.t[j - 1], self.BG[j])

                c = self._calculate_c(j)

                x_znam = np.dot(c, self.b1[j]) + self.b2[j]
                self.x[j] = 0.0 + 0.0j
                for k in range(x_znam.shape[1]):
                    self.x[j] += x_znam[0, k]

                self.x[j] = (1.0 + 0.0j) / self.x[j]

                if self.R and j == (self.N - 1):
                    tA = np.dot(self.t[j - 1], self.A[j - 1])
                    tag = np.dot(tA, self.G[j])
                    tag_sum = 0
                    for t_i in range(tag.shape[1]):
                        tag_sum += tag[0, t_i]
                    self.z[j] = 1.0 / tag_sum
                    self.t[j] = self.z[j] * tag

                else:
                    self.z[j] = np.dot(c, self.x[j])
                    self.t[j] = np.dot(self.z[j], self.b1[j]) + \
                        np.dot(self.x[j], self.b2[j])

            self.x[0] = (1.0 + 0.0j) / self.z[1]

            x_max1 = np.max(self.x)

            if self.verbose:
                print(f"End iter # {self.num_of_iter_}")

        self._calculate_p()
        self._calculate_y()

    def get_p(self) -> list[float]:
        """
        Calculate the probabilities of states 
        """
        return [prob.real for prob in self.p]

    def get_q(self) -> float:
        """
        Calculation of the conditional probability of successful service completion at a node
        """
        return 1.0 - (self.l_neg/self.l_pos)*(1.0-self.p[0].real)

    def get_w(self, derivate=False) -> list[float]:
        """
        Get the waiting time moments
        """

        w = [0.0] * 3

        if derivate:
            for i in range(3):
                w[i] = derivative(self._calc_w_pls, 0,
                                  dx=1e-3 / self.b[0], n=i + 1, order=9)
            return np.array([-w[0].real, w[1].real, -w[2].real])

        for j in range(1, len(self.p) - self.n):
            w[0] += j * self.p[self.n + j]
        for j in range(2, len(self.p) - self.n):
            w[1] += j * (j - 1) * self.p[self.n + j]
        for j in range(3, len(self.p) - self.n):
            w[2] += j * (j - 1) * (j - 2) * self.p[self.n + j]

        for j in range(3):
            w[j] /= math.pow(self.l_pos, j + 1)
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

            key_numbers = self._get_key_numbers(k+1) # for k=1, [(2,0), (1,1), (0, 2)]
            
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
        
        Pn_plus = self._calc_up_probs(self.n + 1) # size=(4, 4)
        
        Pa = Pn_plus*a_n  # 4,4
        
        PdotaT = Pn_plus.dot(a_n.T) # 4,1

        prod = np.eye(self.n+1)
        
        for k in range(self.n, self.N):
            
            Ys = np.array([self.Y[k][0, i] for i in range(self.n + 1)]) # (1, 4)
            
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

            Pa = np.transpose(self._matrix_pow(Pn_plus * a, k - self.n))  # n+1 x n+1

            Ys = [self.Y[k][0, i] for i in range(self.n + 1)]
            aPa = np.dot(a, Pa)
            for i in range(self.n + 1):
                w += Ys[i] * aPa[i]

        return w

    def get_v(self, derivate=False) -> list[float]:
        """
        Get the sojourn time moments
        """
        v = [0.0] * 3

        if derivate:
            for i in range(3):
                v[i] = derivative(self._calc_v_pls, 0,
                                  dx=1e-3 / self.b[0], n=i + 1, order=9)
            return [-v[0], v[1].real, -v[2]]

        w = self.get_w(derivate=True)
        return conv_moments(w, self.b)

    def _initial_probabilities(self):
        """
        Set initial probabilities of microstates
        """
        # Initialize all states to equal probability for each i
        probs = [1.0 / self.cols[i] for i in range(self.N)]
        for i in range(self.N):
            self.t[i][0, :] = probs[i]
        self.x[0] = 0.4

    def _norm_probs(self):
        # Calculate the sum of probabilities in one pass using a generator expression
        total = sum(value for value in self.p)

        # Normalize each probability in a single list comprehension
        self.p = [value / total for value in self.p]

        if self.verbose:
            # Check if the sum is approximately 1 (using a generator expression)
            print(f"Summ of probs = {sum(self.p):.5f}")

    def _calculate_p(self):
        """
        Calculate probabilities p based on current values of x.

        Optimizations:
            - Vectorized operations for better performance.
            - Reduced nested loops using list comprehensions.
            - Improved variable names for readability.
        """

        # Compute initial terms for f1 and znam
        term_f1 = self.y[0] / self.mu[0] + self.y[1] / self.mu[1]

        # Calculate the denominator (znam) more efficiently using list comprehensions
        znam = self.n + sum((self.n - j) *
                            np.prod(self.x[:j]) for j in range(1, self.n))

        if self.R:
            product_r = np.prod([self.x[i] for i in range(self.N)])
            znam -= term_f1 * self.l_pos * product_r

        # Calculate p[0]
        self.p[0] = (self.n - self.l_pos * term_f1) / znam
        summ_p = self.p[0]

        # Compute subsequent probabilities using vectorized operations
        for j in range(self.N - 1):
            self.p[j + 1] = self.p[j] * self.x[j]
            summ_p += self.p[j + 1]

        geom_tail = self.p[self.N - 1] * \
            self.x[self.N - 1]/(1.0 - self.x[self.N - 1])
        summ_p += geom_tail

        if self.verbose:
            print(f'Geom tail = {geom_tail:.5f}')
            print(f"Summ of probs = {summ_p:.5f}")

        self._norm_probs()

    def _calculate_y(self):
        
        for i in range(self.N):
            y_tek = np.dot(self.p[i], self.t[i])
            self.Y.append(y_tek)
            

    def _build_matrices(self):
        """
        Form transition matrices
        """
        for i in range(self.N):
            self.A.append(self._build_big_a_matrix(i))
            self.B.append(self._build_big_b_matrix(i))
            self.C.append(self._build_big_c_matrix(i))
            self.D.append(self._build_big_d_matrix(i))

    def _calc_g_matrices(self):
        self.G = []
        for j in range(0, self.N):
            self.G.append(np.linalg.inv(self.D[j] - self.C[j]))

    def _calc_ag_matrices(self):
        self.AG = [0]
        for j in range(1, self.N):
            self.AG.append(np.dot(self.A[j - 1], self.G[j]))

    def _calc_bg_matrices(self):
        self.BG = [0]
        for j in range(1, self.N):
            if j != (self.N - 1):
                self.BG.append(np.dot(self.B[j + 1], self.G[j]))
            else:
                self.BG.append(np.dot(self.B[j], self.G[j]))

    def _calc_support_matrices(self):
        self._calc_g_matrices()
        self._calc_ag_matrices()
        self._calc_bg_matrices()

    def _calculate_c(self, j):
        """
        Calculate value of variable c participating in the calculation.
        """
        m = np.dot(self.b2[j], self.B[j])
        chisl = np.sum(m[0])

        m = np.dot(self.b1[j], self.B[j])
        znam2 = np.sum(m[0])

        m = np.dot(self.t[j - 1], self.A[j - 1])
        znam = np.sum(m[0])

        return chisl / (znam - znam2)

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

        for i in range(row):
            if num < self.n:
                output[i, i] = self.l_pos * self.y[0]
                output[i, i + 1] = self.l_pos * self.y[1]
            else:
                output[i, i] = self.l_pos

        return output

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

    def _build_big_c_matrix(self, num):
        """
        Create matrix C by the given level number.
        """
        if num < self.n:
            col = self.cols[num]
            row = col
        else:
            col = self.cols[self.n]
            row = col

        output = np.zeros((row, col), dtype=self.dt)
        if num > self.n:
            output = self.C[self.n]
            return output

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
            output[i, i] = self.l_pos + self.l_neg + \
                (num - i) * self.mu[0] + i * self.mu[1]

        return output
