"""
Calculation of the M/H2/n system with H2-warming using the Takahasi-Takagi method.
Use complex parameters, which allow to approximate the service time 
distribution with arbitrary coefficients of variation (>1, <=1).
"""
import math

import numpy as np
from scipy.misc import derivative

from most_queue.rand_distribution import H2Distribution
from most_queue.theory.utils.binom_probs import calc_binom_probs
from most_queue.theory.utils.transforms import lst_exp


class MH2nH2Warm:
    """
    Calculation of the M/H2/n system with H2-warming using the Takahasi-Takagi method.
    Use complex parameters, which allow to approximate the service time 
    distribution with arbitrary coefficients of variation (>1, <=1).
    """

    def __init__(self, l, b, b_warm, n, buffer=None, N=150, accuracy=1e-6, dtype="c16", verbose=False,
                 is_only_first=False):
        """
        Initialization of the M/H2/n system with H2-warming using the Takahasi-Takagi method.
        Use complex parameters, which allow to approximate the service time 
        distribution with arbitrary coefficients of variation (>1, <=1).
        :param l: arrival intensity
        :param b: initial moments of service time distribution
        :param b_warm: initial moments of warming time distribution
        :param n: number of servers
        :param buffer: size of the buffer (optional)
        :param N: number of levels (default is 150)
        :param accuracy: accuracy, parameter for stopping iteration
        :param dtype: data type for calculations (default is complex)
        :param verbose: if True, print intermediate results
        :param is_only_first: if True, calculate only the first level of the hierarchy

        """

        self.dt = np.dtype(dtype)
        if buffer:
            self.R = buffer + n  # maximum number of requests in the system - queue + channels
            self.N = self.R + 1  # number of levels on one more than + zero state
        else:
            self.N = N
            self.R = None

        self.is_only_first = is_only_first

        self.e1 = accuracy
        self.n = n
        self.b = b
        self.verbose = verbose
        self.l = l

        if self.dt == 'c16':
            h2_params_service = H2Distribution.get_params_clx(b)
        else:
            h2_params_service = H2Distribution.get_params(b)

        # Parameters of the H2 distribution
        self.y = [h2_params_service.p1, 1.0 - h2_params_service.p1]
        self.mu = [h2_params_service.mu1, h2_params_service.mu2]

        self.b_warm = b_warm
        if self.dt == 'c16':
            h2_params_warm = H2Distribution.get_params_clx(b_warm)
        else:
            h2_params_warm = H2Distribution.get_params(b_warm)
        self.y_w = [h2_params_warm.p1, 1.0 - h2_params_warm.p1]
        self.mu_w = [h2_params_warm.mu1, h2_params_warm.mu2]

        # Cols - array that stores the number of columns for each level, it is convenient to calculate it once:
        self.cols = [] * N

        # init parameters for the Takahasi-Takagi method
        self.t = []
        self.b1 = []
        self.b2 = []
        if self.dt == 'c16':
            self.x = [0.0 + 0.0j] * N
            self.z = [0.0 + 0.0j] * N

            # probabilities of states of the queueing system
            self.p = [0.0 + 0.0j] * N
        else:
            self.x = [0.0] * N
            self.z = [0.0] * N

            # probabilities of states of the queueing system
            self.p = [0.0] * N

        # Transition matrices for the Takahasi-Takagi method
        self.A = []
        self.B = []
        self.C = []
        self.D = []
        self.Y = []

        for i in range(N):

            if self.is_only_first:
                if i < n + 1:
                    if i == 0:
                        self.cols.append(1)  # 00 state
                    else:
                        self.cols.append(3 * i + 1)  # w1 w2 01 10
                else:
                    self.cols.append(3 * n + 1)
            else:
                if i < n + 1:
                    if i == 0:
                        self.cols.append(1)  # 00 state
                    else:
                        # w1 w2 01 10, 20 11 02, ...
                        self.cols.append(i + 3)
                else:
                    self.cols.append(n + 3)

            self.t.append(np.zeros((1, self.cols[i]), dtype=self.dt))
            self.b1.append(np.zeros((1, self.cols[i]), dtype=self.dt))
            self.b2.append(np.zeros((1, self.cols[i]), dtype=self.dt))
            self.x.append(np.zeros((1, self.cols[i]), dtype=self.dt))

        self._build_matrices()
        self._initial_probabilities()
        # Keys of the levels n, for n=3 [(3,0) (2,1) (1,2) (0,3)]
        self.key_numbers = self._get_key_numbers(self.n)
        self.down_probs = self._calc_down_probs(self.n + 1)

    def get_p(self) -> list[float]:
        """
        Get the probabilities of states of the queueing system.
        """
        self.p = [prob.real for prob in self.p]

        return self.p

    def get_w(self):
        """
        Get the first three moments of the waiting time distribution.
        """
        w = [0.0] * 3

        if self.is_only_first:

            for j in range(1, len(self.p) - self.n):
                w[0] += j * self.p[self.n + j]
            for j in range(2, len(self.p) - self.n):
                w[1] += j * (j - 1) * self.p[self.n + j]
            for j in range(3, len(self.p) - self.n):
                w[2] += j * (j - 1) * (j - 2) * self.p[self.n + j]

            for j in range(3):
                w[j] /= math.pow(self.l, j + 1)
                w[j] = w[j].real

            return w

        else:
            for i in range(3):
                w[i] = derivative(self._calc_w_pls, 0,
                                  dx=1e-3 / self.b[0], n=i + 1, order=9)
            return [-w[0], w[1].real, -w[2]]

    def get_v(self):
        """
        Get the first three initial moments of sojourn time in the queue.
        """
        v = [0.0] * 3
        w = self.get_w()
        b = self.b
        v[0] = w[0] + b[0]
        v[1] = w[1] + 2 * w[0] * b[0] + b[1]
        v[2] = w[2] + 3 * w[1] * b[0] + 3 * w[0] * b[1] + b[2]

        return v

    def _get_key_numbers(self, level):
        key_numbers = []
        for i in range(level + 1):
            key_numbers.append((self.n - i, i))
        return np.array(key_numbers, dtype=self.dt)

    def _calc_down_probs(self, from_level):
        if from_level == self.n:
            return np.eye(self.n + 1)
        b_matrix = self.B[from_level]
        probs = []
        for i in range(2, self.cols[from_level - 1]):
            probs_on_level = []
            for j in range(2, self.cols[from_level - 1]):
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

    def _calc_w_pls(self, s) -> float:
        """
        Calculate Laplace-Stietjes transform of the waiting time distribution.
        :param s: Laplace variable
        :return: Laplace-Stieltjes transform of the waiting time distribution
        """
        w = 0
        # вероятности попадания в состояния обслуживания, вычисляются с помощью биноминального распределения
        probs = calc_binom_probs(self.n + 1, self.y[0])

        # Если заявка попала в фазу разогрева, хотя каналы свободны,
        # ей придется подождать окончание разогрева
        for k in range(1, self.n):
            for i in range(2):
                w += self.Y[k][0, i] * lst_exp(self.mu_w[i], s)

        for k in range(self.n, self.N):

            # Если заявка попала в фазу разогрева и каналы заняты. Также есть k-n заявок в очереди
            # ей придется подождать окончание разогрева + обслуживание всех накопленных заявок
            # ключи яруса n, для n=3 [(3,0) (2,1) (1,2) (0,3)]
            key_numbers = self._get_key_numbers(self.n)

            a = np.array(
                [lst_exp(key_numbers[j][0] * self.mu[0] + key_numbers[j][1] * self.mu[1], s) for j in
                 range(self.n + 1)])

            P = self._calc_down_probs(k)
            Pa = np.transpose(self._matrix_pow(P * a, k - self.n))  # n+1 x n+1

            aP_tilda = a * probs

            pls_service_total = sum(np.dot(aP_tilda, Pa))

            for i in range(2):
                w += self.Y[k][0, i] * \
                    lst_exp(self.mu_w[i], s) * pls_service_total

            # попала в фазу обслуживания
            Ys = [self.Y[k][0, i + 2] for i in range(len(probs))]
            aPa = np.dot(a, Pa)
            for i in range(self.n + 1):
                w += Ys[i] * aPa[i]

        return w

    def _initial_probabilities(self):
        """
        Задаем первоначальные значения вероятностей микросостояний
        """
        # t задаем равновероятными
        for i in range(self.N):
            for j in range(self.cols[i]):
                self.t[i][0, j] = 1.0 / self.cols[i]
        self.x[0] = 0.4

    def _norm_probs(self):
        summ = 0
        for i in range(self.N):
            summ += self.p[i]

        for i in range(self.N):
            self.p[i] /= summ

        if self.verbose:
            summ = 0
            for i in range(self.N):
                summ += self.p[i]

    def _calculate_p(self):
        """
        После окончания итераций находим значения вероятностей p по найденным х
        """
        # version 1
        p_sum = 0
        p0_max = 1.0
        p0_min = 0.0

        while math.fabs(1.0 - p_sum) > 1e-6:
            p0_ = (p0_max + p0_min) / 2.0
            p_sum = p0_
            self.p[0] = p0_
            for j in range(self.N - 1):
                self.p[j + 1] = self.p[j] * self.x[j]
                p_sum += self.p[j + 1]

            if (p_sum > 1.0):
                p0_max = p0_
            else:
                p0_min = p0_

        self._norm_probs()

    def _calculate_y(self):
        for i in range(self.N):
            self.Y.append(np.dot(self.p[i], self.t[i]))

    def _build_matrices(self):
        """
        Формирует матрицы переходов
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

    def run(self):
        """
        Запускает расчет
        """
        if self.dt == 'c16':
            self.b1[0][0, 0] = 0.0 + 0.0j
            self.b2[0][0, 0] = 0.0 + 0.0j
            x_max1 = 0.0 + 0.0j
            x_max2 = 0.0 + 0.0j
        else:
            self.b1[0][0, 0] = 0.0
            self.b2[0][0, 0] = 0.0
            x_max1 = 0.0
            x_max2 = 0.0

        self._calc_support_matrices()

        self.num_of_iter_ = 0  # кол-во итераций алгоритма
        for i in range(self.N):
            if self.x[i].real > x_max1.real:
                x_max1 = self.x[i]

        while math.fabs(x_max2.real - x_max1.real) >= self.e1:
            x_max2 = x_max1
            self.num_of_iter_ += 1

            for j in range(1, self.N):  # по всем ярусам, кроме первого.

                # b':
                self.b1[j] = np.dot(self.t[j - 1], self.AG[j])

                # b":
                if j != (self.N - 1):
                    self.b2[j] = np.dot(self.t[j + 1], self.BG[j])
                else:
                    self.b2[j] = np.dot(self.t[j - 1], self.BG[j])

                c = self._calculate_c(j)

                x_znam = np.dot(c, self.b1[j]) + self.b2[j]
                if self.dt == 'c16':
                    self.x[j] = 0.0 + 0.0j
                else:
                    self.x[j] = 0.0
                for k in range(x_znam.shape[1]):
                    self.x[j] += x_znam[0, k]

                if self.dt == 'c16':
                    self.x[j] = (1.0 + 0.0j) / self.x[j]
                else:
                    self.x[j] = (1.0) / self.x[j]

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

            if self.dt == 'c16':
                self.x[0] = (1.0 + 0.0j) / self.z[1]
            else:
                self.x[0] = 1.0 / self.z[1]

            self.t[0] = self.x[0]*(np.dot(self.t[1], self.B[1]).dot(self.G[0]))

            x_max1 = 0.0 + 0.0j
            for i in range(self.N):
                if self.x[i].real > x_max1.real:
                    x_max1 = self.x[i]

            if self.verbose:
                print(f"End iter # {self.num_of_iter_}")

        self._calculate_p()
        self._calculate_y()

    def _calculate_c(self, j):
        """
        Вычисляет значение переменной с, участвующей в расчете
        """
        chisl = 0
        znam = 0
        znam2 = 0

        m = np.dot(self.b2[j], self.B[j])
        for k in range(m.shape[1]):
            chisl += m[0, k]

        m = np.dot(self.b1[j], self.B[j])
        for k in range(m.shape[1]):
            znam2 += m[0, k]

        m = np.dot(self.t[j - 1], self.A[j - 1])
        for k in range(m.shape[1]):
            znam += m[0, k]

        return chisl / (znam - znam2)

    def _insert_standart_A_into(self, mass, l, y1, left_pos, bottom_pos, level):
        row_num = level
        for i in range(row_num):
            mass[i + left_pos, i + bottom_pos] = l * y1
            mass[i + left_pos, i + bottom_pos + 1] = l * (1.0 - y1)

    def _build_big_a_matrix(self, num):
        """
        Формирует матрицу А по заданному номеру яруса
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
            output[0, 0] = self.l * self.y_w[0]
            output[0, 1] = self.l * self.y_w[1]
        else:

            if num < self.n:
                if self.is_only_first:
                    # first block
                    self._insert_standart_A_into(
                        output, self.l, self.y[0], 0, 0, level=num)
                    # second
                    self._insert_standart_A_into(
                        output, self.l, self.y[0], num, num + 1, level=num)
                    # third
                    self._insert_standart_A_into(
                        output, self.l, self.y[0], 2 * num, 2 * (num + 1), level=num + 1)
                else:
                    # first block
                    output[0, 0] = self.l
                    output[1, 1] = self.l
                    # second
                    self._insert_standart_A_into(
                        output, self.l, self.y[0], 2, 2, level=num + 1)
            else:
                for i in range(row):
                    output[i, i] = self.l

        return output

    def _insert_standart_B_into(self, mass, y, mu, left_pos, bottom_pos, level, n):
        col = level
        for i in range(col):
            if level <= n:
                mass[i + left_pos, i + bottom_pos] = (level - i) * mu[0]
                mass[i + left_pos + 1, i + bottom_pos] = (i + 1) * mu[1]
            else:
                mass[i + left_pos, i +
                     bottom_pos] = (level - i - 1) * mu[0] * y[0] + i * mu[1] * y[1]
                if i != level - 1:
                    mass[i + left_pos, i + bottom_pos +
                         1] = (level - i - 1) * mu[0] * y[1]
                if i != level - 1:
                    mass[i + + left_pos + 1, i +
                         bottom_pos] = (i + 1) * mu[1] * y[0]

    def _build_big_b_matrix(self, num):
        """
            Формирует матрицу B по заданному номеру яруса
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

        if num == 1:
            output[2, 0] = self.mu[0]
            output[3, 0] = self.mu[1]

            if self.is_only_first:
                output[0, 0] = self.mu_w[0]
                output[1, 0] = self.mu_w[1]
        else:

            if num < self.n + 1:

                if self.is_only_first:
                    # first block
                    self._insert_standart_B_into(
                        output, self.y, self.mu, 0, 0, num - 1, self.n)
                    # second
                    self._insert_standart_B_into(
                        output, self.y, self.mu, num, num - 1, num - 1, self.n)
                    # third
                    self._insert_standart_B_into(
                        output, self.y, self.mu, 2 * num, 2 * (num - 1), num, self.n)

                    # warm block 1
                    for i in range(num):
                        output[i, i + 2 * (num - 1)] = self.mu_w[0]

                    # warm block 2
                    for i in range(num):
                        output[i + num, i + 2 * (num - 1)] = self.mu_w[1]
                else:
                    self._insert_standart_B_into(
                        output, self.y, self.mu, 2, 2, num, self.n)

            else:

                if self.is_only_first:
                    # first block
                    self._insert_standart_B_into(
                        output, self.y, self.mu, 0, 0, num - 1, self.n - 1)
                    # second
                    self._insert_standart_B_into(
                        output, self.y, self.mu, num - 1, num - 1, num - 1, self.n - 1)
                    # third
                    self._insert_standart_B_into(
                        output, self.y, self.mu, 2 * (num - 1), 2 * (num - 1), num, self.n)

                    # warm block 1
                    for i in range(num - 1):
                        output[i, i + 2 * (num - 1)] = self.mu_w[0] * self.y[0]
                        output[i, i + 2 * (num - 1) +
                               1] = self.mu_w[0] * self.y[1]

                    # warm block 2
                    for i in range(num - 1):
                        output[i + num - 1, i + 2 *
                               (num - 1)] = self.mu_w[1] * self.y[0]
                        output[i + num - 1, i + 2 *
                               (num - 1) + 1] = self.mu_w[1] * self.y[1]

                else:
                    self._insert_standart_B_into(
                        output, self.y, self.mu, 2, 2, num, self.n)

        return output

    def _build_big_c_matrix(self, num):
        """
        Формирует матрицу C по заданному номеру яруса
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

        if not self.is_only_first:
            if num > 0:
                probs = calc_binom_probs(num + 1, self.y[0])
                for i in range(len(probs)):
                    output[0, 2 + i] = self.mu_w[0] * probs[i]
                    output[1, 2 + i] = self.mu_w[1] * probs[i]

        return output

    def _build_big_d_matrix(self, num):
        """
        Формирует матрицу D по заданному номеру яруса
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
            if self.dt == 'c16':
                sumA = 0.0 + 0.0j
                sumB = 0.0 + 0.0j
                sumC = 0.0 + 0.0j
            else:
                sumA = 0.0
                sumB = 0.0
                sumC = 0.0

            for j in range(self.cols[num + 1]):
                sumA += self.A[num][i, j]

            if num != 0:
                for j in range(self.cols[num - 1]):
                    sumB += self.B[num][i, j]

            for j in range(col):
                sumC += self.C[num][i, j]

            output[i, i] = sumA + sumB + sumC

        return output
