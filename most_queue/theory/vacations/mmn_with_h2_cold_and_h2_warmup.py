"""
MMnHyperExpWarmAndCold class for calculating M/M/n queueing system with H2-warm-up and H2-cooling
using numerical Takahashi-Takagi method.
Complex parameters are used. Complex parameters allow approximating the service time distribution
with arbitrary coefficients of variation (>1, <=1).
"""

import math
import time
from itertools import chain

import numpy as np
from scipy.misc import derivative

from most_queue.random.distributions import H2Distribution
from most_queue.structs import VacationResults
from most_queue.theory.fifo.mgn_takahasi import MGnCalc, TakahashiTakamiParams
from most_queue.theory.utils.conv import conv_moments
from most_queue.theory.utils.transforms import lst_exp


class MMnHyperExpWarmAndCold(MGnCalc):
    """
    Calculation of M/M/n queueing system with H2-warm-up and H2-cooling
    using numerical Takahashi-Takagi method.Complex parameters are used.
    Complex parameters allow approximating the service time distribution
    with arbitrary coefficients of variation (>1, <=1).
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
        calc_params: parameters for the calculation (optional)
        """

        super().__init__(n=n, calc_params=calc_params, buffer=buffer)

        self.calc_params = calc_params or TakahashiTakamiParams()

        self.dt = np.dtype(self.calc_params.dtype)
        if buffer:
            self.R = buffer + n
            self.N = self.R + 1
        else:
            self.N = self.calc_params.N
            self.R = None

        self.e1 = self.calc_params.tolerance
        self.n = n
        self.mu = None
        self.verbose = self.calc_params.verbose
        self.l = None

        self.cols = [] * self.N

        self.t = []
        self.b1 = []
        self.b2 = []
        if self.dt == "c16":
            self.x = [0.0 + 0.0j] * self.N
            self.z = [0.0 + 0.0j] * self.N

            self.p = [0.0 + 0.0j] * self.N
        else:
            self.x = [0.0] * self.N
            self.z = [0.0] * self.N

            self.p = [0.0] * self.N

        # transient matrices for the Markov chain approximation
        self.A = []
        self.B = []
        self.C = []
        self.D = []
        self.Y = []

        self.G = []
        self.AG = []
        self.BG = []

        for i in range(self.N):

            if i < n + 1:
                if i == 0:
                    self.cols.append(3)  # 0 state normal + 0_cold_1 + 0_cold_2
                else:
                    # i_warm_1 + i_warm_2 i state normal + i_cold_1 +
                    # i_cold_2...
                    self.cols.append(5)
            else:
                self.cols.append(5)

            self.t.append(np.zeros((1, self.cols[i]), dtype=self.dt))
            self.b1.append(np.zeros((1, self.cols[i]), dtype=self.dt))
            self.b2.append(np.zeros((1, self.cols[i]), dtype=self.dt))
            self.x.append(np.zeros((1, self.cols[i]), dtype=self.dt))

        self.num_of_iter_ = 0  # number of iterations for the algorithm

        self.y_w, self.mu_w = None, None
        self.y_c, self.mu_c = None, None
        self.b_warm = None
        self.b_cold = None

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """
        Set the arrival rate
        """
        self.l = l
        self.is_sources_set = True

    def set_servers(
        self,
        mu: float,
        b_warm: list[float],
        b_cold: list[float],
    ):  # pylint: disable=arguments-differ
        """
        Set the service rate and raw moments of distributions for vacation states:
        warming time, cooling

        :param mu: service rate of Exp distribution
        :param b_warm: raw moments of warming time distribution
        :param b_cold: raw moments of cooling time distribution
        """
        self.mu = mu
        self.b_warm = b_warm
        self.b_cold = b_cold

        self.b_warm = b_warm
        if self.dt == "c16":
            h2_params_warm = H2Distribution.get_params_clx(b_warm)
        else:
            h2_params_warm = H2Distribution.get_params(b_warm)

        self.y_w = [h2_params_warm.p1, 1.0 - h2_params_warm.p1]
        self.mu_w = [h2_params_warm.mu1, h2_params_warm.mu2]

        self.b_cold = b_cold
        if self.dt == "c16":
            h2_params_cold = H2Distribution.get_params_clx(b_cold)
        else:
            h2_params_cold = H2Distribution.get_params(b_cold)
        self.y_c = [h2_params_cold.p1, 1.0 - h2_params_cold.p1]
        self.mu_c = [h2_params_cold.mu1, h2_params_cold.mu2]

        self.is_servers_set = True

    def run(self) -> VacationResults:
        """
        Run calculation for queueing system.
        """

        start = time.process_time()

        self._check_if_servers_and_sources_set()

        self._build_matrices()
        self._initial_probabilities()

        if self.dt == "c16":
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
                if self.dt == "c16":
                    self.x[j] = 0.0 + 0.0j
                else:
                    self.x[j] = 0.0
                for k in range(x_znam.shape[1]):
                    self.x[j] += x_znam[0, k]

                if self.dt == "c16":
                    self.x[j] = (1.0 + 0.0j) / self.x[j]
                else:
                    self.x[j] = 1.0 / self.x[j]

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
                    self.t[j] = np.dot(self.z[j], self.b1[j]) + np.dot(self.x[j], self.b2[j])

            if self.dt == "c16":
                self.x[0] = (1.0 + 0.0j) / self.z[1]
            else:
                self.x[0] = 1.0 / self.z[1]

            x_max1 = 0
            for i in range(self.N):
                if self.x[i].real > x_max1.real:
                    x_max1 = self.x[i]

            if self.verbose:
                print(f"End iter # {self.num_of_iter_:d}, x_max: {x_max1}")

        self._calculate_p()
        self._calculate_y()
        results = self.get_results()

        results.duration = time.process_time() - start

        return results

    def get_results(self) -> VacationResults:
        """
        Get all results
        """

        self.p = self.get_p()
        self.w = self.get_w()
        self.v = self.get_v()

        utilization = self.get_utilization()

        warmup_prob = self.get_warmup_prob()
        cold_prob = self.get_cold_prob()

        return VacationResults(
            v=self.v,
            w=self.w,
            p=self.p,
            utilization=utilization,
            warmup_prob=warmup_prob,
            cold_prob=cold_prob,
        )

    def get_utilization(self):
        """
        Calc utilization of the queue.
        """

        return self.l / (self.n * self.mu)

    def get_p(self):
        """
        Returns the list of state probabilities
        p[k] - probability of being in the system with exactly k requests
        """
        return [prob.real for prob in self.p]

    def get_w(self, _derivate=False, num_of_moments: int = 4):
        """
        Returns waiting time raw moments
        """
        w = [0.0] * num_of_moments

        for i in range(num_of_moments):
            min_mu = min(
                chain(
                    np.array([mu.real for mu in self.mu_w]).astype("float"),
                    np.array([mu.real for mu in self.mu_c]).astype("float"),
                    [self.mu],
                )
            )
            w[i] = derivative(self._calc_w_pls, 0, dx=1e-3 / min_mu, n=i + 1, order=9)
            if i % 2 == 0:
                w[i] = -w[i]
        return w

    def get_b(self):
        """
        Returns service time raw moments
        """
        return [1.0 / self.mu, 2.0 / pow(self.mu, 2), 6.0 / pow(self.mu, 3), 24.0 / pow(self.mu, 4)]

    def get_v(self, num_of_moments: int = 4):
        """
        Return raw moments of time spent in the system
        """
        w = self.get_w(num_of_moments)
        b = self.get_b()[:num_of_moments]  # return 4 moments

        v = conv_moments(w, b, num=num_of_moments)

        return v

    def get_cold_prob(self):
        """
        Get the probability of being in a cold state.
        """
        p_cold = 0
        for k in range(self.N):
            p_cold += self.Y[k][0, -1] + self.Y[k][0, -2]
        return p_cold.real

    def get_warmup_prob(self):
        """
        Get the probability of being in a warmup state.
        """
        p_warmup = 0
        for k in range(1, self.N):
            p_warmup += self.Y[k][0, 0] + self.Y[k][0, 1]
        return p_warmup.real

    def _calc_w_pls(self, s):
        w = 0

        # вычислим ПЛС заранее
        mu_pls = lst_exp(self.mu * self.n, s)
        mu_w_pls = np.array([lst_exp(self.mu_w[0], s), lst_exp(self.mu_w[1], s)])
        mu_c_pls = np.array([lst_exp(self.mu_c[0], s), lst_exp(self.mu_c[1], s)])

        # Комбо переходов: охлаждение + разогрев
        # [i,j] = охлаждение в i, переход из состояния i охлаждения
        # в j состояние разогрева, разогрев j
        c_to_w = np.zeros((2, 2), dtype=self.dt)
        for c_phase in range(2):
            for w_phase in range(2):
                c_to_w[c_phase, w_phase] = mu_c_pls[c_phase] * self.y_w[w_phase] * mu_w_pls[w_phase]

        # Если заявка попала в состояние [0] ей придется подождать окончание
        # разогрева
        w += self.Y[0][0, 0] * (self.y_w[0] * mu_w_pls[0] + self.y_w[1] * mu_w_pls[1])
        # Если заявка попала в фазу разогрева, хотя каналы свободны,
        # ей придется подождать окончание разогрева
        for k in range(1, self.n):
            for i in range(2):
                w += self.Y[k][0, i] * mu_w_pls[i]

        # Если заявка попала в фазу охлаждения, хотя каналы свободны,
        # ей придется подождать окончание охлаждения и разогрева
        for k in range(0, self.n):
            if k == 0:
                for i in range(2):
                    # Переход в [0] состояние, а из него -> в разогрев
                    w += self.Y[k][0, i + 1] * mu_c_pls[i] * (self.y_w[0] * mu_w_pls[0] + self.y_w[1] * mu_w_pls[1])
            else:
                for c_phase in range(2):
                    for w_phase in range(2):
                        w += self.Y[k][0, 3 + c_phase] * c_to_w[c_phase, w_phase]

        pls_service_total = 1.0
        for k in range(self.n, self.N):

            # попала в фазу обслуживания
            # подождать окончание k+1-n обслуживаний
            pls_service_total *= mu_pls

            w += self.Y[k][0, 2] * pls_service_total

            # попала в фазу разогрева - разогрев + обслуживание

            w += self.Y[k][0, 0] * mu_w_pls[0] * pls_service_total
            w += self.Y[k][0, 1] * mu_w_pls[1] * pls_service_total

            # попала в фазу охлаждения - охлаждение + разогрев + обслуживание
            for c_phase in range(2):
                for w_phase in range(2):
                    w += self.Y[k][0, 3 + c_phase] * c_to_w[c_phase, w_phase] * pls_service_total

        return w

    def _initial_probabilities(self):
        """
        Задаем первоначальные значения вероятностей микросостояний
        """
        # t задаем равновероятными
        for i in range(self.N):
            for j in range(self.cols[i]):
                self.t[i][0, j] = 1.0 / self.cols[i]

        ro = self.l / (self.mu * self.n)
        va = 1.0
        vb = 1.0
        self.x[0] = pow(ro, 2.0 / (va * va + vb * vb))

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
            print(f"Summ of probs = {summ:.5f}")

    def _calculate_p(self):
        """
        После окончания итераций находим значения вероятностей p по найденным х
        """
        # version 1
        p_sum = 0
        p0_max = 1.0
        p0_min = 0.0

        while math.fabs(1.0 - p_sum.real) > 1e-6:
            p0_ = (p0_max + p0_min) / 2.0
            p_sum = p0_
            self.p[0] = p0_
            for j in range(self.N - 1):
                self.p[j + 1] = self.p[j] * self.x[j]
                p_sum += self.p[j + 1]

            if p_sum > 1.0:
                p0_max = p0_
            else:
                p0_min = p0_

        self._norm_probs()

    def _calculate_y(self):
        """
        Calculate the Y matrix based on the probabilities p
        """
        # sum_y = 0.0
        for i in range(self.N):
            self.Y.append(np.dot(self.p[i], self.t[i]))
            # sum_y += np.sum(self.Y[i])
        # print("Sum Y: ", sum_y)

    def _build_matrices(self):
        """
        Формирует матрицы переходов
        """
        for i in range(self.N):
            self.A.append(self._buildA(i))
            self.B.append(self._buildB(i))
            self.C.append(self._buildC(i))
            self.D.append(self._buildD(i))

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

    def _insert_standart_A_into(
        self, mass, l, y1, left_pos, bottom_pos, level
    ):  # pylint: disable=too-many-positional-arguments, too-many-arguments
        row_num = level
        for i in range(row_num):
            mass[i + left_pos, i + bottom_pos] = l * y1
            mass[i + left_pos, i + bottom_pos + 1] = l * (1.0 - y1)

    def _buildA(self, num):
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
            # to warm phase
            output[0, 0] = self.l * self.y_w[0]
            output[0, 1] = self.l * self.y_w[1]
            # cold phase
            output[1, 3] = self.l
            output[2, 4] = self.l
        else:

            for i in range(row):
                output[i, i] = self.l

        return output

    def _buildB(self, num):
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
            # to cold phase
            output[2, 1] = self.mu * self.y_c[0]
            output[2, 2] = self.mu * self.y_c[1]

        else:
            if num < self.n + 1:

                output[2, 2] = self.mu * num

            else:

                output[2, 2] = self.mu * self.n

        return output

    def _buildC(self, num):
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

        if num == 0:
            # cold serving
            output[1, 0] = self.mu_c[0]
            output[2, 0] = self.mu_c[1]
        else:
            # cold serving
            output[3, 0] = self.mu_c[0] * self.y_w[0]
            output[3, 1] = self.mu_c[0] * self.y_w[1]
            output[4, 0] = self.mu_c[1] * self.y_w[0]
            output[4, 1] = self.mu_c[1] * self.y_w[1]

            # warmup end
            output[0, 2] = self.mu_w[0]
            output[1, 2] = self.mu_w[1]

        return output

    def _buildD(self, num):
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
            if self.dt == "c16":
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
