"""
Calculation M/M/2 queue with three classes of requests and absolute priority.
"""

import math
import time

import numpy as np

from most_queue.random.distributions import CoxDistribution
from most_queue.structs import PriorityResults, QueueResults
from most_queue.theory.fifo.mgn_takahasi import MGnCalc, TakahashiTakamiParams
from most_queue.theory.fifo.mmnr import MMnrCalc
from most_queue.theory.priority.preemptive.mmn_2cls_pr_busy_approx import MMnPR2ClsBusyApprox
from most_queue.theory.utils.busy_periods import busy_calc
from most_queue.theory.utils.passage_time import PassageTimeCalculation, TransitionMatrices


class MM2BusyApprox3Classes(MGnCalc):
    """
    Calculation of busy periods for M/M/2 queue with three classes of requests and absolute priority
    using Cox distribution for passage time approximation.
    """

    def __init__(
        self,
        calc_params: TakahashiTakamiParams | None = None,
    ):
        """
        Init method for MM2BusyApprox3Classes class.
        :param calc_params: Parameters for calculation.
        """

        super().__init__(n=2, calc_params=calc_params)

        self.l_H = None
        self.l_M = None
        self.l_L = None
        self.mu_H = None
        self.mu_M = None
        self.mu_L = None

        self.dt = np.dtype(self.calc_params.dtype)
        self.N = self.calc_params.N
        self.e1 = self.calc_params.tolerance

        self.iter_num_ = 0

        self.busy_periods = []  # список из шести наборов начальных моментров ПНЗ B1, B2, ..., B6
        self.busy_periods_cvs = []  # коэффициенты вариации ПНЗ
        self.pp = []  # список из шести вероятностей p2mm, p2mh, phmm, phmh, p2hm, p2hh

        # массив cols хранит число столбцов для каждого яруса, удобней
        # рассчитать его один раз:
        self.cols = [] * self.N

        # переменные
        self.t = []
        self.b1 = []
        self.b2 = []
        self.x = [0.0] * self.N
        self.z = [0.0] * self.N

        # искомые вреоятности состояний СМО
        self.p = [0.0 + 0.0j] * self.N

        # матрицы переходов
        self.A = []
        self.B = []
        self.C = []
        self.D = []
        self.Y = []

        for i in range(self.N):
            self.cols.append(15)
            self.t.append(np.zeros((1, self.cols[i]), dtype=self.dt))
            self.b1.append(np.zeros((1, self.cols[i]), dtype=self.dt))
            self.b2.append(np.zeros((1, self.cols[i]), dtype=self.dt))
            self.x.append(np.zeros((1, self.cols[i]), dtype=self.dt))

        self.high_results = None
        self.med_results = None

    def set_sources(self, l_low: float, l_med: float, l_high: float):  # pylint: disable=arguments-differ
        """
        Set the arrival rates
        :param l_low: intensity of the arrivals with low priority,
        :param l_med: intensity of the arrivals with med priority,
        :param l_high: intensity of the arrivals with high priority,
        """
        self.l_L = l_low
        self.l_M = l_med
        self.l_H = l_high
        self.is_sources_set = True

    def set_servers(self, mu_low: float, mu_med: float, mu_high: float):  # pylint: disable=arguments-differ
        """
        Set the raw moments of service time distribution
        :param mu_low: intensity of service for low-priority requests,
        :param mu_med: intensity of service for med-priority requests,
        :param mu_high: intensity of service for high priority requests,

        """
        self.mu_H = mu_high
        self.mu_M = mu_med
        self.mu_L = mu_low

        self.is_servers_set = True

    def run(self) -> PriorityResults:
        """
        Запускает расчет. Не зависит от структуры матриц
        """

        start = time.process_time()

        self._calc_busy_periods()

        self._build_matrices()
        self._initial_probabilities()

        self.iter_num_ = 0
        self.b1[0][0, 0] = 0
        self.b2[0][0, 0] = 0
        x_max1 = 0
        x_max2 = 0

        for i in range(self.N):
            if self.x[i] > x_max1:
                x_max1 = self.x[i]
        while math.fabs(x_max2.real - x_max1.real) >= self.e1:
            x_max2 = x_max1
            self.iter_num_ += 1
            for j in range(1, self.N):  # по всем ярусам, кроме первого.

                G = np.linalg.inv(self.D[j] - self.C[j])
                # b':
                self.b1[j] = np.dot(self.t[j - 1], np.dot(self.A[j - 1], G))

                # b":
                if j != (self.N - 1):
                    self.b2[j] = np.dot(self.t[j + 1], np.dot(self.B[j + 1], G))
                else:
                    self.b2[j] = np.dot(self.t[j - 1], np.dot(self.B[j], G))

                c = self._calculate_c(j)

                x_znam = np.dot(c, self.b1[j]) + self.b2[j]
                self.x[j] = 0
                for k in range(x_znam.shape[1]):
                    self.x[j] += x_znam[0, k]

                self.x[j] = 1 / self.x[j]

                self.z[j] = np.dot(c, self.x[j])
                self.t[j] = np.dot(self.z[j], self.b1[j]) + np.dot(self.x[j], self.b2[j])

            self.x[0] = 1.0 / self.z[1]

            t1B1 = np.dot(self.t[1], self.B[1])
            self.t[0] = np.dot(self.x[0], t1B1)
            self.t[0] = np.dot(self.t[0], np.linalg.inv(self.D[0] - self.C[0]))

            x_max1 = 0
            for i in range(self.N):
                if self.x[i] > x_max1:
                    x_max1 = self.x[i]

        self._calculate_p()
        self._calculate_y()

        self.high_results = self._calc_high_queue()
        self.med_results = self._calc_med_queue()

        results = self.get_results()
        results.duration = time.process_time() - start
        return results

    def get_w(self, _derivate=False):
        """
        Calculate waiting time moments.

        Note: Currently returns waiting time moments for high and medium priority classes.
        Low priority class returns zero moments as placeholder.
        Full implementation for all priority classes is pending.
        """
        w_med = self.med_results.w[1]
        return [self.high_results.w, w_med, [0, 0, 0, 0]]

    def get_v(self) -> list[float]:
        """
        Calculate sojourn time moments
        """
        v1_low = self.get_low_class_v1()
        v_med = self.med_results.v[1]
        return [self.high_results.v, v_med, [v1_low, 0, 0, 0]]

    def get_utilization(self):

        b_sr = (1.0 / self.mu_H + 1.0 / self.mu_L + 1.0 / self.mu_M) / 3.0
        return (self.l_L + self.l_H + self.l_M) * b_sr / self.n

    def get_results(self) -> PriorityResults:
        """
        Collect all results
        """

        v = self.get_v()
        w = self.get_w()
        utilization = self.get_utilization()
        p = self.get_p()

        return PriorityResults(v=v, w=w, p=p, utilization=utilization)

    def get_low_class_v1(self):
        """
        Возвращает среднее время пребывания для класса L
        """
        l1 = 0
        p_real = self.get_p()
        for i in range(self.N):
            l1 += p_real[i] * i
        return l1 / self.l_L

    def _calculate_y(self):
        for i in range(self.N):
            self.Y.append(np.dot(self.p[i], self.t[i]))

    def _initial_probabilities(self):
        """
        Задаем первоначальные значения вероятностей микросостояний
        """
        # t задаем равновероятными
        for i in range(self.N):
            for j in range(self.cols[i]):
                self.t[i][0, j] = 1.0 / self.cols[i]
        self.x[0] = 0.4

    def _calculate_p(self):
        """
        После окончания итераций находим значения вероятностей p по найденным х
        """
        # version 1
        p_sum = 0 + 0j
        p0_max = 1.0
        p0_min = 0.0
        while math.fabs(1.0 - p_sum.real) > self.calc_params.tolerance:
            p0_ = (p0_max + p0_min) / 2.0
            p_sum = p0_
            self.p[0] = p0_
            for j in range(self.N - 1):
                self.p[j + 1] = np.dot(self.p[j], self.x[j])
                p_sum += self.p[j + 1]
            if p_sum > 1.0:
                p0_max = p0_
            else:
                p0_min = p0_

        self._norm_probs()

    def _calc_busy_periods(self):

        l_M = self.l_M
        l_H = self.l_H
        mu_H = self.mu_H
        mu_M = self.mu_M

        b_mom = [0, 0, 0]

        for j in range(3):
            b_mom[j] = math.factorial(j + 1) / math.pow(2 * mu_H, j + 1)

        busy = busy_calc(l_H, b_mom, 3)

        param_cox = CoxDistribution.get_params(busy)

        y1_cox = param_cox.p1
        mu1_cox = param_cox.mu1
        mu2_cox = param_cox.mu2

        t1 = mu1_cox * (1.0 - y1_cox)
        t12 = mu1_cox * y1_cox
        t2 = mu2_cox

        A = []
        A.append(np.array([[l_H, l_M]], dtype=self.dt))
        A.append(np.array([[l_H, 0, l_M, 0], [0, 0, l_H, l_M]], dtype=self.dt))
        A.append(
            np.array(
                [[l_M, 0, 0, 0], [0, l_M, 0, 0], [l_H, 0, l_M, 0], [0, 0, l_H, l_M]],
                dtype=self.dt,
            )
        )
        A.append(
            np.array(
                [[l_M, 0, 0, 0], [0, l_M, 0, 0], [l_H, 0, l_M, 0], [0, 0, l_H, l_M]],
                dtype=self.dt,
            )
        )

        B = []
        B.append(np.array([[0]], dtype=self.dt))
        B.append(np.array([[mu_H], [mu_M]], dtype=self.dt))
        B.append(np.array([[t1, 0], [t2, 0], [mu_M, mu_H], [0, 2 * mu_M]], dtype=self.dt))
        B.append(
            np.array(
                [[0, 0, t1, 0], [0, 0, t2, 0], [0, 0, mu_M, mu_H], [0, 0, 0, 2 * mu_M]],
                dtype=self.dt,
            )
        )

        C = []
        C.append(np.array([[0]], dtype=self.dt))
        C.append(np.array([[0, 0], [0, 0]], dtype=self.dt))
        C.append(
            np.array(
                [[0, t12, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                dtype=self.dt,
            )
        )
        C.append(
            np.array(
                [[0, t12, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                dtype=self.dt,
            )
        )

        D = []
        for i, c_matrix in enumerate(C):
            d_rows = c_matrix.shape[0]
            D.append(np.zeros((d_rows, d_rows), dtype=self.dt))

            for row in range(d_rows):
                a_sum = 0.0 + 0.0j
                a_cols = A[i].shape[1]
                for j in range(a_cols):
                    a_sum += A[i][row, j]
                b_sum = 0.0 + 0.0j
                b_cols = B[i].shape[1]
                for j in range(b_cols):
                    b_sum += B[i][row, j]
                c_sum = 0.0 + 0.0j
                c_cols = c_matrix.shape[1]
                for j in range(c_cols):
                    c_sum += c_matrix[row, j]
                D[i][row, row] = a_sum + b_sum + c_sum
        pass_time = PassageTimeCalculation(transition_matrices=TransitionMatrices(A, B, C, D))

        pass_time.calc()

        for j in range(6):
            self.busy_periods.append([0, 0, 0])
        for r in range(3):
            self.busy_periods[0][r] = pass_time.Gr[2][r][3, 1]
            self.busy_periods[1][r] = pass_time.Gr[2][r][3, 0]
            self.busy_periods[2][r] = pass_time.Gr[2][r][2, 1]
            self.busy_periods[3][r] = pass_time.Gr[2][r][2, 0]
            self.busy_periods[4][r] = pass_time.Gr[2][r][0, 1]
            self.busy_periods[5][r] = pass_time.Gr[2][r][0, 0]

        for j in range(6):
            cv = (
                math.sqrt(self.busy_periods[j][1].real - pow(self.busy_periods[j][0].real, 2))
                / self.busy_periods[j][0].real
            )
            self.busy_periods_cvs.append(cv.real)

        # pp - список из шести вероятностей p2mm, p2mh, phmm, phmh, p2hm, p2hh
        # берем моменты Gr, поскольку моменты Z - условные, с учетом pp
        self.pp.append(pass_time.G[2][3, 1])
        self.pp.append(pass_time.G[2][3, 0])
        self.pp.append(pass_time.G[2][2, 1])
        self.pp.append(pass_time.G[2][2, 0])
        self.pp.append(pass_time.G[2][0, 1])
        self.pp.append(pass_time.G[2][0, 0])

    def _calc_high_queue(self) -> QueueResults:

        mgn_high = MMnrCalc(n=self.n, r=300)
        mgn_high.set_sources(l=self.l_H)
        mgn_high.set_servers(mu=self.mu_H)

        return mgn_high.run()

    def _calc_med_queue(self) -> PriorityResults:

        mgn_high = MMnPR2ClsBusyApprox(n=self.n, calc_params=self.calc_params)
        mgn_high.set_sources(l_low=self.l_M, l_high=self.l_H)
        mgn_high.set_servers(mu_low=self.mu_L, mu_high=self.mu_H)

        return mgn_high.run()

    def _build_matrices(self):
        """
        Формирует матрицы переходов
        """
        for i in range(self.N):
            self.A.append(self._build_big_a(i))
            self.B.append(self._build_big_b(i))
            self.C.append(self._build_big_c(i))
            self.D.append(self._build_big_d(i))

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

    def _build_big_a(self, num):
        """
        Формирует матрицу А(L) по заданному номеру яруса
        """
        col = self.cols[num]
        row = self.cols[num]

        output = np.zeros((row, col), dtype=self.dt)
        if num > 1:
            output = self.A[0]
            return output
        for i in range(row):
            output[i, i] = self.l_L
        return output

    def _build_big_b(self, num):
        """
        Формирует матрицу B по заданному номеру яруса
        """
        if num == 0:
            return np.zeros((1, 1), dtype=self.dt)

        col = self.cols[num]
        row = self.cols[num]
        output = np.zeros((row, col), dtype=self.dt)
        if num > 2:
            output = self.B[2]
            return output

        # Матрица B - диагональная. Количество ненулевых элементов = 2.
        # Остальные 13 - нулевые
        output[0, 0] = min(2, num) * self.mu_L
        output[1, 1] = self.mu_L

        return output

    def _build_big_c(self, num):
        """
        Формирует матрицу C по заданному номеру яруса
        """
        col = self.cols[num]
        row = col

        output = np.zeros((row, col), dtype=self.dt)
        if num > 1:
            output = self.C[0]
            return output

        lm = self.l_M
        lh = self.l_H
        mum = self.mu_M
        muh = self.mu_H

        # pp список из шести вероятностей p2mm, p2mh, phmm, phmh, p2hm, p2hh
        p2mm = self.pp[0]
        p2mh = self.pp[1]
        phmm = self.pp[2]
        phmh = self.pp[3]
        p2hm = self.pp[4]
        p2hh = self.pp[5]

        # first quad
        output[0, 1] = lm
        output[0, 2] = lh
        output[1, 0] = mum
        output[2, 0] = muh

        # quad 0, 1
        output[1, 3] = lm * p2mm

        # quad 0, 2
        output[1, 5] = lm * p2mh

        # quad 0, 3
        output[1, 7] = lh * phmm  # or lm - in paper
        output[2, 7] = lm * phmm  # or lh - in paper

        # quad 0, 4
        output[1, 9] = lh * phmh  # or lm - in paper
        output[2, 9] = lm * phmh  # or lh- in paper

        # quad 0, 5
        output[2, 11] = lh * p2hm

        # quad 0, 6
        output[2, 13] = lh * p2hh

        t1 = []
        t2 = []
        t12 = []

        for i in range(6):
            cox_param = CoxDistribution.get_params(self.busy_periods[i])
            y1 = cox_param.p1
            mu1 = cox_param.mu1
            mu2 = cox_param.mu2

            t1.append(mu1 * (1.0 - y1))
            t12.append(mu1 * y1)
            t2.append(mu2)

        # left t's blocks
        row = 3
        for i in range(6):
            if (i + 1) % 2 == 0:
                output[row, 2] = t1[i]
                output[row + 1, 2] = t2[i]
            else:
                output[row, 1] = t1[i]
                output[row + 1, 1] = t2[i]
            row += 2

        # T sections
        row = 3
        for i in range(6):
            output[row, row + 1] = t12[i]
            row += 2

        return output

    def _build_big_d(self, num):
        """
        Формирует матрицу D по заданному номеру яруса
        """
        col = self.cols[num]
        row = col
        output = np.zeros((row, col), dtype=self.dt)

        if num > 3:
            output = self.D[3]
            return output

        for i in range(row):
            sumA = 0.0 + 0.0j
            sumB = 0.0 + 0.0j
            sumC = 0.0 + 0.0j

            for j in range(self.cols[num]):
                sumA += self.A[num][i, j]

            if num != 0:
                for j in range(self.cols[num]):
                    sumB += self.B[num][i, j]

            for j in range(self.cols[num]):
                sumC += self.C[num][i, j]

            output[i, i] = sumA + sumB + sumC

        return output
