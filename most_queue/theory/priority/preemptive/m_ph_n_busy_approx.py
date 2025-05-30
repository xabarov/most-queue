"""
Calculation of M/PH, M/n queue with two classes of requests and absolute priority
using the Takahashi-Takagi numerical method based on the approximation 
of the busy-time distribution by a Cox second-order distribution.
"""
import math

import numpy as np

from most_queue.rand_distribution import CoxDistribution, H2Distribution
from most_queue.theory.fifo.mgn_takahasi import MGnCalc
from most_queue.theory.utils.passage_time import PassageTimeCalculation


class MPhNPrty(MGnCalc):
    """
    Calculation of M/PH, M/n queue with two classes of requests and absolute priority
    using the Takahashi-Takagi numerical method based on the approximation 
    of the busy-time distribution by a Cox second-order distribution.
    """

    def __init__(self, mu_L: float, b_high: list[float], l_L: float, l_H: float, n: int, N: int = 250,
                 accuracy: float = 1e-8, max_iter: int = 300, is_cox: bool = True,
                 approx_ee: float = 0.1, approx_e: float = 0.5, is_fitting: bool = True,
                 buffer=None, verbose: bool = True, dtype="c16"):
        """
        Calculation of M/PH, M/n queue with two classes of requests and absolute priority
        based on the Takahashi-Takagi numerical method based on the approximation 
        of the busy-time distribution by a Cox second-order distribution.

        :param l_L: intensity of the arrivals with low priority,
        :param l_H: intensity of the arrivals with high priority,
        :param mu_L: intensity of service for low-priority requests,
        :param b_high: list of E[X^k], k =1,2,3 for service time for high priority jobs
        :param N: number of levels (stages)
        :param accuracy: accuracy parameter for stopping the iteration
        :param max_iter: maximum number of iterations
        :param is_cox: if True, use Cox2 distribution for approximating service time 
            of low-priority jobs, otherwise use H2 distribution.
        :param approx_ee, approx_e: approximation paramters 
        """

        super().__init__(n=n, l=0, b=b_high, buffer=buffer,
                         N=N, accuracy=accuracy, dtype=dtype, verbose=verbose)

        self.l_L = l_L
        self.l_H = l_H
        self.mu_L = mu_L

        cox_param_H = CoxDistribution.get_params(b_high)

        self.mu1_H = cox_param_H.mu1
        self.mu2_H = cox_param_H.mu2
        self.p_H = cox_param_H.p1
        self.max_iter = max_iter
        self.is_cox = is_cox
        self.approx_ee = approx_ee
        self.approx_e = approx_e
        self.is_fitting = is_fitting

        self.busy_periods = []  # list of busy periods initial moments
        self.busy_periods_coevs = []  # list of busy periods coefficients of variation
        self.alphas = []
        self.pp = []  # list of transition probabilities to the busy periods states

        self._calc_busy_periods()

        self.cols = [] * N

        self.run_iterations_num_ = 0
        self.p_iteration_num_ = 0
        self.cols_length_ = 0

    def get_low_class_v1(self) -> float:
        """
        Get average time spent by class L (with low priority) in the system.
        Returns:
            float: Average time spent by class L (with low priority) in the system.
        """
        l1 = 0
        p_real = self.get_p()
        for i in range(self.N):
            l1 += p_real[i] * i
        return l1 / self.l_L

    def _fill_cols(self):
        # Cols depends on number of channels. cols = number of all microstates transitions
        # from Cox2 to level
        self.cols_length_ = 2 * self.n ** 2
        for i in range(self.n):
            self.cols_length_ += i + 1

    def _fill_t_b(self):
        for _i in range(self.N):
            self.t.append(np.zeros((1, self.cols_length_), dtype=self.dt))
            self.b1.append(np.zeros((1, self.cols_length_), dtype=self.dt))
            self.b2.append(np.zeros((1, self.cols_length_), dtype=self.dt))

    def _build_big_a_for_busy_periods(self, num):
        """
        Forms the matrix A at level num for calculating the Busy Period at a given level number.
        :param num: level number
        :return: matrix A(num)
        """
        if num < self.n:
            col = num + 2
            row = num + 1
        elif num == self.n:
            col = num + 1
            row = num + 1
        else:
            output = self.big_a_for_busy[self.n]
            return output

        output = np.zeros((row, col), dtype=self.dt)

        for i in range(row):
            output[i, i] = self.l_H
        return output

    def _build_big_b_for_busy_periods(self, num):
        """
        Forms the matrix B at level num for calculating the Busy Period at a given level number.
        :param num: level number
        :return: matrix B(num)
        """
        if num == 0:
            output = np.zeros((1, 1), dtype=self.dt)
            return output
        elif num <= self.n:
            col = num
            row = num + 1
        elif num == self.n + 1:
            col = num
            row = num
            output = np.zeros((row, col), dtype=self.dt)
            output[:, :self.n] = self.big_b_for_busy[self.n]
            return output
        else:
            output = self.big_b_for_busy[self.n + 1]
            return output

        output = np.zeros((row, col), dtype=self.dt)

        mu1 = self.mu1_H
        mu2 = self.mu2_H
        qH = 1.0 - self.p_H

        for i in range(col):
            output[i, i] = (num - i) * mu1 * qH
            output[i + 1, i] = (i + 1) * mu2
        return output

    def _build_big_c_for_busy_periods(self, num):
        """
        Forms the matrix C at level num for calculating the Busy Period at a given level number.
        :param num: level number
        :return: matrix C(num)
        """
        if num == 0:
            output = np.zeros((1, 1), dtype=self.dt)
            return output
        elif num <= self.n:
            col = num + 1
            row = num + 1
        else:
            output = self.big_c_for_busy[self.n]
            return output

        output = np.zeros((row, col), dtype=self.dt)

        mu1 = self.mu1_H
        pH = self.p_H

        for i in range(col - 1):
            output[i, i + 1] = (num - i) * mu1 * pH
        return output

    def _build_big_d_for_busy_periods(self, num):
        """
        Forms the matrix D at level num for calculating the Busy Period at a given level number.
        :param num: level number
        :return: matrix D(num)
        """
        if num <= self.n:
            col = num + 1
            row = num + 1
        else:
            output = self.big_d_for_busy[self.n]
            return output

        output = np.zeros((row, col), dtype=self.dt)

        for i in range(row):
            sumA = 0.0 + 0.0j
            sumB = 0.0 + 0.0j
            sumC = 0.0 + 0.0j

            for j in range(col):
                sumA += self.big_a_for_busy[num][i, j]

            if num != 0:
                for j in range(col - 1):
                    sumB += self.big_b_for_busy[num][i, j]

            for j in range(col):
                sumC += self.big_c_for_busy[num][i, j]

            output[i, i] = sumA + sumB + sumC

        return output

    def _calc_busy_periods(self):
        """
        Calculate the busy periods for all levels.
        """

        self.big_a_for_busy = []
        self.big_b_for_busy = []
        self.big_c_for_busy = []
        self.big_d_for_busy = []

        for i in range(self.n + 2):
            self.big_a_for_busy.append(self._build_big_a_for_busy_periods(i))
            self.big_b_for_busy.append(self._build_big_b_for_busy_periods(i))
            self.big_c_for_busy.append(self._build_big_c_for_busy_periods(i))
            self.big_d_for_busy.append(self._build_big_d_for_busy_periods(i))

        pass_time = PassageTimeCalculation(self.big_a_for_busy, self.big_b_for_busy,
                                           self.big_c_for_busy, self.big_d_for_busy, is_clx=True, is_verbose=self.verbose)
        pass_time.calc()

        self.pnz_num_ = self.n ** 2

        for j in range(self.pnz_num_):
            self.busy_periods.append([0, 0, 0])

        num = 0
        for i in range(self.n):
            for j in range(self.n):
                for r in range(3):
                    self.busy_periods[num][r] = pass_time.Z[self.n][r][i, j]
                num = num + 1

        for j in range(self.pnz_num_):
            under_sqrt = self.busy_periods[j][1] - self.busy_periods[j][0] ** 2
            if under_sqrt > 0:
                coev = math.sqrt(under_sqrt.real)
                self.alphas.append(1 / (coev ** 2))
                self.busy_periods_coevs.append(coev / self.busy_periods[j][0])
            else:
                self.busy_periods_coevs.append(math.inf)

        if self.verbose:
            print("\nBusy periods:\n")
            for j in range(self.pnz_num_):
                for r in range(3):
                    if math.isclose(self.busy_periods[j][r].imag, 0):
                        print("{0:^8.3g}".format(
                            self.busy_periods[j][r].real), end=" ")
                    else:
                        print("{0:^8.3g}".format(
                            self.busy_periods[j][r]), end=" ")
                if math.isclose(self.busy_periods_coevs[j].imag, 0):
                    print("coev = {0:^4.3g}".format(
                        self.busy_periods_coevs[j].real))
                else:
                    print("coev = {0:^4.3g}".format(
                        self.busy_periods_coevs[j]))

        # pp - список из n**2 вероятностей переходов
        for j in range(self.pnz_num_):
            num = 0
            for i in range(self.n):
                for j in range(self.n):
                    self.pp.append(pass_time.G[self.n][i, j])
                    num = num + 1
        if self.verbose:
            print("\nTransition probabilities of busy periods:\n")
            for j in range(self.pnz_num_):
                if math.isclose(self.pp[j].imag, 0):
                    print("{0:^8.3g}".format(self.pp[j].real), end=" ")
                else:
                    print("{0:^8.3g}".format(self.pp[j]), end=" ")

    def _initial_probabilities(self):
        """
        Set initial values of microstate probabilities.
        """
        # t задаем равновероятными
        for i in range(self.N):
            for j in range(self.cols_length_):
                self.t[i][0, j] = 1.0 / self.cols_length_
        self.x[0] = 0.4

    def _calculate_p(self):
        """
        After the iterations are completed, we find the values of probabilities p by the found x.
        """
        # version 1
        p_sum = 0 + 0j
        p0_max = 1.0
        p0_min = 0.0
        iter_num = 0
        self.p_iteration_num_ = 0
        while math.fabs(1.0 - p_sum.real) > 1e-6 and iter_num < self.max_iter:
            iter_num += 1
            p0_ = (p0_max + p0_min) / 2.0
            p_sum = p0_
            self.p[0] = p0_
            for j in range(self.N - 1):
                self.p[j + 1] = np.dot(self.p[j], self.x[j])
                p_sum += self.p[j + 1]
            if p_sum.real > 1.0:
                p0_max = p0_
            else:
                p0_min = p0_
        self.p_iteration_num_ = iter

    def _build_big_a_matrix(self, num: int):
        """
        Forms the matrix A(L) at a given level number
        :param: num - level number
        return: matrix A(L) at level number 'num'
        """
        col = self.cols_length_
        row = self.cols_length_

        output = np.zeros((row, col), dtype=self.dt)
        if num > 1:
            output = self.A[0]
            return output
        for i in range(row):
            output[i, i] = self.l_L
        return output

    def _build_big_b_matrix(self, num):
        """
        Forms the matrix B(L) at a given level number
        :param: num - level number
        return: matrix B(L) at level number 'num'
        """
        if num == 0:
            return np.zeros((1, 1), dtype=self.dt)

        col = self.cols_length_
        row = self.cols_length_
        output = np.zeros((row, col), dtype=self.dt)

        if num > self.n:
            output = self.B[self.n]
            return output

        # Матрица B - диагональная. Количество ненулевых элементов зависит от числа n.
        lev = 0
        for i in range(self.n):  # количество ярусов до перехода в ПНЗ
            for _ in range(i + 1):  # количество микросочтояний в этом ярусе
                output[lev, lev] = min(self.n - i, num) * self.mu_L
                lev = lev + 1

        return output

    def _build_big_c_matrix(self, num):
        """
        Forms the matrix C(L) at a given level number
        :param: num - level number
        return: matrix C(L) at level number 'num'
        """
        col = self.cols_length_
        row = col

        output = np.zeros((row, col), dtype=self.dt)
        if num >= 1:
            output = self.C[0]
            return output

        lh = self.l_H

        y1_mass = []
        m1_mass = []
        m2_mass = []
        print("\n")

        for i in range(self.pnz_num_):
            if not self.is_cox:
                h2_param = H2Distribution.get_params_clx(
                    self.busy_periods[i], ee=self.approx_ee, e=self.approx_e, is_fitting=self.is_fitting, verbose=self.verbose)
                # h2_param = H2Distribution.get_params(self.busy_periods[i])
                y1_mass.append(h2_param.p1)
                m1_mass.append(h2_param.mu1)
                m2_mass.append(h2_param.mu2)
                if self.verbose:
                    print("Params for B{0}: {1:3.3f}, {2:3.3f}, {3:3.3f}".format(i + 1, h2_param.p1, h2_param.mu1,
                                                                                 h2_param.mu2))
            else:
                cox_params = CoxDistribution.get_params(
                    self.busy_periods[i], ee=self.approx_ee, e=self.approx_e, is_fitting=self.is_fitting, verbose=self.verbose)
                y1_mass.append(cox_params.p1)
                m1_mass.append(cox_params.mu1)
                m2_mass.append(cox_params.mu2)
                if self.verbose:
                    print("Params for B{0}: {1:3.3f}, {2:3.3f}, {3:3.3f}".format(i + 1, cox_params.p1, cox_params.mu1,
                                                                                 cox_params.mu2))

        # first quad

        # lH's = A_for_busy. Но поскольку = l_h, проще здесь посчитать заново
        l_start = 0
        for i in range(self.n - 1):
            for j in range(i + 1):
                output[l_start + j, l_start + j + i + 1] = lh
            l_start += i + 1

        # B_for_busy
        l_start = 1
        l_end = 0
        for i in range(1, self.n):
            rows = self.big_b_for_busy[i].shape[0]
            cols = self.big_b_for_busy[i].shape[1]
            for r in range(rows):
                for c in range(cols):
                    output[l_start + r, l_end + c] = self.big_b_for_busy[i][r, c]
            l_start += i + 1
            l_end += i

        # C_for_busy
        l_start = 1
        for i in range(1, self.n):
            rows = self.big_c_for_busy[i].shape[0]
            cols = self.big_c_for_busy[i].shape[1]
            for r in range(rows):
                for c in range(cols):
                    output[l_start + r, l_start + c] = self.big_c_for_busy[i][r, c]
            l_start += i + 1

        l_start = 0
        for i in range(self.n - 1):
            l_start += i + 1

        l_end = l_start + self.n
        num = 0
        for i in range(self.n):
            for j in range(self.n):
                if not self.is_cox:
                    output[l_start + i, l_end] = lh * \
                        y1_mass[num] * self.pp[num]
                    output[l_start + i, l_end + 1] = lh * \
                        (1 - y1_mass[num]) * self.pp[num]
                else:
                    output[l_start + i, l_end] = lh * self.pp[num]
                    output[l_start + i, l_end + 1] = 0
                l_end += 2
                num += 1

        # left t's blocks
        row = 0
        for i in range(self.n):
            row += i + 1

        col = 0
        for i in range(self.n - 1):
            col += i + 1
        num = 0
        for i in range(self.n):
            for j in range(self.n):
                if not self.is_cox:
                    output[row, col + j] = m1_mass[num]
                    output[row + 1, col + j] = m2_mass[num]
                else:
                    output[row, col + j] = m1_mass[num] * (1.0 - y1_mass[num])
                    output[row + 1, col + j] = m2_mass[num]
                row += 2
                num += 1

        if self.is_cox:
            # central T's block
            row = 0
            for i in range(self.n):
                row += i + 1

            col = 0
            for i in range(self.n):
                col += i + 1
            num = 0
            for i in range(self.n):
                for j in range(self.n):
                    output[row, col + 1] = m1_mass[num] * y1_mass[num]
                    num += 1
                    row += 2
                    col += 2
        return output

    def _build_big_d_matrix(self, num):
        """
        Forms the matrix D(L) at a given level number
        :param: num - level number
        return: matrix D(L) at level number 'num'
        """
        col = self.cols_length_
        row = col
        output = np.zeros((row, col), dtype=self.dt)

        if num > self.n:
            output = self.D[self.n]
            return output

        for i in range(row):
            sumA = 0.0 + 0.0j
            sumB = 0.0 + 0.0j
            sumC = 0.0 + 0.0j

            for j in range(self.cols_length_):
                sumA += self.A[num][i, j]

            if num != 0:
                for j in range(self.cols_length_):
                    sumB += self.B[num][i, j]

            for j in range(self.cols_length_):
                sumC += self.C[num][i, j]

            output[i, i] = sumA + sumB + sumC

        return output
