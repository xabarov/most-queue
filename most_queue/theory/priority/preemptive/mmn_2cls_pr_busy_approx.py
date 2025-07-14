"""
Calculation of M/M/n queue with absolute priority using the numerical method
by Takahashi-Takami based on the approximation of busy periods by Cox's second-order distribution.
"""

import math

import numpy as np

from most_queue.rand_distribution import CoxDistribution
from most_queue.theory.calc_params import TakahashiTakamiParams
from most_queue.theory.utils.passage_time import PassageTimeCalculation


class MMn_PRTY_PNZ_Cox_approx:
    """
    Calculation of M/M/n queue with absolute priority using the numerical method
    by Takahashi-Takami based on the approximation
    of busy periods by Cox's second-order distribution.
    """

    def __init__(
        self,
        n: int,
        mu_L: float,
        mu_H: float,
        l_L: float,
        l_H: float,
        calc_params: TakahashiTakamiParams | None = None,
    ):
        """
        n: number of channels
        l_L, l_H: arrival rates for low and high priority requests
        mu_L, mu_H: service rates for low and high priority requests
        N: number of tiers (levels)
        accuracy: accuracy, parameter for stopping iteration
        """

        if calc_params is None:
            calc_params = TakahashiTakamiParams()
        self.dt = np.dtype(calc_params.dtype)
        self.N = calc_params.N
        self.e1 = calc_params.accuracy
        self.n = n
        self.l_L = l_L
        self.l_H = l_H
        self.mu_L = mu_L
        self.mu_H = mu_H
        self.busy_period = self._get_pnz_markov()
        self.busy_coev = self._get_busy_coev()
        self.param_cox = CoxDistribution.get_params(self.busy_period)
        self.y1_cox = self.param_cox.p1
        self.mu1_cox = self.param_cox.mu1
        self.mu2_cox = self.param_cox.mu2

        self.n_iter_ = 0
        self.inter_level_mom_ = None

        # массив cols хранит число столбцов для каждого яруса, удобней рассчитать его один раз:
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
            self.cols.append(n + 2)
            self.t.append(np.zeros((1, self.cols[i]), dtype=self.dt))
            self.b1.append(np.zeros((1, self.cols[i]), dtype=self.dt))
            self.b2.append(np.zeros((1, self.cols[i]), dtype=self.dt))
            self.x.append(np.zeros((1, self.cols[i]), dtype=self.dt))

        self._build_matrices()
        self._initial_probabilities()

    def _get_pnz_markov(self):

        b_mom = [0, 0, 0]

        for j in range(3):
            b_mom[j] = math.factorial(j + 1) / math.pow(self.n * self.mu_H, j + 1)

        pi = [0, 0, 0]

        ro_load = self.l_H * b_mom[0]

        pi[0] = b_mom[0] / (1.0 - ro_load)
        pi[1] = b_mom[1] / (math.pow(1 - ro_load, 3))
        pi[2] = (b_mom[2] / math.pow(1.0 - ro_load, 4)) + 3.0 * self.l_H * b_mom[1] * b_mom[
            1
        ] / math.pow(1 - ro_load, 5)

        return pi

    def _get_busy_coev(self):
        return (
            math.sqrt(self.busy_period[1] - self.busy_period[0] * self.busy_period[0])
            / self.busy_period[0]
        )

    def get_p(self):
        """
        Возвращает список с вероятностями состояний системы
        p[k] - вероятность пребывания в системе ровно k заявок
        """
        p_real = [p_val.real for p_val in self.p]
        return p_real

    def get_second_class_v1(self):
        """
        Возвращает среднее время пребывания для второго класса
        """
        l1 = 0
        p_real = self.get_p()
        for i in range(self.N):
            l1 += p_real[i] * i
        return l1 / self.l_L

    def _initial_probabilities(self):
        """
        Задаем первоначальные значения вероятностей микросостояний
        """
        # t задаем равновероятными
        for i in range(self.N):
            for j in range(self.cols[i]):
                self.t[i][0, j] = 1.0 / self.cols[i]
        self.x[0] = 0.4

    def calculate_p(self):
        """
        После окончания итераций находим значения вероятностей p по найденным х
        """
        # version 1
        p_sum = 0 + 0j
        p0_max = 1.0
        p0_min = 0.0
        while math.fabs(1.0 - p_sum.real) > 1e-6:
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

    def _calculate_y(self):
        for i in range(self.N):
            self.Y.append(np.dot(self.p[i], self.t[i]))

    def _build_matrices(self):
        """
        Формирует матрицы переходов
        """
        for i in range(self.N):
            self.A.append(self.buildA(i))
            self.B.append(self.buildB(i))
            self.C.append(self.buildC(i))
            self.D.append(self.buildD(i))

    def run(self):
        """
        Запускает расчет
        """
        self.b1[0][0, 0] = 0
        self.b2[0][0, 0] = 0
        x_max1 = 0
        x_max2 = 0
        self.n_iter_ = 0
        for i in range(self.N):
            if self.x[i] > x_max1:
                x_max1 = self.x[i]
        while math.fabs(x_max2.real - x_max1.real) >= self.e1:
            x_max2 = x_max1
            self.n_iter_ += 1
            for j in range(1, self.N):  # по всем ярусам, кроме первого.

                G = np.linalg.inv(self.D[j] - self.C[j])
                # b':
                self.b1[j] = np.dot(self.t[j - 1], np.dot(self.A[j - 1], G))

                # b":
                if j != (self.N - 1):
                    self.b2[j] = np.dot(self.t[j + 1], np.dot(self.B[j + 1], G))
                else:
                    self.b2[j] = np.dot(self.t[j - 1], np.dot(self.B[j], G))

                c = self.calculate_c(j)

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

        self.calculate_p()
        self._calculate_y()

    def _calculate_v(self):
        v = [0, 0, 0]
        A = []
        B = []
        C = []
        D = []
        for num in range(self.n + 1):
            # build A
            col = self.cols[num]
            row = self.cols[num]
            output = np.zeros((row, col), dtype=self.dt)
            if num > 0:
                output = A[0]
            # for i in range(row):
            #     output[i, i] = self.l_L
            A.append(output)

            # build B
            output = np.zeros((row, col), dtype=self.dt)
            if num != 0:
                if num > self.n:
                    output = B[self.n]
                # Матрица B - диагональная.
                # Количество ненулевых элементов = n (все, кроме состояний с ПНЗ)
                else:
                    for i in range(self.n):
                        output[i, i] = min(num, self.n - i) * self.mu_L
            B.append(output)

            # build C
            output = np.zeros((row, col), dtype=self.dt)
            if num > 0:
                output = C[0]
            else:
                for i in range(self.n):
                    output[i, i + 1] = self.l_H

                # Mu_H section. Количество таких элементов = n - 1.

                for i in range(self.n - 1):
                    output[i + 1, i] = self.mu_H * (i + 1)
                # ППНЗ секция: Структура одинаковая. Нужно только определить
                # позицию.Первый элемент - (n, n - 1)

                output[self.n, self.n - 1] = self.mu1_cox * (1.0 - self.y1_cox)
                output[self.n, self.n + 1] = self.mu1_cox * self.y1_cox
                output[self.n + 1, self.n - 1] = self.mu2_cox
            C.append(output)

            output = np.zeros((row, col), dtype=self.dt)

            if num > self.n:
                output = D[self.n]
            else:
                for i in range(row):
                    sumA = 0
                    sumB = 0
                    sumC = 0

                    for j in range(self.cols[num]):
                        sumA += A[num][i, j]

                    if num != 0:
                        for j in range(self.cols[num]):
                            sumB += B[num][i, j]

                    for j in range(self.cols[num]):
                        sumC += C[num][i, j]

                    output[i, i] = sumA + sumB + sumC
            D.append(output)

        pass_time = PassageTimeCalculation(A, B, C, D, is_clx=True)
        pass_time.calc()

        # Z_gap = pass_time.Z_gap_calc(10,0)
        l_start = 10
        l_end = 0
        G_gap = pass_time.G_gap_calc(l_start, l_end)
        Gr_gap = pass_time.Gr_gap_calc(l_start, l_end)

        print(f"\nGr_gap {l_start:d} -> {l_end:d}:\n")

        for r in range(3):
            print(f"r = {r + 1:^1d}")
            rows = Gr_gap[r].shape[0]
            cols = Gr_gap[r].shape[1]
            for j in range(rows):
                for t in range(cols):
                    if t == cols - 1:
                        if math.isclose(Gr_gap[r][j, t].imag, 0):
                            print(f"{Gr_gap[r][j, t].real:^5.3g}  ")
                        else:
                            print(f"{Gr_gap[r][j, t]:^5.3g}  ")
                    else:
                        if math.isclose(Gr_gap[r][j, t].imag, 0):
                            print(f"{Gr_gap[r][j, t].real:^5.3g}  ", end="")
                        else:
                            print(f"{Gr_gap[r][j, t]:^5.3g}  ", end="")

        print(f"\nG_gap {l_start:d} -> {l_end:d}:\n")

        rows = G_gap.shape[0]
        cols = G_gap.shape[1]
        for j in range(rows):
            for t in range(cols):
                if t == cols - 1:
                    if math.isclose(G_gap[j, t].imag, 0):
                        print(f"{G_gap[j, t].real:^5.3g}")
                    else:
                        print(f"{G_gap[j, t]:^5.3g}")
                else:
                    if math.isclose(G_gap[j, t].imag, 0):
                        print(f"{G_gap[j, t].real:^5.3g}", end="")
                    else:
                        print(f"{G_gap[j, t]:^5.3g}", end="")

        l_tilda = self.n
        # b_mom = [1 / self.mu_L, 2 / pow(self.mu_L, 2), 6 / pow(self.mu_L, 3)]

        # вычислим начальные моменты переходов с яруса на ярус
        # с учетом вероятностей микросостояний внутри яруса
        inter_level_mom = []
        for i in range(1, self.N):
            inter_level_mom.append([0, 0, 0])
            if i < l_tilda:
                for j in range(self.cols[i]):
                    for s in range(self.cols[i - 1]):
                        Zs = []
                        for k in range(3):
                            Zs.append(
                                self.t[i][0, j] * pass_time.G[i][j, s] * pass_time.Z[i][k][j, s]
                            )
                            # Zs.append(self.t[i][0, j] * pass_time.Z[i][k][j, s])
                            # Zs.append(self.t[i][0, j] * pass_time.Gr[i][k][j, s])
                        inter_level_mom[i - 1] = self.binom_calc(inter_level_mom[i - 1], Zs)
            else:
                for j in range(self.cols[i]):
                    for s in range(self.cols[i - 1]):
                        Zs = []
                        for k in range(3):
                            Zs.append(
                                self.t[i][0, j]
                                * pass_time.G[l_tilda][j, s]
                                * pass_time.Z[l_tilda][k][j, s]
                            )
                            # Zs.append(self.t[i][0, j] * pass_time.Gr[l_tilda][k][j, s])
                            # Zs.append(self.t[i][0, j] * pass_time.Z[l_tilda][k][j, s])
                        inter_level_mom[i - 1] = self.binom_calc(inter_level_mom[i - 1], Zs)

        self.inter_level_mom_ = inter_level_mom

        v1_pr = 0
        for i in range(1, self.N):
            # Z_gap = pass_time.Z_gap_calc(i, 0)
            Gr_gap = pass_time.Gr_gap_calc(i, 0)
            # G_gap = pass_time.G_gap_calc(i, 0)

            for j in range(self.cols[i]):
                # добавим время перехода из микросостояния [i, j] на ярус выше

                for t in range(self.cols[0]):
                    Zs = []
                    for l in range(3):
                        # Zs.append(self.Y[i][0, j] * G_gap[j, t] * Z_gap[l][j, t])
                        # Zs.append(self.Y[i][0, j] * G_gap[j, t] * Gr_gap[l][j, t])
                        Zs.append(self.Y[i][0, j] * Gr_gap[l][j, t])
                    # Zs.append(self.Y[i][0, j] * Z_gap[l][j, k])
                    v = self.binom_calc(v, Zs)
            v1_new = v[0]
            if math.fabs(v1_new.real - v1_pr.real) < 1e-8:
                print(f"End v calc iteration on step = {i:d}")
                break
            v1_pr = v1_new

        return v

    @staticmethod
    def binom_calc(a, b, num=3):
        """
        Calculate binomial coefficients for given vectors.
        """
        res = []
        if num > 0:
            res.append(a[0] + b[0])
        if num > 1:
            res.append(a[1] + 2 * a[0] * b[0] + b[1])
        if num > 2:
            res.append(a[2] + 3 * a[1] * b[0] + 3 * b[1] * a[0] + b[2])
        return res

    def calculate_c(self, j):
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

    def buildA(self, num):
        """
        Формирует матрицу А по заданному номеру яруса
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

    def buildB(self, num):
        """
        Формирует матрицу B по заданному номеру яруса
        """
        if num == 0:
            return np.zeros((1, 1), dtype=self.dt)

        col = self.cols[num]
        row = self.cols[num]
        output = np.zeros((row, col), dtype=self.dt)
        if num > self.n + 1:
            output = self.B[self.n + 1]
            return output
        # Матрица B - диагональная. Количество ненулевых элементов = n (все, кроме состояний с ПНЗ)
        for i in range(self.n):
            output[i, i] = min(num, self.n - i) * self.mu_L
        return output

    def buildC(self, num):
        """
        Формирует матрицу C по заданному номеру яруса
        """
        col = self.cols[num]
        row = col

        output = np.zeros((row, col), dtype=self.dt)
        if num > self.n:
            output = self.C[self.n]
            return output
        # COX 2:
        # l_H section: Количество переходов заявок 1 - го класса = n
        for i in range(self.n):
            output[i, i + 1] = self.l_H

        # Mu_H section. Количество таких элементов = n - 1.

        for i in range(self.n - 1):
            output[i + 1, i] = self.mu_H * (i + 1)
        # ППНЗ секция: Структура одинаковая. Нужно только определить
        # позицию.Первый элемент - (n, n - 1)

        output[self.n, self.n - 1] = self.mu1_cox * (1.0 - self.y1_cox)
        output[self.n, self.n + 1] = self.mu1_cox * self.y1_cox
        output[self.n + 1, self.n - 1] = self.mu2_cox

        # H2:
        # output[0, 1] = self.l_H
        # output[1, 0] = self.mu_H
        # output[1, 2] = self.l_H * self.y1_H2
        # output[1, 3] = self.l_H * (1.0 - self.y1_H2)
        # output[2, 1] = self.Mu1_H2
        # output[3, 1] = self.Mu2_H2

        return output

    def buildD(self, num):
        """
        Формирует матрицу D по заданному номеру яруса
        """
        col = self.cols[num]
        row = col
        output = np.zeros((row, col), dtype=self.dt)

        if num > self.n:
            output = self.D[self.n]
            return output

        for i in range(row):
            sumA = 0
            sumB = 0
            sumC = 0

            for j in range(self.cols[num]):
                sumA += self.A[num][i, j]

            if num != 0:
                for j in range(self.cols[num]):
                    sumB += self.B[num][i, j]

            for j in range(self.cols[num]):
                sumC += self.C[num][i, j]

            output[i, i] = sumA + sumB + sumC

        return output
