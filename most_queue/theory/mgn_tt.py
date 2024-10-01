import math

import numpy as np

from most_queue.rand_distribution import H2_dist


class MGnCalc:
    """
    Расчет СМО M/H2/n с комплексными параметрами численным методом Такахаси-Таками.
    Комплексные параметры позволяют аппроксимировать распределение времени обслуживания
    с произволиными коэффициентами вариации (>1, <=1)
    """

    def __init__(self, n, l, b, buffer=None, N=150, accuracy=1e-6, dtype="c16", verbose=False):

        """
        n: число каналов
        l: интенсивность вх. потока
        b: начальные моменты времени обслуживания
        N: число ярусов
        accuracy: точность, параметр для остановки итерации
        """
        self.dt = np.dtype(dtype)
        if buffer:
            self.R = buffer + n  # максимальное число заявок в сисетеме - очередь + каналы
            self.N = self.R + 1  # число ярусов на один больше + нулевое состояние
        else:
            self.N = N
            self.R = None  # для проверки задан ли буфер

        self.e1 = accuracy
        self.n = n
        self.b = b
        self.verbose = verbose

        h2_params = H2_dist.get_params_clx(b)
        # параметры H2-распределения:
        self.y = [h2_params[0], 1.0 - h2_params[0]]
        self.l = l
        self.mu = [h2_params[1], h2_params[2]]
        # массив cols хранит число столбцов для каждого яруса, удобней рассчитать его один раз:
        self.cols = [] * N

        # переменные
        self.t = []
        self.b1 = []
        self.b2 = []
        self.x = [0.0 + 0.0j] * N
        self.z = [0.0 + 0.0j] * N

        # искомые вреоятности состояний СМО
        self.p = [0.0 + 0.0j] * N

        # матрицы переходов
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
            self.x.append(np.zeros((1, self.cols[i]), dtype=self.dt))

        self.build_matrices()
        self.initial_probabilities()

    def get_p(self):
        """
        Возвращает список с вероятностями состояний системы
        p[k] - вероятность пребывания в системе ровно k заявок
        """
        for i in range(len(self.p)):
            self.p[i] = self.p[i].real
        return self.p

    def get_w(self):
        """
        Возвращает три первых начальных момента времени ожидания в СМО
        """
        w = [0.0] * 3

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

    def get_v(self):
        """
        Возвращает три первых начальных момента времени пребывания в СМО
        """
        v = [0.0] * 3
        w = self.get_w()
        v[0] = w[0] + self.b[0]
        v[1] = w[1] + 2 * w[0] * self.b[0] + self.b[1]
        v[2] = w[2] + 3 * w[1] * self.b[0] + 3 * w[0] * self.b[1] + self.b[2]

        return v


    def print_mrx(self, mrx):
        row = mrx.shape[0]
        col = mrx.shape[1]

        for i in range(row):
            for j in range(col):
                if math.isclose(mrx[i, j].real, 0.0):
                    print("{0:^5s} | ".format("     "), end="")
                else:
                    print("{0:^5.3f} | ".format(mrx[i, j].real), end="")
            print("\n" + "--------" * col)

    @staticmethod
    def binom_calc(a, b, num=3):
        res = []
        if num > 0:
            res.append(a + b)
        if num > 1:
            res.append(a[1] + 2 * a[0] * b[0] + b[1])
        if num > 2:
            res.append(a[2] + 3 * a[1] * b[0] + 3 * b[1] * a[0] + b[2])
        return res

    def initial_probabilities(self):
        """
        Задаем первоначальные значения вероятностей микросостояний
        """
        # t задаем равновероятными
        for i in range(self.N):
            for j in range(self.cols[i]):
                self.t[i][0, j] = 1.0 / self.cols[i]
        self.x[0] = 0.4

    def norm_probs(self):
        sum = 0
        for i in range(self.N):
            sum += self.p[i]

        for i in range(self.N):
            self.p[i] /= sum

        if self.verbose:
            sum = 0
            for i in range(self.N):
                sum += self.p[i]
            print("Summ of probs = {0:.5f}".format(sum))

    def calculate_p(self):
        """
        После окончания итераций находим значения вероятностей p по найденным х
        """
        # version 1
        # p_sum = 0
        # p0_max = 1.0
        # p0_min = 0.0
        # while math.fabs(1.0 - p_sum) > 1e-6:
        #    p0_ = (p0_max + p0_min) / 2.0
        #    p_sum = p0_
        #    p[0] = p0_
        #    for j in range(self.N-1):
        #        self.p[j + 1] = self.p[j] * self.x[j]
        #        p_sum += self.p[j + 1]
        #
        #    if (p_sum > 1.0):
        #        p0_max = p0_
        #    else:
        #        p0_min = p0_

        # version 2

        f1 = self.y[0] / self.mu[0] + self.y[1] / self.mu[1]
        znam = self.n

        for j in range(1, self.n):
            prod1 = 1
            for i in range(j):
                prod1 = np.dot(prod1, self.x[i])
            znam += np.dot((self.n - j), prod1)

        if self.R:
            prod2 = 1
            for i in range(0, self.N):
                prod2 *= self.x[i]
            znam -= f1 * self.l * prod2

        self.p[0] = (self.n - self.l * f1) / znam

        summ_p = self.p[0]

        for j in range(self.N - 1):
            self.p[j + 1] = np.dot(self.p[j], self.x[j])
            summ_p += self.p[j + 1]

        if self.verbose:
            print("Summ of probs = {0:.5f}".format(summ_p))

        self.norm_probs()

    def calculate_y(self):
        for i in range(self.N):
            self.Y.append(np.dot(self.p[i], self.t[i]))

    def build_matrices(self):
        """
        Формирует матрицы переходов
        """
        for i in range(self.N):
            self.A.append(self.buildA(i))
            self.B.append(self.buildB(i))
            self.C.append(self.buildC(i))
            self.D.append(self.buildD(i))

    def calc_g_matrices(self):
        self.G = []
        for j in range(0, self.N):
            self.G.append(np.linalg.inv(self.D[j] - self.C[j]))

    def calc_ag_matrices(self):
        self.AG = [0]
        for j in range(1, self.N):
            self.AG.append(np.dot(self.A[j - 1], self.G[j]))

    def calc_bg_matrices(self):
        self.BG = [0]
        for j in range(1, self.N):
            if j != (self.N - 1):
                self.BG.append(np.dot(self.B[j + 1], self.G[j]))
            else:
                self.BG.append(np.dot(self.B[j], self.G[j]))

    def calc_support_matrices(self):
        self.calc_g_matrices()
        self.calc_ag_matrices()
        self.calc_bg_matrices()

    def run(self):
        """
        Запускает расчет
        """
        self.b1[0][0, 0] = 0.0 + 0.0j
        self.b2[0][0, 0] = 0.0 + 0.0j
        x_max1 = 0.0 + 0.0j
        x_max2 = 0.0 + 0.0j

        self.calc_support_matrices()


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

                c = self.calculate_c(j)

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
                    self.t[j] = np.dot(self.z[j], self.b1[j]) + np.dot(self.x[j], self.b2[j])

            self.x[0] = (1.0 + 0.0j) / self.z[1]

            t1B1 = np.dot(self.t[1], self.B[1])
            self.t[0] = np.dot(self.x[0], t1B1)
            self.t[0] = np.dot(self.t[0], self.G[0])

            x_max1 = 0.0 + 0.0j

            for i in range(self.N):
                if self.x[i].real > x_max1.real:
                    x_max1 = self.x[i]

            if self.verbose:
                print("End iter # {0:d}".format(self.num_of_iter_))

        self.calculate_p()
        self.calculate_y()

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

    def buildA(self, num, is_v_calc=False):
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
        if is_v_calc:
            return output
        if num > self.n:
            output = self.A[self.n]
            return output
        for i in range(row):
            if num < self.n:
                output[i, i] = self.l * self.y[0]
                output[i, i + 1] = self.l * self.y[1]
            else:
                output[i, i] = self.l

        return output

    def buildB(self, num):
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

        for i in range(col):
            if num <= self.n:
                output[i, i] = (num - i) * self.mu[0]
                output[i + 1, i] = (i + 1) * self.mu[1]
            else:
                output[i, i] = (num - i - 1) * self.mu[0] * self.y[0] + i * self.mu[1] * self.y[1]
                if i != num - 1:
                    output[i, i + 1] = (num - i - 1) * self.mu[0] * self.y[1]
                if i != num - 1:
                    output[i + 1, i] = (i + 1) * self.mu[1] * self.y[0]
        return output

    def buildC(self, num):
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

        return output

    def buildD(self, num, is_v_calc=False):
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
            if is_v_calc:
                output = self.buildD(self.n, is_v_calc=True)
            else:
                output = self.D[self.n]
            return output

        for i in range(row):
            if is_v_calc:
                output[i, i] = (num - i) * self.mu[0] + i * self.mu[1]
            else:
                output[i, i] = self.l + (num - i) * self.mu[0] + i * self.mu[1]

        return output


