import numpy as np
import math
from theory import passage_time
from tqdm import tqdm
from sim import rand_destribution as rd
from scipy import special
from utils.binom_probs import calc_binom_probs
from theory.diff5dots import diff5dots
from scipy.misc import derivative


class Mh2h2Warm:
    """
    Расчет СМО M/H2/n с H2-разогревом численным методом Такахаси-Таками.
    Используются комплексные параметры. Комплексные параметры позволяют аппроксимировать распределение времени обслуживания
    с произвольными коэффициентами вариации (>1, <=1)
    """

    def __init__(self, l, b, b_warm, n, buffer=None, N=150, accuracy=1e-6, dtype="c16", verbose=False,
                 is_only_first=False):

        """
        n: число каналов
        l: интенсивность вх. потока
        b: начальные моменты времени обслуживания
        b_warm: начальные моменты времени разогрева
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

        self.is_only_first = is_only_first

        self.e1 = accuracy
        self.n = n
        self.b = b
        self.verbose = verbose
        self.l = l

        if self.dt == 'c16':
            h2_params_service = rd.H2_dist.get_params_clx(b)
        else:
            h2_params_service = rd.H2_dist.get_params(b)

        # параметры H2-распределения:
        self.y = [h2_params_service[0], 1.0 - h2_params_service[0]]
        self.mu = [h2_params_service[1], h2_params_service[2]]

        self.b_warm = b_warm
        if self.dt == 'c16':
            h2_params_warm = rd.H2_dist.get_params_clx(b_warm)
        else:
            h2_params_warm = rd.H2_dist.get_params(b_warm)
        self.y_w = [h2_params_warm[0], 1.0 - h2_params_warm[0]]
        self.mu_w = [h2_params_warm[1], h2_params_warm[2]]

        # массив cols хранит число столбцов для каждого яруса, удобней рассчитать его один раз:
        self.cols = [] * N

        # переменные
        self.t = []
        self.b1 = []
        self.b2 = []
        if self.dt == 'c16':
            self.x = [0.0 + 0.0j] * N
            self.z = [0.0 + 0.0j] * N

            # искомые вреоятности состояний СМО
            self.p = [0.0 + 0.0j] * N
        else:
            self.x = [0.0] * N
            self.z = [0.0] * N

            # искомые вреоятности состояний СМО
            self.p = [0.0] * N

        # матрицы переходов
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
                        self.cols.append(i + 3)  # w1 w2 01 10, w1 w2 20 11 02, ...
                else:
                    self.cols.append(n + 3)

            self.t.append(np.zeros((1, self.cols[i]), dtype=self.dt))
            self.b1.append(np.zeros((1, self.cols[i]), dtype=self.dt))
            self.b2.append(np.zeros((1, self.cols[i]), dtype=self.dt))
            self.x.append(np.zeros((1, self.cols[i]), dtype=self.dt))

        self.build_matrices()
        self.initial_probabilities()
        self.key_numbers = self.get_key_numbers(self.n)  # ключи яруса n, для n=3 [(3,0) (2,1) (1,2) (0,3)]
        self.down_probs = self.calc_down_probs(self.n + 1)

    def get_p(self):
        """
        Возвращает список с вероятностями состояний системы
        p[k] - вероятность пребывания в системе ровно k заявок
        """
        for i in range(len(self.p)):
            self.p[i] = self.p[i].real
        return self.p

    def pls(self, mu, s):
        return mu / (mu + s)

    def calc_passage_times(self):
        pass_time = passage_time.passage_time_calc(self.A, self.B, self.C, self.D, l_tilda=self.n + 1)
        pass_time.calc()
        print("\nЗначения матриц G:\n")

        g_num = len(pass_time.G)
        for i in range(g_num):

            print("G{0:^1d}".format(i))

            rows = pass_time.G[i].shape[0]
            cols = pass_time.G[i].shape[1]
            for j in range(rows):
                for t in range(cols):
                    if t == cols - 1:
                        print("{0:^5.3g}  ".format(pass_time.G[i][j, t]))
                    else:
                        print("{0:^5.3g}  ".format(pass_time.G[i][j, t]), end=" ")

    def get_key_numbers(self, level):
        key_numbers = []
        for i in range(level + 1):
            key_numbers.append((self.n - i, i))
        return np.array(key_numbers, dtype=self.dt)

    def calc_down_probs(self, from_level):
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

    def matrix_pow(self, matrix, k):
        res = np.eye(self.n + 1, dtype=self.dt)
        for i in range(k):
            res = np.dot(res, matrix)
        return res

    def calc_w_pls(self, s):
        w = 0
        # вероятности попадания в состояния обслуживания, вычисляются с помощью биноминального распределения
        probs = calc_binom_probs(self.n + 1, self.y[0])

        # Если заявка попала в фазу разогрева, хотя каналы свободны,
        # ей придется подождать окончание разогрева
        for k in range(1, self.n):
            for i in range(2):
                w += self.Y[k][0, i] * self.pls(self.mu_w[i], s)

        for k in range(self.n, self.N):

            # Если заявка попала в фазу разогрева и каналы заняты. Также есть k-n заявок в очереди
            # ей придется подождать окончание разогрева + обслуживание всех накопленных заявок
            key_numbers = self.get_key_numbers(self.n)  # ключи яруса n, для n=3 [(3,0) (2,1) (1,2) (0,3)]

            a = np.array(
                [self.pls(key_numbers[j][0] * self.mu[0] + key_numbers[j][1] * self.mu[1], s) for j in
                 range(self.n + 1)])

            P = self.calc_down_probs(k)
            Pa = np.transpose(self.matrix_pow(P * a, k - self.n))  # n+1 x n+1

            aP_tilda = a * probs

            pls_service_total = sum(np.dot(aP_tilda, Pa))

            for i in range(2):
                w += self.Y[k][0, i] * self.pls(self.mu_w[i], s) * pls_service_total

            # попала в фазу обслуживания
            Ys = [self.Y[k][0, i + 2] for i in range(len(probs))]
            aPa = np.dot(a, Pa)
            for i in range(self.n + 1):
                w += Ys[i] * aPa[i]

        return w

    def calc_residual_b(self, mom_num):
        res = self.b[mom_num + 1] / (2.0 * self.b[mom_num])
        return res

    def get_w(self):
        """
        Возвращает три первых начальных момента времени ожидания в СМО
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
                w[i] = derivative(self.calc_w_pls, 0, dx=1e-3 / self.b[0], n=i + 1, order=9)
            return [-w[0].real, w[1].real, -w[2].real]

    def get_v(self):
        """
        Возвращает три первых начальных момента времени пребывания в СМО
        """
        v = [0.0] * 3
        w = self.get_w()
        b = self.b
        v[0] = w[0] + b[0]
        v[1] = w[1] + 2 * w[0] * b[0] + b[1]
        v[2] = w[2] + 3 * w[1] * b[0] + 3 * w[0] * b[1] + b[2]

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
            res.append(a[0] + b[0])
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
        summ = 0
        for i in range(self.N):
            summ += self.p[i]

        for i in range(self.N):
            self.p[i] /= summ

        if self.verbose:
            summ = 0
            for i in range(self.N):
                summ += self.p[i]
            print("Summ of probs = {0:.5f}".format(summ))

    def calculate_p(self):
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
                    self.t[j] = np.dot(self.z[j], self.b1[j]) + np.dot(self.x[j], self.b2[j])

            if self.dt == 'c16':
                self.x[0] = (1.0 + 0.0j) / self.z[1]
            else:
                self.x[0] = 1.0 / self.z[1]

            t1B1 = np.dot(self.t[1], self.B[1])
            self.t[0] = np.dot(self.x[0], t1B1)
            self.t[0] = np.dot(self.t[0], self.G[0])

            if self.dt == 'c16':
                x_max1 = 0.0 + 0.0j
            else:
                x_max1 = 0

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

    def insert_standart_A_into(self, mass, l, y1, left_pos, bottom_pos, level):
        row_num = level
        for i in range(row_num):
            mass[i + left_pos, i + bottom_pos] = l * y1
            mass[i + left_pos, i + bottom_pos + 1] = l * (1.0 - y1)

    def buildA(self, num):
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
                    self.insert_standart_A_into(output, self.l, self.y[0], 0, 0, level=num)
                    # second
                    self.insert_standart_A_into(output, self.l, self.y[0], num, num + 1, level=num)
                    # third
                    self.insert_standart_A_into(output, self.l, self.y[0], 2 * num, 2 * (num + 1), level=num + 1)
                else:
                    # first block
                    output[0, 0] = self.l
                    output[1, 1] = self.l
                    # second
                    self.insert_standart_A_into(output, self.l, self.y[0], 2, 2, level=num + 1)
            else:
                for i in range(row):
                    output[i, i] = self.l

        return output

    def insert_standart_B_into(self, mass, y, mu, left_pos, bottom_pos, level, n):
        col = level
        for i in range(col):
            if level <= n:
                mass[i + left_pos, i + bottom_pos] = (level - i) * mu[0]
                mass[i + left_pos + 1, i + bottom_pos] = (i + 1) * mu[1]
            else:
                mass[i + left_pos, i + bottom_pos] = (level - i - 1) * mu[0] * y[0] + i * mu[1] * y[1]
                if i != level - 1:
                    mass[i + left_pos, i + bottom_pos + 1] = (level - i - 1) * mu[0] * y[1]
                if i != level - 1:
                    mass[i + + left_pos + 1, i + bottom_pos] = (i + 1) * mu[1] * y[0]

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
                    self.insert_standart_B_into(output, self.y, self.mu, 0, 0, num - 1, self.n)
                    # second
                    self.insert_standart_B_into(output, self.y, self.mu, num, num - 1, num - 1, self.n)
                    # third
                    self.insert_standart_B_into(output, self.y, self.mu, 2 * num, 2 * (num - 1), num, self.n)

                    # warm block 1
                    for i in range(num):
                        output[i, i + 2 * (num - 1)] = self.mu_w[0]

                    # warm block 2
                    for i in range(num):
                        output[i + num, i + 2 * (num - 1)] = self.mu_w[1]
                else:
                    self.insert_standart_B_into(output, self.y, self.mu, 2, 2, num, self.n)

            else:

                if self.is_only_first:
                    # first block
                    self.insert_standart_B_into(output, self.y, self.mu, 0, 0, num - 1, self.n - 1)
                    # second
                    self.insert_standart_B_into(output, self.y, self.mu, num - 1, num - 1, num - 1, self.n - 1)
                    # third
                    self.insert_standart_B_into(output, self.y, self.mu, 2 * (num - 1), 2 * (num - 1), num, self.n)

                    # warm block 1
                    for i in range(num - 1):
                        output[i, i + 2 * (num - 1)] = self.mu_w[0] * self.y[0]
                        output[i, i + 2 * (num - 1) + 1] = self.mu_w[0] * self.y[1]

                    # warm block 2
                    for i in range(num - 1):
                        output[i + num - 1, i + 2 * (num - 1)] = self.mu_w[1] * self.y[0]
                        output[i + num - 1, i + 2 * (num - 1) + 1] = self.mu_w[1] * self.y[1]

                else:
                    self.insert_standart_B_into(output, self.y, self.mu, 2, 2, num, self.n)

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

        if not self.is_only_first:
            if num > 0:
                probs = calc_binom_probs(num + 1, self.y[0])
                print(probs)
                for i in range(len(probs)):
                    output[0, 2 + i] = self.mu_w[0] * probs[i]
                    output[1, 2 + i] = self.mu_w[1] * probs[i]

        return output

    def buildD(self, num):
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

