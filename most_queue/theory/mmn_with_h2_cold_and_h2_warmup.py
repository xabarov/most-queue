import numpy as np
import math
from tqdm import tqdm
from sim import rand_destribution as rd
from scipy import special
from utils.binom_probs import calc_binom_probs
from scipy.misc import derivative
from itertools import chain


class MMn_H2warm_H2cold:
    """
    Расчет СМО M/M/n с H2-разогревом и H2-охлаждением численным методом Такахаси-Таками.
    Используются комплексные параметры. Комплексные параметры позволяют аппроксимировать распределение времени обслуживания
    с произвольными коэффициентами вариации (>1, <=1)
    """

    def __init__(self, l, mu, b_warm, b_cold, n, buffer=None, N=150, accuracy=1e-8, dtype="c16", verbose=False):

        """
        n: число каналов
        l: интенсивность вх. потока
        mu: интенсивность обслуживания
        b_warm: начальные моменты времени разогрева
        b_cold: начальные моменты времени охлаждения
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
        self.mu = mu
        self.verbose = verbose
        self.l = l

        self.b_warm = b_warm
        if self.dt == 'c16':
            h2_params_warm = rd.H2_dist.get_params_clx(b_warm)
        else:
            h2_params_warm = rd.H2_dist.get_params(b_warm)

        self.y_w = [h2_params_warm[0], 1.0 - h2_params_warm[0]]
        self.mu_w = [h2_params_warm[1], h2_params_warm[2]]

        self.b_cold = b_cold
        if self.dt == 'c16':
            h2_params_cold = rd.H2_dist.get_params_clx(b_cold)
        else:
            h2_params_cold = rd.H2_dist.get_params(b_cold)
        self.y_c = [h2_params_cold[0], 1.0 - h2_params_cold[0]]
        self.mu_c = [h2_params_cold[1], h2_params_cold[2]]

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

            if i < n + 1:
                if i == 0:
                    self.cols.append(3)  # 0 state normal + 0_cold_1 + 0_cold_2
                else:
                    self.cols.append(5)  # i_warm_1 + i_warm_2 i state normal + i_cold_1 + i_cold_2...
            else:
                self.cols.append(5)

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

    def pls(self, mu, s):
        return mu / (mu + s)

    def calc_w_pls(self, s):
        w = 0

        # вычислим ПЛС заранее
        mu_pls = self.pls(self.mu * self.n, s)
        mu_w_pls = np.array([self.pls(self.mu_w[0], s), self.pls(self.mu_w[1], s)])
        mu_c_pls = np.array([self.pls(self.mu_c[0], s), self.pls(self.mu_c[1], s)])

        # Комбо переходов: охлаждение + разогрев
        # [i,j] = охлаждение в i, переход из состояния i охлаждения в j состояние разогрева, разогрев j
        c_to_w = np.zeros((2, 2), dtype=self.dt)
        for c_phase in range(2):
            for w_phase in range(2):
                c_to_w[c_phase, w_phase] = mu_c_pls[c_phase] * self.y_w[w_phase] * mu_w_pls[w_phase]

        # Если заявка попала в состояние [0] ей придется подождать окончание разогрева
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

    def get_w(self):
        """
        Возвращает три первых начальных момента времени ожидания в СМО
        """
        w = [0.0] * 3

        for i in range(3):
            min_mu = min(chain(np.array(self.mu_w).astype('float'), np.array(self.mu_c).astype('float'), [self.mu]))
            w[i] = derivative(self.calc_w_pls, 0, dx=1e-3 / min_mu, n=i + 1, order=9)
        return [-w[0].real, w[1].real, -w[2].real]

    def get_b(self):
        return [1.0 / self.mu, 2.0 / pow(self.mu, 2), 6.0 / pow(self.mu, 3)]

    def get_v(self):
        """
        Возвращает три первых начальных момента времени пребывания в СМО
        """
        v = [0.0] * 3
        w = self.get_w()
        b = self.get_b()
        v[0] = w[0] + b[0]
        v[1] = w[1] + 2 * w[0] * b[0] + b[1]
        v[2] = w[2] + 3 * w[1] * b[0] + 3 * w[0] * b[1] + b[2]

        return v

    def get_cold_prob(self):
        """
        Возвращает вероятность нахождения в состоянии охлаждения
        """
        p_cold = 0
        for k in range(self.N):
            p_cold += self.Y[k][0, -1] + self.Y[k][0, -2]
        return p_cold.real

    def get_warmup_prob(self):
        """
        Возвращает вероятность нахождения в состоянии разогрева
        """
        p_warmup = 0
        for k in range(1, self.N):
            p_warmup += self.Y[k][0, 0] + self.Y[k][0, 1]
        return p_warmup.real

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

    def calc_source_coev(self):

        return 1.0

    def calc_serv_coev(self):

        return 1.0

    def initial_probabilities(self):
        """
        Задаем первоначальные значения вероятностей микросостояний
        """
        # t задаем равновероятными
        for i in range(self.N):
            for j in range(self.cols[i]):
                self.t[i][0, j] = 1.0 / self.cols[i]

        ro = self.l / (self.mu * self.n)
        va = self.calc_source_coev()
        vb = self.calc_serv_coev()
        self.x[0] = pow(ro, 2.0 / (va * va + vb * vb))

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

        while math.fabs(1.0 - p_sum.real) > 1e-6:
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
        # sum_y = 0.0
        for i in range(self.N):
            self.Y.append(np.dot(self.p[i], self.t[i]))
            # sum_y += np.sum(self.Y[i])
        # print("Sum Y: ", sum_y)

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
                print(f"End iter # {self.num_of_iter_:d}, x_max: {x_max1}")

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
            # to cold phase
            output[2, 1] = self.mu * self.y_c[0]
            output[2, 2] = self.mu * self.y_c[1]

        else:
            if num < self.n + 1:

                output[2, 2] = self.mu * num

            else:

                output[2, 2] = self.mu * self.n

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




    
