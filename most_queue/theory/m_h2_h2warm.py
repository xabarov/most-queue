import numpy as np
import math
import passage_time
from tqdm import tqdm
from most_queue.sim import rand_destribution as rd

class Mh2h2Warm:
    """
    Расчет СМО M/H2/H2warm/n с комплексными параметрами численным методом Такахаси-Таками.
    Комплексные параметры позволяют аппроксимировать распределение времени обслуживания
    с произволиными коэффициентами вариации (>1, <=1)
    """

    def __init__(self, l, b, b_warm, n, buffer=None, N=150, accuracy=1e-6, dtype="c16", verbose=False):

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
            if i < n + 1:
                if i == 0:
                    self.cols.append(1)  # 00 state
                else:
                    self.cols.append(3 * i + 1)  # w1 w2 01 10
            else:
                self.cols.append(3 * n + 1)

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

    def get_b(self):
        b_plus_b_warm = [0.0] * 3
        for i in range(3):
            b_plus_b_warm[i] = self.b[i] * (1.0 - self.p[0]) + self.p[0] * self.b_warm[i]
        return b_plus_b_warm

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

        # self.calc_G_passage()

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
        if num == 0:
            output[0, 0] = self.l * self.y_w[0]
            output[0, 1] = self.l * self.y_w[1]
        else:

            if num < self.n:
                # first block
                self.insert_standart_A_into(output, self.l, self.y[0], 0, 0, level=num)
                # second
                self.insert_standart_A_into(output, self.l, self.y[0], num, num + 1, level=num)
                # third
                self.insert_standart_A_into(output, self.l, self.y[0], 2 * num, 2 * (num + 1), level=num + 1)
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
            output[0, 0] = self.mu_w[0]
            output[1, 0] = self.mu_w[1]
            output[2, 0] = self.mu[0]
            output[3, 0] = self.mu[1]
        else:

            if num < self.n + 1:
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


if __name__ == "__main__":

    from sim import smo_im
    from sim import rand_destribution as rd
    import time

    n = 3  # число каналов
    l = 1.0  # интенсивность вх потока
    ro = 0.7  # коэфф загрузки
    b1 = n * 0.7  # ср время обслуживания
    b1_warm = n * 0.9  # ср время разогрева
    num_of_jobs = 1000000  # число обсл заявок ИМ
    b_coev = [0.8, 1.5]  # коэфф вариации времени обсл
    b_coev_warm = 1.2  # коэфф вариации времени разогрева
    buff = None
    verbose = False

    for k in range(len(b_coev)):
        b = [0.0] * 3
        alpha = 1 / (b_coev[k] ** 2)
        b[0] = b1
        b[1] = math.pow(b[0], 2) * (math.pow(b_coev[k], 2) + 1)
        b[2] = b[1] * b[0] * (1.0 + 2 / alpha)

        b_w = [0.0] * 3
        b_w[0] = b1_warm
        alpha = 1 / (b_coev_warm ** 2)
        b_w[1] = math.pow(b_w[0], 2) * (math.pow(b_coev_warm, 2) + 1)
        b_w[2] = b_w[1] * b_w[0] * (1.0 + 2 / alpha)

        h2_params = rd.H2_dist.get_params_clx(b)

        im_start = time.process_time()
        smo = smo_im.SmoIm(n, buffer=buff)
        smo.set_sources(l, 'M')

        gamma_params = rd.Gamma.get_mu_alpha(b)
        gamma_params_warm = rd.Gamma.get_mu_alpha(b_w)
        smo.set_servers(gamma_params, 'Gamma')
        smo.set_warm(gamma_params_warm, 'Gamma')
        smo.run(num_of_jobs)
        p = smo.get_p()
        v_im = smo.v  # .w -> wait times
        im_time = time.process_time() - im_start

        tt_start = time.process_time()
        tt = Mh2h2Warm(l, b, b_w, n, buffer=buff, verbose=verbose)

        tt.run()
        p_tt = tt.get_p()
        v_tt = tt.get_v()  # .get_w() -> wait times
        tt_time = time.process_time() - tt_start
        num_of_iter = tt.num_of_iter_

        print("\nСравнение результатов расчета методом Такахаси-Таками и ИМ.\n"
              "ИМ - M/Gamma/{0:^2d}\nТакахаси-Таками - M/H2/{0:^2d}"
              "с комплексными параметрами\n"
              "Коэффициент загрузки: {1:^1.2f}".format(n, ro))
        print(f'Коэффициент вариации времени обслуживания {b_coev[k]:0.3f}')
        print(f'Коэффициент вариации времени разогрева {b_coev_warm:0.3f}')
        print("Количество итераций алгоритма Такахаси-Таками: {0:^4d}".format(num_of_iter))
        print("Время работы алгоритма Такахаси-Таками: {0:^5.3f} c".format(tt_time))
        print("Время ИМ: {0:^5.3f} c".format(im_time))
        print("{0:^25s}".format("Первые 10 вероятностей состояний СМО"))
        print("{0:^3s}|{1:^15s}|{2:^15s}".format("№", "Числ", "ИМ"))
        print("-" * 32)
        for i in range(11):
            print("{0:^4d}|{1:^15.3g}|{2:^15.3g}".format(i, p_tt[i], p[i]))

        print("\n")
        print("{0:^25s}".format("Начальные моменты времени пребывания в СМО"))
        print("{0:^3s}|{1:^15s}|{2:^15s}".format("№", "Числ", "ИМ"))
        print("-" * 32)
        for i in range(3):
            print("{0:^4d}|{1:^15.3g}|{2:^15.3g}".format(i + 1, v_tt[i], v_im[i]))