import numpy as np
import math
from most_queue.theory import passage_time
from tqdm import tqdm
from most_queue.sim import rand_destribution as rd


class MGnCalc:
    """
    Расчет СМО M/H2/n с комплексными параметрами численным методом Такахаси-Таками.
    Комплексные параметры позволяют аппроксимировать распределение времени обслуживания
    с произволиными коэффициентами вариации (>1, <=1)
    """

    def __init__(self, n, l, b, N=150, accuracy=1e-6, dtype="c16"):

        """
        n: число каналов
        l: интенсивность вх. потока
        h2_params: содержит список параметров H2-распределения [y1, mu1, mu2]
        N: число ярусов
        accuracy: точность, параметр для остановки итерации
        """
        self.dt = np.dtype(dtype)
        self.N = N
        self.e1 = accuracy
        self.n = n
        self.b = b
        h2_params = rd.H2_dist.get_params_clx(b)
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

    def get_v_pt(self):
        """
        Нахождение высших моментов с помощью алгоритма Ньютса - passage_time

        Согласно теореме PASTA (Poisson Arrivals See Time Averages) пребывающая заявка с вероятностью
        p[№ яруса, № микрососояния] наблюдает систему в микросостоянии [№ яруса, № микрососояния] и переводит
        ее в состояние на следующий ярус. Значит, необходимо пройтись по всем микросостояниям
        [№ яруса, № микрососояния], соответствующие вероятности p[№ яруса, № микрососояния] умножить на сумму
        вероятностей перехода в состояние [№+1 , № микрососояния] и время перехода из состояния
        [№+1 , № микрососояния] на 0 ярус. Время перехода вычисляется с помощью алгоритма Ньютса,
        реализованного в passage_time.py
        """
        A = []
        B = []
        C = []
        D = []

        for i in range(self.n + 2):
            A.append(self.buildA(i, is_v_calc=True))
            B.append(self.buildB(i))
            C.append(self.buildC(i))
            D.append(self.buildD(i, is_v_calc=True))

        pt = passage_time.passage_time_calc(A, B, C, D)
        pt.calc()

        F = []
        for num in range(self.n + 1):
            if num < self.n:
                col = self.cols[num + 1]
                row = self.cols[num]
            else:
                col = self.cols[self.n]
                row = self.cols[self.n]
            output = np.matrix(np.zeros((row, col)), dtype=self.dt)

            for i in range(row):
                if num < self.n:
                    output[i, i] = self.y[0]
                    output[i, i + 1] = self.y[1]
                else:
                    output[i, i] = 1
            F.append(output)

        # Gr_gap = pt.Gr_gap_calc(1, 0)
        # print("Gr gap mrx from {0:d} to {1:d}".format(1, 0))
        # b = [0,0,0]
        # for i in range(3):
        #     b[i] += F[0][0, 0]*Gr_gap[i][0, 0] + F[0][0, 1]*Gr_gap[i][1, 0]
        # print("Начальные моменты времени обслуживания b, вычисленные через passage time:")
        # for i in range(3):
        #     print("b[{0:d}] = {1:5.3f}".format(i + 1, b[i]))

        # print("Y[0]:")
        # self.print_mrx(self.Y[0])
        # print("Y[1]:")
        # self.print_mrx(self.Y[1])
        #
        # self.print_mrx(Gr_gap[0])
        # Gr_gap = pt.Gr_gap_calc(3, 0)
        # print("Gr gap mrx from {0:d} to {1:d}".format(3, 0))
        # self.print_mrx(Gr_gap[0])
        # Gr_gap = pt.Gr_gap_calc(5, 0)
        # print("Gr gap mrx from {0:d} to {1:d}".format(5, 0))
        # self.print_mrx(Gr_gap[0])

        v = [0 + 0j, 0 + 0j, 0 + 0j]
        print("Calc v")
        v_old = 0
        # inter_level_means = []
        # inter_level_delta_means = []

        for i in tqdm(range(0, self.N - 1)):
            Zr_gap = pt.Gr_gap_calc(i + 1, 0)
            # G_gap = pt.G_gap_calc(i+1, 0)
            # inter_level_means.append(0)
            # # for j in range(self.cols[i+1]):
            # #     inter_level_means[i] += self.t[i+1][0, j] * Zr_gap[0][j, 0]
            # # if i!=0:
            # #     inter_level_delta_means.append(inter_level_means[i].real-inter_level_means[i-1].real)
            # for j in range(self.cols[i+1]):
            #     if i < pt.l_tilda-1:
            #         inter_level_means[i] += pt.G[i+1][j, 0] * pt.Z[i+1][0][j, 0]
            #     else:
            #         inter_level_means[i] += pt.G[pt.l_tilda][j, 0] * pt.Z[pt.l_tilda][0][j, 0]
            # формируем переходы из состояния [i, j] на ярус ниже
            # G_gap = pt.G_gap_calc(i+1, 0) == 1, одно состояние
            for j in range(self.cols[i]):
                for k in range(self.cols[i + 1]):
                    if i > self.n:
                        v[0] = v[0] + F[self.n][j, k] * self.Y[i][0, j] * Zr_gap[0][k, 0]
                        # v[0] = v[0] + self.Y[i][0, j] * Gr_gap[0][k, 0]
                    else:
                        v[0] = v[0] + F[i][j, k] * self.Y[i][0, j] * Zr_gap[0][k, 0]
                        # v[0] = v[0] + self.Y[i][0, j] * Gr_gap[0][k, 0]
                # if i >= pt.l_tilda:
                #     v[0] = v[0] + self.Y[i][0, j] * Gr_gap[0][j, 0]
                # else:
                #     v[0] = v[0] + self.Y[i][0, j] * Gr_gap[0][j, 0]

            v_new = v[0]
            if math.fabs(v_new.real - v_old.real) < 1e-10:
                print("Stop v calc at i = {0:d}".format(i))
                break
            v_old = v_new

        # print(inter_level_means)

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

        self.p[0] = (self.n - self.l * f1) / znam

        for j in range(self.N - 1):
            self.p[j + 1] = np.dot(self.p[j], self.x[j])

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

    def run(self):
        """
        Запускает расчет
        """
        self.b1[0][0, 0] = 0.0 + 0.0j
        self.b2[0][0, 0] = 0.0 + 0.0j
        x_max1 = 0.0 + 0.0j
        x_max2 = 0.0 + 0.0j
        self.num_of_iter_ = 0  # кол-во итераций алгоритма
        for i in range(self.N):
            if self.x[i].real > x_max1.real:
                x_max1 = self.x[i]
        while math.fabs(x_max2.real - x_max1.real) >= self.e1:
            x_max2 = x_max1
            self.num_of_iter_ += 1
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
                self.x[j] = 0.0 + 0.0j
                for k in range(x_znam.shape[1]):
                    self.x[j] += x_znam[0, k]

                self.x[j] = (1.0 + 0.0j) / self.x[j]

                self.z[j] = np.dot(c, self.x[j])
                self.t[j] = np.dot(self.z[j], self.b1[j]) + np.dot(self.x[j], self.b2[j])

            self.x[0] = (1.0 + 0.0j) / self.z[1]

            t1B1 = np.dot(self.t[1], self.B[1])
            self.t[0] = np.dot(self.x[0], t1B1)
            self.t[0] = np.dot(self.t[0], np.linalg.inv(self.D[0] - self.C[0]))

            x_max1 = 0.0 + 0.0j

            for i in range(self.N):
                if self.x[i].real > x_max1.real:
                    x_max1 = self.x[i]

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


if __name__ == "__main__":
    from most_queue.sim import smo_im
    from most_queue.sim import rand_destribution as rd
    import time

    n = 3  # число каналов
    l = 1.0  # интенсивность вх потока
    ro = 0.8  # коэфф загрузки
    b1 = n * ro / l  # ср время обслуживания
    num_of_jobs = 800000  # число обсл заявок ИМ
    b_coev = [0.42, 1.5]  # коэфф вариации времени обсл

    for k in range(len(b_coev)):

        b = [0.0] * 3
        alpha = 1 / (b_coev[k] ** 2)
        b[0] = b1
        b[1] = math.pow(b[0], 2) * (math.pow(b_coev[k], 2) + 1)
        b[2] = b[1] * b[0] * (1.0 + 2 / alpha)

        im_start = time.process_time()
        smo = smo_im.SmoIm(n)
        smo.set_sources(l, 'M')
        gamma_params = rd.Gamma.get_mu_alpha([b[0], b[1]])
        smo.set_servers(gamma_params, 'Gamma')
        smo.run(num_of_jobs)
        p = smo.get_p()
        v_im = smo.v
        im_time = time.process_time() - im_start

        h2_params = rd.H2_dist.get_params_clx(b)

        tt_start = time.process_time()
        tt = MGnCalc(n, l, b)
        tt.run()
        p_tt = tt.get_p()
        v_tt = tt.get_v()
        tt_time = time.process_time() - tt_start
        num_of_iter = tt.num_of_iter_

        print("\nСравнение результатов расчета методом Такахаси-Таками и ИМ.\n"
              "ИМ - M/Gamma/{0:^2d}\nТакахаси-Таками - M/H2/{0:^2d}"
              "с комплексными параметрами\n"
              "Коэффициент загрузки: {1:^1.2f}\nКоэффициент вариации времени обслуживания: {2:^1.2f}\n".format(n, ro,
                                                                                                               b_coev[
                                                                                                                   k]))
        print("Количество итераций алгоритма Такахаси-Таками: {0:^4d}".format(num_of_iter))
        print("Время работы алгоритма Такахаси-Таками: {0:^5.3f} c".format(tt_time))
        print("Время работы ИМ: {0:^5.3f} c".format(im_time))
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
