import numpy as np
import math
import passage_time
import rand_destribution as rd


class m_ph_n_prty:
    """
    Расчет СМО M/PH, M/n с 2-мя классами заявок, абсолютным приоритетом
    численным методом Такахаси-Таками на основе аппроксимации ПНЗ распределением Кокса второго порядка
    """

    def __init__(self, mu_L, mu1_H, mu2_H, p_H, l_L, l_H, n, N=250, accuracy=1e-8, max_iter=300, is_cox=True,
                 verbose=True):

        """
        Численный расчет СМО M/Ph/n с абсолютным приоритетом на основе аппроксимации периодов непрерывной занятости
        l_L, l_H: интенсивности вх. потока заявок с низким и высоким приоритетами
        mu_L: интенсивность обслуживания заявок с низким приоритетом,
        mu1_H, mu2_H, p_H: параметры Cox2 (H2) - распределения для заявок высокого класса
        N: число ярусов
        accuracy: точность, параметр для остановки итерации
        is_cox:
            True -  для аппроксимации времени обслуживания заявок первого класса используется
                    распределение Кокса 2-го порядка,
            False - гиперэкспоненциальное распределение 2-го порядка H2
        """
        # self.dt = np.dtype("f8")
        self.dt = np.dtype("c16")
        self.n = n  # количество каналов
        self.N = N
        self.e1 = accuracy
        self.l_L = l_L
        self.l_H = l_H
        self.mu_L = mu_L
        self.mu1_H = mu1_H
        self.mu2_H = mu2_H
        self.p_H = p_H
        self.max_iter = max_iter
        self.is_cox = is_cox
        self.verbose = verbose

        self.busy_periods = []  # список из наборов начальных моментров ПНЗ B1, B2, ...
        self.busy_periods_coevs = []  # коэффициенты вариации ПНЗ
        self.alphas = []
        self.pp = []  # список из вероятностей переходов в ПНЗ

        self.calc_busy_periods()

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

        # cols зависит от числа каналов. cols = количество всех микросостояний переходов для Cox2 до яруса
        # с номером N-1 + число ПНЗ * 2. Число ПНЗ = n**2
        self.cols_length_ = 2 * self.n ** 2
        for i in range(self.n):
            self.cols_length_ += i + 1

        for i in range(N):
            self.t.append(np.zeros((1, self.cols_length_), dtype=self.dt))
            self.b1.append(np.zeros((1, self.cols_length_), dtype=self.dt))
            self.b2.append(np.zeros((1, self.cols_length_), dtype=self.dt))
            self.x.append(np.zeros((1, self.cols_length_), dtype=self.dt))

        self.build_matrices()
        self.initial_probabilities()

    def build_A_for_busy_periods(self, num):
        """
            Формирует матрицу А(L) для расчета ПНЗ по заданному номеру яруса
            num - номер яруса
        """
        if num < self.n:
            col = num + 2
            row = num + 1
        elif num == self.n:
            col = num + 1
            row = num + 1
        else:
            output = self.A_for_busy[self.n]
            return output

        output = np.zeros((row, col), dtype=self.dt)

        for i in range(row):
            output[i, i] = self.l_H
        return output

    def build_B_for_busy_periods(self, num):
        """
            Формирует матрицу B(num) для расчета ПНЗ  по заданному номеру яруса
            num - номер яруса
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
            output[:, :self.n] = self.B_for_busy[self.n]
            return output
        else:
            output = self.B_for_busy[self.n + 1]
            return output

        output = np.zeros((row, col), dtype=self.dt)

        mu1 = self.mu1_H
        mu2 = self.mu2_H
        qH = 1.0 - self.p_H

        for i in range(col):
            output[i, i] = (num - i) * mu1 * qH
            output[i + 1, i] = (i + 1) * mu2
        return output

    def build_C_for_busy_periods(self, num):
        """
            Формирует матрицу C(num) для расчета ПНЗ по заданному номеру яруса
            num - номер яруса
        """
        if num == 0:
            output = np.zeros((1, 1), dtype=self.dt)
            return output
        elif num <= self.n:
            col = num + 1
            row = num + 1
        else:
            output = self.C_for_busy[self.n]
            return output

        output = np.zeros((row, col), dtype=self.dt)

        mu1 = self.mu1_H
        pH = self.p_H

        for i in range(col - 1):
            output[i, i + 1] = (num - i) * mu1 * pH
        return output

    def build_D_for_busy_periods(self, num):
        """
            Формирует матрицу D для ПНЗ по заданному номеру яруса num
        """
        if num <= self.n:
            col = num + 1
            row = num + 1
        else:
            output = self.D_for_busy[self.n]
            return output

        output = np.zeros((row, col), dtype=self.dt)

        for i in range(row):
            sumA = 0.0 + 0.0j
            sumB = 0.0 + 0.0j
            sumC = 0.0 + 0.0j

            for j in range(col):
                sumA += self.A_for_busy[num][i, j]

            if num != 0:
                for j in range(col - 1):
                    sumB += self.B_for_busy[num][i, j]

            for j in range(col):
                sumC += self.C_for_busy[num][i, j]

            output[i, i] = sumA + sumB + sumC

        return output

    def calc_busy_periods(self):

        self.A_for_busy = []
        self.B_for_busy = []
        self.C_for_busy = []
        self.D_for_busy = []

        for i in range(self.n + 2):
            self.A_for_busy.append(self.build_A_for_busy_periods(i))
            self.B_for_busy.append(self.build_B_for_busy_periods(i))
            self.C_for_busy.append(self.build_C_for_busy_periods(i))
            self.D_for_busy.append(self.build_D_for_busy_periods(i))

        pass_time = passage_time.passage_time_calc(self.A_for_busy, self.B_for_busy,
                                                   self.C_for_busy, self.D_for_busy, is_clx=True, is_verbose=True)
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
            print("\nПериоды НЗ:\n")
            for j in range(self.pnz_num_):
                for r in range(3):
                    if math.isclose(self.busy_periods[j][r].imag, 0):
                        print("{0:^8.3g}".format(self.busy_periods[j][r].real), end=" ")
                    else:
                        print("{0:^8.3g}".format(self.busy_periods[j][r]), end=" ")
                if math.isclose(self.busy_periods_coevs[j].imag, 0):
                    print("coev = {0:^4.3g}".format(self.busy_periods_coevs[j].real))
                else:
                    print("coev = {0:^4.3g}".format(self.busy_periods_coevs[j]))

        # pp - список из n**2 вероятностей переходов
        for j in range(self.pnz_num_):
            num = 0
            for i in range(self.n):
                for j in range(self.n):
                    self.pp.append(pass_time.G[self.n][i, j])
                    num = num + 1
        if self.verbose:
            print("\nВероятности переходов в ПНЗ:\n")
            for j in range(self.pnz_num_):
                if math.isclose(self.pp[j].imag, 0):
                    print("{0:^8.3g}".format(self.pp[j].real), end=" ")
                else:
                    print("{0:^8.3g}".format(self.pp[j]), end=" ")

    def get_p(self):
        """
        Возвращает список с вероятностями состояний системы
        p[k] - вероятность пребывания в системе ровно k заявок класса L
        """
        p_real = [0.0] * self.N
        for i in range(len(self.p)):
            p_real[i] = self.p[i].real
        return p_real

    def get_low_class_v1(self):
        """
        Возвращает среднее время пребывания для класса L
        """
        l1 = 0
        p_real = self.get_p()
        for i in range(self.N):
            l1 += p_real[i] * i
        return l1 / self.l_L

    def initial_probabilities(self):
        """
        Задаем первоначальные значения вероятностей микросостояний
        """
        # t задаем равновероятными
        for i in range(self.N):
            for j in range(self.cols_length_):
                self.t[i][0, j] = 1.0 / self.cols_length_
        self.x[0] = 0.4

    def calculate_p(self):
        """
        После окончания итераций находим значения вероятностей p по найденным х
        """
        # version 1
        p_sum = 0 + 0j
        p0_max = 1.0
        p0_min = 0.0
        iter = 0
        self.p_iteration_num_ = 0
        while math.fabs(1.0 - p_sum.real) > 1e-6 and iter < self.max_iter:
            iter += 1
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
        Запускает расчет. Алгоритм расчета не зависит от структуры матриц
        """
        self.b1[0][0, 0] = 0.0 + 0.0j
        self.b2[0][0, 0] = 0.0 + 0.0j

        x_ave1 = 0.0 + 0.0j
        x_ave2 = 0.0 + 0.0j

        iter = 0
        self.run_iterations_num_ = 0
        for i in range(self.N):
            x_ave1 += self.x[i]
        x_ave1 /= self.N
        while math.fabs(x_ave2.real - x_ave1.real) >= self.e1 and iter < self.max_iter:
            if self.verbose:
                print("Start numeric iteration {0:d}".format(iter))
            iter += 1
            x_ave2 = x_ave1
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

                self.x[j] = 1.0 / self.x[j]

                self.z[j] = np.dot(c, self.x[j])
                self.t[j] = np.dot(self.z[j], self.b1[j]) + np.dot(self.x[j], self.b2[j])

            self.x[0] = 1.0 / self.z[1]

            t1B1 = np.dot(self.t[1], self.B[1])
            self.t[0] = np.dot(self.x[0], t1B1)
            self.t[0] = np.dot(self.t[0], np.linalg.inv(self.D[0] - self.C[0]))

            x_ave1 = 0.0 + 0.0j

            for i in range(self.N):
                x_ave1 += self.x[i]
            x_ave1 /= self.N

        self.run_iterations_num_ = iter
        self.calculate_p()
        self.calculate_y()

    def calculate_c(self, j):
        """
        Вычисляет значение переменной с, участвующей в расчете
        """
        chisl = 0.0 + 0.0j
        znam = 0.0 + 0.0j
        znam2 = 0.0 + 0.0j

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
        Формирует матрицу А(L) по заданному номеру яруса num
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

    def buildB(self, num):
        """
            Формирует матрицу B по заданному номеру яруса
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
            for j in range(i + 1):  # количество микросочтояний в этом ярусе
                output[lev, lev] = min(self.n - i, num) * self.mu_L
                lev = lev + 1

        return output

    def buildC(self, num):
        """
            Формирует матрицу C по заданному номеру яруса
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
                h2_param = rd.H2_dist.get_params_clx(self.busy_periods[i])
                # h2_param = rd.H2_dist.get_params(self.busy_periods[i])
                y1_mass.append(h2_param[0])
                m1_mass.append(h2_param[1])
                m2_mass.append(h2_param[2])
                if self.verbose:
                    print("Параметры для B{0}: {1:3.3f}, {2:3.3f}, {3:3.3f}".format(i + 1, h2_param[0], h2_param[1],
                                                                                    h2_param[2]))
            else:
                cox_params = rd.Cox_dist.get_params_clx(self.busy_periods[i])
                y1_mass.append(cox_params[0])
                m1_mass.append(cox_params[1])
                m2_mass.append(cox_params[2])
                if self.verbose:
                    print("Параметры для B{0}: {1:3.3f}, {2:3.3f}, {3:3.3f}".format(i + 1, cox_params[0], cox_params[1],
                                                                                    cox_params[2]))

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
            rows = self.B_for_busy[i].shape[0]
            cols = self.B_for_busy[i].shape[1]
            for r in range(rows):
                for c in range(cols):
                    output[l_start + r, l_end + c] = self.B_for_busy[i][r, c]
            l_start += i + 1
            l_end += i

        # C_for_busy
        l_start = 1
        for i in range(1, self.n):
            rows = self.C_for_busy[i].shape[0]
            cols = self.C_for_busy[i].shape[1]
            for r in range(rows):
                for c in range(cols):
                    output[l_start + r, l_start + c] = self.C_for_busy[i][r, c]
            l_start += i + 1

        l_start = 0
        for i in range(self.n - 1):
            l_start += i + 1

        l_end = l_start + self.n
        num = 0
        for i in range(self.n):
            for j in range(self.n):
                if not self.is_cox:
                    output[l_start + i, l_end] = lh * y1_mass[num] * self.pp[num]
                    output[l_start + i, l_end + 1] = lh * (1 - y1_mass[num]) * self.pp[num]
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

    def print_mrx(self, mrx, is_short=False):
        row = mrx.shape[0]
        col = mrx.shape[1]

        for i in range(row):
            for j in range(col):
                if math.isclose(mrx[i, j].real, 0.0):
                    if is_short:
                        print("{0:^3s} | ".format(""), end="")
                    else:
                        print("{0:^5s} | ".format(""), end="")
                else:
                    if is_short:
                        print("{0:^3.1f} | ".format(mrx[i, j].real), end="")
                    else:
                        print("{0:^5.3f} | ".format(mrx[i, j].real), end="")
            if is_short:
                print("\n" + "------" * col)
            else:
                print("\n" + "--------" * col)

    def buildD(self, num):
        """
            Формирует матрицу D по заданному номеру яруса
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


if __name__ == "__main__":
    import smo_im_prty
    import rand_destribution as rd
    import prty_calc
    import time

    num_of_jobs = 800000  # число обсл заявок ИМ

    is_cox = False  # использовать для аппроксимации ПНЗ распределение Кокса или Н2-распределение
    max_iter = 100  # максимальное число итераций численного метода
    # Исследование влияния среднего времени пребывания заявок 2-го класса от коэффициента загрузки
    n = 7  # количество каналов
    K = 2  # количество классов
    ros = 0.85  # коэффициент загрузки СМО
    bH_to_bL = 2  # время обслуживания класса H меньше L в это число раз
    lH_to_lL = 1.5  # интенсивность поступления заявок класса H ниже L в это число раз
    l_H = 1.0  # интенсивность вх потока заявок 1-го класса
    l_L = lH_to_lL * l_H  # интенсивность вх потока заявок 2-го класса
    bH_coev = [0.63, 0.82]  # исследуемые коэффициенты вариации обсл заявок 1 класса
    iteration = 1  # кол-во итераций ИМ для получения более точных оценок ИМ

    v1_im_mass = []
    v2_im_mass = []
    v2_tt_mass = []
    iter_num = []
    tt_times = []
    im_times = []
    invar_times = []
    v2_invar_mass = []

    for k in range(len(bH_coev)):

        print("coev =  {0:5.3f}".format(bH_coev[k]))

        lsum = l_L + l_H
        bsr = n * ros / lsum
        bH1 = lsum * bsr / (l_L * bH_to_bL + l_H)
        bL1 = bH_to_bL * bH1
        bH = [0.0] * 3
        alpha = 1 / (bH_coev[k] ** 2)
        bH[0] = bH1
        bH[1] = math.pow(bH[0], 2) * (math.pow(bH_coev[k], 2) + 1)
        bH[2] = bH[1] * bH[0] * (1.0 + 2 / alpha)

        gamma_params = rd.Gamma.get_mu_alpha([bH[0], bH[1]])

        mu_L = 1.0 / bL1

        # задание ИМ:
        v1_sum = 0
        v2_sum = 0

        cox_params = rd.Cox_dist.get_params_clx(bH)

        # расчет численным методом:
        tt_start = time.process_time()
        tt = m_ph_n_prty(mu_L, cox_params[1], cox_params[2], cox_params[0], l_L, l_H, n=n, is_cox=is_cox, max_iter=max_iter)
        tt.run()
        tt_times.append(time.process_time() - tt_start)

        iter_num.append(tt.run_iterations_num_)
        p_tt = tt.get_p()
        v_tt = tt.get_low_class_v1()
        v2_tt_mass.append(v_tt)
        print("{0:^25s}".format("Вероятности состояний для заявок 2-го класса"))
        print("{0:^3s}|{1:^15s}|{2:^15s}".format("№", "Числ", "ИМ"))
        print("-" * 32)
        for i in range(11):
            print("{0:^4d}|{1:^15.3g}|{2:^15.3g}".format(i, p_tt[i], 0))
        print("{0:^15.3f}|{1:^15.6g}|{2:^15.6g}|{3:^15d}".format(bH_coev[k], v2_tt_mass[k], 0, iter_num[k]))

        mu_L = 1.0 / bL1

        bL = rd.Exp_dist.calc_theory_moments(mu_L, 3)

        b = []
        b.append(bH)
        b.append(bL)

        L = [l_H, l_L]

        invar_start = time.process_time()
        v = prty_calc.get_v_prty_invar(L, b, n=n, type='PR', num=2)
        v2_invar_mass.append(v[1][0])
        invar_times.append(time.process_time() - invar_start)

        im_start = time.process_time()

        for i in range(iteration):
            print("Start IM iteration: {0:d}".format(i + 1))

            smo = smo_im_prty.SmoImPrty(n, K, "PR")
            sources = []
            servers_params = []
            l = [l_H, l_L]

            sources.append({'type': 'M', 'params': l_H})
            sources.append({'type': 'M', 'params': l_L})
            servers_params.append({'type': 'Gamma', 'params': gamma_params})
            servers_params.append({'type': 'M', 'params': mu_L})

            smo.set_sources(sources)
            smo.set_servers(servers_params)

            # запуск ИМ:
            smo.run(num_of_jobs)

            # получение результатов ИМ:
            p = smo.get_p()
            v_im = smo.v
            v1_sum += v_im[0][0]
            v2_sum += v_im[1][0]

        v1 = v1_sum / iteration
        v2 = v2_sum / iteration
        # расчет численным методом:
        v2_im_mass.append(v2)
        im_times.append(time.process_time() - im_start)

    print("\nСравнение результатов расчета численным методом с аппроксимацией ПНЗ "
          "\nраспределением Кокса второго порядка и ИМ.")
    print("ro: {0:1.2f}".format(ros))
    print("n : {0:d}".format(n))
    print("Количество обслуженных заявок для ИМ: {0:d}\n".format(num_of_jobs))

    print("\n")
    print("{0:^35s}".format("Средние времена пребывания в СМО для заявок 2-го класса"))
    print("-" * 128)
    print("{0:^15s}|{1:^15s}|{5:^15s}|{2:^15s}|{3:^15s}|{5:^15s}|{4:^15s}|{5:^15s}".format("coev", "Числ",
                                                                                           "Кол-во итер алг", "ИМ",
                                                                                           "Инвар", "t, c"))
    print("-" * 128)
    for k in range(len(bH_coev)):
        print("{0:^15.3f}|{1:^15.6g}|{2:^15.6g}|{3:^15d}|{4:^15.6g}|{5:^15.6g}|{6:^15.6g}|{7:^15.6g}".format(
            bH_coev[k],
            v2_tt_mass[k], tt_times[k], iter_num[k],
            v2_im_mass[k], im_times[k],
            v2_invar_mass[k], invar_times[k]
        ))
