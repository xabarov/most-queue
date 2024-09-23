from sim import flow_sum_sim

import copy
import matplotlib.pyplot as plt
import sim.rand_destribution as rd
import math

import matplotlib
matplotlib.use('TkAgg')


class SummatorNumeric:

    def __init__(self, a, verbose=False, is_semi=False):
        self.n = len(a)
        self.a = a
        self.num_of_moments = len(a[0])
        self.is_semi = is_semi
        self.verbose = verbose
        self.coevs = []
        self.result_flow = []
        self.flows_ = []
        self.a1_sum = []
        self.a2_sum = []

    def sum_flows(self):
        """
        суммирование n потоков
        a[i][j] - i - номер потока, j номер начального момента интервалов между соседникми заявками i потока
        """
        n = len(self.a)  # число суммируемых потоков

        for i in range(n - 1):
            if self.verbose:
                print("Summation of flows. Start calculation {0} from {1}. ".format(i + 1, n - 1))
            if not self.is_semi:
                f1 = self.sum_2_H2_flows(self.a[0], self.a[1])
            else:
                f1 = self.sum_2_flows_semiinvariants(self.a[0], self.a[1])

            self.flows_.append(f1)
            self.coevs.append(SummatorNumeric.get_coev(f1))
            f = []
            f.append(f1)

            for j in range(len(self.a) - 2):
                f.append(self.a[j + 2])

            self.a = copy.deepcopy(f)

        for i in range(len(self.flows_)):
            self.a1_sum.append(self.flows_[i][0])
            self.a2_sum.append(self.flows_[i][1])

        self.result_flow = self.a[0]

    @staticmethod
    def get_residual_distr_moments(a):
        f = [0.0] * (len(a) - 1)
        for i in range(len(f)):
            f[i] = a[i + 1] / ((i + 2) * a[0])
        return f

    @staticmethod
    def get_v(k):
        """
        Вычисление нормированных семиинвариантов числа событий рекуррентного потока
        """
        v = []
        v.append(1.0 / k[0])
        if len(k) > 1:
            v.append(k[1] / pow(k[0], 3))
        if len(k) > 2:
            v3 = 3 * pow(k[1], 2) / pow(k[0], 5)
            v3 -= k[2] / pow(k[0], 4)
            v.append(v3)
        if len(k) > 3:
            v4 = k[3] / pow(k[0], 5)
            v4 -= 10 * k[1] * k[2] / pow(k[0], 6)
            v4 += 15 * pow(k[1], 3) / pow(k[0], 7)
            v.append(v4)

        return v

    @staticmethod
    def get_k(a):
        """
        Вычисление семиинвариантов
        """
        k = []
        k.append(a[0])
        if len(a) > 1:
            k.append(a[1] - pow(a[0], 2))
        if len(a) > 2:
            k.append(a[2] - 3 * a[1] * a[0] + 2 * pow(a[0], 3))
        if len(a) > 3:
            k.append(a[3] - 4 * a[2] * a[0] - 3 * pow(a[1], 2) + 12 * a[1] * pow(a[0], 2) - 6 * pow(a[0], 4))

        return k

    @staticmethod
    def get_k_by_v(v):
        """
        Вычисление семиинвариантов по v -нормированным семиинвариантам числа событий рекуррентного потока
        """
        k = []
        k.append(1.0 / v[0])
        if len(v) > 1:
            k.append(v[1] / pow(v[0], 3))
        if len(v) > 2:
            k.append((3 * pow(v[1], 2) - v[0] * v[2]) / pow(v[0], 5))
        if len(v) > 3:
            k4 = v[3] / pow(v[0], 2) + 5 * v[1] * (3 * pow(v[1], 2) - 2 * v[0] * v[2])
            k4 = k4 / pow(v[0], 7)
            k.append(k4)
        return k

    @staticmethod
    def get_a(k):
        """
        Вычисление нач моментов по семиинвариантам
        """
        a = []
        a.append(k[0])
        if len(k) > 1:
            a.append(k[1] + pow(k[0], 2))
        if len(k) > 2:
            a.append(k[2] + 3 * k[0] * k[1] + pow(k[0], 3))
        if len(k) > 3:
            a.append(k[3] + 4 * k[2] * k[0] + 3 * pow(k[1], 2) + 6 * k[1] * pow(k[0], 2) + pow(k[0], 4))

        return a

    @staticmethod
    def get_coev(a):
        D = a[1] - pow(a[0], 2)
        coev = math.sqrt(D.real) / a[0].real
        return coev

    @staticmethod
    def sum_2_flows_semiinvariants(a1, a2):

        k1 = SummatorNumeric.get_k(a1)
        k2 = SummatorNumeric.get_k(a2)

        min_len = min(len(a1), len(a2))

        v1 = SummatorNumeric.get_v(k1)
        v2 = SummatorNumeric.get_v(k2)

        v_sum = []
        for i in range(min_len):
            v_sum.append(v1[i] + v2[i])

        k_sum = SummatorNumeric.get_k_by_v(v_sum)
        a_sum = SummatorNumeric.get_a(k_sum)

        return a_sum

    @staticmethod
    def get_error(test, real):
        return 100*(test-real)/real

    @staticmethod
    def sum_2_H2_flows(a1, a2):
        """
        суммирование двух потоков c произвольными коэффициентами вариации 
        Аппроксимация H2-распределением с комплексными параметрами
        a1 - список из начальных моментов интервалов между соседникми заявками первого потока
        a2 - список из начальных моментов интервалов между соседникми заявками второго потока
        """

        # Вычисляем начальныем моменты первого произведения A1, A2*
        y1_mus = rd.H2_dist.get_params_clx(a1)
        y = [y1_mus[0], 1.0 - y1_mus[0]]
        mu = [y1_mus[1], y1_mus[2]]
        y2_mus = rd.H2_dist.get_params_clx(a2)
        u1_lambdas = rd.H2_dist.get_residual_params(y2_mus)
        u = [u1_lambdas[0], 1.0 - u1_lambdas[0]]
        lambdas = [u1_lambdas[1], u1_lambdas[2]]

        f_first = []
        for i in range(len(a1)):
            f_first.append(0)

        for k in range(len(a1)):
            summ = 0
            for i in range(2):
                for j in range(2):
                    summ += y[i] * u[j] * math.factorial(k + 1) / pow(mu[i] + lambdas[j], k + 1)
            f_first[k] = summ

        # Вычисляем начальныем моменты второго произведения A1*, A2

        y1_mus = rd.H2_dist.get_params_clx(a2)
        y = [y1_mus[0], 1.0 - y1_mus[0]]
        mu = [y1_mus[1], y1_mus[2]]
        y2_mus = rd.H2_dist.get_params_clx(a1)
        u1_lambdas = rd.H2_dist.get_residual_params(y2_mus)
        u = [u1_lambdas[0], 1.0 - u1_lambdas[0]]
        lambdas = [u1_lambdas[1], u1_lambdas[2]]

        f_second = []
        for i in range(len(a1)):
            f_second.append(0)

        for k in range(len(a1)):
            summ = 0
            for i in range(2):
                for j in range(2):
                    summ += y[i] * u[j] * math.factorial(k + 1) / pow(mu[i] + lambdas[j], k + 1)
            f_second[k] = summ

        # Итоговые моменты:
        f = []
        for i in range(len(a1)):
            f.append(0)

        l1 = 1 / a1[0]
        l2 = 1 / a2[0]
        l_sum = l1 + l2
        for k in range(len(a1)):
            f[k] = (l1 * f_first[k] + l2 * f_second[k]) / l_sum

        return f


if __name__ == "__main__":

    # Тестирование суммирования потоков

    # Задаем следующие параметры:
    n_nums = 10  # число суммируемых потоков
    coev = 0.74  # коэффициент вариации каждого потока
    mean = 1  # среднее каждого потока
    num_of_jobs = 400000  # количество заявок для ИМ
    is_semi = False  # True, если необходимо использовать метод семиинвариантов вместо H2
    distr_im = "Gamma"
    ns = [x + 2 for x in range(n_nums - 1)]

    a = []
    for i in range(n_nums):
        params1 = rd.Gamma.get_mu_alpha_by_mean_and_coev(mean, coev)
        a1 = rd.Gamma.calc_theory_moments(*params1, 4)
        a.append(a1)

    s = SummatorNumeric(a, is_semi=is_semi)
    s_im = flow_sum_sim.FlowSumSim(a, distr=distr_im, num_of_jobs=num_of_jobs)

    s.sum_flows()
    s_im.sum_flows()

    coevs_im = s_im.coevs
    coevs_num = s.coevs
    errors1 = []
    errors2 = []
    errors_coev = []

    str_f = "{0:^18s}|{1:^10.3f}|{2:^10.3f}|{3:^10.3f}|{4:^10.3f}|{5:^10.3f}"
    print("{0:^18s}|{1:^10s}|{2:^10s}|{3:^10s}|{4:^10s}|{5:^10s}".format("-", "a1", "a2", "a3", "a4", "coev"))
    print("-" * 80)

    for i in range(n_nums - 1):
        print("{0:^80s}".format("Сумма " + str(i + 2) + " потоков"))
        print("-" * 80)
        print(str_f.format("ИМ", s_im.flows_[i][0], s_im.flows_[i][1], s_im.flows_[i][2], s_im.flows_[i][3],
                           coevs_num[i]))
        print("-" * 80)
        print(str_f.format("Числ", s.flows_[i][0].real, s.flows_[i][1].real, s.flows_[i][2].real, s.flows_[i][3].real,
                           coevs_im[i]))
        print("-" * 80)
        errors1.append(SummatorNumeric.get_error(s.flows_[i][0].real, s_im.flows_[i][0]))
        errors2.append(SummatorNumeric.get_error(s.flows_[i][1].real, s_im.flows_[i][1]))
        errors_coev.append(SummatorNumeric.get_error(coevs_num[i], coevs_im[i]))

    fig, ax = plt.subplots()
    linestyles = ["solid", "dotted", "dashed", "dashdot"]

    # ax.plot(ns, coevs_im, label="ИМ")
    # ax.plot(ns, coevs_num, label="Числ")
    #
    # ax.plot(ns, s_im.a1_sum, label="ИМ a1", linestyle=linestyles[0])
    # ax.plot(ns, s.a1_sum, label="Числ a1", linestyle=linestyles[1])
    #
    # ax.plot(ns, s_im.a2_sum, label="ИМ a2", linestyle=linestyles[2])
    # ax.plot(ns, s.a2_sum, label="Числ a2", linestyle=linestyles[3])

    ax.plot(ns, errors1, label="error a1", linestyle=linestyles[0])
    ax.plot(ns, errors2, label="error a2", linestyle=linestyles[1])
    ax.plot(ns, errors_coev, label="error coev", linestyle=linestyles[2])

    plt.legend()
    str_title = "Отн. ошибка от числа сумм-х потоков, %"
    if is_semi:
        str_title += ". Метод семиинвариантов"
    else:
        str_title += ". Метод H2"
    plt.title(str_title)
    plt.show()
