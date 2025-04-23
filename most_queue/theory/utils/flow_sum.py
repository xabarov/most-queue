import copy
import math

import matplotlib
import matplotlib.pyplot as plt

from most_queue.rand_distribution import H2Distribution

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
                print("Summation of flows. Start calculation {0} from {1}. ".format(
                    i + 1, n - 1))
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
            k.append(a[3] - 4 * a[2] * a[0] - 3 * pow(a[1], 2) +
                     12 * a[1] * pow(a[0], 2) - 6 * pow(a[0], 4))

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
            k4 = v[3] / pow(v[0], 2) + 5 * v[1] * \
                (3 * pow(v[1], 2) - 2 * v[0] * v[2])
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
            a.append(k[3] + 4 * k[2] * k[0] + 3 * pow(k[1], 2) +
                     6 * k[1] * pow(k[0], 2) + pow(k[0], 4))

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
        y1_mus = H2Distribution.get_params_clx(a1)
        y = [y1_mus.p1, 1.0 - y1_mus.p1]
        mu = [y1_mus.mu1, y1_mus.mu2]
        y2_mus = H2Distribution.get_params_clx(a2)
        u1_lambdas = H2Distribution.get_residual_params(y2_mus)
        u = [u1_lambdas.p1, 1.0 - u1_lambdas.p1]
        lambdas = [u1_lambdas.mu1, u1_lambdas.mu2]

        f_first = []
        for i in range(len(a1)):
            f_first.append(0)

        for k in range(len(a1)):
            summ = 0
            for i in range(2):
                for j in range(2):
                    summ += y[i] * u[j] * \
                        math.factorial(k + 1) / pow(mu[i] + lambdas[j], k + 1)
            f_first[k] = summ

        # Вычисляем начальныем моменты второго произведения A1*, A2

        y1_mus = H2Distribution.get_params_clx(a2)
        y = [y1_mus.p1, 1.0 - y1_mus.p1]
        mu = [y1_mus.mu1, y1_mus.mu2]
        y2_mus = H2Distribution.get_params_clx(a1)
        u1_lambdas = H2Distribution.get_residual_params(y2_mus)
        u = [u1_lambdas.p1, 1.0 - u1_lambdas.p1]
        lambdas = [u1_lambdas.mu1, u1_lambdas.mu2]

        f_second = []
        for i in range(len(a1)):
            f_second.append(0)

        for k in range(len(a1)):
            summ = 0
            for i in range(2):
                for j in range(2):
                    summ += y[i] * u[j] * \
                        math.factorial(k + 1) / pow(mu[i] + lambdas[j], k + 1)
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
