"""
Class to calculate M/G/1 queue with preemptive (absolute) priority.
"""
import math

from most_queue.theory.utils.busy_periods import busy_calc, busy_calc_warm_up


class MG1PreemtiveCalculation:
    """
    Class to calculate M/G/1 queue with preemptive (absolute) priority.
    """

    def __init__(self, l: list[float], b: list[list[float]]):
        """
        :param l: list[float]  - list of arrival intensities for each job class 
        :param b: list[list[float]] - list of initial moments for each job class
        """
        self.l = l
        self.b = b

    def get_w1(self):
        """
        Calculation of the average waiting time in M/G/1 with absolute priority.

        """
        k = len(self.l)
        w1 = [0.0] * k
        R = [0.0] * k
        ro = [0.0] * k

        for j in range(k):
            ro[j] = self.l[j] * self.b[j][0]
            s = 0
            for i in range(j + 1):
                s += ro[i]
            R[j] = s
        for j in range(k):
            s = 0
            for i in range(j + 1):
                s += self.l[i] * self.b[i][1]
            if j == 0:
                w1[j] = s / (2 * (1 - R[j]))
            else:
                w1[j] = s / (2 * (1 - R[j]) * (1 - R[j - 1]))

            if j != 0:
                w1[j] += self.b[j][0] * R[j - 1] / (1 - R[j - 1])
        return w1

    def climov_waiting_times(self) -> list[float]:
        """
        Calculate waiting times using Climov's method.
        Returns:
            list: waiting times
        """
        # только два первых момента
        k_num = len(self.l)
        ro_k_j = []
        ro_k = []
        w = []
        for k in range(k_num):
            ro_k.append([])
            ro_k_j.append([0.0] * 3)
            for j in range(3):
                for i in range(k + 1):
                    ro_k_j[k][j] += self.l[i] * self.b[i][j]
            ro_k[k] = 1 - ro_k_j[k][0]
        for k in range(k_num):
            w.append([0.0] * 2)
            if k != 0:
                w[k][0] = ro_k_j[k][1] / (2 * ro_k[k - 1] * ro_k[k])
                w[k][1] = ro_k_j[k][2] / (3 * ro_k[k]) + (ro_k_j[k][2]) / (
                    2 * math.pow(ro_k[k], 2)) + \
                    (ro_k_j[k][1]) / (2 * ro_k[k])

            else:
                w[k][0] = ro_k_j[k][1] / (2 * ro_k[k])
                w[k][1] = ro_k_j[k][2] / (3 * math.pow(ro_k[k - 1], 3) * ro_k[k]) + (ro_k_j[k][2] * ro_k_j[k - 1][1]) / (
                    2 * math.pow(ro_k[k - 1], 2) * math.pow(ro_k[k], 2)) + \
                    (ro_k_j[k][1] * ro_k_j[k - 1][1]) / \
                    (2 * math.pow(ro_k[k - 1], 3) * ro_k[k])

        return w

    def calc_all(self, num=3):
        """
        Calculate initial moments of sojourn time, waiting 
        for service without and with interruptions, active time
          and busy period in M/G/1 with absolute priority.
        :param num: number of moments to calculate
        :return:
         - sojourn initial moments of continuous busy period
         - waiting initial moments of continuous busy period
         - active time initial moments of continuous busy period
         - busy period initial moments of continuous busy period

        return res:
        res['v'][k][j] -initial moments of sojourn time
        res['w'][k][j] - initial moments of waiting for service without interruptions
        res['h'][k][j] - initial moments of active time
        res['busy'][k][j] - initial moments of busy period
        res['w_with_pr'][k][j] - initial moments of waiting for service with interruptions
        """
        num_of_cl = len(self.l)
        L = []
        for i in range(num_of_cl):
            summ = 0
            for j in range(i + 1):
                summ += self.l[j]
            L.append(summ)

        pi_j_i = []
        pi_j_i.append([])
        pi_j = []
        w = []
        v = []
        h = []

        pi_j.append(busy_calc(self.l[0], self.b[0]))

        # Формула Полячека - Хинчина. Заявки первого
        # класса не прерываются
        w.append([0.0] * num)

        for i in range(num):
            summ = self.b[0][i + 1] / (i + 2)
            for s in range(1, i + 1):
                summ += self.b[0][i + 1 - s] * w[0][s] * \
                    math.factorial(i + 1) / (math.factorial(s)
                                             * math.factorial(i + 2 - s))
            w[0][i] = summ * self.l[0] / (1 - self.l[0] * self.b[0][0])

        v.append([0.0] * num)
        v[0][0] = w[0][0] + self.b[0][0]
        v[0][1] = w[0][1] + 2 * w[0][0] * self.b[0][0] + self.b[0][1]
        if num > 2:
            v[0][2] = w[0][2] + 3 * w[0][1] * \
                self.b[0][0] + 3 * w[0][0] * self.b[0][1] + self.b[0][2]

        h.append(self.b[0])

        for j in range(1, num_of_cl):
            pi_j.append([0.0] * (num + 1))
            h.append(busy_calc_warm_up(L[j - 1], self.b[j], pi_j[j - 1]))

            pi_j_i.append([])
            for _ in range(j + 1):
                pi_j_i[j].append([])

            pi_j_i[j][j] = busy_calc(self.l[j], h[j])

            for i in range(j):
                if j == 1:
                    pi_j_i[j][i] = busy_calc_warm_up(
                        self.l[j], pi_j[0], pi_j_i[j][j])
                else:
                    pi_j_i[j][i] = busy_calc_warm_up(
                        self.l[j], pi_j_i[j - 1][i], pi_j_i[j][j])

            for moment in range(num + 1):
                summ = 0
                for i in range(j + 1):
                    summ += self.l[i] * pi_j_i[j][i][moment]
                pi_j[j][moment] = summ / L[j]

            w.append([0.0] * (num + 1))
            w[j][0] = 1
            v.append([0.0] * num)

            c = (1.0 - self.l[j] * h[j][0]) / (1.0 + L[j - 1] * pi_j[j - 1][0])
            for i in range(1, num + 1):
                summ = 0
                for m in range(i):
                    summ += w[j][m] * h[j][i - m] * \
                        math.factorial(i) / (math.factorial(m) *
                                             math.factorial(i + 1 - m))

                w[j][i] = (c * L[j] * pi_j[j - 1][i] / (i + 1) +
                           self.l[j] * summ) / (1.0 - self.l[j] * h[j][0])
            w[j] = w[j][1:]
            v[j][0] = w[j][0] + h[j][0]
            v[j][1] = w[j][1] + 2 * w[j][0] * h[j][0] + h[j][1]
            if num > 2:
                v[j][2] = w[j][2] + 3 * w[j][1] * \
                    h[j][0] + 3 * w[j][0] * h[j][1] + h[j][2]

        res = {}
        res['v'] = v
        res['w'] = w
        res['h'] = h
        w_with_pr = []
        for j in range(num_of_cl):
            w_with_pr.append([0.0] * 3)
            w_with_pr[j][0] = v[j][0] - self.b[j][0]
            w_with_pr[j][1] = v[j][1] - 2 * \
                w_with_pr[j][0] * self.b[j][0] - self.b[j][1]
            if num > 2:
                w_with_pr[j][2] = v[j][2] - 3 * w_with_pr[j][1] * \
                    self.b[j][0] - 3 * w_with_pr[j][0] * \
                    self.b[j][1] - self.b[j][2]
        res['w_with_pr'] = w_with_pr
        res['busy'] = pi_j

        return res
