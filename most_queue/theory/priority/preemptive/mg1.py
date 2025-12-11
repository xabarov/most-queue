"""
Class to calculate M/G/1 queue with preemptive (absolute) priority.
"""

import math

from most_queue.structs import PriorityResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.utils.busy_periods import busy_calc, busy_calc_warm_up


class MG1PreemptiveCalc(BaseQueue):
    """
    Class to calculate M/G/1 queue with preemptive (absolute) priority.
    """

    def __init__(self) -> None:
        """
        Initialize the MG1PreemptiveCalc class.
        """
        super().__init__(n=1)
        self.l = None
        self.b = None

    def set_sources(self, l: list[float]) -> None:  # pylint: disable=arguments-differ
        """
        Set the arrival rates for each priority class.

        Args:
            l: List of arrival rates for each priority class.
        """
        self.l = l
        self.is_sources_set = True

    def set_servers(self, b: list[list[float]]) -> None:  # pylint: disable=arguments-differ
        """
        Set the raw moments of service time distribution for each priority class.

        Args:
            b: List of raw moments of service time distribution for each priority class.
        """
        self.b = b
        self.is_servers_set = True

    def get_w1(self):
        """
        Calculation of the average waiting time in M/G/1 with absolute priority.

        """

        self._check_if_servers_and_sources_set()
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

    def run(self, num: int = 3) -> PriorityResults:
        """
        Calculate raw moments of sojourn time, waiting for service without and with
        interruptions, active time and busy period in M/G/1 with absolute priority.

        Args:
            num: Number of moments to calculate.

        Returns:
            PriorityResults containing:
            - v: Raw moments of sojourn time for each class
            - w: Raw moments of waiting for service without interruptions for each class
            - h: Raw moments of active time for each class
            - busy: Raw moments of busy period for each class
            - w_with_pr: Raw moments of waiting for service with interruptions for each class
        """
        start = self._measure_time()

        self._check_if_servers_and_sources_set()
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
                summ += (
                    self.b[0][i + 1 - s]
                    * w[0][s]
                    * math.factorial(i + 1)
                    / (math.factorial(s) * math.factorial(i + 2 - s))
                )
            w[0][i] = summ * self.l[0] / (1 - self.l[0] * self.b[0][0])

        v.append([0.0] * num)
        v[0][0] = w[0][0] + self.b[0][0]
        v[0][1] = w[0][1] + 2 * w[0][0] * self.b[0][0] + self.b[0][1]
        if num > 2:
            v[0][2] = w[0][2] + 3 * w[0][1] * self.b[0][0] + 3 * w[0][0] * self.b[0][1] + self.b[0][2]

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
                    pi_j_i[j][i] = busy_calc_warm_up(self.l[j], pi_j[0], pi_j_i[j][j])
                else:
                    pi_j_i[j][i] = busy_calc_warm_up(self.l[j], pi_j_i[j - 1][i], pi_j_i[j][j])

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
                    summ += w[j][m] * h[j][i - m] * math.factorial(i) / (math.factorial(m) * math.factorial(i + 1 - m))

                w[j][i] = (c * L[j] * pi_j[j - 1][i] / (i + 1) + self.l[j] * summ) / (1.0 - self.l[j] * h[j][0])
            w[j] = w[j][1:]
            v[j][0] = w[j][0] + h[j][0]
            v[j][1] = w[j][1] + 2 * w[j][0] * h[j][0] + h[j][1]
            if num > 2:
                v[j][2] = w[j][2] + 3 * w[j][1] * h[j][0] + 3 * w[j][0] * h[j][1] + h[j][2]

        w_with_pr = []
        for j in range(num_of_cl):
            w_with_pr.append([0.0] * 3)
            w_with_pr[j][0] = v[j][0] - self.b[j][0]
            w_with_pr[j][1] = v[j][1] - 2 * w_with_pr[j][0] * self.b[j][0] - self.b[j][1]
            if num > 2:
                w_with_pr[j][2] = (
                    v[j][2] - 3 * w_with_pr[j][1] * self.b[j][0] - 3 * w_with_pr[j][0] * self.b[j][1] - self.b[j][2]
                )

        utilization = self.get_utilization()

        result = PriorityResults(v=v, w=w, h=h, w_with_pr=w_with_pr, busy=pi_j, utilization=utilization)
        self._set_duration(result, start)
        return result

    def get_utilization(self) -> float:
        """
        Calc utilization factor
        """
        b_ave = sum(b[0] for b in self.b) / len(self.b)
        l_sum = sum(self.l)

        return l_sum * b_ave
