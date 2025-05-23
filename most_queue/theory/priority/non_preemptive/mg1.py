"""
Class to calculate the average waiting time in an M/G/1 queue with non-preemptive priority.
"""
from most_queue.rand_distribution import GammaDistribution
from most_queue.theory.utils.busy_periods import busy_calc
from most_queue.theory.utils.conv import conv_moments
from most_queue.theory.utils.diff5dots import diff5dots
from most_queue.theory.utils.transforms import lst_gamma


class MG1NonPreemtiveCalculation:
    """
    Class to calculate the average waiting time in an M/G/1 queue with non-preemptive priority.
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
        Calculation of the average waiting time in M/G/1 with non-preemptive priority.
        """
        k = len(self.l)
        w1 = [0.0] * k
        R = [0, 0] * k
        ro = [0.0] * k

        for j in range(k):
            ro[j] = self.l[j] * self.b[j][0]
            s = 0
            for i in range(j + 1):
                s += ro[i]
            R[j] = s
        for j in range(k):
            s = 0
            for i in range(k):
                s += self.l[i] * self.b[i][1]
            if j == 0:
                w1[j] = s / (2 * (1 - R[j]))
            else:
                w1[j] = s / (2 * (1 - R[j]) * (1 - R[j - 1]))
        return w1

    def get_v(self, num=3) -> list[list[float]]:
        """
        Calculation of initial moments of sojourn time in M/G/1 with non-preemptive priority.
        :param num: number of moments to calculate
        :return: list of initial moments of sojourn time for each class
        """
        k = len(self, self.l)
        v = []
        w = self.get_w()

        v = [conv_moments(w[i], self.b[i], num) for i in range(k)]

        return v

    def get_w(self) -> list[list[float]]:
        """
        Calculation of initial moments of waiting time in M/G/1 with non-preemptive priority.
        :return: list of initial moments of waiting time for each class
        """
        # a - lower pr
        # j - the same
        # e - higher pr

        num_of_cl = len(self.l)
        w = []
        ro = 0
        L = []
        for i in range(num_of_cl):
            ro += self.l[i] * self.b[i][0]
            summ = 0
            for s in range(i + 1):
                summ += self.l[s]
            L.append(summ)

        for j in range(num_of_cl):
            w.append([])
            w[j] = [0.0] * (len(self.b[j]) - 1)

            la = 0
            for i in range(j):
                la += self.l[i]

            lb = 0
            for i in range(j + 1, num_of_cl):
                lb += self.l[i]

            b_i = self.b[j]
            num_of_mom = len(b_i)

            b_a = [0.0] * num_of_mom
            for m in range(num_of_mom):
                if j == 0:
                    b_a[m] = 0
                else:
                    summ = 0
                    for i in range(j):
                        summ += self.l[i] * self.b[i][m]
                    b_a[m] = summ / la

            b_b = [0.0] * num_of_mom
            for m in range(num_of_mom):
                if j == num_of_cl - 1:
                    b_b[m] = 0
                else:
                    summ = 0
                    for i in range(j + 1, num_of_cl):
                        summ += self.l[i] * self.b[i][m]
                    b_b[m] = summ / lb

            h = 0.0001
            steps = 5

            if j != num_of_cl - 1:
                b_b_param = GammaDistribution.get_params(b_b)
            else:
                b_b_param = 0

            b_k_param = GammaDistribution.get_params(self.b[j])

            nu_a_busy = busy_calc(la, b_a)

            if j != 0:
                nu_a_param = GammaDistribution.get_params(nu_a_busy)
            else:
                nu_a_param = 0

            w_pls = []

            for c in range(1, steps):
                s = h * c

                if j != 0:
                    nu_a = lst_gamma(nu_a_param, s)
                    summ = s + la - la * nu_a
                else:
                    summ = s

                chisl = (1 - ro) * summ

                if j != len(self.l) - 1:
                    chisl += lb * \
                        (1 - lst_gamma(b_b_param, summ))

                znam = self.l[j] * \
                    lst_gamma(b_k_param, summ) - self.l[j] + s

                w_pls.append(chisl / znam)

                w[j] = diff5dots(w_pls, h)
                w[j][0] = -w[j][0]
                if len(self.b[j]) > 2:
                    w[j][2] = -w[j][2]

        return w
