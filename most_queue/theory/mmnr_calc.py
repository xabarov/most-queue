import math

from most_queue.general_utils.conv import get_moments
from most_queue.rand_distribution import Exp_dist


class MMnr_calc:

    @staticmethod
    def getPI(l, mu, n, r):

        ro = l / mu
        p = MMnr_calc.get_p(l, mu, n, r)

        chisl = math.pow(ro, n + r) * p[0]
        znam = math.factorial(n) * math.pow(n, r)
        return chisl / znam

    @staticmethod
    def getQ(l, mu, n, r):
        ro = l / mu
        p = MMnr_calc.get_p(l, mu, n, r)
        sum = 0
        for i in range(1, r + 1):
            sum += i * math.pow(ro / n, i)
        return p[n] * sum

    @staticmethod
    def get_qs(l, mu, n, r, q_num=3):
        p = MMnr_calc.get_p(l, mu, n, r)
        q_s = []
        for k in range(1, q_num + 1):
            summ = 0
            for nn in range(k, r + 1):
                summ += (math.factorial(nn) / math.factorial(nn - k)) * p[nn+n]
            q_s.append(summ)
        return q_s

    @staticmethod
    def get_w(l, mu, n, r, num=3):
        qs = MMnr_calc.get_qs(l, mu, n, r, q_num=num)
        w = [0] * num
        for k in range(num):
            w[k] = qs[k] / pow(l, k + 1)
        return w

    @staticmethod
    def get_v(l, mu, n, r):
        w = MMnr_calc.get_w(l, mu, n, r)
        b = Exp_dist.calc_theory_moments(mu)
        v = get_moments(w, b)
        return v

    @staticmethod
    def get_p(l, mu, n, r):

        p = [0] * (n + r + 1)
        ro = l / mu

        summ1 = 0
        for i in range(n):
            summ1 += pow(ro, i) / math.factorial(i)

        chisl = 1 - pow(ro / n, r + 1)
        coef = pow(ro, n) / math.factorial(n)
        znam = 1 - (ro / n)

        p[0] = 1.0 / (summ1 + coef * chisl / znam)

        for i in range(n):
            p[i] = pow(ro, i) * p[0] / math.factorial(i)

        for i in range(n, n + r + 1):
            p[i] = pow(ro, i) * p[0] / (math.factorial(n) * pow(n, i - n))

        return p
