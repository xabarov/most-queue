import math

"""
Расчет СМО М/M/1 с групповым прибытием заявок
"""


class BatchMM1:

    def __init__(self, lam, mu, ls, p_num=100000, tol=1e-12):
        self.lam = lam
        self.mu = mu
        self.ls = ls
        self.p_num = p_num
        self.tol = tol
        moments = self.calc_l_moments()
        self.l = moments[0]
        self.l2 = moments[1]

    def calc_l(self):
        return sum(self.ls) / len(self.ls)

    def calc_l_moments(self):
        moments = [0.0, 0.0]
        for j in range(len(moments)):
            for i in range(len(self.ls)):
                moments[j] += pow(i + 1, j + 1)*self.ls[i]
        return moments

    def calc_L(self):
        Ls = []
        Ls.append(1)
        summ = 0
        for i in range(len(self.ls)):
            summ += self.ls[i]
            Ls.append(1.0 - summ)

        return Ls

    def get_p(self):
        ro = self.lam / self.mu
        p0 = 1.0 - self.l * ro
        Ls = self.calc_L()
        rs = []
        rs.append(1)

        ps = []
        ps.append(p0)

        for i in range(1, self.p_num):
            summ = 0
            for j in range(i):
                num = i - j - 1
                if num < len(self.ls):
                    summ += rs[j] * Ls[num]

            r_tek = ro * summ
            rs.append(r_tek)

            ps.append(p0 * r_tek)
            if ps[i] < self.tol:
                break

        return ps

    def get_N(self):
        ro = self.l * self.lam / self.mu
        ro_tilda = self.lam / self.mu
        return (self.l2 + self.l) * ro_tilda / (2 * (1.0 - ro))

    def get_Q(self):
        N = self.get_N()
        ro = self.l * self.lam / self.mu
        return N - ro

    def get_w1(self):
        return self.get_Q() / (self.l * self.lam)

    def get_v1(self):
        return self.get_N() / (self.l * self.lam)

