import math

"""
Calculation of M/M/1 QS with batch arrival 
"""


class BatchMM1:
    """
    Calculation of M/M/1 QS with batch arrival 
    """
    def __init__(self, lam, mu, batch_probs, p_num=100000, tol=1e-12):
        """_
            lam: (float): arrival Exp distribution param
            mu (float): serving Exp distribution param
            batch_probs (list[float]): batch probs 
            p_num (int, optional): size of probs to calc. Defaults to 100000.
            tol (float, optional): tol, precision of calcs. Defaults to 1e-12.
        """
        self.lam = lam
        self.mu = mu
        self.ls = list(batch_probs)
        self.p_num = p_num
        self.tol = tol
        moments = self.calc_l_moments()
        self.l = moments[0]
        self.l2 = moments[1]

    def calc_l(self):
        """
        Mean batch size
        """
        return sum(self.ls) / len(self.ls)

    def calc_l_moments(self):
        """
        Initial moments of the number of jobs in the batch
        """
        moments = [0.0, 0.0]
        for j, mom in enumerate(moments):
            for i, prob in enumerate(self.ls):
                moments[j] += pow(i + 1, j + 1)*prob
        return moments

    def calc_L(self):
        """
        Returns list of probabilities
        of arrival more than i applications in a group
        """
        Ls = []
        Ls.append(1)
        summ = 0
        for i, prob in enumerate(self.ls):
            summ += prob
            Ls.append(1.0 - summ)

        return Ls

    def get_p(self):
        """
        Probs of QS states
        """
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
        """
        Mean jobs in QS
        """
        ro = self.l * self.lam / self.mu
        ro_tilda = self.lam / self.mu
        return (self.l2 + self.l) * ro_tilda / (2 * (1.0 - ro))

    def get_Q(self):
        """
        Mean queue length
        """
        N = self.get_N()
        ro = self.l * self.lam / self.mu
        return N - ro

    def get_w1(self):
        """
        Mean wait time
        """
        return self.get_Q() / (self.l * self.lam)

    def get_v1(self):
        """
        Mean sojourn time
        """
        return self.get_N() / (self.l * self.lam)

