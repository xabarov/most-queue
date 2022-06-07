import math

class M_M_n_formula:

    @staticmethod
    def getPI(l, mu, n, r):

        ro = l / mu
        p = M_M_n_formula.getp(l, mu, n, r)

        chisl = math.pow(ro, n + r) * p[0]
        znam = math.factorial(n) * math.pow(n, r)
        return chisl / znam

    @staticmethod
    def getQ(l, mu, n, r):
        ro = l / mu
        p = M_M_n_formula.getp(l, mu, n, r)
        sum = 0
        for i in range(1, r+1):
            sum += i * math.pow(ro / n, i)
        return p[n] * sum

    @staticmethod
    def getW(l, mu, n, r):
        q = M_M_n_formula.getQ(l, mu, n , r)
        w = q*l
        return w

    @staticmethod
    def getV(l, mu, n, r):
        w = M_M_n_formula.getW(l, mu, n, r)
        return w + 1/mu

    @staticmethod
    def getp(l, mu, n, r):

        p = []
        sum1 = 0
        ro = l / mu
        sum2 = 0
        for i in range(1, r+1):
            sum2 +=math.pow(ro/n, i)
        sum2 *= (math.pow(ro, n)/math.factorial(n))

        for i in range(n+1):
            sum1 += math.pow(ro, i) / math.factorial(i)

        p.append(1 / (sum1 + sum2))

        for i in range(1, n+r+1):
            if (i <= n):
                p.append(math.pow(ro, i) * p[0] / math.factorial(i))
            else:
                p.append(math.pow(ro, i) * p[0] / math.factorial(n) * math.pow(n, i - n))
        return p
