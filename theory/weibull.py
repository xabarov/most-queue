import math
import rand_destribution as rd
import numpy as np


class Weibull:
    @staticmethod
    def get_params(t, num=2):
        # t - начальные моменты СВ
        a = t[1] / (t[0] * t[0])
        u0 = math.log(2 * a) / (2.0 * math.log(2))
        ee = 1e-6
        u1 = (1.0 / (2 * math.log(2))) * math.log(
            a * math.sqrt(math.pi) * rd.Gamma.get_gamma(u0 + 1) / rd.Gamma.get_gamma(u0 + 0.5))
        delta = u1 - u0
        while math.fabs(delta) > ee:
            u1 = (1.0 / (2 * math.log(2))) * math.log(
                a * math.sqrt(math.pi) * rd.Gamma.get_gamma(u0 + 1) / rd.Gamma.get_gamma(u0 + 0.5))
            delta = u1 - u0
            u0 = u1
        k = 1 / u1
        T = math.pow(t[0] / rd.Gamma.get_gamma(u1 + 1), k)
        weibullParam = [k, T]
        if num > 2:
            b = [0, 0, 0]
            for i in range(3):
                b[i] = k * t[i] / (i + 1)
            A = np.matrix(np.zeros((3, 3)))
            B = np.matrix(np.zeros((3, 1)))
            for i in range(3):
                B[i, 0] = b[i]

            for j in range(3):
                for i in range(3):
                    A[i, j] = rd.Gamma.get_gamma((i + j + 1) / k) * math.pow(T, (i + j + 1) / k)
            G = A.Invert() * B
            for i in range(num - 2):
                weibullParam.append(G[i, 0])
        return weibullParam

    @staticmethod
    def get_tail_one_value(weibull_params, x):
        p = 0
        k = weibull_params[0]
        T = weibull_params[1]
        g_count = len(weibull_params) - 2
        if g_count != 0:
            g = weibull_params[2:]
            for j in range(g_count):
                p += g[j] * math.pow(x, j)
            p = p * math.exp(-math.pow(x, k) / T)
        else:
            p = math.exp(-math.pow(x, k) / T)
        return p

    @staticmethod
    def get_tail(weibull_params, x_mass):
        res = []
        for x in x_mass:
            res.append(Weibull.get_tail_one_value(weibull_params, x))
        return res

    @staticmethod
    def get_cdf(weibull_params, x_mass):
        res = []
        for x in x_mass:
            res.append(1.0 - Weibull.get_tail_one_value(weibull_params, x))
        return res
