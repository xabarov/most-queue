import math
import most_queue.rand_distribution as rd
import numpy as np


class Weibull:
    """
    Class for working with the Weibull distribution
    """
    @staticmethod
    def get_params(t, num=2):
        """
        Parameter selection for the distribution based on initial moments of the distribution (default is two)
        When num>2 and with an appropriate number of initial moments, in addition to the parameters of the distribution k and T
        it returns values of g for the correction polynomial
        
        params:
            t: initial moments of the random variable
            num: number of initial moments

       return:
            k, T: parameters of the Weibull distribution
            g: coefficients of the correction polynomial (if num>2)

        """
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
            G = A.I * B
            for i in range(num - 2):
                weibullParam.append(G[i, 0])
        return weibullParam

    @staticmethod
    def get_tail_one_value(weibull_params, x):
        """
        Calculate the tail probability of a Weibull distribution at a given value.
         Args:
             weibull_params (list): A list containing the parameters of the Weibull distribution.
             x (float): The value at which to calculate the tail probability.
         Returns:
             float: The tail probability at the given value.

        """
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
        """
        Calculate the tail probability of a Weibull distribution for a given set of values.
         Args:
             weibull_params (list): A list containing the parameters of the Weibull distribution.
             x_mass (list): A list of values at which to calculate the tail probabilities.
         Returns:
             list: The tail probabilities at the given values.

        """
        res = []
        for x in x_mass:
            res.append(Weibull.get_tail_one_value(weibull_params, x))
        return res

    @staticmethod
    def get_cdf(weibull_params, x_mass):
        """
        Calculate the cumulative distribution function (CDF) of a Weibull distribution for a given set of values.
         Args:
             weibull_params (list): A list containing the parameters of the Weibull distribution.
             x_mass (list): A list of values at which to calculate the CDFs.
         Returns:
             list: The CDFs at the given values.

        """
        res = []
        for x in x_mass:
            res.append(1.0 - Weibull.get_tail_one_value(weibull_params, x))
        return res

    @staticmethod
    def get_params_by_mean_and_coev(f1, coev, num=2):
        """
        Subselect parameters of Weibull distribution by the given mean and coefficient of variation.
         Args:
             f1 (float): The first parameter of the Weibull distribution.
             coev (float): The coefficient of variation.
         Returns:
             list: A list of parameters of the Weibull distribution.
        """

        f = [0, 0, 0]
        alpha = 1 / (coev ** 2)
        f[0] = f1
        f[1] = pow(f[0], 2) * (pow(coev, 2) + 1)
        f[2] = f[1] * f[0] * (1.0 + 2 / alpha)

        return Weibull.get_params(f, num)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    mean = 1.0
    coevs = [1.0, 2.1, 3.2]

    fig, ax = plt.subplots()

    for coev in coevs:
        k, T = Weibull.get_params_by_mean_and_coev(mean, coev)

        print("Параметры распределения Вейбулла при коэфф вариации {0:1.3f}: k = {1:1.3f} T = {2:1.3f}".format(coev, k,
                                                                                                               T))

        t = np.linspace(0, 3 * coevs[len(coevs)-1], 100)
        dfr = Weibull.get_tail([k, T], t)

        ax.plot(t, dfr, label="$\\nu$  = {0:1.1f}".format(coev))

    ax.set_xlabel('t')
    ax.set_ylabel('ДФР')
    plt.legend()
    plt.show()
