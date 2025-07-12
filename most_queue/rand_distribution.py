"""
Random distributions for simulation.
"""
import cmath
import math

import numpy as np
import scipy.special as sp
from scipy import stats

from most_queue.general.distribution_params import (
    Cox2Params,
    ErlangParams,
    GammaParams,
    GaussianParams,
    H2Params,
    ParetoParams,
    UniformParams,
    WeibullParams
)
from most_queue.general.interfaces import Distribution


class Weibull(Distribution):
    """
    Class for working with the Weibull distribution
    """

    def __init__(self, params: WeibullParams, generator=None):
        """
        :param params: WeibullParams object containing shape parameter (k) and scale parameter (T)
        :param generator: random number generator. If None, np.random is used.
        """

        self.k = params.k
        self.W = params.W
        self.params = params
        self.type = 'Weibull'
        self.generator = generator

    @staticmethod
    def generate_static(params: WeibullParams, generator=None):
        """
        Generates a random number according to the Weibull distribution.
        """
        if generator:
            p = generator.random()
        else:
            p = np.random.rand()
        return math.pow(-math.log(p)*params.W, -1 / params.k)

    def generate(self) -> float:
        """
        Generates a random number according to the Weibull distribution.
        """

        return self.generate_static(self.params, self.generator)

    @staticmethod
    def calc_theory_moments(params: WeibullParams, num: int) -> list[float]:
        """
        Calculates theoretical moments of the distribution up to the specified order.
        :param params: WeibullParams parameters for the distribution.
        :param num: number of moments to calculate.
        :return: list[float]
        """
        f = [0.0]*num
        for i in range(num):
            f[i] = math.gamma(1 + i/params.k) * math.pow(params.W, i/params.k)
        return f

    @staticmethod
    def get_params(moments: list[float]) -> WeibullParams:
        """
        Parameter selection for the distribution based 
        on initial moments of the distribution 

        params:
            moments: initial moments of the random variable

       return:
            WeibullParams 

        """
        a = moments[1] / (moments[0] * moments[0])
        u0 = math.log(2 * a) / (2.0 * math.log(2))
        ee = 1e-6
        u1 = (1.0 / (2 * math.log(2))) * math.log(
            a * math.sqrt(math.pi) * math.gamma(u0 + 1) / math.gamma(u0 + 0.5))
        delta = u1 - u0
        while math.fabs(delta) > ee:
            u1 = (1.0 / (2 * math.log(2))) * math.log(
                a * math.sqrt(math.pi) * math.gamma(u0 + 1) / math.gamma(u0 + 0.5))
            delta = u1 - u0
            u0 = u1
        k = 1 / u1
        big_w = math.pow(moments[0] / math.gamma(u1 + 1), k)

        return WeibullParams(k=k, W=big_w)

    @staticmethod
    def get_params_by_mean_and_coev(f1: float, coev: float) -> WeibullParams:
        """
        Subselect parameters of Weibull distribution by the given mean and coefficient of variation.
         Args:
             f1 (float): The first parameter of the Weibull distribution.
             coev (float): The coefficient of variation.
         Returns:
             WeibullParams
        """

        f = [0, 0, 0]
        alpha = 1 / (coev ** 2)
        f[0] = f1
        f[1] = pow(f[0], 2) * (pow(coev, 2) + 1)
        f[2] = f[1] * f[0] * (1.0 + 2 / alpha)

        return Weibull.get_params(f)

    @staticmethod
    def get_tail(params: WeibullParams, t: float) -> float:
        """
        Calculate the tail probability of a Weibull distribution at a given value.
         Args:
            params (WeibullParams): The parameters of Weibull distribution
            t: The value at which to calculate the tail probability.
        """
        k = params.k
        big_w = params.W

        return math.exp(-math.pow(t, k) / big_w)

    @staticmethod
    def get_cdf(params: WeibullParams, t: float) -> float:
        """
        Calculate the cumulative distribution function (CDF) of a Weibull distribution for a given set of values.
         Args:
            params (WeibullParams): The parameters of Weibull distribution
            t: The value at which to calculate the cdf probability.

        """
        k = params.k
        big_w = params.W

        return 1.0 - math.exp(-math.pow(t, k) / big_w)


class NormalDistribution(Distribution):
    """
    Gaussian distribution. Generates random numbers from a normal (Gaussian) distribution.
    """

    def __init__(self, params: GaussianParams, generator=None):
        """
        :param params: list of parameters [mean, std_dev]
        :param generator: random number generator. If None, np.random is used.
        """

        self.mean = params.mean
        self.std_dev = params.std_dev
        self.params = params
        self.type = 'Normal'
        self.generator = generator

    def generate(self) -> float:
        """
        Generates a random number from the normal distribution.
        :return: float
        """

        return self.generate_static(self.params, self.generator)

    @staticmethod
    def generate_static(params: GaussianParams, generator=None):
        """
        Генерация псевдо-случайных чисел
        Статический метод
        """
        if generator:
            return generator.normal(params.mean, params.std_dev)
        return np.random.normal(params.mean, params.std_dev)

    @staticmethod
    def calc_theory_moments(params: GaussianParams, num=3) -> list[float]:
        """
        Method calculates theoretical initial moments for the uniform distribution.
        :param params: list of parameters [mean, half_interval]
        :param num: number of moments to calculate
        :return: list of theoretical moments
        """

        f = [0.0] * num
        f[0] = params.mean
        f[1] = params.mean ** 2 + params.std_dev ** 2
        return f

    @staticmethod
    def get_params(moments: list[float]):
        """
        :param moments: list of theoretical moments.
        :return: Parameters for the distribution that correspond to the given moments.
        """
        mean = moments[0]
        std_dev = np.sqrt(moments[1] - mean ** 2)
        return GaussianParams(mean, std_dev)

    @staticmethod
    def get_params_by_mean_and_coev(f1: float, coev: float):
        """
        :param f1: mean of the distribution.
        :param coev: coefficient of variation (std_dev / mean).
        :return: Parameters for the distribution that correspond to the given mean and coefficient of variation.
        """
        mean = f1
        std_dev = coev * mean
        return GaussianParams(mean, std_dev)


class UniformDistribution(Distribution):
    """
    Uniform distribution class. Generates random numbers according to the uniform distribution.
    """

    def __init__(self, params: UniformParams, generator=None):
        """
        :param params: list of parameters [mean, half_interval]
        :param generator: random number generator object (optional)
        """

        self.mean = params.mean
        self.half_interval = params.half_interval
        self.params = params
        self.type = 'Uniform'
        self.generator = generator

    def generate(self):
        """
        Generates a random number according to the uniform distribution.
        """

        return self.generate_static(self.params, self.generator)

    @staticmethod
    def generate_static(params: UniformParams, generator=None):
        """
        Generates a random number according to the uniform distribution.
        :param params: list of parameters [mean, half_interval]
        :param generator: random number generator object (optional)

        """
        mean, half_interval = params.mean, params.half_interval

        if generator:
            return generator.uniform(mean - half_interval, mean + half_interval)
        return np.random.uniform(mean - half_interval, mean + half_interval)

    @staticmethod
    def calc_theory_moments(params: UniformParams, num: int = 3) -> list[float]:
        """
        Calculates theoretical moments for the uniform distribution.
        """
        mean, half_interval = params.mean, params.half_interval
        f = [0.0] * num
        for i in range(num):
            f[i] = (pow(mean + half_interval, i + 2) - pow(mean -
                    half_interval, i + 2)) / (2 * half_interval * (i + 2))
        return f

    @staticmethod
    def get_params(moments: list[float]) -> UniformParams:
        """
        Get parameters of the uniform distribution by moments.
        """

        D = moments[1] - moments[0] * moments[0]
        mean = moments[0]
        half_interval = math.sqrt(3 * D)

        return UniformParams(mean=mean, half_interval=half_interval)

    @staticmethod
    def get_params_by_mean_and_coev(f1: float, coev: float) -> UniformParams:
        """
        Get parameters of the uniform distribution by mean and coefficient of variation.
        Returns mean and half interval.
        """

        D = pow(coev * f1, 2)
        half_interval = math.sqrt(3 * D)

        return UniformParams(mean=f1, half_interval=half_interval)

    @staticmethod
    def get_pdf(params: UniformParams, t: float) -> float:
        """
        Get probability density function (PDF) value for a given time t
        """
        mean, half_interval = params.mean, params.half_interval

        a = mean - half_interval
        b = mean + half_interval
        if t < a or t > b:
            return 0
        return 1.0 / (b - a)

    @staticmethod
    def get_cdf(params: UniformParams, t: float) -> float:
        """
        Get cummulative distribution function (CDF) value for a given time t
        """
        mean, half_interval = params.mean, params.half_interval
        a = mean - half_interval
        b = mean + half_interval
        if t < a:
            return 0
        if t > b:
            return 1

        return (t - a) / (b - a)

    @staticmethod
    def get_tail(params: UniformParams, t: float) -> float:
        """
        Get the tail probability of the distribution.
        """

        return 1.0 - UniformDistribution.get_cdf(params, t)


class H2Distribution(Distribution):
    """
    Hyperexponential distribution of the second order. (H2)
    """

    def __init__(self, params: H2Params, generator=None):
        """
        Get the parameters of the distribution.
        :param params: Parameters of the H2 distribution
        :param generator: Random number generator (optional)
        """

        self.params = params
        self.type = 'H'
        self.generator = generator

    def generate(self) -> float:
        """
        Generation of pseudo-random numbers according to the H2 distribution.
        """

        return self.generate_static(self.params, self.generator)

    @staticmethod
    def generate_static(params: H2Params, generator=None) -> float:
        """
        Generation of pseudo-random numbers according to the H2 distribution.
        """

        if generator:
            r = generator.random()
            res = -np.log(generator.random())
        else:
            r = np.random.rand()
            res = -np.log(np.random.rand())

        if r < params.p1:
            if params.mu1 != 0:
                res = res / params.mu1
            else:
                res = 1e10
        else:
            if params.mu2 != 0:
                res = res / params.mu2
            else:
                res = 1e10
        return res

    @staticmethod
    def calc_theory_moments(params: H2Params, num=3) -> list[float]:
        """
        Calculates theoretical moments of the H2 distribution.
        """
        f = [0.0] * num
        y2 = 1.0 - params.p1

        for i in range(num):
            f[i] = math.factorial(
                i + 1) * (params.p1 / pow(params.mu1, i + 1) + y2 / pow(params.mu2, i + 1))
        return f

    @staticmethod
    def get_residual_params(params: H2Params) -> H2Params:
        """
        Calculates the  parameters of residual of the H2 distribution.
        """
        y1, y2, mu1, mu2 = params.p1, 1.0 - params.p1, params.mu1, params.mu2

        return H2Params(p1=y1 * mu2 / (y1 * mu2 + y2 * mu1), mu1=mu1, mu2=mu2)

    @staticmethod
    def get_params(moments: list[float]) -> H2Params:
        """
        Aliev's method for calculating the parameters of H2 distribution by given initial moments.
        Only real parameters are selected.
        Returns a list with parameters [y1, mu1, mu2].
        """

        v = moments[1] - moments[0] * moments[0]
        v = math.sqrt(v) / moments[0]
        res = [0.0] * 3
        if v < 1.0:
            return H2Params(p1=0, mu1=0, mu2=0)

        q_max = 2.0 / (1.0 + v * v)
        t_min = 1.5 * ((1 + v * v) ** 2) * math.pow(moments[0], 3)
        q_min = 0.0
        tn = 0.0

        if t_min > moments[2]:
            # one phase distibution
            q_new = q_max
            mu1 = (1.0 - math.sqrt(q_new * (v * v - 1.0) /
                   (2 * (1.0 - q_new)))) * moments[0]
            if math.isclose(mu1, 0):
                mu1 = 1e10
            else:
                mu1 = 1.0 / mu1
            res = H2Params(p1=q_max, mu1=mu1, mu2=1e6)
            return res
        else:
            max_iteration = 10000
            tec = 0
            t1 = 0
            t2 = 0
            while abs(tn - moments[2]) > 1e-8 and tec < max_iteration:
                tec += 1
                q_new = (q_max + q_min) / 2.0
                t1 = (1.0 + math.sqrt((1.0 - q_new) *
                      (v * v - 1.0) / (2 * q_new))) * moments[0]
                t2 = (1.0 - math.sqrt(q_new * (v * v - 1.0) /
                      (2 * (1.0 - q_new)))) * moments[0]

                tn = 6 * (q_new * math.pow(t1, 3) +
                          (1.0 - q_new) * math.pow(t2, 3))

                if tn - moments[2] > 0:
                    q_min = q_new
                else:
                    q_max = q_new

            res = H2Params(p1=q_max, mu1=1.0/t1, mu2=1.0/t2)
            return res

    @staticmethod
    def get_params_clx(moments: list[float], verbose=True, ee=0.001, e=0.02, e_percent=0.15, is_fitting=True) -> H2Params:
        """
        Method of fitting H2 distribution parameters to given initial moments.
        Uses the method of moments and optimization to fit the parameters.
        Returns H2Params object with fitted parameters.
        """

        f = [0.0] * 3
        for i in range(3):
            f[i] = complex(moments[i] / math.factorial(i + 1))
        znam = (f[1] - pow(f[0], 2))
        c0 = (f[0] * f[2] - pow(f[1], 2)) / znam
        c1 = (f[0] * f[1] - f[2]) / znam

        d = pow(c1 / 2.0, 2.0) - c0

        if is_fitting:
            # проверка на близость распределения к экспоненциальному
            coev = cmath.sqrt(moments[1] - moments[0] ** 2) / moments[0]
            if math.fabs(coev.real - 1.0) < ee:
                if verbose:
                    print(
                        f"H2 is close to Exp. Multiply moments to (1+je), coev = {coev:5.3f}, e = {e:5.3f}.")
                f = []
                for i, mom in enumerate(moments):
                    f.append(mom * complex(1, (i + 1) * e))

                return H2Distribution.get_params_clx(f, verbose=verbose, ee=ee, e=e * (1.0 + e_percent), e_percent=e_percent,
                                                     is_fitting=is_fitting)

            coev = cmath.sqrt(moments[1] - moments[0] ** 2) / moments[0]

            # проверка на близость распределения к Эрланга 2-го порядка
            if math.fabs(coev.real - 1.0 / math.sqrt(2.0)) < ee:
                if verbose:
                    print(
                        f"H2 is close to E2. Multiply moments to (1+je), coev = {coev:5.3f}, e = {e:5.3f}.")
                f = []
                for i, mom in enumerate(moments):
                    f.append(mom * complex(1, (i + 1) * e))
                return H2Distribution.get_params_clx(f, verbose=verbose, ee=ee, e=e * (1.0 + e_percent), e_percent=e_percent,
                                                     is_fitting=is_fitting)

        res = [0, 0, 0]  # y1, mu1, mu2
        c1 = complex(c1)
        x1 = -c1 / 2 + cmath.sqrt(d)
        x2 = -c1 / 2 - cmath.sqrt(d)
        y1 = (f[0] - x2) / (x1 - x2)

        res = H2Params(p1=y1, mu1=1.0/x1, mu2=1.0/x2)

        return res

    @staticmethod
    def get_params_by_mean_and_coev(f1: float, coev: float, is_clx=False) -> H2Params:
        """
        Get parameters of the H2 distribution by mean and coefficient of variation.
        """

        f = [0, 0, 0]
        alpha = 1 / (coev ** 2)

        f[0] = f1
        f[1] = pow(f[0], 2) * (pow(coev, 2) + 1)
        f[2] = f[1] * f[0] * (1.0 + 2 / alpha)

        if is_clx:
            return H2Distribution.get_params_clx(f)
        return H2Distribution.get_params(f)

    @staticmethod
    def get_cdf(params: H2Params, t: float) -> float:
        """
        Get the cumulative distribution function (CDF) of the H2 distribution.
        """
        if t < 0:
            return 0
        y = [params.p1, 1 - params.p1]
        mu = [params.mu1, params.mu2]
        res = 0
        for i in range(2):
            res += y[i] * math.exp(-mu[i] * t)
        return 1.0 - res

    @staticmethod
    def get_pdf(params: H2Params, t: float) -> float:
        """
        Get probability density function (PDF) of the H2 distribution.
        """
        if t < 0:
            return 0
        y = [params.p1, 1 - params.p1]
        mu = [params.mu1, params.mu2]
        res = 0
        for i in range(2):
            res += y[i] * mu[i] * math.exp(-mu[i] * t)
        return res

    @staticmethod
    def get_tail(params: H2Params, t: float) -> float:
        """
        Get the tail probability of the H2 distribution.
        """
        return 1.0 - H2Distribution.get_cdf(params, t)


class CoxDistribution(Distribution):
    """
    Coxian distribution of the second order.

    """

    def __init__(self, params: Cox2Params, generator=None):
        """
        :param params: Parameters of the distribution.
        :param generator: Random number generator.
        """
        self.params = params
        self.type = 'C'
        self.generator = generator

    def generate(self) -> float:
        """
        generate random number according to Coxian distribution.
        """
        return self.generate_static(self.params, self.generator)

    @staticmethod
    def generate_static(params: Cox2Params, generator=None) -> float:
        """
        Generate random number according to Coxian distribution. Static method.
        """

        p1, m1, m2 = params.p1, params.mu1, params.mu2

        if generator:
            r = generator.random()
            res = (-1.0 / m1) * np.log(generator.random())
        else:
            r = np.random.rand()
            res = (-1.0 / m1) * np.log(np.random.rand())
        if r < p1:
            if generator:
                res = res + (-1.0 / m2) * np.log(generator.random())
            else:
                res = res + (-1.0 / m2) * np.log(np.random.rand())
        return res

    @staticmethod
    def calc_theory_moments(params: Cox2Params, num: int = 3) -> list[float]:
        """
        Calculates the theoretical initial moments of the Coxian distribution.
        """
        y1, m1, m2 = params.p1, params.mu1, params.mu2

        y2 = 1.0 - y1
        f = [0.0] * 3
        f[0] = y2 / m1 + y1 * (1.0 / m1 + 1.0 / m2)
        f[1] = 2.0 * (y2 / math.pow(m1, 2) + y1 * (1.0 /
                      math.pow(m1, 2) + 1.0 / (m1 * m2) + 1.0 / math.pow(m2, 2)))
        f[2] = 6.0 * (y2 / (math.pow(m1, 3)) + y1 * (1.0 / math.pow(m1, 3) + 1.0 / (math.pow(m1, 2) * m2) +
                                                     1.0 / (math.pow(m2, 2) * m1) + 1.0 / math.pow(m2, 3)))

        return f

    @staticmethod
    def get_params(moments: list[float], ee=0.001, e=0.5, e_percent=0.25, verbose=True, is_fitting=True) -> Cox2Params:
        """
        Calculates Cox-2 distribution parameters by three given initial moments [moments].
        """
        f = [0.0] * 3

        if is_fitting:
            # проверка на близость распределения к экспоненциальному
            coev = cmath.sqrt(moments[1] - moments[0] ** 2) / moments[0]
            if abs(moments[1] - moments[0] * moments[0]) < ee:
                if verbose:
                    print(
                        f"Cox special 1. Multiply moments to (1+je), coev = {coev:5.3f}  e = {e:5.3f}.")
                f = []
                for i, mom in enumerate(moments):
                    f.append(mom * complex(1, (i + 1) * e))

                return CoxDistribution.get_params(f, verbose=verbose, ee=ee, e=e * (1.0 + e_percent), e_percent=e_percent,
                                                  is_fitting=is_fitting)

            coev = cmath.sqrt(moments[1] - moments[0] ** 2) / moments[0]

            # проверка на близость распределения к Эрланга 2-го порядка
            if abs(moments[1] - (3.0 / 4) * moments[0] * moments[0]) < ee:
                if verbose:
                    print(
                        f"Cox special 2. Multiply moments to (1+je), coev = {coev:5.3f}, e = {e:5.3f}.")
                f = []
                for i, mom in enumerate(moments):
                    f.append(mom * complex(1, (i + 1) * e))
                return CoxDistribution.get_params(f, verbose=verbose, ee=ee, e=e * (1.0 + e_percent), e_percent=e_percent,
                                                  is_fitting=is_fitting)

        for i in range(3):
            f[i] = moments[i] / math.factorial(i + 1)

        d = np.power(f[2] - f[0] * f[1], 2) - 4.0 * (f[1] -
                                                     np.power(f[0], 2)) * (f[0] * f[2] - np.power(f[1], 2))
        mu2 = f[0] * f[1] - f[2] + cmath.sqrt(d)
        mu2 /= 2.0 * (np.power(f[1], 2) - f[0] * f[2])
        mu1 = (mu2 * f[0] - 1.0) / (mu2 * f[1] - f[0])
        y1 = (mu1 * f[0] - 1.0) * mu2 / mu1

        return Cox2Params(p1=y1, mu1=mu1, mu2=mu2)

    @staticmethod
    def get_params_by_mean_and_coev(f1: float, coev: float) -> Cox2Params:
        """
        Get parameters of C2 distribution by mean and coefficient of variation
        """

        f = [0, 0, 0]
        alpha = 1 / (coev ** 2)
        f[0] = f1
        f[1] = pow(f[0], 2) * (pow(coev, 2) + 1)
        f[2] = f[1] * f[0] * (1.0 + 2 / alpha)

        return CoxDistribution.get_params(f)


class DeterministicDistribution(Distribution):
    """Deterministic distribution class. Generates constant value."""

    def __init__(self, b: float):
        """
        :param b: constant value to generate.
        """
        self.b = b
        self.type = 'D'

    def generate(self):
        """
        Generate a constant value b.
        """
        return DeterministicDistribution.generate_static(self.b)

    @staticmethod
    def generate_static(params, generator=None):
        """
        Generate a constant value b. Static method for use without creating an instance of the class.
        """
        return params

    @staticmethod
    def calc_theory_moments(params, num: int) -> list[float]:
        """
        Calculates theoretical moments of the distribution up to the specified order.
        :param params = b: constant value.
        :param num: number of moments to calculate.
        :return: list[float]
        """
        moments = [params**i for i in range(1, num+1)]
        return moments

    @staticmethod
    def get_params(moments: list[float]):
        """
        :param moments: list of theoretical moments.
        :return: Parameters for the distribution that correspond to the given moments.
        """
        return moments[0]

    @staticmethod
    def get_params_by_mean_and_coev(f1: float, coev: float):
        """
        :param f1: mean of the distribution.
        :param coev: coefficient of variation (std_dev / mean).
        :return: Parameters for the distribution that correspond to the given mean and coefficient of variation.
        """
        b = f1
        return b


class ParetoDistribution(Distribution):
    """
    Pareto distribution class. Generates values according to Pareto distribution.
    """

    def __init__(self, params: ParetoParams, generator=None):
        """
        :param params: ParetoParams object with alpha and K parameters.
        :param generator: random number generator. If None, then default is used.
        """
        self.a = params.alpha
        self.k = params.K
        self.params = params
        self.type = 'Pa'
        self.generator = generator

    def generate(self) -> float:
        """
        Generate a random value according to Pareto distribution.
        :return: generated random value.
        """
        return ParetoDistribution.generate_static(self.params, self.generator)

    @staticmethod
    def get_pdf(params: ParetoParams, t: float) -> float:
        """
        Get probability density function value for given time t.
        """
        a, k = params.alpha, params.K

        if t < 0:
            return 0
        return a * math.pow(k, a) / math.pow(t, a + 1)

    @staticmethod
    def get_cdf(params: ParetoParams, t: float) -> float:
        """
        Get cummulative distribution function value for given time t.
        """
        return 1.0 - ParetoDistribution.get_tail(params, t)

    @staticmethod
    def get_tail(params: ParetoParams, t: float) -> float:
        """
        Get tail of distribution function value for given time t. 
        Tail is defined as P(X>t).
        """
        if t < 0:
            return 0
        a = params.alpha
        k = params.K
        return math.pow(k / t, a)

    @staticmethod
    def calc_theory_moments(params: ParetoParams, num: int = 3) -> list[float]:
        """
        Calc theoretical moments of the distribution.
        """

        a = params.alpha
        k = params.K
        f = []
        for i in range(num):
            if a > i + 1:
                f.append(a * math.pow(k, i + 1) / (a - i - 1))
            else:
                return f
        return f

    @staticmethod
    def generate_static(params: ParetoParams,  generator=None) -> float:
        """
        Generate static value according to Pareto distribution.
         :param params: parameters of the distribution
         :param generator: random number generator. If None, then np.random.rand() is used.
          :return: generated value

        """
        a = params.alpha
        k = params.K
        if generator:
            return k * math.pow(generator.random(), -1 / a)
        return k * math.pow(np.random.rand(), -1 / a)

    @staticmethod
    def get_params(moments: list[float]):
        """
        Calc parameters of the distribution. 
        :param moments: list of initial moments
        """
        d = moments[1] - moments[0] * moments[0]
        c = moments[0] * moments[0] / d
        disc = 4 * (1 + c)
        a = (2 + math.sqrt(disc)) / 2
        k = (a - 1) * moments[0] / a
        return ParetoParams(alpha=a, K=k)

    @staticmethod
    def get_params_by_mean_and_coev(f1: float, coev: float):
        """
        Get parameters of the distribution by mean and coefficient of variation.
        :param f1: float, mean value of the distribution
        :param coev: float, coefficient of variation (mean / std)
        """
        d = pow(f1 * coev, 2)
        c = pow(f1, 2) / d
        disc = 4 * (1 + c)
        a = (2 + math.sqrt(disc)) / 2
        k = (a - 1) * f1 / a
        return ParetoParams(alpha=a, K=k)


class ErlangDistribution(Distribution):
    """
    Erlang distribution class.
    """

    def __init__(self, params: ErlangParams, generator=None):
        """
        :param params: ErlangParams, parameters of the distribution
        :param generator: random number generator, default is None (use numpy.random)
        """
        self.r = params.r
        self.mu = params.mu
        self.params = params
        self.type = 'E'
        self.generator = generator

    def generate(self) -> float:
        """
        Generate a random number from the Erlang distribution.
        :return: float, generated random number
        """
        return self.generate_static(self.params, self.generator)

    @staticmethod
    def generate_static(params: ErlangParams, generator=None) -> float:
        """
        Generator of pseudo-random numbers. Static method.
        """
        r, mu = params.r, params.mu

        prod = 1
        for _ in range(r):
            if generator:
                prod *= generator.random()
            else:
                prod *= np.random.rand()

        return -(1.0 / mu) * np.log(prod)

    @staticmethod
    def get_cdf(params: ErlangParams, t: float) -> float:
        """
        Get cummulative distribution function value of the random variable.
        """
        if t < 0:
            return 0
        r = params.r
        mu = params.mu
        res = 0
        for i in range(r):
            res += math.pow(mu * t, i) * math.exp(-mu * t) / math.factorial(i)
        return 1.0 - res

    @staticmethod
    def get_tail(params: ErlangParams, t: float) -> float:
        """
        Returns the value of the tail distribution function.
        """
        return 1.0 - ErlangDistribution.get_cdf(params, t)

    @staticmethod
    def calc_theory_moments(params: ErlangParams, num: int = 3) -> list[float]:
        """
        Calculates theoretical initial moments of the distribution. By default - first three.
        """
        r, mu = params.r, params.mu

        f = [0.0] * num
        for i in range(num):
            prod = 1
            for k in range(i + 1):
                prod *= r + k
            f[i] = prod / math.pow(mu, i + 1)
        return f

    @staticmethod
    def get_params(moments: list[float]) -> ErlangParams:
        """
        Calculates parameters of the Erlang distribution by initial moments. 
        """
        r = int(math.floor(moments[0] * moments[0] /
                (moments[1] - moments[0] * moments[0]) + 0.5))
        mu = r / moments[0]
        return ErlangParams(r=r, mu=mu)

    @staticmethod
    def get_params_by_mean_and_coev(f1: float, coev: float) -> ErlangParams:
        """
        Method selects the parameters of the Erlang distribution 
        by mean and coefficient of variation.
        """
        f = [0, 0]
        f[0] = f1
        f[1] = (math.pow(coev, 2) + 1) * math.pow(f[0], 2)
        return ErlangDistribution.get_params(f)


class ExpDistribution(Distribution):
    """
    Exponential distribution. It is a special case of the Erlang distribution with r = 1.
    """

    def __init__(self, mu: float, generator=None):
        """
        :param mu: rate parameter (inverse of the mean)
        :param generator: random number generator (optional)
        """
        self.erl = ErlangDistribution(
            ErlangParams(r=1, mu=mu), generator=generator)
        self.params = mu
        self.type = 'M'
        self.generator = generator

    def generate(self) -> float:
        """
        Generates a random number from the exponential distribution.
        """
        return self.erl.generate()

    @staticmethod
    def generate_static(params: float, generator=None):
        """
        Generates a random number from the exponential distribution.
        :param params (mu): rate parameter (inverse of the mean)
        """
        return ErlangDistribution.generate_static(ErlangParams(r=1, mu=params), generator)

    @staticmethod
    def calc_theory_moments(params: float, num: int = 3) -> list[float]:
        """
        Calculates theoretical moments of the exponential distribution.
        :param params (mu): rate parameter (inverse of the mean)

        """
        return ErlangDistribution.calc_theory_moments(ErlangParams(r=1, mu=params), num=num)

    @staticmethod
    def get_params(moments: list[float]):
        """
        Get parameters of the exponential distribution from theoretical moments.
        :param moments: list of theoretical moments
        """
        return ErlangDistribution.get_params(moments).mu

    @staticmethod
    def get_params_by_mean_and_coev(f1: float, coev: float) -> float:
        """
        Get parameters of the exponential distribution from mean and coefficient of variation.
        :param f1: mean value
        :param coev: coefficient of variation (coefficient of dispersion)
        """
        return 1 / f1


class GammaDistribution(Distribution):
    """
    Gamma distribution class.
    """

    def __init__(self, params: GammaParams, generator=None):
        self.mu = params.mu
        self.alpha = params.alpha
        self.is_corrective = False
        self.g = params.g if params.g is not None else []

        self.params = params
        self.type = 'Gamma'
        self.generator = generator

    @staticmethod
    def get_params(moments: list[float]) -> GammaParams:
        """
        Get parameters of the Gamma distribution from theoretical moments.
        """
        d = moments[1] - moments[0] * moments[0]
        mu = moments[0] / d
        alpha = mu * moments[0]
        if len(moments) > 2:
            # подбор коэффициентов g
            A = []
            B = []
            for i in range(len(moments) + 1):
                A.append([])
                if i == 0:
                    B.append(1)
                else:
                    B.append(moments[i - 1])
                for j in range(len(moments) + 1):
                    A[i].append(GammaDistribution.get_gamma(alpha + i + j) /
                                (pow(mu, i + j) * GammaDistribution.get_gamma(alpha)))
            g = np.linalg.solve(A, B)
            return GammaParams(mu=mu, alpha=alpha, g=g)

        return GammaParams(mu=mu, alpha=alpha)

    @staticmethod
    def get_params_by_mean_and_coev(f1: float, coev: float) -> GammaParams:
        """
        Get parameters of gamma distribution by mean and coefficient of variation.
        """
        d = pow(f1 * coev, 2)
        mu = f1 / d
        alpha = mu * f1
        return GammaParams(mu=mu, alpha=alpha)

    def generate(self) -> float:
        """
        Generate gamma distributed random number.
        """
        return self.generate_static(self.params, self.generator)

    @staticmethod
    def generate_static(params: GammaParams, generator=None) -> float:
        """
        Generate gamma distributed random number.
        """
        theta = 1 / params.mu
        if generator:
            return generator.gamma(params.alpha, theta)
        return np.random.gamma(params.alpha, theta)

    @staticmethod
    def get_cdf(params: GammaParams, t: float) -> float:
        """
        Get cummulative distribution function of gamma distribution.
        """
        return stats.gamma.cdf(params.mu * t, params.alpha)

    @staticmethod
    def get_pdf(params: GammaParams, t: float) -> float:
        """
        Get probability density function of gamma distribution.
        """
        return GammaDistribution.get_f(params, t)

    @staticmethod
    def get_f(parmas: GammaParams, t: float) -> float:
        """
        Function of probability density of gamma distribution.
        """
        mu, alpha = parmas.mu, parmas.alpha

        fract = sp.gamma(alpha)
        if math.fabs(fract) > 1e-12:
            if math.fabs(mu * t) > 1e-12:
                main = mu * math.pow(mu * t, alpha - 1) * \
                    math.exp(-mu * t) / fract
            else:
                main = 0
        else:
            main = 0
        return main

    @staticmethod
    def get_f_corrective(params: GammaParams, gs: list[float], t: float) -> float:
        """
        Function of probability density of gamma distribution with corrective polynomial.
        :param params: GammaParams object with parameters of gamma distribution.
        :param gs: List of coefficients of corrective polynomial.
        :param t: Time point.
        """
        mu, alpha = params.mu, params.alpha

        fract = sp.gamma(alpha)
        if math.fabs(fract) > 1e-12:
            if math.fabs(mu * t) > 1e-12:
                main = mu * math.pow(mu * t, alpha - 1) * \
                    math.exp(-mu * t) / fract
            else:
                main = 0
        else:
            main = 0
        summ = 0
        for i, g in enumerate(gs):
            summ += g * pow(t, i)

        return main * summ

    @staticmethod
    def calc_theory_moments(params: GammaParams, num: int = 3) -> list[float]:
        """
        Calc theoretical moments of gamma distribution.
        :param params: GammaParams object with parameters of gamma distribution.
        :param num: Count of moments to calculate.
        :return: List of theoretical moments.
        """
        f = [0.0] * num
        for i in range(num):
            prod = 1
            for k in range(i + 1):
                prod *= params.alpha + k
            f[i] = prod / math.pow(params.mu, i + 1)
        return f

    @staticmethod
    def get_lst(params: GammaParams, s: float) -> float:
        """
        Calculate Laplace-Stieljets transform for gamma distribution.
        :param params: GammaParams object with parameters of gamma distribution.
        :param s: Argument of Laplace-Stieljets transform.
        """
        return math.pow(params.mu / (params.mu + s), params.alpha)

    @staticmethod
    def get_gamma_incomplete(x, z, e=1e-12):

        return GammaDistribution.get_gamma(x) - GammaDistribution.get_gamma_small(x, z, e)

    @staticmethod
    def get_gamma_small(x, z, e=1e-12):
        summ = 0
        n = 0
        while True:
            elem = pow(-z, n) / (math.factorial(n) * (x + n))
            summ += elem
            if math.fabs(elem) < e:
                break
            n += 1
        gamma = summ * pow(z, x)
        return gamma

    @staticmethod
    def get_minus_gamma(x):
        gamma = sp.gamma(x)
        fraction = -math.pi / (x * math.sin(math.pi * x))
        return fraction / gamma

    @staticmethod
    def get_gamma(x):
        """
        Gamma-function Г(x)
        """
        if (x > 0.0) & (x < 1.0):
            return GammaDistribution.get_gamma(x + 1.0) / x
        if x > 2:
            return (x - 1) * GammaDistribution.get_gamma(x - 1)
        if x <= 0:
            return math.pi / (math.sin(math.pi * x) * GammaDistribution.get_gamma(1 - x))
        return GammaDistribution.gamma_approx(x)

    @staticmethod
    def gamma_approx(x):
        """
        Get approximation of Gamma function for x in [1,2]
        """
        p = [-1.71618513886549492533811e+0,
             2.47656508055759199108314e+1,
             -3.79804256470945635097577e+2,
             6.29331155312818442661052e+2,
             8.66966202790413211295064e+2,
             -3.14512729688483657254357e+4,
             -3.61444134186911729807069e+4,
             6.6456143820240544627855e+4]
        q = [-3.08402300119738975254354e+1,
             3.15350626979604161529144e+2,
             -1.01515636749021914166146e+3,
             -3.10777167157231109440444e+3,
             2.253811842098015100330112e+4,
             4.75584667752788110767815e+3,
             -1.34659959864969306392456e+5,
             -1.15132259675553483497211e+5]
        z = x - 1.0
        a = 0.0
        b = 1.0
        for i in range(0, 8):
            a = (a + p[i]) * z
            b = b * z + q[i]
        return a / b + 1.0
