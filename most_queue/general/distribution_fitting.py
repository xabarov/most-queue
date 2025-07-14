"""
Methods for fitting distributions to given initial moments.
"""

import cmath
import math
from dataclasses import dataclass

import numpy as np

from most_queue.general.distribution_params import (Cox2Params, ErlangParams,
                                                    GammaParams, H2Params,
                                                    ParetoParams,
                                                    WeibullParams)


@dataclass
class FittingParams:
    """
    Parameters for distribution fitting.
    """

    verbose: bool = True
    ee: float = 0.001
    e: float = 0.02
    e_percent: float = 0.15
    is_fitting: bool = True


def fit_h2(moments: list[float]) -> H2Params:
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
        mu1 = (1.0 - math.sqrt(q_new * (v * v - 1.0) / (2 * (1.0 - q_new)))) * moments[0]
        if math.isclose(mu1, 0):
            mu1 = 1e10
        else:
            mu1 = 1.0 / mu1
        res = H2Params(p1=q_max, mu1=mu1, mu2=1e6)
        return res

    max_iteration = 10000
    tec = 0
    t1 = 0
    t2 = 0
    while abs(tn - moments[2]) > 1e-8 and tec < max_iteration:
        tec += 1
        q_new = (q_max + q_min) / 2.0
        t1 = (1.0 + math.sqrt((1.0 - q_new) * (v * v - 1.0) / (2 * q_new))) * moments[0]
        t2 = (1.0 - math.sqrt(q_new * (v * v - 1.0) / (2 * (1.0 - q_new)))) * moments[0]

        tn = 6 * (q_new * math.pow(t1, 3) + (1.0 - q_new) * math.pow(t2, 3))

        if tn - moments[2] > 0:
            q_min = q_new
        else:
            q_max = q_new

    res = H2Params(p1=q_max, mu1=1.0 / t1, mu2=1.0 / t2)
    return res


def fit_h2_clx(moments: list[float], fitting_params: FittingParams | None = None) -> H2Params:
    """
    Method of fitting H2 distribution parameters to given initial moments.
    Uses the method of moments and optimization to fit the parameters.
    Returns H2Params object with fitted parameters.
    """

    if fitting_params is None:
        fitting_params = FittingParams()

    f = [0.0] * 3
    for i in range(3):
        f[i] = complex(moments[i] / math.factorial(i + 1))
    znam = f[1] - pow(f[0], 2)
    c0 = (f[0] * f[2] - pow(f[1], 2)) / znam
    c1 = (f[0] * f[1] - f[2]) / znam

    d = pow(c1 / 2.0, 2.0) - c0

    if fitting_params.is_fitting:
        # проверка на близость распределения к экспоненциальному
        coev = cmath.sqrt(moments[1] - moments[0] ** 2) / moments[0]
        if math.fabs(coev.real - 1.0) < fitting_params.ee:
            if fitting_params.verbose:
                print(
                    f"H2 is close to Exp. Multiply moments to (1+je)"
                    f"coev = {coev:5.3f}, e = {fitting_params.e:5.3f}."
                )
            f = []
            for i, mom in enumerate(moments):
                f.append(mom * complex(1, (i + 1) * fitting_params.e))

            return fit_h2_clx(f, fitting_params=fitting_params)

        coev = cmath.sqrt(moments[1] - moments[0] ** 2) / moments[0]

        # проверка на близость распределения к Эрланга 2-го порядка
        if math.fabs(coev.real - 1.0 / math.sqrt(2.0)) < fitting_params.ee:
            if fitting_params.verbose:
                print(
                    f"H2 is close to E2. Multiply moments to (1+je)"
                    f"coev = {coev:5.3f}, e = {fitting_params.e:5.3f}."
                )
            f = []
            for i, mom in enumerate(moments):
                f.append(mom * complex(1, (i + 1) * fitting_params.e))
            return fit_h2_clx(f, fitting_params=fitting_params)

    res = [0, 0, 0]  # y1, mu1, mu2
    c1 = complex(c1)
    x1 = -c1 / 2 + cmath.sqrt(d)
    x2 = -c1 / 2 - cmath.sqrt(d)
    y1 = (f[0] - x2) / (x1 - x2)

    res = H2Params(p1=y1, mu1=1.0 / x1, mu2=1.0 / x2)

    return res


def fit_cox(moments: list[float], fitting_params: FittingParams | None = None) -> Cox2Params:
    """
    Calculates Cox-2 distribution parameters by three given initial moments [moments].
    """

    if not fitting_params:
        fitting_params = FittingParams()

    f = [0.0] * 3

    if fitting_params.is_fitting:
        # проверка на близость распределения к экспоненциальному
        coev = cmath.sqrt(moments[1] - moments[0] ** 2) / moments[0]
        if abs(moments[1] - moments[0] * moments[0]) < fitting_params.ee:
            if fitting_params.verbose:
                print(
                    f"Cox special 1. Multiply moments to (1+je)"
                    f"coev = {coev:5.3f}  e = {fitting_params.e:5.3f}."
                )
            f = []
            for i, mom in enumerate(moments):
                f.append(mom * complex(1, (i + 1) * fitting_params.e))

            return fit_cox(f, fitting_params=fitting_params)

        coev = cmath.sqrt(moments[1] - moments[0] ** 2) / moments[0]

        # проверка на близость распределения к Эрланга 2-го порядка
        if abs(moments[1] - (3.0 / 4) * moments[0] * moments[0]) < fitting_params.ee:
            if fitting_params.verbose:
                print("Cox special 2. Multiply moments to (1+je)")
                print(f"\tcoev = {coev:5.3f}, e = {fitting_params.e:5.3f}")
            f = []
            for i, mom in enumerate(moments):
                f.append(mom * complex(1, (i + 1) * fitting_params.e))
            return fit_cox(f, fitting_params=fitting_params)

    for i in range(3):
        f[i] = moments[i] / math.factorial(i + 1)

    d = np.power(f[2] - f[0] * f[1], 2) - 4.0 * (f[1] - np.power(f[0], 2)) * (
        f[0] * f[2] - np.power(f[1], 2)
    )
    mu2 = f[0] * f[1] - f[2] + cmath.sqrt(d)
    mu2 /= 2.0 * (np.power(f[1], 2) - f[0] * f[2])
    mu1 = (mu2 * f[0] - 1.0) / (mu2 * f[1] - f[0])
    y1 = (mu1 * f[0] - 1.0) * mu2 / mu1

    return Cox2Params(p1=y1, mu1=mu1, mu2=mu2)


def fit_pareto_moments(moments: list[float]):
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


def fit_pareto_by_mean_and_coev(f1: float, coev: float):
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


def fit_erlang(moments: list[float]) -> ErlangParams:
    """
    Calculates parameters of the Erlang distribution by initial moments.
    """
    r = int(math.floor(moments[0] * moments[0] / (moments[1] - moments[0] * moments[0]) + 0.5))
    mu = r / moments[0]
    return ErlangParams(r=r, mu=mu)


def calc_gamma_func(x):
    """
    Gamma-function Г(x)
    """
    if (x > 0.0) & (x < 1.0):
        return calc_gamma_func(x + 1.0) / x
    if x > 2:
        return (x - 1) * calc_gamma_func(x - 1)
    if x <= 0:
        return math.pi / (math.sin(math.pi * x) * calc_gamma_func(1 - x))
    return gamma_approx(x)


def gamma_approx(x):
    """
    Get approximation of Gamma function for x in [1,2]
    """
    p = [
        -1.71618513886549492533811e0,
        2.47656508055759199108314e1,
        -3.79804256470945635097577e2,
        6.29331155312818442661052e2,
        8.66966202790413211295064e2,
        -3.14512729688483657254357e4,
        -3.61444134186911729807069e4,
        6.6456143820240544627855e4,
    ]
    q = [
        -3.08402300119738975254354e1,
        3.15350626979604161529144e2,
        -1.01515636749021914166146e3,
        -3.10777167157231109440444e3,
        2.253811842098015100330112e4,
        4.75584667752788110767815e3,
        -1.34659959864969306392456e5,
        -1.15132259675553483497211e5,
    ]
    z = x - 1.0
    a = 0.0
    b = 1.0
    for i in range(0, 8):
        a = (a + p[i]) * z
        b = b * z + q[i]
    return a / b + 1.0


def fit_gamma(moments: list[float]) -> GammaParams:
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
                A[i].append(
                    calc_gamma_func(alpha + i + j) / (pow(mu, i + j) * calc_gamma_func(alpha))
                )
        g = np.linalg.solve(A, B)
        return GammaParams(mu=mu, alpha=alpha, g=g)

    return GammaParams(mu=mu, alpha=alpha)


def fit_weibull(moments: list[float]) -> WeibullParams:
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
        a * math.sqrt(math.pi) * math.gamma(u0 + 1) / math.gamma(u0 + 0.5)
    )
    delta = u1 - u0
    while math.fabs(delta) > ee:
        u1 = (1.0 / (2 * math.log(2))) * math.log(
            a * math.sqrt(math.pi) * math.gamma(u0 + 1) / math.gamma(u0 + 0.5)
        )
        delta = u1 - u0
        u0 = u1
    k = 1 / u1
    big_w = math.pow(moments[0] / math.gamma(u1 + 1), k)

    return WeibullParams(k=k, W=big_w)


def gamma_moments_by_mean_and_coev(mean: float, coev: float) -> list[float]:
    """
    Calculate the first three moments of a gamma distribution
    given the mean and coefficient of variation.
    """
    f = [0, 0, 0]
    alpha = 1 / (coev**2)
    f[0] = mean
    f[1] = pow(f[0], 2) * (pow(coev, 2) + 1)
    f[2] = f[1] * f[0] * (1.0 + 2 / alpha)
    return f
