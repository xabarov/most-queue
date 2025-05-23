"""
Calculate q[j] - probabilities of servicing exactly j of requests 
during the interval between arrivals of adjacent requests
"""
import math

from most_queue.rand_distribution import GammaDistribution


def get_q_gamma(l, mu, alpha, num=100):
    """
    Calculate q[j] for Gamma-distributed inter-arrival times
    :param l: arrival rate
    :param mu: parameter of Gamma distribution
    :param alpha: shape parameter of Gamma distribution
    :param num: number of moments to calculate
    """
    q = [0.0] * num
    q[0] = math.pow(mu / (mu + l), alpha)
    for j in range(1, num):
        q[j] = q[j - 1] * l * (alpha + j - 1) / ((l + mu) * j)

    return q


def get_q_uniform(l, mean, half_interval, num=100):
    """
    Calculate q[j] for uniform-distributed inter-arrival times
    :param l: arrival rate
    :param mean: mean of uniform distribution
    :param half_interval: half of the interval length
    :param num: number of moments to calculate
    """
    q = [0.0] * num
    for j in range(num):
        summ1 = 0
        for i in range(j + 1):
            summ1 += l * pow(mean - half_interval, i) * \
                math.exp(-l * (mean - half_interval)) / math.factorial(i)
        summ2 = 0
        for i in range(j + 1):
            summ2 += l * pow(mean + half_interval, i) * \
                math.exp(-l * (mean + half_interval)) / math.factorial(i)
        q[j] = (1.0 / (2 * l * half_interval)) * (summ1 - summ2)

    return q


def get_q_pareto(l, alpha, K, num=100):
    """
    Calculate q[j] for Pareto-distributed inter-arrival times
    l - arrival rate
    K, alpha - parameters of the Pareto distribution
    num - number of moments to calculate
    """
    q = [0.0] * num
    gammas = [0.0] * num
    z = l * K
    gammas[0] = GammaDistribution.get_minus_gamma(
        alpha) - GammaDistribution.get_gamma_small(-alpha, z)

    for j in range(1, num):
        gammas[j] = (j - alpha - 1) * gammas[j - 1] + \
            pow(z, j - alpha - 1) * math.exp(-z)
    forw = alpha * pow(z, alpha)
    for j in range(num):
        q[j] = forw * gammas[j] / math.factorial(j)

    return q
