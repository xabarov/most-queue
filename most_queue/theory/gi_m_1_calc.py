import math

from most_queue.general_utils.conv import get_moments_minus
from most_queue.rand_distribution import Gamma, Pareto_dist
from most_queue.theory.utils.q_poisson_arrival_calc import get_q_Gamma


def get_pi(a, mu, num=100, e=1e-10, approx_distr="Gamma"):
    """
    Вычисление вероятностей состояний СМО
    a - список начальных моментов распределения интервало рекуррентного вх потока заявок
    mu - интенсивность обслуживания
    num - число вероятностей
    """
    pi = [0.0] * num

    v, alpha = Gamma.get_mu_alpha(a)

    q = get_q_Gamma(mu, v, alpha)
    summ = 0
    w = get_w_param(a, mu, e, approx_distr)
    for i in range(len(q)):
        summ += q[i] * pow(w, i)
    pi[0] = 1.0 - summ
    for k in range(1, num):
        pi[k] = (1.0 - w) * pow(w, k)
    return pi


def get_v(a, mu, num=3, e=1e-10, approx_distr="Gamma"):
    w_param = get_w_param(a, mu, e, approx_distr)
    v = [0.0] * num
    for k in range(num):
        v[k] = math.factorial(k + 1) / pow(mu * (1 - w_param), k + 1)
    return v


def get_w(a, mu, num=3, e=1e-10, approx_distr="Gamma"):
    v = get_v(a, mu, num, e, approx_distr)
    b = [1.0 / mu, 2.0 / pow(mu, 2), 6.0 / pow(mu, 3), 24.0 / pow(mu, 4)]
    w = get_moments_minus(v, b, num)

    return w


def get_p(a, mu, num=100, e=1e-10, approx_distr="Gamma"):
    ro = 1.0 / (a[0] * mu)
    p = [0.0] * num
    p[0] = 1 - ro
    w_param = get_w_param(a, mu, e, approx_distr)
    for i in range(1, num):
        p[i] = ro * (1.0 - w_param) * pow(w_param, i - 1)
    return p


def get_w_param(a, mu, e=1e-10, approx_distr="Gamma"):
    ro = 1.0 / (a[0] * mu)
    coev_a = math.sqrt(a[1] - pow(a[0], 2)) / a[0]
    w_old = pow(ro, 2.0 / (pow(coev_a, 2) + 1.0))

    if approx_distr == "Gamma":
        v, alpha, g = Gamma.get_params(a)
        while True:
            summ = 0
            for i in range(len(g)):
                summ += (g[i] / pow(mu * (1.0 - w_old) + v, i)) * (
                            Gamma.get_gamma(alpha + i) / Gamma.get_gamma(alpha))
            left = pow(v / (mu * (1.0 - w_old) + v), alpha)
            w_new = left * summ
            if math.fabs(w_new - w_old) < e:
                break
            w_old = w_new
        return w_new

    elif approx_distr == "Pa":
        alpha, K = Pareto_dist.get_a_k(a)
        while True:
            left = alpha * pow(K * mu * (1.0 - w_old), alpha)
            w_new = left * Gamma.get_gamma_incomplete(-alpha, K * mu * (1.0 - w_old))
            if math.fabs(w_new - w_old) < e:
                break
            w_old = w_new
        return w_new

    else:
        print("w_param calc. Unknown type of distr_type")

    return 0

