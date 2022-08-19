import numpy as np
import math
import rand_destribution as rd
import q_poisson_arrival_calc
import sv_sum_calc


def get_pi(a, mu, num=100, e=1e-10, approx_distr="Gamma"):
    """
    Вычисление вероятностей состояний СМО
    a - список начальных моментов распределения интервало рекуррентного вх потока заявок
    mu - интенсивность обслуживания
    num - число вероятностей
    """
    pi = [0.0] * num

    v, alpha = rd.Gamma.get_mu_alpha(a)

    q = q_poisson_arrival_calc.get_q_Gamma(mu, v, alpha)
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
    w = sv_sum_calc.get_moments_minus(v, b, num)

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
        v, alpha, g = rd.Gamma.get_params(a)
        while True:
            summ = 0
            for i in range(len(g)):
                summ += (g[i] / pow(mu * (1.0 - w_old) + v, i)) * (
                            rd.Gamma.get_gamma(alpha + i) / rd.Gamma.get_gamma(alpha))
            left = pow(v / (mu * (1.0 - w_old) + v), alpha)
            w_new = left * summ
            if math.fabs(w_new - w_old) < e:
                break
            w_old = w_new
        return w_new

    elif approx_distr == "Pa":
        alpha, K = rd.Pareto_dist.get_a_k(a)
        while True:
            left = alpha * pow(K * mu * (1.0 - w_old), alpha)
            w_new = left * rd.Gamma.get_gamma_incomplete(-alpha, K * mu * (1.0 - w_old))
            if math.fabs(w_new - w_old) < e:
                break
            w_old = w_new
        return w_new

    else:
        print("w_param calc. Unknown type of distr_type")

    return 0


if __name__ == '__main__':

    import smo_im

    l = 1
    a1 = 1 / l
    b1 = 0.9
    mu = 1 / b1
    a_coev = 1.6
    num_of_jobs = 800000

    v, alpha = rd.Gamma.get_mu_alpha_by_mean_and_coev(a1, a_coev)
    a = rd.Gamma.calc_theory_moments(v, alpha)
    v_ch = get_v(a, mu)
    p_ch = get_p(a, mu)

    smo = smo_im.SmoIm(1)
    smo.set_sources([v, alpha], "Gamma")
    smo.set_servers(mu, "M")
    smo.run(num_of_jobs)
    v_im = smo.v
    p_im = smo.get_p()

    print("\nGamma. Значения начальных моментов времени пребывания заявок в системе:\n")

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 45)
    for j in range(3):
        print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, v_ch[j], v_im[j]))

    w_ch = get_w(a, mu)
    w_im = smo.w

    print("\nЗначения начальных моментов времени ожидания заявок в системе:\n")

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 45)
    for j in range(3):
        print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, w_ch[j], w_im[j]))

    print("{0:^25s}".format("Вероятности состояний СМО"))
    print("{0:^3s}|{1:^15s}|{2:^15s}".format("№", "Числ", "ИМ"))
    print("-" * 32)
    for i in range(11):
        print("{0:^4d}|{1:^15.3g}|{2:^15.3g}".format(i, p_ch[i], p_im[i]))


    # Pareto test
    alpha, K = rd.Pareto_dist.get_a_k_by_mean_and_coev(a1, a_coev)
    a = rd.Pareto_dist.calc_theory_moments(alpha, K)
    v_ch = get_v(a, mu, approx_distr="Pa")
    p_ch = get_p(a, mu, approx_distr="Pa")

    smo = smo_im.SmoIm(1)
    smo.set_sources([alpha, K], "Pa")
    smo.set_servers(mu, "M")
    smo.run(num_of_jobs)
    v_im = smo.v
    p_im = smo.get_p()

    print("\nPareto. Значения начальных моментов времени пребывания заявок в системе:\n")

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 45)
    for j in range(3):
        print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, v_ch[j], v_im[j]))

    w_ch = get_w(a, mu, approx_distr="Pa")
    w_im = smo.w

    print("\nЗначения начальных моментов времени ожидания заявок в системе:\n")

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 45)
    for j in range(3):
        print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, w_ch[j], w_im[j]))

    print("{0:^25s}".format("Вероятности состояний СМО"))
    print("{0:^3s}|{1:^15s}|{2:^15s}".format("№", "Числ", "ИМ"))
    print("-" * 32)
    for i in range(11):
        print("{0:^4d}|{1:^15.3g}|{2:^15.3g}".format(i, p_ch[i], p_im[i]))