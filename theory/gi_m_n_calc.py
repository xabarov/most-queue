import numpy as np
import math
import rand_destribution as rd
import q_poisson_arrival_calc
import sv_sum_calc
import diff5dots


def get_pi(a, mu, n, num=100, e=1e-10, approx_distr="Gamma"):
    """
    Вычисление вероятностей состояний СМО
    a - список начальных моментов распределения интервало рекуррентного вх потока заявок
    mu - интенсивность обслуживания
    num - число вероятностей
    """
    pi = [0.0] * num
    w = get_w_param(a, mu, n, e, approx_distr)
    A = np.zeros((n + 1, n + 1))
    B = np.zeros(n + 1)

    B[0] = 1
    for i in range(n):
        A[0, i] = 1

    A[0, n] = 1.0 / (1.0 - w)

    for k in range(1, n + 1):
        for j in range(k - 1, n):
            A[k, j] = 0
            for i in range(j + 2 - k):
                A[k, j] += pow(-1, i) * math.factorial(j + 1) * get_b0(a, k + i, mu, approx_distr) / (
                        math.factorial(k) * math.factorial(i) * math.factorial(j + 1 - (k + i)))

        A[k, n] = 0
        for i in range(n - k + 1):
            A[k, n] += pow(-1, i) * math.factorial(n) * (get_b0(a, k + i, mu, approx_distr) - w) / (
                    math.factorial(k) * math.factorial(i) * math.factorial(n - k - i) * (n * (1 - w) - (k + i)))
        A[k, n] = n * A[k, n]
        A[k, k] = A[k, k] - 1
    pi_to_n = np.linalg.solve(A, B)
    for i in range(n + 1):
        pi[i] = pi_to_n[i]
    for i in range(n + 1, num):
        pi[i] = pi[n] * pow(w, i - n)
    return pi


def get_b0(a, j, mu, approx_distr="Gamma"):
    if approx_distr == "Gamma":
        v, alpha, g = rd.Gamma.get_params(a)
        summ = 0
        for i in range(len(g)):
            summ += (g[i] / pow(mu * j + v, i)) * (
                    rd.Gamma.get_gamma(alpha + i) / rd.Gamma.get_gamma(alpha))
        left = pow(v / (mu * j + v), alpha)
        b0 = left * summ
        return b0

    elif approx_distr == "Pa":
        alpha, K = rd.Pareto_dist.get_a_k(a)
        left = alpha * pow(K * mu * j, alpha)
        b0 = left * rd.Gamma.get_gamma_incomplete(-alpha, K * mu * j)
        return b0

    else:
        print("w_param calc. Unknown type of distr_type")

    return 0


def get_v(a, mu, n, num=100, e=1e-10, approx_distr="Gamma"):
    w = get_w(a, mu, n, num, e, approx_distr)
    b = [1 / mu, 2 / pow(mu, 2), 6 / pow(mu, 3), 24 / pow(mu, 4)]
    v = sv_sum_calc.get_moments(w, b, len(w))
    return v


def get_w_pls(n, mu, pn, w, s):
    return n * mu * pn / (n * mu * (1.0 - w) + s)


def get_w(a, mu, n, num=100, e=1e-10, approx_distr="Gamma"):
    pi = get_pi(a, mu, n, num, e, approx_distr)
    pn = pi[n]
    pls = []
    w_param = get_w_param(a, mu, n, e, approx_distr)
    h = 0.001
    s = 0
    for i in range(5):
        pls.append(get_w_pls(n, mu, pn, w_param, s))
        s += h
    w = diff5dots.diff5dots(pls, h)
    w[0] = - w[0]
    w[2] = - w[2]
    return w


def get_p(a, mu, n, num=100, e=1e-10, approx_distr="Gamma"):
    pi = get_pi(a, mu, n, num, e, approx_distr)
    p = [0.0] * num
    for i in range(1, n + 1):
        p[i] = pi[i - 1] / (i * mu * a[0])
    for i in range(n, num):
        p[i] = pi[i - 1] / (n * mu * a[0])
    summ = 0
    for k in range(1, n):
        summ += (n - k) * p[k]
    summ = summ + 1.0 / (mu * a[0])
    p[0] = 1.0 - summ / n
    return p


def get_w_param(a, mu, n, e=1e-10, approx_distr="Gamma"):
    ro = 1.0 / (a[0] * mu * n)
    coev_a = math.sqrt(a[1] - pow(a[0], 2)) / a[0]
    w_old = pow(ro, 2.0 / (pow(coev_a, 2) + 1.0))

    if approx_distr == "Gamma":
        v, alpha, g = rd.Gamma.get_params(a)
        while True:
            summ = 0
            for i in range(len(g)):
                summ += (g[i] / pow(mu * n * (1.0 - w_old) + v, i)) * (
                        rd.Gamma.get_gamma(alpha + i) / rd.Gamma.get_gamma(alpha))
            left = pow(v / (mu * n * (1.0 - w_old) + v), alpha)
            w_new = left * summ
            if math.fabs(w_new - w_old) < e:
                break
            w_old = w_new
        return w_new

    elif approx_distr == "Pa":
        alpha, K = rd.Pareto_dist.get_a_k(a)
        while True:
            left = alpha * pow(K * mu * n * (1.0 - w_old), alpha)
            w_new = left * rd.Gamma.get_gamma_incomplete(-alpha, K * mu * n * (1.0 - w_old))
            if math.fabs(w_new - w_old) < e:
                break
            w_old = w_new
        return w_new

    else:
        print("w_param calc. Unknown type of distr_type")

    return 0


if __name__ == '__main__':

    import smo_im

    l = 1.0
    a1 = 1.0 / l
    n = 4
    ro = 0.8
    b1 = ro * n / l
    mu = 1 / b1
    a_coev = 1.6

    num_of_jobs = 800000

    v, alpha = rd.Gamma.get_mu_alpha_by_mean_and_coev(a1, a_coev)
    a = rd.Gamma.calc_theory_moments(v, alpha)
    v_ch = get_v(a, mu, n)
    p_ch = get_p(a, mu, n)

    smo = smo_im.SmoIm(n)
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

    w_ch = get_w(a, mu, n)
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
    v_ch = get_v(a, mu, n, approx_distr="Pa")
    p_ch = get_p(a, mu, n, approx_distr="Pa")

    smo = smo_im.SmoIm(n)
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

    w_ch = get_w(a, mu, n, approx_distr="Pa")
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
