from most_queue.theory.gi_m_1_calc import *
from most_queue.sim import rand_destribution as rd
from most_queue.sim import smo_im


def test():
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


if __name__ == "__main__":
    test()
