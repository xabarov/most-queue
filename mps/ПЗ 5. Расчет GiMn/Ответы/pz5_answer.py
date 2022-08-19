import smo_im
import rand_destribution as rd
import numpy as np
import matplotlib.pyplot as plt
import gi_m_n_calc
import time
from matplotlib.ticker import MaxNLocator

def print_head(head):
    print("-" * len(head))
    print(head)
    print("-" * len(head))


def get_cost(v_cost, n_cost, v1, n):
    return v_cost * v1 + n_cost * n


if __name__ == "__main__":

    a1 = 1
    coev = 2.7
    num_of_jobs = 100000
    n = 5

    vs_ch = []
    vs_im = []
    p_ch_mass = []
    im_times = []
    ch_times = []

    roes = np.linspace(0.1, 0.9, 6)

    for ro in roes:

        mu = 1 / (ro * n)
        v, alpha = rd.Gamma.get_mu_alpha_by_mean_and_coev(a1, coev)
        a = rd.Gamma.calc_theory_moments(v, alpha)

        start = time.process_time()
        v_ch = gi_m_n_calc.get_v(a, mu, n)
        ch_times.append(time.process_time() - start)

        p_ch = gi_m_n_calc.get_p(a, mu, n)
        p_ch_mass.append(p_ch)

        start = time.process_time()
        smo = smo_im.SmoIm(n)
        smo.set_sources([v, alpha], "Gamma")
        smo.set_servers(mu, "M")
        smo.run(num_of_jobs)
        v_im = smo.v
        im_times.append(time.process_time() - start)

        p_im = smo.get_p()

        vs_ch.append(v_ch[0])
        vs_im.append(v_im[0])
        v_errors = []
        for j in range(3):
            v_errors.append(100 * (v_im[j] - v_ch[j]) / v_ch[j])

        print("\nЗначения начальных моментов времени пребывания заявок в системе:\n")

        head = "{0:^15s}|{1:^15s}|{2:^15s}|{3:^15s}".format("№ момента", "Числ", "ИМ", "Err, %")
        print_head(head)

        for j in range(3):
            print("{0:^16d}|{1:^15.5g}|{2:^15.5g}|{3:^15.5g}".format(j + 1, v_ch[j], v_im[j], v_errors[j]))

        print("{0:^25s}".format("Вероятности состояний СМО"))
        head = "{0:^3s}|{1:^15s}|{2:^15s}".format("№", "Числ", "ИМ")
        print_head(head)
        for i in range(11):
            print("{0:^4d}|{1:^15.3g}|{2:^15.3g}".format(i, p_ch[i], p_im[i]))

    im_time_mean = sum(im_times) / len(im_times)
    ch_time_mean = sum(ch_times) / len(ch_times)
    print("Среднее время моделирования, сек {0:1.3f}".format(im_time_mean))
    print("Среднее время численного расчета, сек {0:1.3f}".format(ch_time_mean))
    print("Ускорение в {0:1.3f} раз".format(im_time_mean / ch_time_mean))

    fig, ax = plt.subplots()
    ax.plot(roes, vs_im, label="ИМ")
    ax.plot(roes, vs_ch, label="Числ")
    ax.set_xlabel('ro')
    ax.set_ylabel('v1')

    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    for i, ro in enumerate(roes):
        p_s = []
        for j in range(10):
            p_s.append(p_ch_mass[i][j])
        xs = [x for x in range(10)]
        ax.plot(xs, p_s, label="ro={0:1.2f}".format(ro))
    ax.set_xlabel('i')
    ax.set_ylabel('p')
    plt.legend()
    plt.show()

    ns = [x for x in range(1, 10)]

    ro_one_channel = 0.7
    mu = 1 / (ro_one_channel)
    v_cost = 1
    n_cost = 0.7

    total_costs = []
    for n in ns:
        v, alpha = rd.Gamma.get_mu_alpha_by_mean_and_coev(a1, coev)
        a = rd.Gamma.calc_theory_moments(v, alpha)
        v_ch = gi_m_n_calc.get_v(a, mu, n)

        v1 = v_ch[0]
        total_costs.append(get_cost(v_cost, n_cost, v1, n))

    print("Минимальное значение стоимости: {0:1.3f} при n = {1:d}".format(min(total_costs), np.argmin(total_costs)+1))
    fig, ax = plt.subplots()
    ax.plot(ns, total_costs)
    ax.set_xlabel('n')
    ax.set_ylabel('Стоимость')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Распределение стоимости обслуживания системы")
    plt.show()
