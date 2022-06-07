import smo_im
import rand_destribution as rd
import numpy as np
import matplotlib.pyplot as plt
import gi_m_1_calc

a = [1, 4, 140]
roes = np.linspace(0.1, 0.9, 6)
num_of_jobs = 800000

vs_ch = []
vs_im = []

v_errors = []

for ro in roes:

    mu = 1 / ro
    v, alpha = rd.Gamma.get_mu_alpha(a)
    a = rd.Gamma.calc_theory_moments(v, alpha)
    v_ch = gi_m_1_calc.get_v(a, mu)
    p_ch = gi_m_1_calc.get_p(a, mu)

    smo = smo_im.SmoIm(1)
    smo.set_sources([v, alpha], "Gamma")
    smo.set_servers(mu, "M")
    smo.run(num_of_jobs)
    v_im = smo.v
    p_im = smo.get_p()

    v_ch = gi_m_1_calc.get_v(a, mu)
    v_im = smo.v

    vs_ch.append(v_ch[0])
    vs_im.append(v_im[0])
    # v_errors.append(100 * (v_im[0] - v_ch[0]) / v_ch[0])

    print("\nЗначения начальных моментов времени ожидания заявок в системе:\n")

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 45)
    for j in range(3):
        print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, v_ch[j], v_im[j]))

    print("{0:^25s}".format("Вероятности состояний СМО"))
    print("{0:^3s}|{1:^15s}|{2:^15s}".format("№", "Числ", "ИМ"))
    print("-" * 32)
    for i in range(11):
        print("{0:^4d}|{1:^15.3g}|{2:^15.3g}".format(i, p_ch[i], p_im[i]))

vs_ch_2 = []
vs_im_2 = []
a[2] = 2*a[2]

for ro in roes:

    mu = 1 / ro

    v, alpha = rd.Gamma.get_mu_alpha(a)
    a = rd.Gamma.calc_theory_moments(v, alpha)
    v_ch = gi_m_1_calc.get_v(a, mu)
    p_ch = gi_m_1_calc.get_p(a, mu)

    smo = smo_im.SmoIm(1)
    smo.set_sources([v, alpha], "Gamma")
    smo.set_servers(mu, "M")
    smo.run(num_of_jobs)
    v_im = smo.v
    p_im = smo.get_p()

    v_ch = gi_m_1_calc.get_v(a, mu)
    v_im = smo.v

    vs_ch_2.append(v_ch[0])
    vs_im_2.append(v_im[0])
    # v_errors.append(100 * (v_im[0] - v_ch[0]) / v_ch[0])

    print("\nЗначения начальных моментов времени ожидания заявок в системе:\n")

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 45)
    for j in range(3):
        print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, v_ch[j], v_im[j]))

    print("{0:^25s}".format("Вероятности состояний СМО"))
    print("{0:^3s}|{1:^15s}|{2:^15s}".format("№", "Числ", "ИМ"))
    print("-" * 32)
    for i in range(11):
        print("{0:^4d}|{1:^15.3g}|{2:^15.3g}".format(i, p_ch[i], p_im[i]))

fig, ax = plt.subplots()
ax.plot(roes, vs_im, label="ИМ")
ax.plot(roes, vs_ch, label="Числ")
ax.plot(roes, vs_im_2, label="ИМ2")
ax.plot(roes, vs_ch_2, label="Числ2")

# ax.plot(roes, v_errors, label="относ ошибка ИМ")
plt.legend()
plt.show()





