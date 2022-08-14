import smo_im
import numpy as np
import matplotlib.pyplot as plt


def get_w_teor(mu, l):
    v = 1.0 / (mu - l)
    b = 1.0 / mu
    return v - b


def get_p(mu, l, count=100):
    ro = l / mu
    p = [0.0] * count
    p[0] = 1.0 - ro
    for i in range(count - 1):
        p[i + 1] = p[i] * ro
    return p

if __name__=="__main__":
    n = 1
    l = 1.0

    jobs_count = 100000

    roes = np.linspace(0.1, 0.95, 20)
    w1_im_mass = []
    w1_teor_mass = []
    error = []

    p_095_teor = []
    p_095_im = []
    for i, ro in enumerate(roes):
        smo = smo_im.SmoIm(n)
        smo.set_sources(l, 'M')
        mu = l / (ro * n)
        smo.set_servers(mu, 'M')
        smo.run(jobs_count)
        w1_teor = get_w_teor(mu, l)
        w1_teor_mass.append(w1_teor)
        w1_im = smo.w[0]
        w1_im_mass.append(w1_im)
        error.append(100 * (w1_teor - w1_im) / w1_teor)
        if i == len(roes) - 1:
            p_095_im = smo.get_p()
            p_095_teor = get_p(mu, l)

    print("Среднее время ожидания в СМО от коэффициента загрузки")
    print("-" * 62)
    print("{0:^15s}|{1:^15s}|{2:^15s}|{3:^15s}".format("ro", "ИМ", "Теор", "Отн. ошибка, %"))
    str_f = "{0:^15.3f}|{1:^15.3f}|{2:^15.3f}|{3:^15.3f}"
    print("-" * 62)
    for i in range(len(roes)):
        print(str_f.format(roes[i], w1_im_mass[i], w1_teor_mass[i], error[i]))
    print("-" * 62)

    print("Первые 20 вероятностей состояний в СМО при коэффициенте загрузки rо=0.95")
    print("-" * 62)
    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№", "ИМ", "Теор"))
    str_f = "{0:^15d}|{1:^15.3f}|{2:^15.3f}"
    print("-" * 62)
    for i in range(20):
        print(str_f.format(i, p_095_im[i], p_095_teor[i]))
    print("-" * 62)

    fig, ax = plt.subplots()

    ax.plot(roes, w1_im_mass, label="ИМ")
    ax.plot(roes, w1_teor_mass, label="Числ")
    ax.set_xlabel('ro')
    ax.set_ylabel('w1')
    plt.legend()
    plt.show()
