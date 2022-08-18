import smo_im
import numpy as np
import matplotlib.pyplot as plt
import math


def get_p(l, mu, n, r):
    """
    Расчет вероятностей состояния СМО М/М/n/r
    l - интенсивность вх. потока
    mu - интенсивность обслуживания заявок
    n - число каналов обслуживания
    r - размер буфера (очереди)
    """

    p = [0] * (n + r + 1)
    ro = l / mu

    summ1 = 0
    for i in range(n):
        summ1 += pow(ro, i) / math.factorial(i)

    chisl = 1 - pow(ro / n, r + 1)
    coef = pow(ro, n) / math.factorial(n)
    znam = 1 - (ro / n)

    p[0] = 1.0 / (summ1 + coef * chisl / znam)

    for i in range(n):
        p[i] = pow(ro, i) * p[0] / math.factorial(i)

    for i in range(n, n + r + 1):
        p[i] = pow(ro, i) * p[0] / (math.factorial(n) * pow(n, i - n))

    return p


def getQ(l, mu, n, r):
    """
    Расчет средней длины очереди
    """

    ro = l / mu
    p = get_p(l, mu, n, r)
    sum = 0
    for i in range(1, r + 1):
        sum += i * math.pow(ro / n, i)
    return p[n] * sum


def print_head(head):
    print("-" * len(head))
    print(head)
    print("-" * len(head))


if __name__ == "__main__":

    n = 1
    r = 50
    l = 1.0

    jobs_count = 100000

    roes = np.linspace(0.1, 0.9, 20)
    w1_im = []
    w1_teor = []
    error = []
    for ro in roes:
        smo = smo_im.SmoIm(n, r)
        smo.set_sources(l, 'M')
        mu = l / (ro * n)
        smo.set_servers(mu, 'M')
        smo.run(jobs_count)
        q = getQ(l, mu, n, r)
        w1_teor.append(q / l)
        w1_im.append(smo.w[0])
        error.append(100 * (q / l - smo.w[0]) / (q / l))

    head = "\nСреднее время ожидания в СМО от коэффициента загрузки системы"
    print(head)

    head = "{0:^15s}|{1:^15s}|{2:^15s}|{3:^15s}".format("ro", "ИМ", "Теор", "Err")

    str_f = "{0:^15.3f}|{1:^15.3f}|{2:^15.3f}|{3:^15.3f}"
    print_head(head)

    for i in range(len(roes)):
        print(str_f.format(roes[i], w1_im[i], w1_teor[i], error[i]))

    print("-" * len(head))

    fig, ax = plt.subplots()

    ax.set_title("Среднее время ожидания в СМО от коэффициента загрузки системы")
    ax.plot(roes, w1_im, label="ИМ")
    ax.plot(roes, w1_teor, label="Числ")
    ax.set_xlabel('ro')
    ax.set_ylabel('w1')
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(roes, error, label="относ ошибка ИМ")
    ax.set_xlabel('ro')
    ax.set_ylabel('err, %')
    plt.legend()
    plt.show()

    head = "Исследование эффекта дробления производительности"
    print(head)

    ns = [x for x in range(1, 20)]

    v1_teor = []
    w1_teor = []
    ro = 0.75

    for n in ns:
        mu = l / (ro * n)
        q = getQ(l, mu, n, r)
        w1_teor.append(q / l)
        v1_teor.append(q / l + 1.0 / mu)

    head = "{0:^15s}|{1:^15s}|{2:^15s}".format("n", "w1", "v1")
    print_head(head)

    str_f = "{0:^15d}|{1:^15.3f}|{2:^15.3f}"

    for i in range(len(ns)):
        print(str_f.format(ns[i], w1_teor[i], v1_teor[i]))

    print("-" * len(head))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.title("Исследование эффекта дробления производительности", loc="center")

    axes[0].plot(ns, w1_teor)
    axes[0].set_xlabel('n')
    axes[0].set_ylabel('w1')

    axes[1].plot(ns, v1_teor)
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('v1')
    plt.show()
