import matplotlib
import matplotlib.pyplot as plt

from most_queue.sim.queueing_systems.fork_join import ForkJoinSim
from most_queue.theory.fork_join.m_m_n import (
    get_v1_fj_nelson_tantawi,
    get_v1_fj_varki_merchant,
    get_v1_fj_varma,
)

matplotlib.use('TkAgg')


def test_fj_calc():
    """
    Тестирование аппроксимаций для системы Fork-Join
    В пакете theory.fj_calc модержатся следующие методы:

        get_v1_fj_nelson_tantawi -  R. D. Nelson and A. N. Tantawi, “Approximate analysis of fork/join
                                    synchronization in parallel queues.” IEEE Trans. Computers, vol. 37, no. 6, pp. 739–743, 1988.

        get_v1_fj_varma  -  S. Varma and A. M. Makowski, “Interpolation approximations for
                            symmetric fork-join queues.” Perform. Eval., vol. 20, no. 1, pp. 245–265, 1994.

        get_v1_fj_varki_merchant  -  E. Varki, “Response time analysis of parallel computer and storage
                                     systems.” IEEE Trans. Parallel Distrib. Syst., 2001.

    Для верификации используем имитационное моделирование (ИМ).

    """

    ro = 0.8  # коэффициент загрузки СМО
    n = [x for x in range(2, 15)]  # число каналов
    num_of_jobs = 1000000  # количество заявок для ИМ. Чем больше, тем выше точночть ИМ

    # интенсивность обслуживания и начальные моменты
    mu = 1.0
    b = [1 / mu, 2 / pow(mu, 2), 6 / pow(mu, 3)]

    l = ro / b[0]  # интенсивность вх потока

    # массивы для накопления средних времен пребывания в СМО
    v_sim = []
    v_varma = []
    v_varki = []
    v_nelson = []

    str_f = "{0:^15s}|{1:^15s}|{2:^15s}|{3:^15s}"
    str_f_v = "\n{0:^15.3f}|{1:^15.3f}|{2:^15.3f}|{3:^15.3f}"
    print(str_f.format("ИМ", "Varki", "Varma", "Nelson"))

    for i, nn in enumerate(n):
        # для верификации используем ИМ.
        # создаем экземпляр класса ИМ, передаем число каналов обслуживания
        # ИМ поддерживает СМО типа Fork-Join (n, k). В нашем случае k = n

        qs = ForkJoinSim(nn, nn, False)

        # задаем входной поток. Методу нужно передать параметры распределения и тип распределения. М - экспоненциальное
        qs.set_sources(l, 'M')

        # задаем каналы обслуживания. Методу нужно передать параметры распределения и тип распределения. М - экспоненциальное
        qs.set_servers(mu, 'M')

        # запускаем ИМ
        qs.run(num_of_jobs)

        # получаем список начальных моментов времени пребывания заявок в СМО и сохраняем
        # в массив только среднее (первый нач момент)
        v = qs.v
        v_sim.append(v[0])

        # расчет средних времен пребывания с помощью аппроксимаций. \
        # На вход каждого из методов - l, mu, nn (число каналов СМО)

        v_varki.append(get_v1_fj_varki_merchant(l, mu, nn))
        v_varma.append(get_v1_fj_varma(l, mu, nn))
        v_nelson.append(get_v1_fj_nelson_tantawi(l, mu, nn))

        print(str_f_v.format(v_sim[i], v_varki[i], v_varma[i], v_nelson[i]))

    # строим графики и сохраняем в текущую директорию:

    fig, ax = plt.subplots()

    linestyles = ["solid", "dotted", "dashed", "dashdot"]

    ax.plot(n, v_sim, label="ИМ", linestyle=linestyles[0])
    ax.plot(n, v_varki, label="Varki", linestyle=linestyles[1])
    ax.plot(n, v_varma, label="Varma", linestyle=linestyles[2])
    ax.plot(n, v_nelson, label="Nelson", linestyle=linestyles[3])

    ax.set_ylabel("v1")
    ax.set_xlabel("n")

    plt.legend()
    plt.savefig("v1_from_n_with_ro = {0:^4.2f}.png".format(ro), dpi=300)

    # errors
    v_nelson_err = []
    v_varma_err = []
    v_varki_err = []

    for i in range(len(v_sim)):
        v_varma_err.append(100 * (v_varma[i] - v_sim[i]) / v_sim[i])
        v_varki_err.append(100 * (v_varki[i] - v_sim[i]) / v_sim[i])
        v_nelson_err.append(100 * (v_nelson[i] - v_sim[i]) / v_sim[i])

    fig, ax = plt.subplots()
    ax.plot(n, v_varki_err, label="Varki", linestyle=linestyles[0])
    ax.plot(n, v_varma_err, label="Varma", linestyle=linestyles[1])
    ax.plot(n, v_nelson_err, label="Nelson", linestyle=linestyles[2])

    ax.set_ylabel("error, %")
    ax.set_xlabel("n")

    plt.legend()
    plt.savefig("error_from_n_with_ro = {0:^4.2f}.png".format(ro), dpi=300)


if __name__ == "__main__":
    
    test_fj_calc()