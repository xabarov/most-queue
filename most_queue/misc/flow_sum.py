import matplotlib
import matplotlib.pyplot as plt

from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.flow_sum import FlowSumSim
from most_queue.theory.utils.flow_sum import SummatorNumeric

matplotlib.use('TkAgg')


def test():
    """
    Тестирование суммирования потоков
    """

    # Задаем следующие параметры:
    n_nums = 10  # число суммируемых потоков
    coev = 0.74  # коэффициент вариации каждого потока
    mean = 1  # среднее каждого потока
    num_of_jobs = 400000  # количество заявок для ИМ
    is_semi = False  # True, если необходимо использовать метод семиинвариантов вместо H2
    distr_sim = "Gamma"  # распределение, используемое для ИМ

    # число суммируемых потоков
    ns = [x + 2 for x in range(n_nums - 1)]

    # начальные моменты суммируемых потоков. В нашем случае все потоки одинаково распределены
    a = []
    for i in range(n_nums):
        params1 = GammaDistribution.get_params_by_mean_and_coev(mean, coev)
        a1 = GammaDistribution.calc_theory_moments(params1, 4)
        a.append(a1)

    # Численный расчет
    s = SummatorNumeric(a, is_semi=is_semi)
    # в  s._flows[i][j] содержатся начальные моменты суммируемых потокоы,
    s.sum_flows()
    # i - кол-во суммируемых потоков, j - номер начального момента

    # ИМ
    s_sim = FlowSumSim(a, distr=distr_sim, num_of_jobs=num_of_jobs)
    # в  s_sim._flows[i][j] содержатся начальные моменты суммируемых потокоы,
    s_sim.sum_flows()
    # i - кол-во суммируемых потоков, j - номер начального момента

    # Расчет ошибок и отображение результатов
    coevs_sim = s_sim.coevs
    coevs_num = s.coevs
    errors1 = []
    errors2 = []
    errors_coev = []

    str_f = "{0:^18s}|{1:^10.3f}|{2:^10.3f}|{3:^10.3f}|{4:^10.3f}|{5:^10.3f}"
    print("{0:^18s}|{1:^10s}|{2:^10s}|{3:^10s}|{4:^10s}|{5:^10s}".format(
        "-", "a1", "a2", "a3", "a4", "coev"))
    print("-" * 80)

    for i in range(n_nums - 1):
        print("{0:^80s}".format("Сумма " + str(i + 2) + " потоков"))
        print("-" * 80)
        print(str_f.format("ИМ", s_sim.flows_[i][0], s_sim.flows_[i][1], s_sim.flows_[i][2], s_sim.flows_[i][3],
                           coevs_num[i]))
        print("-" * 80)
        print(str_f.format("Числ", s.flows_[i][0].real, s.flows_[i][1].real, s.flows_[i][2].real, s.flows_[i][3].real,
                           coevs_sim[i]))
        print("-" * 80)
        errors1.append(SummatorNumeric.get_error(
            s.flows_[i][0].real, s_sim.flows_[i][0]))
        errors2.append(SummatorNumeric.get_error(
            s.flows_[i][1].real, s_sim.flows_[i][1]))
        errors_coev.append(SummatorNumeric.get_error(
            coevs_num[i], coevs_sim[i]))

    fig, ax = plt.subplots()
    linestyles = ["solid", "dotted", "dashed", "dashdot"]

    ax.plot(ns, s_sim.a1_sum, label="ИМ a1", linestyle=linestyles[0])
    ax.plot(ns, s.a1_sum, label="Числ a1", linestyle=linestyles[1])

    plt.legend()
    str_title = "Среднее сумм-х потоков, %"
    if is_semi:
        str_title += ". Метод семиинвариантов"
    else:
        str_title += ". Метод H2"
    plt.title(str_title)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(ns, errors1, label="error a1", linestyle=linestyles[0])
    ax.plot(ns, errors2, label="error a2", linestyle=linestyles[1])
    ax.plot(ns, errors_coev, label="error coev", linestyle=linestyles[2])

    plt.legend()
    str_title = "Отн. ошибка от числа сумм-х потоков, %"
    if is_semi:
        str_title += ". Метод семиинвариантов"
    else:
        str_title += ". Метод H2"
    plt.title(str_title)
    plt.show()


if __name__ == "__main__":
    test()
