import numpy as np

from most_queue.rand_distribution import Gamma, Pareto_dist
from most_queue.sim.qs_sim import QueueingSystemSimulator
from most_queue.theory.gi_m_n_calc import get_p, get_v, get_w


def test_gi_m_n():
    """
    Тестирование расчета СМО GI/M/n
    Для верификации используем имитационное моделирование (ИМ).
    """

    l = 1.0  # интенсивность вх потока
    a1 = 1.0 / l  # средний интревал между заявками
    n = 4   # число каналов
    ro = 0.8  # коэффициент загрузки СМО
    b1 = ro * n / l  # среднее время обслуживания из заданного ro
    mu = 1 / b1  # интенсивность обслуживания
    a_coev = 1.6  # коэфф вариации вх потока

    num_of_jobs = 300000  # количество заявок для ИМ. Чем больше, тем выше точночть ИМ

    # расчет параметров аппроксимирующего Гамма-распределения для вх птокоа по заданным среднему и коэфф вариации
    v, alpha = Gamma.get_mu_alpha_by_mean_and_coev(a1, a_coev)
    a = Gamma.calc_theory_moments(v, alpha)

    # расчет начальных моментов времени пребывания и ожидания в СМО
    v_ch = get_v(a, mu, n)
    w_ch = get_w(a, mu, n)

    # расчет вероятностей состояний СМО
    p_ch = get_p(a, mu, n)

    # для верификации используем ИМ.
    # создаем экземпляр класса ИМ, передаем число каналов обслуживания
    qs = QueueingSystemSimulator(n)

    # задаем входной поток. Методу нужно передать параметры распределения списком и тип распределения.
    qs.set_sources([v, alpha], "Gamma")

    # задаем каналы обслуживания. На вход параметры (в нашем случае интенсивность обслуживания)
    # и тип распределения - М (экспоненциальное).
    qs.set_servers(mu, "M")

    # запускаем ИМ:
    qs.run(num_of_jobs)

    # получаем список начальных моментов времени пребывания и ожидания в СМО
    v_sim = qs.v
    w_sim = qs.w

    # получаем распределение вероятностей состояний СМО
    p_sim = qs.get_p()

    # Вывод результатов
    print("\nGamma. Значения начальных моментов времени пребывания заявок в системе:\n")

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 45)
    for j in range(3):
        print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, v_ch[j], v_sim[j]))

    print("\nЗначения начальных моментов времени ожидания заявок в системе:\n")

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 45)
    for j in range(3):
        print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, w_ch[j], w_sim[j]))

    print("{0:^25s}".format("Вероятности состояний СМО"))
    print("{0:^3s}|{1:^15s}|{2:^15s}".format("№", "Числ", "ИМ"))
    print("-" * 32)
    for i in range(11):
        print("{0:^4d}|{1:^15.3g}|{2:^15.3g}".format(i, p_ch[i], p_sim[i]))

    # Тоже для Парето
    alpha, K = Pareto_dist.get_a_k_by_mean_and_coev(a1, a_coev)
    a = Pareto_dist.calc_theory_moments(alpha, K)
    v_ch = get_v(a, mu, n, approx_distr="Pa")
    p_ch = get_p(a, mu, n, approx_distr="Pa")

    qs = QueueingSystemSimulator(n)
    qs.set_sources([alpha, K], "Pa")
    qs.set_servers(mu, "M")
    qs.run(num_of_jobs)
    v_sim = qs.v
    p_sim = qs.get_p()

    print("\nPareto. Значения начальных моментов времени пребывания заявок в системе:\n")

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 45)
    for j in range(3):
        print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, v_ch[j], v_sim[j]))

    w_ch = get_w(a, mu, n, approx_distr="Pa")
    w_sim = qs.w
    
    assert np.allclose(np.array(p_sim[:10]), np.array(p_ch[:10]), rtol=1e-1)

    print("\nЗначения начальных моментов времени ожидания заявок в системе:\n")

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 45)
    for j in range(3):
        print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, w_ch[j], w_sim[j]))

    print("{0:^25s}".format("Вероятности состояний СМО"))
    print("{0:^3s}|{1:^15s}|{2:^15s}".format("№", "Числ", "ИМ"))
    print("-" * 32)
    for i in range(11):
        print("{0:^4d}|{1:^15.3g}|{2:^15.3g}".format(i, p_ch[i], p_sim[i]))


if __name__ == "__main__":
    test_gi_m_n()
