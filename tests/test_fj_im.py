import numpy as np

from most_queue.rand_distribution import H2_dist, Erlang_dist
from most_queue.sim.fj_sim import ForkJoinSim
from most_queue.theory import fj_calc, mg1_calc


def test_fj_sim():
    """
    Тестирование расчета системы Split-Join
        | Рыжиков, Ю. И. Метод расчета длительности обработки задач в системе массового обслуживания
        | с учетом процессов Split-Join / Ю. И. Рыжиков, В. А. Лохвицкий, Р. С. Хабаров
        | Известия высших учебных заведений. Приборостроение. – 2019. – Т. 62. – № 5. – С. 419-423. –
        | DOI 10.17586/0021-3454-2019-62-5-419-423.

    Для верификации используем имитационное моделирование (ИМ).

    """
    n = 3  # число каналов
    l = 1.0  # интенсивность вх потока
    num_of_jobs = 300000  # количество заявок для ИМ. Чем больше, тем выше точночть ИМ

    # Подберем начальные моменты времени обслуживания "b" по среднему и коэфф вариации
    # с помощью метода most_queue.sim.rand_distribution.H2_dist.get_params_by_mean_and_coev()
    b1 = 0.37  # среднее время обслуживания
    coev = 1.5  # коэфф вариации времени обслуживания

    params = H2_dist.get_params_by_mean_and_coev(b1, coev)  # параметры H2-распределения [y1, mu1, mu2]

    # рассчитаем 4 начальных момента, нужно на один больше требуемых моментов времени пребывания в СМО
    b = H2_dist.calc_theory_moments(*params, 4)


    # для верификации используем ИМ.
    # создаем экземпляр класса ИМ, передаем число каналов обслуживания
    # ИМ поддерживает СМО типа Fork-Join (n, k). В нашем случае k = n
    # Для задания СМО Split-Join необходимо передать третий параметр True, иначе по умаолчанию - Fork-Join

    qs = ForkJoinSim(n, n, True)

    # задаем входной поток. Методу нужно передать параметры распределения и тип распределения. М - экспоненциальное
    qs.set_sources(l, 'M')

    # задаем каналы обслуживания. Методу нужно передать параметры распределения и тип распределения.
    # H - гиперэкспоненциальное второго параядка
    qs.set_servers(params, 'H')

    # запускаем ИМ
    qs.run(num_of_jobs)

    # получаем список начальных моментов времени пребывания заявок в СМО
    v_sim = qs.v

    # расчет начальных моментов распределения максимума с помощью метода fj_calc.getMaxMoments.
    # На вход число каналов, список начальных моментов
    b_max = fj_calc.getMaxMoments(n, b)
    ro = l * b_max[0]

    # далее расчет как обычной СМО M/G/1 начальными моментами распределения максимума СВ
    v_ch = mg1_calc.get_v(l, b_max)

    print("\n")
    print("-" * 60)
    print("{:^60s}".format('СМО Split-Join'))
    print("-" * 60)
    print("Коэфф вариации времени обслуживания: ", coev)
    print("Коэффициент загрузки: {:4.3f}".format(ro))
    print("Начальные моменты времени пребывания заявок в системе:")
    print("-" * 60)
    print("{0:^15s}|{1:^20s}|{2:^20s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 60)
    for j in range(min(len(v_ch), len(v_sim))):
        print("{0:^16d}|{1:^20.5g}|{2:^20.5g}".format(j + 1, v_ch[j], v_sim[j]))
    print("-" * 60)

    # тоже для коэфф вариации < 1 (аппроксимируем распределением Эрланга)
    coev = 0.8
    b1 = 0.5
    params = Erlang_dist.get_params_by_mean_and_coev(b1, coev)
    b = Erlang_dist.calc_theory_moments(*params, 4)

    qs = ForkJoinSim(n, n, True)
    qs.set_sources(l, 'M')
    qs.set_servers(params, 'E')
    qs.run(num_of_jobs)
    v_sim = qs.v

    b_max = fj_calc.getMaxMoments(n, b)
    ro = l * b_max[0]
    v_ch = mg1_calc.get_v(l, b_max)

    print("\n\nКоэфф вариации времени обслуживания: ", coev)
    print("Коэффициент загрузки: {:4.3f}".format(ro))
    print("Начальные моменты времени пребывания заявок в системе:")
    print("-" * 60)
    print("{0:^15s}|{1:^20s}|{2:^20s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 60)
    for j in range(min(len(v_ch), len(v_sim))):
        print("{0:^16d}|{1:^20.5g}|{2:^20.5g}".format(j + 1, v_ch[j], v_sim[j]))
    print("-" * 60)
    
    assert 100*abs(v_ch[0] - v_sim[0])/max(v_ch[0], v_sim[0]) < 10

if __name__ == "__main__":
    test_fj_sim()