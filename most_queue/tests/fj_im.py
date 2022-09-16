from most_queue.theory import mg1_calc
from most_queue.theory import fj_calc
from most_queue.sim.fj_im import SmoFJ
from most_queue.sim import rand_destribution as rd


def test():
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
    num_of_jobs = 1000000  # количество заявок для ИМ. Чем больше, тем выше точночть ИМ

    # Подберем начальные моменты времени обслуживания "b" по среднему и коэфф вариации
    # с помощью метода most_queue.sim.rand_destribution.H2_dist.get_params_by_mean_and_coev()
    b1 = 0.37  # среднее время обслуживания
    coev = 1.5  # коэфф вариации времени обслуживания

    params = rd.H2_dist.get_params_by_mean_and_coev(b1, coev)  # параметры H2-распределения [y1, mu1, mu2]

    # рассчитаем 4 начальных момента, нужно на один больше требуемых моментов времени пребывания в СМО
    b = rd.H2_dist.calc_theory_moments(*params, 4)


    # для верификации используем ИМ.
    # создаем экземпляр класса ИМ, передаем число каналов обслуживания
    # ИМ поддерживает СМО типа Fork-Join (n, k). В нашем случае k = n
    # Для задания СМО Split-Join необходимо передать третий параметр True, иначе по умаолчанию - Fork-Join

    smo = SmoFJ(n, n, True)

    # задаем входной поток. Методу нужно передать параметры распределения и тип распределения. М - экспоненциальное
    smo.set_sources(l, 'M')

    # задаем каналы обслуживания. Методу нужно передать параметры распределения и тип распределения.
    # H - гиперэкспоненциальное второго параядка
    smo.set_servers(params, 'H')

    # запускаем ИМ
    smo.run(num_of_jobs)

    # получаем список начальных моментов времени пребывания заявок в СМО
    v_im = smo.v

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
    for j in range(min(len(v_ch), len(v_im))):
        print("{0:^16d}|{1:^20.5g}|{2:^20.5g}".format(j + 1, v_ch[j], v_im[j]))
    print("-" * 60)

    # тоже для коэфф вариации < 1 (аппроксимируем распределением Эрланга)
    coev = 0.8
    b1 = 0.5
    params = rd.Erlang_dist.get_params_by_mean_and_coev(b1, coev)
    b = rd.Erlang_dist.calc_theory_moments(*params, 4)

    smo = SmoFJ(n, n, True)
    smo.set_sources(l, 'M')
    smo.set_servers(params, 'E')
    smo.run(num_of_jobs)
    v_im = smo.v

    b_max = fj_calc.getMaxMoments(n, b)
    ro = l * b_max[0]
    v_ch = mg1_calc.get_v(l, b_max)

    print("\n\nКоэфф вариации времени обслуживания: ", coev)
    print("Коэффициент загрузки: {:4.3f}".format(ro))
    print("Начальные моменты времени пребывания заявок в системе:")
    print("-" * 60)
    print("{0:^15s}|{1:^20s}|{2:^20s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 60)
    for j in range(min(len(v_ch), len(v_im))):
        print("{0:^16d}|{1:^20.5g}|{2:^20.5g}".format(j + 1, v_ch[j], v_im[j]))
    print("-" * 60)


if __name__ == "__main__":
    test()
