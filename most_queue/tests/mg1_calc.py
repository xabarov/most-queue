from most_queue.theory.mg1_calc import *


def test():
    """
    Тестирование расчета СМО M/G/1
    Для верификации используем имитационное моделирование (ИМ).
    """
    l = 1  # интенсивность входного потока
    b1 = 0.9  # среднее время обслуживания
    coev = 1.6  # коэфф вариации времени обслуживания
    num_of_jobs = 800000  # количество заявок для ИМ

    # подбор параметров аппроксимирующего H2-распределения для времени обслуживания [y1, mu1, mu2]:
    params = rd.H2_dist.get_params_by_mean_and_coev(b1, coev)
    b = rd.H2_dist.calc_theory_moments(*params, 4)

    # вычисление численными методами
    w_ch = get_w(l, b)
    p_ch = get_p(l, b, 100)
    v_ch = get_v(l, b)

    # запуск ИМ для верификации результатов
    smo = smo_im.SmoIm(1)
    smo.set_servers(params, "H")
    smo.set_sources(l, "M")
    smo.run(num_of_jobs)
    w_im = smo.w
    p_im = smo.get_p()
    v_im = smo.v

    # вывод результатов

    print("\nH2. Значения начальных моментов времени ожидания заявок в системе:\n")

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 45)
    for j in range(3):
        print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, w_ch[j], w_im[j]))

    print("\nЗначения начальных моментов времени пребывания заявок в системе:\n")

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 45)
    for j in range(3):
        print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, v_ch[j], v_im[j]))

    print("{0:^25s}".format("Вероятности состояний СМО"))
    print("{0:^3s}|{1:^15s}|{2:^15s}".format("№", "Числ", "ИМ"))
    print("-" * 32)
    for i in range(11):
        print("{0:^4d}|{1:^15.3g}|{2:^15.3g}".format(i, p_ch[i], p_im[i]))


    # Тоже самое, но для других распределений времени обслуживания
    params = rd.Uniform_dist.get_params_by_mean_and_coev(b1, coev)
    b = rd.Uniform_dist.calc_theory_moments(*params, 4)
    w_ch = get_w(l, b)
    p_ch = get_p(l, b, 100, dist_type="Uniform")
    v_ch = get_v(l, b)

    smo = smo_im.SmoIm(1)
    smo.set_servers(params, "Uniform")
    smo.set_sources(l, "M")
    smo.run(num_of_jobs)
    w_im = smo.w
    p_im = smo.get_p()
    v_im = smo.v

    print("\nUniform. Значения начальных моментов времени ожидания заявок в системе:\n")

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 45)
    for j in range(3):
        print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, w_ch[j], w_im[j]))

    print("\nЗначения начальных моментов времени пребывания заявок в системе:\n")

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 45)
    for j in range(3):
        print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, v_ch[j], v_im[j]))

    print("{0:^25s}".format("Вероятности состояний СМО"))
    print("{0:^3s}|{1:^15s}|{2:^15s}".format("№", "Числ", "ИМ"))
    print("-" * 32)
    for i in range(11):
        print("{0:^4d}|{1:^15.3g}|{2:^15.3g}".format(i, p_ch[i], p_im[i]))

    a, K = rd.Pareto_dist.get_a_k_by_mean_and_coev(b1, coev)
    b = rd.Pareto_dist.calc_theory_moments(a, K, 4)
    w_ch = get_w(l, b)
    p_ch = get_p(l, b, 100, dist_type="Pa")
    v_ch = get_v(l, b)

    smo = smo_im.SmoIm(1)
    smo.set_servers([a, K], "Pa")
    smo.set_sources(l, "M")
    smo.run(num_of_jobs)
    w_im = smo.w
    p_im = smo.get_p()
    v_im = smo.v

    print("\nPareto. Значения начальных моментов времени ожидания заявок в системе:\n")

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 45)
    for j in range(len(w_ch)):
        print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, w_ch[j], w_im[j]))


    print("\nЗначения начальных моментов времени пребывания заявок в системе:\n")

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 45)
    for j in range(len(w_ch)):
        print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, v_ch[j], v_im[j]))

    print("{0:^25s}".format("Вероятности состояний СМО"))
    print("{0:^3s}|{1:^15s}|{2:^15s}".format("№", "Числ", "ИМ"))
    print("-" * 32)
    for i in range(11):
        print("{0:^4d}|{1:^15.3g}|{2:^15.3g}".format(i, p_ch[i], p_im[i]))


if __name__ == "__main__":
    test()
