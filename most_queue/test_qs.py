import math
import time

import numpy as np

from sim import rand_destribution as rd
from sim.priority_queue_sim import PriorityQueueSimulator
from sim.qs_sim import QueueingSystemSimulator
from theory import m_d_n_calc, mmnr_calc, priority_calc
from theory.m_ph_n_prty import m_ph_n_prty
from theory.mgn_tt import MGnCalc
from theory.mg1_calc import get_p, get_v, get_w
from utils.tables import probs_print, times_print


def test_sim():
    """
    Тестирование имитационной модели СМО
    Для верификации - сравнение с результатами расчета СМО M/M/3, M/D/3
    """
    n = 3  # число каналов
    l = 1.0  # интенсивность вх потока
    r = 30  # длина очереди
    ro = 0.8  # коэфф загрузки
    mu = l / (ro * n)  # интенсивность обслуживания

    # создвем экземпляр класса ИМ
    qs = QueueingSystemSimulator(n, buffer=r)

    # задаем вх поток - параметры и тип распределения.
    qs.set_sources(l, "M")
    # задаем распределение обслуживания - параметры и тип распределения.
    qs.set_servers(mu, "M")

    # запуск ИМ - передаем кол-во заявок для обслуживания
    qs.run(1000000)
    # получаем начальные моменты времени ожидания. Также можно получить время пребывания .v,
    # вероятности состояний .get_p(), периоды непрерывной занятости .pppz
    w_sim = qs.w

    # для сравнения расчитаем теже начальные моменты численно
    w = mmnr_calc.MMnr_calc.get_w(l, mu, n, r)

    # вывод результатов

    times_print(w_sim, w, True)

    print("\n\nДанные ИМ::\n")
    print(qs)

    # тоже для детерминированного обслуживания

    qs = QueueingSystemSimulator(n)

    qs.set_sources(l, "M")
    qs.set_servers(1.0 / mu, "D")

    qs.run(1000000)

    mdn = m_d_n_calc.M_D_n(l, 1 / mu, n)
    p_ch = mdn.calc_p()
    p_sim = qs.get_p()

    probs_print(p_sim, p_ch, 10)

    assert np.allclose(np.array(p_sim[:10]), np.array(p_ch[:10]), atol=1e-2)


def test_mgn_tt():
    """
    Тестирование метода Такахаси-Таками для расчета СМО М/H2/n

    При коэфф вариации времени обслуживания < 1 параметры аппроксимирующего Н2-распределения
    являются комплексными, что не мешает получению осмысленных результатов

    Для верификации используется ИМ

    """
    n = 3  # число каналов
    l = 1.0  # интенсивность вх потока
    ro = 0.7  # коэфф загрузки
    b1 = n * ro / l  # ср время обслуживания
    num_of_jobs = 1000000  # число обсл заявок ИМ
    # два варианта коэфф вариации времени обсл, запустим расчет и ИМ для каждого из них
    b_coev_mass = [0.8, 1.2]

    for b_coev in b_coev_mass:
        #  расчет начальных моментов времени обслуживания по заданному среднему и коэфф вариации
        b = [0.0] * 3
        alpha = 1 / (b_coev**2)
        b[0] = b1
        b[1] = math.pow(b[0], 2) * (math.pow(b_coev, 2) + 1)
        b[2] = b[1] * b[0] * (1.0 + 2 / alpha)

        tt_start = time.process_time()
        #  запуск метода Такахаси-Таками
        tt = MGnCalc(n, l, b)
        tt.run()
        # получение результатов численных расчетов
        p_tt = tt.get_p()
        v_tt = tt.get_v()
        tt_time = time.process_time() - tt_start
        # также можно узнать сколько итераций потребовалось
        num_of_iter = tt.num_of_iter_

        # запуск ИМ для верификации результатов
        im_start = time.process_time()

        qs = QueueingSystemSimulator(n, buffer_type="deque")

        # задаем вх поток заявок. М - экспоненциальный с интенсивностью l
        qs.set_sources(l, "M")

        # задаем параметры каналов обслуживания Гамма-распределением.
        # Параметры распределения подбираем с помощью метода библиотеки random_distribution
        gamma_params = rd.Gamma.get_mu_alpha([b[0], b[1]])
        qs.set_servers(gamma_params, "Gamma")

        # Запуск ИМ
        qs.run(num_of_jobs)

        # Получение результатов
        p = qs.get_p()
        v_sim = qs.v
        im_time = time.process_time() - im_start

        # вывод результатов

        print("\nСравнение результатов расчета методом Такахаси-Таками и ИМ.")
        print(
            f"ИМ - M/Gamma/{n:^2d}\nТакахаси-Таками - M/H2/{n:^2d} с комплексными параметрами"
        )
        print(f"Коэффициент загрузки: {ro:^1.2f}")
        print(f"Коэффициент вариации времени обслуживания: {b_coev:^1.2f}")
        print(f"Количество итераций алгоритма Такахаси-Таками: {num_of_iter:^4d}")
        print(f"Время работы алгоритма Такахаси-Таками: {tt_time:^5.3f} c")
        print(f"Время работы ИМ: {im_time:^5.3f} c")
        probs_print(p, p_tt, 10)

        times_print(v_sim, v_tt, False)

        assert 100 * abs(v_tt[0] - v_sim[0]) / max(v_tt[0], v_sim[0]) < 10


def test_m_ph_n_prty():
    """
    Тестирование расчета СМО M/PH, M/n с 2-мя классами заявок, абсолютным приоритетом
    численным методом Такахаси-Таками на основе аппроксимации ПНЗ распределением Кокса второго порядка
    Для верификации используем имитационное моделирование (ИМ).
    """
    num_of_jobs = 300000  # число обсл заявок ИМ

    is_cox = False  # использовать для аппроксимации ПНЗ распределение Кокса или Н2-распределение
    max_iter = 100  # максимальное число итераций численного метода
    # Исследование влияния среднего времени пребывания заявок 2-го класса от коэффициента загрузки
    n = 4  # количество каналов
    K = 2  # количество классов
    ros = 0.75  # коэффициент загрузки СМО
    bH_to_bL = 2  # время обслуживания класса H меньше L в это число раз
    lH_to_lL = 1.5  # интенсивность поступления заявок класса H ниже L в это число раз
    l_H = 1.0  # интенсивность вх потока заявок 1-го класса
    l_L = lH_to_lL * l_H  # интенсивность вх потока заявок 2-го класса
    bH_coev = [1.2]  # исследуемые коэффициенты вариации обсл заявок 1 класса
    iteration = 1  # кол-во итераций ИМ для получения более точных оценок ИМ

    v1_im_mass = []
    v2_im_mass = []
    v2_tt_mass = []
    iter_num = []
    tt_times = []
    im_times = []
    invar_times = []
    v2_invar_mass = []

    for k in range(len(bH_coev)):

        print("coev =  {0:5.3f}".format(bH_coev[k]))

        lsum = l_L + l_H
        bsr = n * ros / lsum
        bH1 = lsum * bsr / (l_L * bH_to_bL + l_H)
        bL1 = bH_to_bL * bH1
        bH = [0.0] * 3
        alpha = 1 / (bH_coev[k] ** 2)
        bH[0] = bH1
        bH[1] = math.pow(bH[0], 2) * (math.pow(bH_coev[k], 2) + 1)
        bH[2] = bH[1] * bH[0] * (1.0 + 2 / alpha)

        gamma_params = rd.Gamma.get_mu_alpha([bH[0], bH[1]])

        mu_L = 1.0 / bL1

        # задание ИМ:
        v1_sum = 0
        v2_sum = 0

        cox_params = rd.Cox_dist.get_params(bH)

        # расчет численным методом:
        tt_start = time.process_time()
        tt = m_ph_n_prty(
            mu_L,
            cox_params[1],
            cox_params[2],
            cox_params[0],
            l_L,
            l_H,
            n=n,
            is_cox=is_cox,
            max_iter=max_iter,
        )
        tt.run()
        tt_times.append(time.process_time() - tt_start)

        iter_num.append(tt.run_iterations_num_)
        p_tt = tt.get_p()
        v_tt = tt.get_low_class_v1()
        v2_tt_mass.append(v_tt)
        # print("{0:^25s}".format("Вероятности состояний для заявок 2-го класса"))
        # print("{0:^3s}|{1:^15s}|{2:^15s}".format("№", "Числ", "ИМ"))
        # print("-" * 32)
        # for i in range(11):
        #     print("{0:^4d}|{1:^15.3g}|{2:^15.3g}".format(i, p_tt[i], 0))
        # print("{0:^15.3f}|{1:^15.6g}|{2:^15.6g}|{3:^15d}".format(bH_coev[k], v2_tt_mass[k], 0, iter_num[k]))

        mu_L = 1.0 / bL1

        bL = rd.Exp_dist.calc_theory_moments(mu_L, 3)

        b = []
        b.append(bH)
        b.append(bL)

        L = [l_H, l_L]

        invar_start = time.process_time()
        v = priority_calc.get_v_prty_invar(L, b, n=n, type="PR", num=2)
        v2_invar_mass.append(v[1][0])
        invar_times.append(time.process_time() - invar_start)

        im_start = time.process_time()

        for i in range(iteration):
            print("Start IM iteration: {0:d}".format(i + 1))

            qs = PriorityQueueSimulator(n, K, "PR")
            sources = []
            servers_params = []
            l = [l_H, l_L]

            sources.append({"type": "M", "params": l_H})
            sources.append({"type": "M", "params": l_L})
            servers_params.append({"type": "Gamma", "params": gamma_params})
            servers_params.append({"type": "M", "params": mu_L})

            qs.set_sources(sources)
            qs.set_servers(servers_params)

            # запуск ИМ:
            qs.run(num_of_jobs)

            # получение результатов ИМ:
            p = qs.get_p()
            v_im = qs.v
            v1_sum += v_im[0][0]
            v2_sum += v_im[1][0]

        v1 = v1_sum / iteration
        v2 = v2_sum / iteration
        # расчет численным методом:
        v2_im_mass.append(v2)
        im_times.append(time.process_time() - im_start)

    print(
        "\nСравнение результатов расчета численным методом с аппроксимацией ПНЗ "
        "\nраспределением Кокса второго порядка и ИМ."
    )
    print("ro: {0:1.2f}".format(ros))
    print("n : {0:d}".format(n))
    print("Количество обслуженных заявок для ИМ: {0:d}\n".format(num_of_jobs))

    print("\n")
    print("{0:^35s}".format("Средние времена пребывания в СМО для заявок 2-го класса"))
    print("-" * 128)
    print(
        "{0:^15s}|{1:^15s}|{5:^15s}|{2:^15s}|{3:^15s}|{5:^15s}|{4:^15s}|{5:^15s}".format(
            "coev", "Числ", "Кол-во итер алг", "ИМ", "Инвар", "t, c"
        )
    )
    print("-" * 128)
    for k in range(len(bH_coev)):
        print(
            "{0:^15.3f}|{1:^15.6g}|{2:^15.6g}|{3:^15d}|{4:^15.6g}|{5:^15.6g}|{6:^15.6g}|{7:^15.6g}".format(
                bH_coev[k],
                v2_tt_mass[k],
                tt_times[k],
                iter_num[k],
                v2_im_mass[k],
                im_times[k],
                v2_invar_mass[k],
                invar_times[k],
            )
        )


def test_mg1():
    """
    Тестирование расчета СМО M/G/1
    Для верификации используем имитационное моделирование (ИМ).
    """
    l = 1  # интенсивность входного потока
    b1 = 0.7  # среднее время обслуживания
    coev = 1.2  # коэфф вариации времени обслуживания
    num_of_jobs = 1000000  # количество заявок для ИМ

    # подбор параметров аппроксимирующего H2-распределения для времени обслуживания [y1, mu1, mu2]:
    params = rd.H2_dist.get_params_by_mean_and_coev(b1, coev)
    b = rd.H2_dist.calc_theory_moments(*params, 4)

    # вычисление численными методами
    w_ch = get_w(l, b)
    p_ch = get_p(l, b, 100)
    v_ch = get_v(l, b)

    # запуск ИМ для верификации результатов
    qs = QueueingSystemSimulator(1)
    qs.set_servers(params, "H")
    qs.set_sources(l, "M")
    qs.run(num_of_jobs)
    w_sim = qs.w
    p_sim = qs.get_p()
    v_sim = qs.v

    # вывод результатов
    print("M/H2/1")

    times_print(w_sim, w_ch, True)
    times_print(v_sim, v_ch, False)
    probs_print(p_sim, p_ch, 10)

    # Тоже самое, но для других распределений времени обслуживания
    print("Uniform")
    params = rd.Uniform_dist.get_params_by_mean_and_coev(b1, coev)
    b = rd.Uniform_dist.calc_theory_moments(*params, 4)
    w_ch = get_w(l, b)
    p_ch = get_p(l, b, 100, dist_type="Uniform")
    v_ch = get_v(l, b)

    qs = QueueingSystemSimulator(1)
    qs.set_servers(params, "Uniform")
    qs.set_sources(l, "M")
    qs.run(num_of_jobs)
    w_sim = qs.w
    p_sim = qs.get_p()
    v_sim = qs.v

    times_print(w_sim, w_ch, True)
    times_print(v_sim, v_ch, False)
    probs_print(p_sim, p_ch, 10)

    print("Pareto")

    a, K = rd.Pareto_dist.get_a_k_by_mean_and_coev(b1, coev)
    b = rd.Pareto_dist.calc_theory_moments(a, K, 4)
    w_ch = get_w(l, b)
    p_ch = get_p(l, b, 100, dist_type="Pa")
    v_ch = get_v(l, b)

    qs = QueueingSystemSimulator(1)
    qs.set_servers([a, K], "Pa")
    qs.set_sources(l, "M")
    qs.run(num_of_jobs)
    w_sim = qs.w
    p_sim = qs.get_p()
    v_sim = qs.v

    assert np.allclose(np.array(p_sim[:10]), np.array(p_ch[:10]), atol=1e-2)

    times_print(w_sim, w_ch, True)
    times_print(v_sim, v_ch, False)
    probs_print(p_sim, p_ch, 10)


if __name__ == "__main__":

    # test_sim()
    # test_mgn_tt()
    test_mg1()
    # test_m_ph_n_prty()
