import math
import time

from most_queue.rand_distribution import Cox_dist, Gamma, Exp_dist
from most_queue.sim.priority_queue_sim import PriorityQueueSimulator
from most_queue.theory import priority_calc
from most_queue.theory.m_ph_n_prty import m_ph_n_prty


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

        gamma_params = Gamma.get_mu_alpha([bH[0], bH[1]])

        mu_L = 1.0 / bL1

        # задание ИМ:
        v1_sum = 0
        v2_sum = 0

        cox_params = Cox_dist.get_params(bH)

        # расчет численным методом:
        tt_start = time.process_time()
        tt = m_ph_n_prty(mu_L, cox_params[1], cox_params[2], cox_params[0], l_L, l_H, n=n, is_cox=is_cox,
                         max_iter=max_iter)
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

        bL = Exp_dist.calc_theory_moments(mu_L, 3)

        b = []
        b.append(bH)
        b.append(bL)

        L = [l_H, l_L]

        invar_start = time.process_time()
        v = priority_calc.get_v_prty_invar(L, b, n=n, type='PR', num=2)
        v2_invar_mass.append(v[1][0])
        invar_times.append(time.process_time() - invar_start)

        im_start = time.process_time()

        for i in range(iteration):
            print("Start IM iteration: {0:d}".format(i + 1))

            qs = PriorityQueueSimulator(n, K, "PR")
            sources = []
            servers_params = []
            l = [l_H, l_L]

            sources.append({'type': 'M', 'params': l_H})
            sources.append({'type': 'M', 'params': l_L})
            servers_params.append({'type': 'Gamma', 'params': gamma_params})
            servers_params.append({'type': 'M', 'params': mu_L})

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

    print("\nСравнение результатов расчета численным методом с аппроксимацией ПНЗ "
          "\nраспределением Кокса второго порядка и ИМ.")
    print("ro: {0:1.2f}".format(ros))
    print("n : {0:d}".format(n))
    print("Количество обслуженных заявок для ИМ: {0:d}\n".format(num_of_jobs))

    print("\n")
    print("{0:^35s}".format("Средние времена пребывания в СМО для заявок 2-го класса"))
    print("-" * 128)
    print("{0:^15s}|{1:^15s}|{5:^15s}|{2:^15s}|{3:^15s}|{5:^15s}|{4:^15s}|{5:^15s}".format("coev", "Числ",
                                                                                           "Кол-во итер алг", "ИМ",
                                                                                           "Инвар", "t, c"))
    print("-" * 128)
    for k in range(len(bH_coev)):
        print("{0:^15.3f}|{1:^15.6g}|{2:^15.6g}|{3:^15d}|{4:^15.6g}|{5:^15.6g}|{6:^15.6g}|{7:^15.6g}".format(
            bH_coev[k],
            v2_tt_mass[k], tt_times[k], iter_num[k],
            v2_im_mass[k], im_times[k],
            v2_invar_mass[k], invar_times[k]
        ))


if __name__ == "__main__":
    test_m_ph_n_prty()