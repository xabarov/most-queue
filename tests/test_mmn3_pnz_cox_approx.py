from most_queue.sim.queueing_systems.priority import PriorityQueueSimulator
from most_queue.theory.queueing_systems.priority.preemptive.mmn_2cls_pr_busy_approx import MMn_PRTY_PNZ_Cox_approx
from most_queue.theory.queueing_systems.priority.preemptive.mmn_3cls_busy_approx import Mmn3_pnz_cox
from most_queue.theory.queueing_systems.fifo.mmnr import MMnrCalc


def test_mmn3():
    num_of_jobs = 200000
    n = 2  # количество каналов
    K = 3  # количество классов
    mu_L = 1.3  # интенсивность обслуживания заявок 3-го класса
    mu_M = 1.4  # интенсивность обслуживания заявок 2-го класса
    mu_H = 1.5  # интенсивность обслуживания заявок 1-го класса
    l_L = 0.7  # интенсивность вх потока заявок 3-го класса
    l_M = 0.8  # интенсивность вх потока заявок 2-го класса
    l_H = 0.9  # интенсивность вх потока заявок 1-го класса

    l_sum = l_H + l_M + l_L
    b1_H = 1 / mu_H
    b1_L = 1 / mu_L
    b1_M = 1 / mu_M
    b_ave = (l_L / l_sum) * b1_L + (l_H / l_sum) * b1_H + (l_M / l_sum) * b1_M
    ro = l_sum * b_ave / n

    # задание ИМ:
    qs = PriorityQueueSimulator(n, K, "PR")
    sources = []
    servers_params = []
    l = [l_H, l_M, l_L]
    mu = [mu_H, mu_M, mu_L]
    for j in range(K):
        sources.append({'type': 'M', 'params': l[j]})
        servers_params.append({'type': 'M', 'params': mu[j]})

    qs.set_sources(sources)
    qs.set_servers(servers_params)

    # запуск ИМ:
    qs.run(num_of_jobs)

    # получение результатов ИМ:
    p = qs.get_p()
    v_sim = qs.v

    # расчет численным методом:
    tt = Mmn3_pnz_cox(mu_L, mu_M, mu_H, l_L, l_M, l_H)
    tt_for_second = MMn_PRTY_PNZ_Cox_approx(
        2, mu_M, mu_H, l_M, l_H)
    tt_for_second.run()

    tt.run()
    p_tt = tt.get_p()
    v_tt = tt.get_low_class_v1()
    v_2 = tt_for_second.get_second_class_v1()
    
    mmnr = MMnrCalc(l_H, mu_H, 2, 100)

    v_1 = mmnr.get_v()[0]

    print("\nСравнение результатов расчета численным методом с аппроксимацией ПНЗ "
          "\nраспределением Кокса второго порядка и ИМ.")
    print("Коэффициент загрузки: {0:^1.2f}".format(ro))
    print("Количество обслуженных заявок для ИМ: {0:d}\n".format(num_of_jobs))

    print("{0:^25s}".format("Вероятности состояний для заявок 3-го класса"))
    print("{0:^3s}|{1:^15s}|{2:^15s}".format("№", "Числ", "ИМ"))
    print("-" * 32)
    for i in range(11):
        print("{0:^4d}|{1:^15.3g}|{2:^15.3g}".format(i, p_tt[i], p[2][i]))

    print("\n")
    print("{0:^35s}".format("Средние времена пребывания в СМО"))
    print("-" * 38)
    print("{0:^10s}|{1:^15s}|{2:^15s}".format("N класса", "Числ", "ИМ"))
    print("-" * 38)
    print("{0:^10d}|{1:^15.3g}|{2:^15.3g}".format(0, v_1, v_sim[0][0]))
    print("{0:^10d}|{1:^15.3g}|{2:^15.3g}".format(1, v_2, v_sim[1][0]))
    print("{0:^10d}|{1:^15.3g}|{2:^15.3g}".format(2, v_tt, v_sim[2][0]))

    assert 100*abs(v_1 - v_sim[0][0])/max(v_1, v_sim[0][0]) < 10


if __name__ == "__main__":

    test_mmn3()
