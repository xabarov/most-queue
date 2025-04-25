from most_queue.theory.queueing_systems.priority.preemptive.mmn_2cls_pr_busy_approx import MMn_PRTY_PNZ_Cox_approx
from most_queue.sim.queueing_systems.priority import PriorityQueueSimulator


def test_mmn_prty():
    
    num_of_jobs = 100000
    n = 3  # количество каналов
    K = 2  # количество классов
    mu_L = 1.3  # интенсивность обслуживания заявок 2-го класса
    mu_H = 1.5  # интенсивность обслуживания заявок 1-го класса
    l_H = 1.0  # интенсивность вх потока заявок 1-го класса
    l_L = 1.4  # интенсивность вх потока заявок 2-го класса
    ro = 0.8
    b1_H = 1 / mu_H
    b1_L = (ro * n - l_H * b1_H) / l_L
    l_sum = l_H + l_L
    # b_ave = (l_L / l_sum) * b1_L + (l_H / l_sum) * b1_H

    # задание ИМ:
    qs = PriorityQueueSimulator(n, K, "PR")
    sources = []
    servers_params = []
    l = [l_H, l_L]
    mu = [mu_H, mu_L]
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
    tt = MMn_PRTY_PNZ_Cox_approx(n, mu_L, mu_H, l_L, l_H)
    tt.run()
    p_tt = tt.get_p()
    v_tt = tt.get_second_class_v1()
    # v = tt.calculate_v()

    print("\nСравнение результатов расчета численным методом с аппроксимацией ПНЗ "
          "\nраспределением Кокса второго порядка и ИМ.")
    print("Коэффициент загрузки: {0:^1.2f}".format(ro))
    print("Количество обслуженных заявок для ИМ: {0:d}\n".format(num_of_jobs))

    print("{0:^25s}".format("Вероятности состояний для заявок 2-го класса"))
    print("{0:^3s}|{1:^15s}|{2:^15s}".format("№", "Числ", "ИМ"))
    print("-" * 32)
    for i in range(11):
        print("{0:^4d}|{1:^15.3g}|{2:^15.3g}".format(i, p_tt[i], p[1][i]))

    print("\n")
    print("{0:^25s}".format("Среднее время пребывания в СМО заявок 2-го класса"))
    print("{0:^15s}|{1:^15s}".format("Числ", "ИМ"))
    print("-" * 32)
    print("{0:^15.3g}|{1:^15.3g}".format(v_tt, v_sim[1][0]))
    # print("{0:^15.3g}|{1:^15.3g}".format(v[0].real, v_sim[1][0]))
    # print("{0:^15.3g}|{1:^15.3g}".format(v[1].real, v_sim[1][1]))
    # print("{0:^15.3g}|{1:^15.3g}".format(v[2].real, v_sim[1][2]))
    
    assert 100*abs(v_tt - v_sim[1][0])/max(v_tt, v_sim[1][0]) < 10



if __name__ == "__main__":
    
    test_mmn_prty()