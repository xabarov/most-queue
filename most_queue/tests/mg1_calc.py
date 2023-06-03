from most_queue.theory.mg1_calc import *
from most_queue.sim.qs_sim import QueueingSystemSimulator
from most_queue.utils.tables import probs_print, times_print


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

    times_print(w_sim, w_ch, True)
    times_print(v_sim, v_ch, False)
    probs_print(p_sim, p_ch, 10)


if __name__ == "__main__":
    test()
