import math
import time

from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.queueing_systems.vacations import VacationQueueingSystemSimulator
from most_queue.theory.queueing_systems.vacations.mmn_with_h2_cold_and_h2_warmup import MMn_H2warm_H2cold


def test_mmn_h2cold_h2_warm():

    n = 1  # число каналов
    l = 1.0  # интенсивность вх потока
    ro = 0.2  # коэфф загрузки
    b1 = 0.001  # n * ro / l  # ср время обслуживания
    mu = 1.0 / b1
    b1_warm = n * 0.3 / l  # ср время разогрева
    b1_cold = n * 0.0001 / l  # ср время охлаждения
    num_of_jobs = 1000000  # число обсл заявок ИМ
    b_coev_warm = 1.01  # коэфф вариации времени разогрева
    b_coev_cold = 1.01  # коэфф вариации времени охлаждения
    buff = None
    verbose = False

    b_w = [0.0] * 3
    b_w[0] = b1_warm
    alpha = 1 / (b_coev_warm ** 2)
    b_w[1] = math.pow(b_w[0], 2) * (math.pow(b_coev_warm, 2) + 1)
    b_w[2] = b_w[1] * b_w[0] * (1.0 + 2 / alpha)

    b_c = [0.0] * 3
    b_c[0] = b1_cold
    alpha = 1 / (b_coev_cold ** 2)
    b_c[1] = math.pow(b_c[0], 2) * (math.pow(b_coev_cold, 2) + 1)
    b_c[2] = b_c[1] * b_c[0] * (1.0 + 2 / alpha)

    im_start = time.process_time()
    smo = VacationQueueingSystemSimulator(n, buffer=buff)
    smo.set_sources(l, 'M')

    gamma_params_warm = GammaDistribution.get_params(b_w)
    gamma_params_cold = GammaDistribution.get_params(b_c)

    smo.set_servers(mu, 'M')
    smo.set_warm(gamma_params_warm, 'Gamma')
    smo.set_cold(gamma_params_cold, 'Gamma')

    smo.run(num_of_jobs)
    p = smo.get_p()
    w_im = smo.w  # .w -> wait times
    im_time = time.process_time() - im_start

    tt_start = time.process_time()
    tt = MMn_H2warm_H2cold(l, mu, b_w, b_c, n, buffer=buff,
                           verbose=verbose, accuracy=1e-14)

    tt.run()
    p_tt = tt.get_p()
    w_tt = tt.get_w()  # .get_w() -> wait times
    # print(w_tt)

    tt_time = time.process_time() - tt_start
    num_of_iter = tt.num_of_iter_

    print('warms starts', smo.warm_phase.starts_times)
    print('warms after cold starts', smo.warm_after_cold_starts)
    print('cold starts', smo.cold_phase.starts_times)
    print("zero wait arrivals num", smo.zero_wait_arrivals_num)

    print("\nСравнение результатов расчета методом Такахаси-Таками и ИМ.\n"
          "ИМ - M/Gamma/{0:^2d} с Gamma разогревом\nТакахаси-Таками - M/M/{0:^2d} c H2-разогревом и H2-охлаждением "
          "с комплексными параметрами\n"
          "Коэффициент загрузки: {1:^1.2f}".format(n, ro))
    print(f'Коэффициент вариации времени разогрева {b_coev_warm:0.3f}')
    print(f'Коэффициент вариации времени охлаждения {b_coev_cold:0.3f}')
    print(
        "Количество итераций алгоритма Такахаси-Таками: {0:^4d}".format(num_of_iter))
    print(
        f"Вероятность нахождения в состоянии разогрева\n\tИМ: {smo.get_warmup_prob():0.3f}\n\tЧисл: {tt.get_warmup_prob():0.3f}")
    print(
        f"Вероятность нахождения в состоянии охлаждения\n\tИМ: {smo.get_cold_prob():0.3f}\n\tЧисл: {tt.get_cold_prob():0.3f}")
    print(
        "Время работы алгоритма Такахаси-Таками: {0:^5.3f} c".format(tt_time))
    print("Время ИМ: {0:^5.3f} c".format(im_time))
    print("{0:^25s}".format("Первые 10 вероятностей состояний СМО"))
    print("{0:^3s}|{1:^15s}|{2:^15s}".format("№", "Числ", "ИМ"))
    print("-" * 32)
    for i in range(11):
        print("{0:^4d}|{1:^15.3g}|{2:^15.3g}".format(i, p_tt[i], p[i]))

    print("\n")
    print("{0:^25s}".format("Начальные моменты времени ожидания в СМО"))
    print("{0:^3s}|{1:^15s}|{2:^15s}".format("№", "Числ", "ИМ"))
    print("-" * 32)
    for i in range(3):
        calc_mom = w_tt[i].real if isinstance(w_tt[i], complex) else w_tt[i]
        sim_mom = w_im[i].real if isinstance(w_im[i], complex) else w_im[i]
        print(f"{i+1:^4d}|{calc_mom:^15.3g}|{sim_mom:^15.3g}")

    assert abs(w_tt[0] - w_im[0]) < 1e-2


if __name__ == "__main__":
    test_mmn_h2cold_h2_warm()
