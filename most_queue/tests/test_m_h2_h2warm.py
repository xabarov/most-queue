import math
import time

import numpy as np

from most_queue.general_utils.tables import probs_print, times_print
from most_queue.rand_distribution import Gamma
from most_queue.sim.qs_sim import QueueingSystemSimulator
from most_queue.theory.m_h2_h2warm import Mh2h2Warm


def test_m_h2_h2warm():
    n = 5  # число каналов
    l = 1.0  # интенсивность вх потока
    ro = 0.7  # коэфф загрузки
    b1 = n * 0.7  # ср время обслуживания
    b1_warm = n * 0.1  # ср время разогрева
    num_of_jobs = 1000000  # число обсл заявок ИМ
    b_coevs = [1.5]  # коэфф вариации времени обсл
    b_coev_warm = 1.2  # коэфф вариации времени разогрева
    buff = None  # очередь - неограниченная
    verbose = False  # не выводить пояснения при расчетах

    for b_coev in b_coevs:
        b = [0.0] * 3
        alpha = 1 / (b_coev ** 2)
        b[0] = b1
        b[1] = math.pow(b[0], 2) * (math.pow(b_coev, 2) + 1)
        b[2] = b[1] * b[0] * (1.0 + 2 / alpha)

        b_w = [0.0] * 3
        b_w[0] = b1_warm
        alpha = 1 / (b_coev_warm ** 2)
        b_w[1] = math.pow(b_w[0], 2) * (math.pow(b_coev_warm, 2) + 1)
        b_w[2] = b_w[1] * b_w[0] * (1.0 + 2 / alpha)

        im_start = time.process_time()
        qs = QueueingSystemSimulator(n, buffer=buff)
        qs.set_sources(l, 'M')

        gamma_params = Gamma.get_mu_alpha(b)
        gamma_params_warm = Gamma.get_mu_alpha(b_w)
        qs.set_servers(gamma_params, 'Gamma')
        qs.set_warm(gamma_params_warm, 'Gamma')
        qs.run(num_of_jobs)
        p = qs.get_p()
        v_sim = qs.v
        im_time = time.process_time() - im_start

        tt_start = time.process_time()
        tt = Mh2h2Warm(l, b, b_w, n, buffer=buff, verbose=verbose)

        tt.run()
        p_tt = tt.get_p()
        v_tt = tt.get_v()
        tt_time = time.process_time() - tt_start

        num_of_iter = tt.num_of_iter_

        print("\nСравнение результатов расчета методом Такахаси-Таками и ИМ.\n"
              "ИМ - M/Gamma/{0:^2d}\nТакахаси-Таками - M/H2/{0:^2d}"
              "с комплексными параметрами\n"
              "Коэффициент загрузки: {1:^1.2f}".format(n, ro))
        print(f'Коэффициент вариации времени обслуживания {b_coev:0.3f}')
        print(f'Коэффициент вариации времени разогрева {b_coev_warm:0.3f}')
        print(
            "Количество итераций алгоритма Такахаси-Таками: {0:^4d}".format(num_of_iter))
        print(
            "Время работы алгоритма Такахаси-Таками: {0:^5.3f} c".format(tt_time))
        print("Время ИМ: {0:^5.3f} c".format(im_time))
        probs_print(p, p_tt, 10)
        times_print(v_sim, v_tt, False)

        assert np.allclose(np.array(v_sim), np.array(v_tt), rtol=1e-1)


if __name__ == "__main__":
    test_m_h2_h2warm()
