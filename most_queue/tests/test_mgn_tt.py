"""
    Тестирование метода Такахаси-Таками для расчета СМО М/H2/n

    При коэфф вариации времени обслуживания < 1 параметры аппроксимирующего Н2-распределения
    являются комплексными, что не мешает получению осмысленных результатов

    Для верификации используется ИМ

"""
import math
import time

from most_queue.rand_distribution import Gamma
from most_queue.sim.qs_sim import QueueingSystemSimulator
from most_queue.theory.mgn_tt import MGnCalc
from most_queue.general_utils.tables import probs_print, times_print


def test_mgn_tt():
    """
    Тестирование метода Такахаси-Таками для расчета СМО М/H2/n
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
        alpha = 1 / (b_coev ** 2)
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

        qs = QueueingSystemSimulator(n)

        # задаем вх поток заявок. М - экспоненциальный с интенсивностью l
        qs.set_sources(l, 'M')

        # задаем параметры каналов обслуживания Гамма-распределением.
        # Параметры распределения подбираем с помощью метода библиотеки random_distribution
        gamma_params = Gamma.get_mu_alpha([b[0], b[1]])
        qs.set_servers(gamma_params, 'Gamma')

        # Запуск ИМ
        qs.run(num_of_jobs)

        # Получение результатов
        p = qs.get_p()
        v_sim = qs.v
        im_time = time.process_time() - im_start

        # вывод результатов

        print("\nСравнение результатов расчета методом Такахаси-Таками и ИМ.")
        print(
            f"ИМ - M/Gamma/{n:^2d}\nТакахаси-Таками - M/H2/{n:^2d} с комплексными параметрами")
        print(f"Коэффициент загрузки: {ro:^1.2f}")
        print(f"Коэффициент вариации времени обслуживания: {b_coev:^1.2f}")
        print(
            f"Количество итераций алгоритма Такахаси-Таками: {num_of_iter:^4d}")
        print(f"Время работы алгоритма Такахаси-Таками: {tt_time:^5.3f} c")
        print(f"Время работы ИМ: {im_time:^5.3f} c")
        probs_print(p, p_tt, 10)

        times_print(v_sim, v_tt, False)

        assert 100*abs(v_tt[0] - v_sim[0])/max(v_tt[0], v_sim[0]) < 10


if __name__ == "__main__":
    test_mgn_tt()