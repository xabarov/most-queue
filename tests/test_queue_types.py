"""
Compare of queue implementation 
"""
import math
import time

from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.queueing_systems.fifo import QueueingSystemSimulator


def compare_calc_times():
    """Compare queue calc times
    """
    n = 3  # число каналов
    l = 1.0  # интенсивность вх потока
    ro = 0.7  # коэфф загрузки
    b1 = n * ro / l  # ср время обслуживания
    num_of_jobs = 1000000  # число обсл заявок ИМ
    # два варианта коэфф вариации времени обсл, запустим расчет и ИМ для каждого из них
    buffer_types = ['list', 'deque']
    b_coev = 1.2

    calc_times = {'list': 0, 'deque': 0}

    for b_type in buffer_types:
        #  расчет начальных моментов времени обслуживания по заданному среднему и коэфф вариации
        b = [0.0] * 3
        alpha = 1 / (b_coev ** 2)
        b[0] = b1
        b[1] = math.pow(b[0], 2) * (math.pow(b_coev, 2) + 1)
        b[2] = b[1] * b[0] * (1.0 + 2 / alpha)

        # запуск ИМ для верификации результатов
        im_start = time.process_time()

        qs = QueueingSystemSimulator(n, buffer_type=b_type)

        # задаем вх поток заявок. М - экспоненциальный с интенсивностью l
        qs.set_sources(l, 'M')

        # задаем параметры каналов обслуживания Гамма-распределением.
        # Параметры распределения подбираем с помощью метода библиотеки random_distribution
        gamma_params = GammaDistribution.get_params([b[0], b[1]])
        qs.set_servers(gamma_params, 'Gamma')

        # Запуск ИМ
        qs.run(num_of_jobs)

        calc_times[b_type] = time.process_time() - im_start

    print(calc_times)


if __name__ == "__main__":
    compare_calc_times()
