import numpy as np

from most_queue.general_utils.tables import probs_print, times_print_with_classes
from most_queue.rand_distribution import Gamma
from most_queue.sim.priority_queue_sim import PriorityQueueSimulator
from most_queue.theory import priority_calc


def test_sim():
    """
    Тестирование имитационной модели СМО с приоритетами
    Для верификации - сравнение с результатами расчета СМО с методом инвариантов отношения:
        Рыжиков Ю.И., Хомоненко А.Д. Расчет многоканальных систем обслуживания с абсолютным и
        относительным приоритетами на основе инвариантов отношения // Интеллектуальные технологии
        на транспорте. 2015. №3
    """
    n = 5  # число каналов
    k = 3  # число классов заявок
    l = [0.2, 0.3, 0.4]  # интенсивности пребытия заявок по классам
    lsum = sum(l)
    num_of_jobs = 300000  # количество заявок для ИМ

    # Зададим параметры обслуживания по начальным моментам.
    # Зададим средние времена обслуживания по классам
    b1 = [0.45 * n, 0.9 * n, 1.35 * n]

    # Коэфф вариации времени обслуживания пусть будет одинаковый для всех классов
    coev = 0.577

    # вторые начальные моменты
    b2 = [0] * k
    for i in range(k):
        b2[i] = (b1[i] ** 2) * (1 + coev ** 2)

    b_sr = sum(b1) / k

    # получим коэфф загрузки
    ro = lsum * b_sr / n

    # теперь по заданным двум нач моментам подберем параметры аппроксимирующего Гамма-распределения
    # и добавим в список параметров params
    params = []
    for i in range(k):
        params.append(Gamma.get_mu_alpha([b1[i], b2[i]]))

    b = []
    for j in range(k):
        b.append(Gamma.calc_theory_moments(params[j][0], params[j][1], 4))

    print("\nСравнение данных ИМ и результатов расчета методом инвариантов отношения (Р) \n"
          "времени пребывания в многоканальной СМО с приоритетами")
    print("Число каналов: " + str(n) + "\nЧисло классов: " + str(k) + "\nКоэффициент загрузки: {0:<1.2f}".format(ro) +
          "\nКоэффициент вариации времени обслуживания: " + str(coev) + "\n")
    print("Абсолютный приоритет")

    # при создании ИМ передаем число каналов, число классов и тип приоритета.
    # PR - абсолютный с дообслуживанием заявок
    qs = PriorityQueueSimulator(n, k, "PR")

    # для задания источников заявок и каналов обслуживания нужно задать набор словарей с полями
    # type - тип распределения,
    # params - его параметры.
    # Число таких словарей в списках sources и servers_params соответствует числу классов

    sources = []
    servers_params = []
    for j in range(k):
        sources.append({'type': 'M', 'params': l[j]})
        servers_params.append({'type': 'Gamma', 'params': params[j]})

    qs.set_sources(sources)
    qs.set_servers(servers_params)

    # запуск ИМ
    qs.run(num_of_jobs)

    # получение начальных моментов времени пребывания

    v_sim = qs.v

    # расчет их же методом инвариантов отношения (для сравнения)
    v_teor = priority_calc.get_v_prty_invar(l, b, n, 'PR')

    assert abs(v_sim[0][0] - v_teor[0][0]) < 0.3

    times_print_with_classes(v_sim, v_teor, False)

    print("Относительный приоритет")

    # Тоже самое для относительного приоритета (NP)
    qs = PriorityQueueSimulator(n, k, "NP")
    sources = []
    servers_params = []
    for j in range(k):
        sources.append({'type': 'M', 'params': l[j]})
        servers_params.append({'type': 'Gamma', 'params': params[j]})

    qs.set_sources(sources)
    qs.set_servers(servers_params)

    qs.run(num_of_jobs)

    v_sim = qs.v

    v_teor = priority_calc.get_v_prty_invar(l, b, n, 'NP')

    times_print_with_classes(v_sim, v_teor, False)

    assert abs(v_sim[0][0] - v_teor[0][0]) < 0.3


if __name__ == "__main__":
    test_sim()
