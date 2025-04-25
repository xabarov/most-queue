import numpy as np

from most_queue.general.tables import probs_print, times_print
from most_queue.sim.queueing_systems.fifo import QueueingSystemSimulator
from most_queue.theory.queueing_systems.fifo.mmnr import MMnrCalc
from most_queue.theory.queueing_systems.fifo.m_d_n import MDn


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
    qs.set_sources(l, 'M')
    # задаем распределение обслуживания - параметры и тип распределения.
    qs.set_servers(mu, 'M')

    # запуск ИМ - передаем кол-во заявок для обслуживания
    qs.run(300000)
    # получаем начальные моменты времени ожидания. Также можно получить время пребывания .v,
    # вероятности состояний .get_p(), периоды непрерывной занятости .pppz
    w_sim = qs.w

    mmnr = MMnrCalc(l, mu, n, r)

    # для сравнения расчитаем теже начальные моменты численно
    w = mmnr.get_w()

    # вывод результатов

    times_print(w_sim, w, True)

    print("\n\nДанные ИМ::\n")
    # print(qs)

    # тоже для детерминированного обслуживания

    qs = QueueingSystemSimulator(n)

    qs.set_sources(l, 'M')
    qs.set_servers(1.0 / mu, 'D')

    qs.run(1000000)

    mdn = MDn(l, 1 / mu, n)
    p_ch = mdn.calc_p()
    p_sim = qs.get_p()

    probs_print(p_sim, p_ch, 10)

    assert np.allclose(np.array(p_sim[:10]), np.array(p_ch[:10]), atol=1e-2)


if __name__ == "__main__":
    test_sim()
