import numpy as np

from most_queue.general_utils.tables import probs_print
from most_queue.rand_distribution import Erlang_dist
from most_queue.sim.qs_sim import QueueingSystemSimulator
from most_queue.theory.ek_d_n_calc import Ek_D_n


def test_EkDn():
    """
    Тестирование численного расчета многоканальной системы Ek/D/n
    с детерминированным обслуживанием

    Для вызова - используйте класс Ek_D_n пакета most_queue.theory.ek_d_n_calc
    Для верификации используем имитационное моделирование (ИМ).

    """

    ro = 0.8  # коэффициент загрузки СМО
    num_of_jobs = 300000  # количество заявок для ИМ. Чем больше, тем выше точночть ИМ
    n = 4  # число каналов

    # при создании экземпляра класса Ek_D_n нужно передать 4 параметра:
    # - l , k - параметры распределения Эрланга вх потока
    # - b - время обслуживания (детерминированное)
    # - n - число каналов обслуживания

    # Подберем параметры k и l аппроксимирующего распределения по среднему значению и коэфф вариации
    # с помощью метода most_queue.sim.rand_distribution.Erlang_dist.get_params_by_mean_and_coev()

    a1 = 1  # среднее время между заявками вх потока
    coev_a = 0.56  # коэффициент вариации вх потока
    k, l = Erlang_dist.get_params_by_mean_and_coev(a1, coev_a)

    # время обслуживания определим на основе заданного коэффициента загрузки
    # В вашем случае параметры l, k, b и n могут быть заданы непосредственно.
    # Обязательно проверьте, чтобы коэфф загрузки СМО не превышал 1.0

    b = a1 * n * ro

    # создаем экземпляр класса для численного расчета
    ekdn = Ek_D_n(l, k, b, n)

    # запускаем расчет вероятностей состояний СМО
    p_ch = ekdn.calc_p()

    # для верификации используем ИМ.
    # создаем экземпляр класса ИМ, передаем число каналов обслуживания
    qs = QueueingSystemSimulator(n)

    # задаем входной поток. Методу нужно передать параметры распределения списком и тип распределения. E - Эрланг
    qs.set_sources([k, l], "E")
    # задаем каналы обслуживания. На вход время обслуживания и тип распределения - D.
    qs.set_servers(b, "D")

    # запускаем ИМ:
    qs.run(num_of_jobs)

    # получаем параметры - начальные моменты времени пребывания и распределение веротяностей состояния системы
    v_sim = qs.v
    p_sim = qs.get_p()

    # выводим полученные значения:
    probs_print(p_sim, p_ch, 10)

    assert np.allclose(np.array(p_sim[:10]), np.array(p_ch[:10]), atol=1e-2)


if __name__ == "__main__":
    test_EkDn()