import numpy as np

from most_queue.general_utils.tables import probs_print, times_print_with_classes
from most_queue.rand_distribution import H2_dist
from most_queue.sim.priority_network import PriorityNetwork
from most_queue.theory import network_calc


def test_network():
    """
    Тестирование ИМ СеМО с приоритетами в узлах
    Сравнение с численным расчетом методом декомпозиции
    """
    k_num = 3  # число классов
    n_num = 5  # число узлов

    n = [3, 2, 3, 4, 3]  # распределение числа каналов в узлах сети
    # список матриц вероятностей переходов между узлами сети для каждого класса.
    R = []
    b = []  # список массивов начальных моментов распределения времени обслуживания [k, node, j]
    for i in range(k_num):
        # задаем одинаковые матрицы для всех классов звявок. Можно задать различные
        R.append(np.matrix([
            [1, 0, 0, 0, 0, 0],
            [0, 0.4, 0.6, 0, 0, 0],
            [0, 0, 0, 0.6, 0.4, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ]))
    # интенсивности поступления заявок в сеть по каждому из классов
    L = [0.1, 0.3, 0.4]
    nodes_prty = []  # распределение приоритетов между заявками для каждого из узлов сети [m][x1, x2 .. x_k],
    # m - номер узла, xi- приоритет для i-го класса, k - число классов
    # Например: [0][0,1,2] - для первого узла задан прямой порядок приоритетов,
    # [2][0,2,1] - для третьего узла задан такой порядок приоритетов: для первого класса - самый старший (0),
    # для второго - младший (2), для третьего - средний (1)

    jobs_num = 100000  # число заявок для обслуживания в ИМ
    # параметры каналов обслуживания сети [m]{type: string, params: []},
    serv_params = []
    # где m - номер узла, type - тип распределения, params - параметры распределения.
    # Подробнее о параметрах распределения:
    #        Вид распределения                   Тип[types]     Параметры [params]
    #         Экспоненциальное                      'М'              mu
    #         Гиперэкспоненциальное 2-го порядка    'Н'         [y1, mu1, mu2]
    #         Эрланга                               'E'           [r, mu]
    #         Гамма-распределение                  'Gamma'        [mu, alpha]
    #         Кокса 2-го порядка                    'C'         [y1, mu1, mu2]
    #         Парето                                'Pa'         [alpha, K]
    #         Равномерное                         'Uniform'     [mean, half_interval]
    #         Детерминированное                      'D'         [b]

    h2_params = []
    for m in range(n_num):
        nodes_prty.append([])
        for j in range(k_num):
            if m % 2 == 0:
                nodes_prty[m].append(j)
            else:
                nodes_prty[m].append(k_num - j - 1)

        b1 = 0.9 * n[m] / sum(L)
        coev = 1.2
        h2_params.append(H2_dist.get_params_by_mean_and_coev(b1, coev))

        serv_params.append([])
        for i in range(k_num):
            serv_params[m].append({'type': 'H', 'params': h2_params[m]})

    for k in range(k_num):
        b.append([])
        for m in range(n_num):
            b[k].append(H2_dist.calc_theory_moments(*h2_params[m], 4))

    # список, содержащий тип приоритета для каждого узла сети.
    prty = ['NP'] * n_num
    # "NP" - относительный приоритет. Также доступны абсолютные ("PR", "RW", "RS") и без приоритета "No"

    # Создаем экземпляр модели СеМО
    qn = PriorityNetwork(k_num, L, R, n, prty, serv_params, nodes_prty)

    #  Запуск ИМ:
    qn.run(jobs_num)

    #  Получение нач. моментов пребывания в СеМО
    v_im = qn.v_network

    #  Получение нач. моментов пребывания в СеМО с помощью метода инвариантов отношения
    semo_calc = network_calc.network_prty_calc(R, b, n, L, prty, nodes_prty)
    v_ch = semo_calc['v']

    #  получения коэфф загрузки каждого узла
    loads = semo_calc['loads']

    #  вывод результатов
    print("\n")
    print("-" * 60)
    print("{0:^60s}\n{1:^60s}".format("Сравнение данных ИМ и результатов расчета времени пребывания",
                                      "в СеМО с многоканальными узлами и приоритетами"))
    print("-" * 60)
    print("Количество каналов в узлах:")
    for nn in n:
        print("{0:^1d}".format(nn), end=" ")
    print("\nКоэффициенты загрузки узлов :")
    for load in loads:
        print("{0:^1.3f}".format(load), end=" ")
    print("\n")
    print("-" * 60)
    print("{0:^60s}".format("Относительный приоритет"))

    assert 100*(abs(v_im[0][0] - v_ch[0][0]))/max(v_im[0][0], v_ch[0][0]) < 10

    times_print_with_classes(v_im, v_ch, False)
    #  Теперь для абсолютного приоритета:
    prty = ['PR'] * n_num
    qn = PriorityNetwork(k_num, L, R, n, prty, serv_params, nodes_prty)

    qn.run(jobs_num)

    v_im = qn.v_network

    semo_calc = network_calc.network_prty_calc(R, b, n, L, prty, nodes_prty)
    v_ch = semo_calc['v']

    print("-" * 60)
    print("{0:^60s}".format("Абсолютный приоритет"))

    times_print_with_classes(v_im, v_ch, False)

    assert 100*(abs(v_im[0][0] - v_ch[0][0]))/max(v_im[0][0], v_ch[0][0]) < 10


if __name__ == "__main__":

    test_network()
