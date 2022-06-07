import smo_im_prty
import rand_destribution as rd
import numpy as np
import math
from smo_im_prty import Task
import network_calc
from network_im_prty import NetworkPrty
import time
import set_lamdas_dyn
from hyper_visor import HyperVisor
import copy

class NetworkPrtyDynArrival(NetworkPrty):
    """
    Имитационная модель СеМО с многоканальными узлами и приоритетами
    """

    def __init__(self, k_num, L, R, n, prty, serv_params, nodes_prty):

        NetworkPrty.__init__(self, k_num, L, R, n, prty, serv_params, nodes_prty)
        self.sigmas = None
        self.delta_jobs = None
        self.arrived_previous_arr = 0
        self.is_set_probab = False
        self.lambdas_data_change = None
        self.tek_change_mom = 0
        self.params_dict = {'k_num': k_num, 'L': L, 'R': R, 'n': n, 'prty': prty, 'serv_params': serv_params,
                            'nodes_prty': nodes_prty}
        self.hv = None
        self.hv_delta_jobs = None
        self.arr_hv_pr = 0

    def hyper_visor_set(self, v1_treb, hv_delta_job_run):
        self.hv = HyperVisor(self.params_dict, v1_treb)
        self.hv_delta_jobs = hv_delta_job_run

    def set_arrival_dyn(self, sigmas, delta_jobs):
        """
            Задает параметры изменения интерснивностей вх потоков
            sigmas - СКО (норм распределение)
            delta_jobs - через сколько прибывших заявок происходит изменение интенсивностей
        """
        self.sigmas = sigmas
        self.delta_jobs = delta_jobs
        self.is_set_probab = True

    def set_arrival_dyn_by_data(self, lambdas, delta_jobs):
        """
            Задает параметры изменения интерснивностей вх потоков
            lambdas[t][k] - массив c интенсивностями, t - дискертный момент времени, k- номер класса
            delta_jobs - через сколько прибывших заявок происходит изменение интенсивностей
        """
        self.lambdas_data_change = lambdas
        self.delta_jobs = delta_jobs

    def run_one_step(self):
        NetworkPrty.run_one_step(self)

        if self.delta_jobs:
            if sum(self.arrived) > self.delta_jobs + self.arrived_previous_arr:
                if self.is_set_probab:
                    for k in range(self.k_num):
                        delta = np.random.normal(0, self.sigmas[k])
                        r = np.random.rand()
                        if r <0.5:
                            delta = -delta
                        if self.L[k] + delta < 0:
                            continue
                        self.L[k] += delta
                else:
                    new_lambdas = self.lambdas_data_change[self.tek_change_mom]
                    self.tek_change_mom += 1
                    if self.tek_change_mom == len(self.lambdas_data_change):
                        self.tek_change_mom = 0
                    for k in range(self.k_num):
                        self.L[k] = new_lambdas[k]
                self.arrived_previous_arr += self.delta_jobs

        if self.hv:
            if sum(self.arrived) > self.hv_delta_jobs + self.arr_hv_pr:
                self.hv.arr_intens_meter(self.arrived, self.ttek, self)
                self.arr_hv_pr += self.hv_delta_jobs


if __name__ == '__main__':
    k_num = 3
    n_num = 5
    n = [3, 2, 3, 4, 3]
    R = []
    b = []  # k, node, j
    for i in range(k_num):
        R.append(np.matrix([
            [1, 0, 0, 0, 0, 0],
            [0, 0.4, 0.6, 0, 0, 0],
            [0, 0, 0, 0.6, 0.4, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ]))
    L = [0.2, 0.3, 0.4]
    L_first = L.copy()
    nodes_prty = []
    jobs_num = 10000
    serv_params = []
    h2_params = []
    for m in range(n_num):
        nodes_prty.append([])
        for j in range(k_num):
            nodes_prty[m].append(j)

        b1 = 0.7*n[m]/sum(L)
        coev = 1.2
        h2_params.append(rd.H2_dist.get_params_by_mean_and_coev(b1, coev))

        serv_params.append([])
        for i in range(k_num):
            serv_params[m].append({'type': 'H', 'params': h2_params[m]})

    nodes_prty_first = copy.deepcopy(nodes_prty)

    for k in range(k_num):
        b.append([])
        for m in range(n_num):
            b[k].append(rd.H2_dist.calc_theory_moments(*h2_params[m], 4))

    prty = ['NP'] * n_num

    # sigmas = [0.02, 0.02, 0.02]

    delta_jobs = 100

    # set_num = int(jobs_num/delta_jobs)

    lambdas = set_lamdas_dyn.load_lambdas_from_file('lambdas_trend_set.txt')

    semo_im = NetworkPrtyDynArrival(k_num, L, R, n, prty, serv_params, nodes_prty)

    # semo_im.set_arrival_dyn(sigmas, delta_jobs)

    semo_im.set_arrival_dyn_by_data(lambdas, delta_jobs)
    semo_im.hyper_visor_set([15, 15, 9], 1000)

    semo_im.run(jobs_num)

    v_im = semo_im.v_semo

    semo_calc = network_calc.network_prty_calc(R, b, n, L, prty, nodes_prty)
    v_ch = semo_calc['v']
    loads = semo_calc['loads']

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
    print("\nИсходные значения интенсивностей:")
    for ll in L_first:
        print("{0:^1.3f}".format(ll), end=" ")
    print("\nНовые значения интенсивностей:")
    for ll in semo_im.L:
        print("{0:^1.3f}".format(ll), end=" ")
    print("\nИсходные значения приоритетов в узлах:")
    for pr in nodes_prty_first:
        print(pr, end=' ')
    print("\nНовые значения приоритетов в узлах:")
    for pr in nodes_prty:
        print(pr, end=' ')
    print("\nОтчет hv. Кол-во (превышение интен-ей/нарушений треб врем пребывания/успешных запусков модуля prty):")
    print(semo_im.hv.num_of_arr_exceeds,  '/',
          semo_im.hv.num_of_seen_v1_prob, '/', semo_im.hv.num_of_prty_success,  end=' ')
    print("\n")
    print("-" * 60)
    print("{0:^60s}".format("Относительный приоритет"))

    print("-" * 60)
    print("{0:^11s}|{1:^47s}|".format('', 'Номер начального момента'))
    print("{0:^10s}| ".format('№ кл'), end="")
    print("-" * 45 + " |")

    print(" " * 11 + "|", end="")
    for j in range(3):
        s = str(j + 1)
        print("{:^15s}|".format(s), end="")
    print("")
    print("-" * 60)

    for i in range(k_num):
        print(" " * 5 + "|", end="")
        print("{:^5s}|".format("ИМ"), end="")
        for j in range(3):
            print("{:^15.3g}|".format(v_im[i][j]), end="")
        print("")
        print("{:^5s}".format(str(i + 1)) + "|" + "-" * 54)

        print(" " * 5 + "|", end="")
        print("{:^5s}|".format("Р"), end="")
        for j in range(3):
            print("{:^15.3g}|".format(v_ch[i][j]), end="")
        print("")
        print("-" * 60)

    print("\n")

    # prty = ['PR'] * n_num
    # semo_im = NetworkPrty(k_num, L, R, n, prty, serv_params, nodes_prty)
    #
    # semo_im.run(jobs_num)
    #
    # v_im = semo_im.v_semo
    #
    # semo_calc = network_calc.network_prty_calc(R, b, n, L, prty, nodes_prty)
    # v_ch = semo_calc['v']
    #
    # print("-" * 60)
    # print("{0:^60s}".format("Абсолютный приоритет"))
    #
    # print("-" * 60)
    # print("{0:^11s}|{1:^47s}|".format('', 'Номер начального момента'))
    # print("{0:^10s}| ".format('№ кл'), end="")
    # print("-" * 45 + " |")
    #
    # print(" " * 11 + "|", end="")
    # for j in range(3):
    #     s = str(j + 1)
    #     print("{:^15s}|".format(s), end="")
    # print("")
    # print("-" * 60)
    #
    # for i in range(k_num):
    #     print(" " * 5 + "|", end="")
    #     print("{:^5s}|".format("ИМ"), end="")
    #     for j in range(3):
    #         print("{:^15.3g}|".format(v_im[i][j]), end="")
    #     print("")
    #     print("{:^5s}".format(str(i + 1)) + "|" + "-" * 54)
    #
    #     print(" " * 5 + "|", end="")
    #     print("{:^5s}|".format("Р"), end="")
    #     for j in range(3):
    #         print("{:^15.3g}|".format(v_ch[i][j]), end="")
    #     print("")
    #     print("-" * 60)
    #
    # print("\n")