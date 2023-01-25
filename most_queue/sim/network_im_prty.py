from most_queue.sim.smo_im_prty import SmoImPrty
import most_queue.sim.rand_destribution as rd
import numpy as np
import math
from most_queue.sim.smo_im_prty import Task
from most_queue.theory import network_calc
import time
from tqdm import tqdm
import sys

class NetworkPrty:
    """
    Имитационная модель СеМО с многоканальными узлами и приоритетами
    """

    def __init__(self, k_num, L, R, n, prty, serv_params, nodes_prty):

        self.k_num = k_num  # число классов
        self.L = L  # L[k] - вх интенсивности
        self.R = R  # R[k] - маршрутные матрицы
        self.n_num = len(n)  # количество узлов
        self.nodes = n  # n[i] - количество каналов в узлах
        self.prty = prty  # prty[n] - тип приоритета в узле. 'PR', 'NP'
        self.serv_params = serv_params  # начальные моменты и типы распределений времени обслуживания по узлам, классам
        # serv_params[node][k][{'params':[...], 'type':'...'}]
        self.nodes_prty = nodes_prty  # [node][prty_numbers_in_new_order] перестановки исходных
        # номеров приоритетов по узлам

        self.smos = []

        for m in range(self.n_num):
            self.smos.append(SmoImPrty(n[m], k_num, prty[m]))
            param_serv_reset = []  # из-за смены порядка приоритетов в узле
            # для расчета необходимо преобразовать список с параметрами вр обслуживания в узле
            for k in range(k_num):
                param_serv_reset.append(serv_params[m][nodes_prty[m][k]])

            self.smos[m].set_servers(param_serv_reset)
            time.sleep(0.1)

        self.arrival_time = []
        self.sources = []
        self.v_semo = []
        self.w_semo = []
        for k in range(k_num):
            self.sources.append(rd.Exp_dist(L[k]))
            self.arrival_time.append(self.sources[k].generate())
            time.sleep(0.1)
            self.v_semo.append([0.0] * 3)
            self.w_semo.append([0.0] * 3)

        self.ttek = 0
        self.total = 0
        self.served = [0] * self.k_num
        self.in_sys = [0] * self.k_num
        self.t_old = [0] * self.k_num
        self.arrived = [0] * self.k_num

    def play_next_node(self, real_class, current_node):
        sum_p = 0
        p = np.random.rand()
        for i in range(self.R[real_class].shape[0]):
            sum_p += self.R[real_class][current_node + 1, i]
            if sum_p > p:
                return i
        return 0

    def refresh_v_stat(self, k, new_a):
        for i in range(3):
            self.v_semo[k][i] = self.v_semo[k][i] * (1.0 - (1.0 / self.served[k])) + math.pow(new_a, i + 1) / \
                                self.served[k]

    def refresh_w_stat(self, k, new_a):
        for i in range(3):
            self.w_semo[k][i] = self.w_semo[k][i] * (1.0 - (1.0 / self.served[k])) + math.pow(new_a, i + 1) / \
                                self.served[k]

    def run_one_step(self):
        num_of_serv_ch_earlier = -1  # номер канала узла, мин время до окончания обслуживания
        num_of_k_earlier = -1  # номер класса, прибывающего через мин время
        num_of_node_earlier = -1  # номер узла, в котором раньше всех закончится обслуживание
        arrival_earlier = 1e10  # момент прибытия ближайшего
        serving_earlier = 1e10  # момент ближайшего обслуживания

        for k in range(self.k_num):
            if self.arrival_time[k] < arrival_earlier:
                num_of_k_earlier = k
                arrival_earlier = self.arrival_time[k]

        for node in range(self.n_num):
            for c in range(self.nodes[node]):
                if self.smos[node].servers[c].time_to_end_service < serving_earlier:
                    serving_earlier = self.smos[node].servers[c].time_to_end_service
                    num_of_serv_ch_earlier = c
                    num_of_node_earlier = node

        if arrival_earlier < serving_earlier:

            self.ttek = arrival_earlier
            self.arrived[num_of_k_earlier] += 1
            self.in_sys[num_of_k_earlier] += 1

            self.arrival_time[num_of_k_earlier] = self.ttek + self.sources[num_of_k_earlier].generate()

            next_node = self.play_next_node(num_of_k_earlier, -1)

            ts = Task(num_of_k_earlier, self.ttek, True)

            next_node_class = self.nodes_prty[next_node][num_of_k_earlier]

            ts.in_node_class_num = next_node_class

            self.smos[next_node].arrival(next_node_class, self.ttek, ts)

        else:
            self.ttek = serving_earlier
            ts = self.smos[num_of_node_earlier].serving(num_of_serv_ch_earlier, True)

            real_class = ts.k
            next_node = self.play_next_node(real_class, num_of_node_earlier)

            if next_node == self.n_num:
                self.served[real_class] += 1
                self.in_sys[real_class] -= 1

                self.refresh_v_stat(real_class, self.ttek - ts.arr_semo)
                self.refresh_w_stat(real_class, ts.wait_semo)

            else:
                next_node_class = self.nodes_prty[next_node][real_class]

                self.smos[next_node].arrival(next_node_class, self.ttek, ts)

    def run(self, job_served, is_real_served=False):
        if is_real_served:
            while (sum(self.served) < job_served):
                self.run_one_step()
                sys.stderr.write('\rStart simulation. Job served: %d/%d' % (sum(self.served), job_served))
                sys.stderr.flush()
        else:
            print("Start simulation...\n")
            for i in tqdm(range(job_served)):
                self.run_one_step()



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
    L = [0.1, 0.3, 0.4]
    nodes_prty = []
    jobs_num = 100000
    serv_params = []
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
        h2_params.append(rd.H2_dist.get_params_by_mean_and_coev(b1, coev))

        serv_params.append([])
        for i in range(k_num):
            serv_params[m].append({'type': 'H', 'params': h2_params[m]})

    for k in range(k_num):
        b.append([])
        for m in range(n_num):
            b[k].append(rd.H2_dist.calc_theory_moments(*h2_params[m], 4))

    prty = ['NP'] * n_num
    semo_im = NetworkPrty(k_num, L, R, n, prty, serv_params, nodes_prty)

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

    prty = ['PR'] * n_num
    semo_im = NetworkPrty(k_num, L, R, n, prty, serv_params, nodes_prty)

    semo_im.run(jobs_num)

    v_im = semo_im.v_semo

    semo_calc = network_calc.network_prty_calc(R, b, n, L, prty, nodes_prty)
    v_ch = semo_calc['v']

    print("-" * 60)
    print("{0:^60s}".format("Абсолютный приоритет"))

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
