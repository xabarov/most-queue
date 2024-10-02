import math
import sys
import time

import numpy as np
from colorama import Fore, Style, init
from tqdm import tqdm

from most_queue.rand_distribution import Exp_dist
from most_queue.sim.priority_queue_sim import PriorityQueueSimulator, Task

init()


class PriorityNetwork:
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
        # начальные моменты и типы распределений времени обслуживания по узлам, классам
        self.serv_params = serv_params
        # serv_params[node][k][{'params':[...], 'type':'...'}]
        # [node][prty_numbers_in_new_order] перестановки исходных
        self.nodes_prty = nodes_prty
        # номеров приоритетов по узлам

        self.smos = []

        for m in range(self.n_num):
            self.smos.append(PriorityQueueSimulator(n[m], k_num, prty[m]))
            param_serv_reset = []  # из-за смены порядка приоритетов в узле
            # для расчета необходимо преобразовать список с параметрами вр обслуживания в узле
            for k in range(k_num):
                param_serv_reset.append(serv_params[m][nodes_prty[m][k]])

            self.smos[m].set_servers(param_serv_reset)
            time.sleep(0.1)

        self.arrival_time = []
        self.sources = []
        self.v_network = []
        self.w_network = []
        for k in range(k_num):
            self.sources.append(Exp_dist(L[k]))
            self.arrival_time.append(self.sources[k].generate())
            time.sleep(0.1)
            self.v_network.append([0.0] * 3)
            self.w_network.append([0.0] * 3)

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
            self.v_network[k][i] = self.v_network[k][i] * (1.0 - (1.0 / self.served[k])) + math.pow(new_a, i + 1) / \
                self.served[k]

    def refresh_w_stat(self, k, new_a):
        for i in range(3):
            self.w_network[k][i] = self.w_network[k][i] * (1.0 - (1.0 / self.served[k])) + math.pow(new_a, i + 1) / \
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

            self.arrival_time[num_of_k_earlier] = self.ttek + \
                self.sources[num_of_k_earlier].generate()

            next_node = self.play_next_node(num_of_k_earlier, -1)

            ts = Task(num_of_k_earlier, self.ttek, True)

            next_node_class = self.nodes_prty[next_node][num_of_k_earlier]

            ts.in_node_class_num = next_node_class

            self.smos[next_node].arrival(next_node_class, self.ttek, ts)

        else:
            self.ttek = serving_earlier
            ts = self.smos[num_of_node_earlier].serving(
                num_of_serv_ch_earlier, True)

            real_class = ts.k
            next_node = self.play_next_node(real_class, num_of_node_earlier)

            if next_node == self.n_num:
                self.served[real_class] += 1
                self.in_sys[real_class] -= 1

                self.refresh_v_stat(real_class, self.ttek - ts.arr_network)
                self.refresh_w_stat(real_class, ts.wait_network)

            else:
                next_node_class = self.nodes_prty[next_node][real_class]

                self.smos[next_node].arrival(next_node_class, self.ttek, ts)

    def run(self, job_served, is_real_served=True):
        """
        Run simulation
        """
        if is_real_served:
            last_percent = 0

            with tqdm(total=100, unit='jobs') as pbar:
                while sum(self.served) < job_served:
                    self.run_one_step()
                    percent = int(100*(sum(self.served)/job_served))
                    if last_percent != percent:
                        last_percent = percent
                        pbar.update(1)
                        pbar.set_description(Fore.MAGENTA + '\rJob served: ' +
                                             Fore.YELLOW + f'{sum(self.served)}/{job_served}' + Fore.LIGHTGREEN_EX)
        else:
            print(Fore.GREEN + '\rStart simulation')
            print(Style.RESET_ALL)
            # print(Back.YELLOW + 'на желтом фоне')

            for i in tqdm(range(job_served)):
                self.run_one_step()

            print(Fore.GREEN + '\rSimulation is finished')
            print(Style.RESET_ALL)
