import math
import sys
import time

import numpy as np
from colorama import Fore, Style, init
from tqdm import tqdm

from most_queue.sim.utils.distribution_utils import create_distribution

init()


class PriorityQueueSimulator:
    """
    Имитационная модель СМО GI/G/n/r и GI/G/n с приоритетами
    """

    def __init__(self, num_of_channels, num_of_classes, prty_type='No', buffer=None, calc_next_event_time=False):
        """
        num_of_channels - количество каналов СМО
        num_of_classes - количество классов заявок
        prty_type - тип приоритета:
            No  - без приоритетов, FIFO
            PR  -  preemptive resume, с дообслуживанием прерванной заявки
            RS  -  preemptive repeat with resampling, обслуживание заново с новой случайной длительностью
            RW  - preemptive repeat without resampling, обслуживание заново с прежней длительностью
            NP  - non preemptive, относительный приоритет
        buffer - максимальная длина очереди

        Для запуска ИМ необходимо:
        - вызвать конструктор с параметрами
        - задать вх поток с помощью метода set_sorces() экземпляра созданного класса PriorityQueueSimulator
        - задать распределение обслуживания с помощью метода set_servers() экземпляра класса PriorityQueueSimulator
        - запустить ИМ с помощью метода run() экземпляра созданного класса PriorityQueueSimulator,
        которому нужно передать число требуемых к обслуживанию заявок

        """
        self.n = num_of_channels
        self.k = num_of_classes
        self.buffer = buffer
        self.prty_type = prty_type
        self.free_channels = self.n
        self.num_of_states = 100000
        self.load = 0  # коэффициент загрузки системы

        # для отслеживания длины периода непрерывной занятости каналов:
        self.start_ppnz = -1
        self.ppnz_moments = [0] * self.k
        self.ppnz = []
        self.queue = []
        self.class_ppnz_started = -1
        self.w = []  # начальные моменты времени ожидания в СМО
        self.v = []  # начальные моменты времени пребывания в СМО
        # вероятности состояний СМО (нахождения в ней j заявок):
        self.p = []
        if self.prty_type == "No":
            # общая очередь для всех классов заявок:
            self.queue = []  # очередь, класс заявок - Task
        else:
            for i in range(self.k):
                self.queue.append([])

        for i in range(self.k):
            self.ppnz.append([0, 0, 0])
            self.w.append([0, 0, 0])
            self.v.append([0, 0, 0])
            self.p.append([0.0] * self.num_of_states)

        self.ttek = 0  # текущее время моделирования
        self.total = 0

        # количество заявок, принятых на обслуживание
        self.taked = [0] * self.k
        self.served = [0] * self.k  # количество заявок, обслуженных системой
        self.in_sys = [0] * self.k  # кол-во заявок в системе
        self.t_old = [0.0] * self.k  # момент предыдущего события
        self.arrived = [0] * self.k  # кол-во поступивших заявок
        # кол-во заявок, получивших отказ в обслуживании
        self.dropped = [0] * self.k
        self.arrival_time = [0.0] * self.k  # момент прибытия следущей заявки

        self.servers = []  # каналы обслуживания, список с классами Server

        self.sources = []
        self.sources_params = None
        self.servers_params = None

        self.is_set_source_params = False
        self.is_set_server_params = False

        self.warm_up = None
        self.is_warm_up_set = False

        self.time_to_next_event = 0
        self.is_next_calc = calc_next_event_time

        self.generator = np.random.default_rng()

    def set_sources(self, sources):
        """
        Задает тип и параметры распределения интервала поступления заявок для каждого из классов.
        sources - список источников. Каждый источник - словарь вида {'type': 'Some letter', 'params': [x, y, z]}

        Вид распределения                   Тип[type]     Параметры [params]
        --------------------------------------------------------------------
        Экспоненциальное                      'М'              mu
        Гиперэкспоненциальное 2-го порядка    'Н'         [y1, mu1, mu2]
        Гамма-распределение                  'Gamma'        [mu, alpha]
        Эрланга                               'E'           [r, mu]
        Кокса 2-го порядка                    'C'         [y1, mu1, mu2]
        Парето                                'Pa'         [alpha, K]
        Равномерное                         'Uniform'     [mean, half_interval]
        Детерминированное                      'D'         [b]
        """
        self.sources_params = sources
        self.is_set_source_params = True

        for i in range(len(sources)):
            source_type = sources[i]['type']
            params = sources[i]['params']

            self.sources.append(create_distribution(
                params, source_type, self.generator))

            self.arrival_time[i] = self.sources[i].generate()
            time.sleep(0.1)

    def set_servers(self, servers_params):
        """
        Задает тип и параметры распределения времени обслуживания.
        servers - список параметров серверов. Каждый элемент списка
         - словарь вида {'type': 'Some letter', 'params': [x, y, z]}
        Вид распределения                   Тип[types]     Параметры [params]
        Экспоненциальное                      'М'              mu
        Гиперэкспоненциальное 2-го порядка    'Н'         [y1, mu1, mu2]
        Гамма-распределение                  'Gamma'        [mu, alpha]
        Эрланга                               'E'           [r, mu]
        Кокса 2-го порядка                    'C'         [y1, mu1, mu2]
        Парето                                'Pa'         [alpha, K]
        Равномерное                         'Uniform'     [mean, half_interval]
        Детерминированное                      'D'         [b]
        """
        self.servers_params = servers_params

        self.is_set_server_params = True

        for i in range(self.n):
            self.servers.append(
                Server(self.servers_params, self.prty_type, self.generator))
            time.sleep(0.1)

    def set_warm_up(self, warm_up_params):
        """
        Задает начальные моменты времени разогрева
        warm_up_params - список словарей типа {'type': 'Some letter', 'params': [x, y, z]}
        Вид распределения                   Тип[types]     Параметры [params]
        Экспоненциальное                      'М'              mu
        Гиперэкспоненциальное 2-го порядка    'Н'         [y1, mu1, mu2]
        Эрланга                               'E'           [r, mu]
        Гамма-распределение                  'Gamma'        [mu, alpha]
        Кокса 2-го порядка                    'C'         [y1, mu1, mu2]
        Парето                                'Pa'         [alpha, K]
        Равномерное                         'Uniform'     [mean, half_interval]
        Детерминированное                      'D'         [b]
        """

        self.is_warm_up_set = True
        self.warm_up = []

        for i in range(len(warm_up_params)):
            warm_up_type = warm_up_params[i]['type']
            params = warm_up_params[i]['params']

            self.warm_up.append(create_distribution(
                params, warm_up_type, self.generator))

    def calc_load(self):
        """
        Вычисляет коэффициент загрузки СМО
        """
        l_sum = 0
        b1_sr = 0

        for i in range(self.k):

            if self.sources_params[i]['type'] == "M":
                l_sum += self.sources_params[i]['params']
            elif self.sources_params[i]['type'] == "H":
                y1 = self.sources_params[i]['params'][0]
                y2 = 1.0 - y1
                mu1 = self.sources_params[i]['type'][1]
                mu2 = self.sources_params[i]['type'][2]

                f1 = y1 / mu1 + y2 / mu2
                l_sum += 1.0 / f1

            elif self.sources_params[i]['type'] == "E":
                r = self.sources_params[i]['params'][0]
                mu = self.sources_params[i]['params'][1]
                l_sum += mu / r

            elif self.sources_params[i]['type'] == "Gamma":
                mu = self.sources_params[i]['params'][0]
                alpha = self.sources_params[i]['params'][1]
                l_sum += mu / alpha

            elif self.sources_params[i]['type'] == "C":
                y1 = self.sources_params[i]['params'][0]
                y2 = 1.0 - y1
                mu1 = self.sources_params[i]['params'][1]
                mu2 = self.sources_params[i]['params'][2]

                f1 = y2 / mu1 + y1 * (1.0 / mu1 + 1.0 / mu2)
                l_sum += 1.0 / f1
            elif self.sources_params[i]['type'] == "Pa":
                if self.sources_params[i]['params'][0] < 1:
                    return None
                else:
                    a = self.sources_params[i]['params'][0]
                    k = self.sources_params[i]['params'][1]
                    f1 = a * k / (a - 1)
                    l_sum += 1.0 / f1
            elif self.sources_params[i]['type'] == "Uniform":
                f1 = self.sources_params[i]['type'][0]
                l_sum += 1.0 / f1

            elif self.sources_params[i]['type'] == "D":
                f1 = self.sources_params[i]['type']
                l_sum += 1.0 / f1

            if self.servers_params[i]['type'] == "M":
                mu = self.servers_params[i]['params']
                b1_sr += 1.0 / mu

            elif self.servers_params[i]['type'] == "H":
                y1 = self.servers_params[i]['params'][0]
                y2 = 1.0 - y1
                mu1 = self.servers_params[i]['params'][1]
                mu2 = self.servers_params[i]['params'][2]

                b1_sr += y1 / mu1 + y2 / mu2

            elif self.servers_params[i]['type'] == "Gamma":
                mu = self.servers_params[i]['params'][0]
                alpha = self.servers_params[i]['params'][1]
                b1_sr += alpha / mu

            elif self.servers_params[i]['type'] == "E":
                r = self.servers_params[i]['params'][0]
                mu = self.servers_params[i]['params'][1]
                b1_sr += r / mu

            elif self.servers_params[i]['type'] == "Uniform":
                f1 = self.servers_params[i]['params'][0]
                b1_sr += 1.0 / f1

            elif self.servers_params[i]['type'] == "D":
                f1 = self.servers_params[i]['type']
                b1_sr += 1.0 / f1

            elif self.servers_params[i]['type'] == "C":
                y1 = self.servers_params[i]['params'][0]
                y2 = 1.0 - y1
                mu1 = self.servers_params[i]['params'][1]
                mu2 = self.servers_params[i]['params'][2]

                b1_sr += y2 / mu1 + y1 * (1.0 / mu1 + 1.0 / mu2)
            elif self.servers_params[i]['type'] == "Pa":
                if self.servers_params[i]['params'][0] < 1:
                    return math.inf
                else:
                    a = self.servers_params[i]['params'][0]
                    k = self.servers_params[i]['params'][1]
                    b1_sr += a * k / (a - 1)

        return l_sum * b1_sr / (self.n * self.k)

    def arrival(self, k, moment=None, ts=None):
        """
        Действия по прибытию заявки в СМО.
        k - номер класса прибывшей заявки
        если переданы 2 и 3 параметр - значит СМО входит в состав СеМО
        и k - номер класса внутри СМО. Истинный номер класса - в ts.k
        """
        if moment:
            self.ttek = moment
            self.p[k][self.in_sys[k]] += moment - self.t_old[k]
            new_tsk = ts
            new_tsk.in_node_class_num = k
            new_tsk.arr_time = moment
            # в текущем узле обнуляем время ожидания. Общее время ожидания - ts.wait_network
            new_tsk.wait_time = 0
            new_tsk.is_pr = False
            new_tsk.start_waiting_time = -1
            new_tsk.time_to_end_service = 0

        else:
            self.ttek = self.arrival_time[k]
            self.p[k][self.in_sys[k]] += self.arrival_time[k] - self.t_old[k]
            self.arrival_time[k] = self.ttek + self.sources[k].generate()
            new_tsk = Task(k, self.ttek)

        self.arrived[k] += 1
        self.in_sys[k] += 1
        self.t_old[k] = self.ttek

        # Дальнейшая логика зависит от типа приоритета.
        if self.free_channels == 0:

            if self.prty_type == 'No':

                if not self.buffer:  # не задана длина очередиб т.е бесконечная очередь
                    new_tsk.start_waiting_time = self.ttek
                    self.queue.append(new_tsk)
                else:

                    if len(self.queue) < self.buffer:
                        new_tsk.start_waiting_time = self.ttek
                        self.queue.append(new_tsk)
                    else:
                        self.dropped[k] += 1
                        self.in_sys[k] -= 1

            elif self.prty_type == "PR" or self.prty_type == "RS" or self.prty_type == "RW":

                if self.free_channels == 0:
                    # смотрим, есть ли на обслуживании заявки младшего класса
                    is_found_weekier = False
                    for c in self.servers:
                        if moment:
                            class_on_service = c.tsk_on_service.in_node_class_num
                        else:
                            class_on_service = c.tsk_on_service.k

                        if class_on_service > k:
                            time_to_end = c.time_to_end_service
                            total_time = c.total_time_to_serve

                            dropped_tsk = c.end_service()
                            self.taked[k] += 1

                            if k != self.class_ppnz_started and self.class_ppnz_started != -1 and self.in_sys[
                                    k] == self.n:
                                self.ppnz_moments[self.class_ppnz_started] += 1
                                self.refresh_ppnz_stat(
                                    self.class_ppnz_started, self.ttek - self.start_ppnz)
                                self.start_ppnz = self.ttek
                                self.class_ppnz_started = k

                            dropped_tsk.start_waiting_time = self.ttek
                            dropped_tsk.is_pr = True
                            if self.prty_type == 'PR':
                                dropped_tsk.time_to_end_service = time_to_end - self.ttek
                            elif self.prty_type == "RS":
                                dropped_tsk.time_to_end_service = c.dist[k].generate(
                                )
                            elif self.prty_type == "RW":
                                dropped_tsk.time_to_end_service = total_time

                            is_found_weekier = True
                            if moment:
                                self.queue[dropped_tsk.in_node_class_num].append(
                                    dropped_tsk)
                                c.start_service(
                                    new_tsk, self.ttek, is_network=True)
                            else:
                                self.queue[dropped_tsk.k].append(dropped_tsk)
                                c.start_service(new_tsk, self.ttek)

                            break
                    if not is_found_weekier:
                        if not self.buffer:  # не задана длина очередиб т.е бесконечная очередь
                            new_tsk.start_waiting_time = self.ttek
                            self.queue[k].append(new_tsk)
                        else:
                            total_queue_length = 0
                            for q in self.queue:
                                total_queue_length += len(q)

                            if total_queue_length < self.buffer:
                                new_tsk.start_waiting_time = self.ttek
                                self.queue[k].append(new_tsk)
                            else:
                                self.dropped[k] += 1
                                self.in_sys[k] -= 1

            elif self.prty_type == "NP":
                if self.free_channels == 0:
                    new_tsk.start_waiting_time = self.ttek
                    self.queue[k].append(new_tsk)

        else:  # there are free channels:
            if self.free_channels == self.n and self.is_warm_up_set == True:
                self.taked[k] += 1
                if moment:
                    self.servers[0].start_service(
                        new_tsk, self.ttek, self.warm_up[k], is_network=True)
                else:
                    self.servers[0].start_service(
                        new_tsk, self.ttek, self.warm_up[k])
                self.free_channels -= 1
            else:
                for s in self.servers:
                    if s.is_free:
                        self.taked[k] += 1
                        if moment:
                            s.start_service(new_tsk, self.ttek,
                                            is_network=True)
                        else:
                            s.start_service(new_tsk, self.ttek)
                        self.free_channels -= 1
                        break
            # Проверям, не наступил ли ПНЗ:
            if self.free_channels == 0:
                if self.in_sys[k] == self.n:
                    self.start_ppnz = self.ttek
                    self.class_ppnz_started = k

    def serving(self, c, is_network=False):
        """
        Дейтсвия по поступлению заявки на обслуживание
        с - номер канала
        is_network - является ли СМО частью СеМО
        """
        time_to_end = self.servers[c].time_to_end_service
        end_ts = self.servers[c].end_service()
        if is_network:
            k = end_ts.in_node_class_num
        else:
            k = end_ts.k

        self.p[k][self.in_sys[k]] += time_to_end - self.t_old[k]

        self.ttek = time_to_end
        self.t_old[k] = self.ttek
        self.served[k] += 1
        self.total += 1
        self.free_channels += 1
        self.refresh_v_stat(k, self.ttek - end_ts.arr_time)
        self.refresh_w_stat(k, end_ts.wait_time)
        self.in_sys[k] -= 1

        if len(self.queue[k]) == 0 and self.free_channels == 1:
            if self.in_sys[k] == self.n - 1 and self.class_ppnz_started != -1:
                # Конец ПНЗ
                self.ppnz_moments[k] += 1
                self.refresh_ppnz_stat(k, self.ttek - self.start_ppnz)

        if self.prty_type != "No":

            start_number = 0
            if self.prty_type == "PR" or self.prty_type == "RS" or self.prty_type == "RW":
                # можно просматривать только с текущего номера класса
                start_number = k

            for kk in range(start_number, self.k):
                if len(self.queue[kk]) != 0:

                    que_ts = self.queue[kk].pop(0)

                    if self.free_channels == 1 and kk != end_ts.k:
                        self.start_ppnz = self.ttek
                        self.class_ppnz_started = kk

                    self.taked[kk] += 1
                    que_ts.wait_time += self.ttek - que_ts.start_waiting_time
                    if is_network:
                        que_ts.wait_network += self.ttek - que_ts.start_waiting_time
                        self.servers[c].start_service(
                            que_ts, self.ttek, is_network=True)
                    else:
                        self.servers[c].start_service(que_ts, self.ttek)

                    self.free_channels -= 1
                    break
        else:
            # одна очередь
            if len(self.queue) != 0:

                que_ts = self.queue.pop(0)

                if self.free_channels == 1 and k != end_ts.k:
                    self.start_ppnz[k] = self.ttek
                    self.class_ppnz_started = k

                self.taked[k] += 1
                que_ts.wait_time += self.ttek - que_ts.start_waiting_time
                self.servers[c].start_service(que_ts, self.ttek)
                self.free_channels -= 1
        if is_network:
            return end_ts

    def swop_queue(self, last_class, new_class):
        buf = self.queue[last_class]
        self.queue[last_class] = self.queue[new_class]
        self.queue[new_class] = buf

    def calc_time_to_next_event(self):
        serv_earl = 1e10
        arrival_earlier = 1e10

        for kk in range(self.k):
            if self.arrival_time[kk] < arrival_earlier:
                arrival_earlier = self.arrival_time[kk]

        for c in range(self.n):
            if self.servers[c].time_to_end_service < serv_earl:
                serv_earl = self.servers[c].time_to_end_service

        # Key moment:

        if arrival_earlier < serv_earl:
            self.time_to_next_event = arrival_earlier - self.ttek

        else:
            self.time_to_next_event = serv_earl - self.ttek

    def run_one_step(self):

        num_of_server_earlier = -1
        serv_earl = 1e10

        k_earlier = -1
        arrival_earlier = 1e10

        for kk in range(self.k):
            if self.arrival_time[kk] < arrival_earlier:
                arrival_earlier = self.arrival_time[kk]
                k_earlier = kk

        for c in range(self.n):
            if self.servers[c].time_to_end_service < serv_earl:
                serv_earl = self.servers[c].time_to_end_service
                num_of_server_earlier = c

        # Key moment:

        if arrival_earlier < serv_earl:
            self.arrival(k_earlier)

        else:
            self.serving(num_of_server_earlier)

        if self.is_next_calc:
            self.calc_time_to_next_event()

    def run(self, total_served, is_real_served=True):

        print(Fore.GREEN + '\rStart simulation')
        if is_real_served:

            last_percent = 0

            with tqdm(total=100, unit='jobs') as pbar:
                while sum(self.served) < total_served:
                    self.run_one_step()
                    percent = int(100*(sum(self.served)/total_served))
                    if last_percent != percent:
                        last_percent = percent
                        pbar.update(1)
                        pbar.set_description(Fore.MAGENTA + '\rJob served: ' +
                                             Fore.YELLOW + f'{sum(self.served)}/{total_served}' + Fore.LIGHTGREEN_EX)

        else:
            for i in tqdm(range(total_served)):
                self.run_one_step()

        print(Fore.GREEN + '\rSimulation is finished')
        print(Style.RESET_ALL)

    def refresh_ppnz_stat(self, k, new_a):
        for i in range(3):
            self.ppnz[k][i] = self.ppnz[k][i] * (1.0 - (1.0 / self.ppnz_moments[k])) + \
                math.pow(new_a, i + 1) / self.ppnz_moments[k]

    def refresh_v_stat(self, k, new_a):
        for i in range(3):
            self.v[k][i] = self.v[k][i] * \
                (1.0 - (1.0 / self.served[k])) + \
                math.pow(new_a, i + 1) / self.served[k]

    def refresh_w_stat(self, k, new_a):
        for i in range(3):
            self.w[k][i] = self.w[k][i] * \
                (1.0 - (1.0 / self.served[k])) + \
                math.pow(new_a, i + 1) / self.served[k]

    def get_p(self):
        """
        Возвращает список с вероятностями состояний СМО
        p[k][j] - вероятность того, что в СМО в случайный момент времени будет ровно j заявок k-го класса
        """
        res = []
        for kk in range(self.k):
            res.append([0.0] * len(self.p[kk]))
            for j in range(0, self.num_of_states):
                res[kk][j] = self.p[kk][j] / self.ttek
        return res

    def __str__(self, is_short=False):

        res = "Queueing system "
        is_the_same_source = True
        first_source_type = self.sources_params[0]['type']
        for kk in range(1, self.k):
            if self.sources_params[kk]['type'] != first_source_type:
                is_the_same_source = False
        if is_the_same_source:
            res += first_source_type + "*/"
        else:
            for kk in range(self.k - 1):
                res += self.sources_params[kk]['type'] + ","
            res += self.sources_params[self.k - 1]['type'] + "/"

        is_the_same_serving_type = True
        first_serv_type = self.servers_params[0]['type']
        for kk in range(1, self.k):
            if self.servers_params[kk]['type'] != first_serv_type:
                is_the_same_serving_type = False
        if is_the_same_serving_type:
            res += first_serv_type + "/"
        else:
            for kk in range(self.k - 1):
                res += self.servers_params[kk]['type'] + ","
            res += self.servers_params[self.k - 1]['type'] + "/"

        res += str(self.n)

        if self.buffer != None:
            res += "/" + str(self.buffer)
        if self.prty_type != 'No':
            res += "/" + self.prty_type

        res += "\nLoad: " + "{0:.3f}\n".format(self.calc_load())
        if not is_short:
            res += "Current Time " + "{0:.3f}\n".format(self.ttek)
        for kk in range(self.k):
            res += "\nClass " + str(kk + 1) + "\n"
            if not is_short:
                res += "\tArrival time: " + \
                    "{0:.3f}\n".format(self.arrival_time[kk])
            res += "\tSojourn moments:\n"
            for i in range(3):
                res += "\t{0:8.4g}\t".format(self.v[kk][i])

            res += "\n\tWait moments:\n"
            for i in range(3):
                res += "\t{0:8.4g}\t".format(self.w[kk][i])
            res += "\n"
            if not is_short:
                res += "\tStationary prob:\n\t"
                for i in range(10):
                    res += "{0:8.4g}".format(self.p[kk][i] / self.ttek) + "   "
                res += "\n"
                res += "\tArrived: " + str(self.arrived[kk]) + "\n"
                if self.buffer != None:
                    res += "\tDropped: " + str(self.dropped[kk]) + "\n"
                res += "\tTaken: " + str(self.taked[kk]) + "\n"
                res += "\tServed: " + str(self.served[kk]) + "\n"
                res += "\tIn System:" + str(self.in_sys[kk]) + "\n"
                res += "\tPPNZ moments:\n\t"
                for j in range(3):
                    res += "{0:8.4g}    ".format(self.ppnz[kk][j])
                res += "\n"
        for c in range(self.n):
            if self.servers[c].is_free:
                res += "Server " + str(c + 1) + ": Free\n"
            else:
                res += str(self.servers[c]) + "\n"
                for kk in range(self.k):
                    res += "Queue # " + \
                        str(kk + 1) + " count " + \
                        str(len(self.queue[kk])) + "\n"

        return res


class Task:
    """
    Заявка
    """
    id = 0

    def __init__(self, k, arr_time, is_network=False):
        """
        arr_time: Момент прибытия в СМО
        k - номер класса
        """
        if is_network:
            self.arr_network = arr_time
            self.wait_network = 0
            self.in_node_class_num = -1

        self.arr_time = arr_time
        self.k = k
        self.start_waiting_time = -1

        self.wait_time = 0
        self.time_to_end_service = 0
        Task.id += 1
        self.id = Task.id
        self.is_pr = False

    def __str__(self, tab=''):
        res = "Task #" + str(self.id) + "  Class: " + str(self.k + 1) + "\n"
        res += tab + "\tArrival moment: " + \
            "{0:8.3f}".format(self.arr_time) + "\n"
        if self.time_to_end_service != 0:
            res += "{0:s}\tEnd service moment: {1:.3f}\n".format(
                tab, self.time_to_end_service)
        return res


class Server:
    """
    Канал обслуживания
    """
    id = 0

    def __init__(self, server_params, prty_type, generator):
        """
        server_params - словарь, содержащий ключи
        params - параметры распределения
        types -  тип распределения
        """
        self.dist = []  # для каждого класса заявок - свой тип распределения
        self.generator = generator
        for i in range(len(server_params)):
            dist_type = server_params[i]['type']
            params = server_params[i]['params']
            self.dist.append(create_distribution(
                params, dist_type, self.generator))

        self.time_to_end_service = 1e10
        self.total_time_to_serve = 0
        # Сохранение типа дисциплины обслуживания необходимо для
        # корректного назначения времени обслуживания после прерывания
        self.prty_type = prty_type
        self.is_free = True
        self.class_on_service = None
        self.tsk_on_service = None
        Server.id += 1
        self.id = Server.id

    def start_service(self, ts, ttek, warm_up=None, is_network=False):

        self.tsk_on_service = ts
        if is_network:
            self.class_on_service = ts.in_node_class_num
        else:
            self.class_on_service = ts.k
        if warm_up:
            self.total_time_to_serve = warm_up.generate()
            self.time_to_end_service = ttek + self.total_time_to_serve
            self.tsk_on_service.time_to_end_service = self.time_to_end_service
        else:
            if ts.is_pr:
                self.time_to_end_service = ttek + ts.time_to_end_service
                self.tsk_on_service.time_to_end_service = self.time_to_end_service
            else:
                self.total_time_to_serve = self.dist[self.class_on_service].generate(
                )
                self.time_to_end_service = ttek + self.total_time_to_serve
                self.tsk_on_service.time_to_end_service = self.time_to_end_service
        self.is_free = False

    def end_service(self):
        self.time_to_end_service = 1e10
        self.is_free = True
        ts = self.tsk_on_service
        self.tsk_on_service = None
        self.class_on_service = None
        self.total_time_to_serve = 0
        return ts

    def __str__(self, ttek=0):
        res = "\nServer #" + str(self.id) + "\n"
        if self.is_free:
            res += "\tFree\n"
        else:
            res += "\tServing...\n"
            res += "\tTask on service:\n"
            res += "\t\t" + str(self.tsk_on_service.__str__("\t"))
        return res
