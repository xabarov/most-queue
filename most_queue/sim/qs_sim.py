"""
    Имитационная модель СМО GI/G/n/r и GI/G/n
"""
import math
import sys
import time

import numpy as np
from colorama import Fore, Style, init
from tqdm import tqdm

from sim.utils.distribution_utils import calc_qs_load, create_distribution
from sim.utils.exceptions import QsSourseSettingException, QsWrongQueueTypeException
from sim.utils.phase import QsPhase
from sim.utils.qs_queue import QsQueueList, QsQueueDeque

init()


class QueueingSystemSimulator:
    """
    Имитационная модель СМО GI/G/n/r и GI/G/n
    """

    def __init__(self, num_of_channels,
                 buffer=None,
                 verbose=True,
                 buffer_type="list"):
        """
        num_of_channels - количество каналов СМО
        buffer - максимальная длина очереди
        verbose - вывод комментариев в процессе ИМ

        Для запуска ИМ необходимо:
        - вызвать конструктор с параметрами
        - задать вх поток с помощью метода set_sorces() 
        - задать распределение обслуживания с помощью метода set_servers() 
        - запустить ИМ с помощью метода run()
        которому нужно передать число требуемых к обслуживанию заявок

        """
        self.n = num_of_channels
        self.buffer = buffer
        self.verbose = verbose  # выводить ли текстовые сообщения о работе

        self.generator = np.random.default_rng()

        self.free_channels = self.n
        self.num_of_states = 100000
        self.load = 0  # коэффициент загрузки системы

        # для отслеживания длины периода непрерывной занятости каналов:
        self.start_ppnz = 0
        self.ppnz = [0, 0, 0]
        self.ppnz_moments = 0

        self.ttek = 0  # текущее время моделирования
        self.total = 0

        self.w = [0, 0, 0]  # начальные моменты времени ожидания в СМО
        self.v = [0, 0, 0]  # начальные моменты времени пребывания в СМО

        # вероятности состояний СМО (нахождения в ней j заявок):
        self.p = [0.0] * self.num_of_states

        self.taked = 0  # количество заявок, принятых на обслуживание
        self.served = 0  # количество заявок, обслуженных системой
        self.in_sys = 0  # кол-во заявок в системе
        self.t_old = 0  # момент предыдущего события
        self.arrived = 0  # кол-во поступивших заявок
        self.dropped = 0  # кол-во заявок, получивших отказ в обслуживании
        self.arrival_time = 0  # момент прибытия следущей заявки

        self.time_to_next_event = 0

        # очередь, класс заявок - Task
        if buffer_type == "list":
            self.queue = QsQueueList()
        elif buffer_type == "deque":
            self.queue = QsQueueDeque()
        else:
            raise QsWrongQueueTypeException("Неизвестный тип очереди")

        self.servers = []  # каналы обслуживания, список с классами Server

        self.source = None
        self.source_params = None
        self.source_types = None

        self.server_params = None
        self.server_types = None

        self.is_set_source_params = False
        self.is_set_server_params = False

        self.warm_phase = QsPhase("WarmUp")
        self.cold_phase = QsPhase("Cold")
        self.cold_delay_phase = QsPhase("ColdDelay")

        self.warm_after_cold_starts = 0
        self.zero_wait_arrivals_num = 0

        self.time_spent = 0

    def set_warm(self, params, types):
        """
            Задает тип и параметры распределения времени обслуживания с разогревом
            --------------------------------------------------------------------
            Вид распределения                   Тип[types]     Параметры [params]
            Экспоненциальное                      'М'             [mu]
            Гиперэкспоненциальное 2-го порядка    'Н'         [y1, mu1, mu2]
            Эрланга                               'E'           [r, mu]
            Кокса 2-го порядка                    'C'         [y1, mu1, mu2]
            Парето                                'Pa'         [alpha, K]
            Детерминированное                      'D'         [b]
            Равномерное                         'Uniform'     [mean, half_interval]
            Нормальное                            'Norm'    [mean, standard_deviation]
        """
        dist = create_distribution(params, types, self.generator)
        self.warm_phase.set_dist(dist)

    def set_cold(self, params, types):
        """
        Задает тип и параметры распределения времени охлаждения
        --------------------------------------------------------------------
        Вид распределения                   Тип[types]     Параметры [params]
        Экспоненциальное                      'М'             [mu]
        Гиперэкспоненциальное 2-го порядка    'Н'         [y1, mu1, mu2]
        Эрланга                               'E'           [r, mu]
        Кокса 2-го порядка                    'C'         [y1, mu1, mu2]
        Парето                                'Pa'         [alpha, K]
        Детерминированное                      'D'         [b]
        Равномерное                         'Uniform'     [mean, half_interval]
        Нормальное                            'Norm'    [mean, standard_deviation]


        """
        dist = create_distribution(params, types, self.generator)
        self.cold_phase.set_dist(dist)

    def set_cold_delay(self, params, types):
        """
        Задает тип и параметры распределения времени задержки начала охлаждения
        --------------------------------------------------------------------
        Вид распределения                   Тип[types]     Параметры [params]
        Экспоненциальное                      'М'             [mu]
        Гиперэкспоненциальное 2-го порядка    'Н'         [y1, mu1, mu2]
        Эрланга                               'E'           [r, mu]
        Кокса 2-го порядка                    'C'         [y1, mu1, mu2]
        Парето                                'Pa'         [alpha, K]
        Детерминированное                      'D'         [b]
        Равномерное                         'Uniform'     [mean, half_interval]
        Нормальное                            'Norm'    [mean, standard_deviation]


        """

        if not self.cold_phase.is_set:
            raise QsSourseSettingException(
                "Необходимо сперва задать время охлаждения. Используйте метод set_cold()")

        dist = create_distribution(params, types, self.generator)
        self.cold_delay_phase.set_dist(dist)

    def set_sources(self, params, types):
        """
        Задает тип и параметры распределения интервала поступления заявок.
        --------------------------------------------------------------------
        Вид распределения                   Тип[types]     Параметры [params]
        Экспоненциальное                      'М'             [mu]
        Гиперэкспоненциальное 2-го порядка    'Н'         [y1, mu1, mu2]
        Гамма-распределение                   'Gamma'       [mu, alpha]
        Эрланга                               'E'           [r, mu]
        Кокса 2-го порядка                    'C'         [y1, mu1, mu2]
        Парето                                'Pa'         [alpha, K]
        Детерминированное                      'D'         [b]
        Равномерное                         'Uniform'     [mean, half_interval]
        Нормальное                            'Norm'    [mean, standard_deviation]
        """
        self.source_params = params
        self.source_types = types

        self.is_set_source_params = True

        self.source = create_distribution(params, types, self.generator)

        self.arrival_time = self.source.generate()

    def set_servers(self, params, types):
        """
        Задает тип и параметры распределения времени обслуживания.
        Вид распределения                   Тип[types]     Параметры [params]
        Экспоненциальное                      'М'             [mu]
        Гиперэкспоненциальное 2-го порядка    'Н'         [y1, mu1, mu2]
        Гамма-распределение                   'Gamma'       [mu, alpha]
        Эрланга                               'E'           [r, mu]
        Кокса 2-го порядка                    'C'         [y1, mu1, mu2]
        Парето                                'Pa'         [alpha, K]
        Равномерное                         'Uniform'     [mean, half_interval]
        Детерминированное                      'D'         [b]
        Нормальное                            'Norm'    [mean, standard_deviation]
        """
        self.server_params = params
        self.server_types = types

        self.is_set_server_params = True

        self.servers = [Server(self.server_params, self.server_types,
                               generator=self.generator) for i in range(self.n)]

    def calc_load(self):
        """
        Вычисляет коэффициент загрузки СМО
        """

        return calc_qs_load(self.source_types,
                            self.source_params,
                            self.server_types,
                            self.server_params, self.n)

    def send_task_to_channel(self, is_warm_start=False):
        """
        Отправляет заявку в канал обслуживания
        is_warm_start- нужен ли разогрев
        """
        for s in self.servers:
            if s.is_free:
                tsk = Task(self.ttek)
                tsk.wait_time = 0
                self.taked += 1
                self.refresh_w_stat(tsk.wait_time)
                self.zero_wait_arrivals_num += 1

                s.start_service(tsk, self.ttek, is_warm_start)
                self.free_channels -= 1

                # Проверям, не наступил ли ПНЗ:
                if self.free_channels == 0:
                    if self.in_sys == self.n:
                        self.start_ppnz = self.ttek

                break

    def send_task_to_queue(self):
        """
        Send Task to Queue
        """
        if self.buffer is None:  # не задана длина очереди, т.е бесконечная очередь
            new_tsk = Task(self.ttek)
            new_tsk.start_waiting_time = self.ttek
            self.queue.append(new_tsk)
        else:
            if self.queue.size() < self.buffer:
                new_tsk = Task(self.ttek)
                new_tsk.start_waiting_time = self.ttek
                self.queue.append(new_tsk)
            else:
                self.dropped += 1
                self.in_sys -= 1

    def arrival(self):
        """
        Действия по прибытию заявки в СМО.
        """

        self.arrived += 1
        self.p[self.in_sys] += self.arrival_time - self.t_old

        self.in_sys += 1
        self.ttek = self.arrival_time
        self.t_old = self.ttek
        self.arrival_time = self.ttek + self.source.generate()

        if self.free_channels == 0:
            self.send_task_to_queue()

        else:  # there are free channels:

            if self.cold_phase.is_set:
                if self.cold_phase.is_start:
                    # Еще не закончено охлаждение. В очередь
                    self.send_task_to_queue()
                    return

            if self.cold_delay_phase.is_set:
                if self.cold_delay_phase.is_start:
                    # Заявка пришла раньше окончания времени задержки начала охлаждения
                    self.cold_delay_phase.is_start = False
                    self.cold_delay_phase.end_time = 1e16
                    self.cold_delay_phase.prob += self.ttek - self.cold_delay_phase.start_mom
                    self.send_task_to_channel()
                    return

            if self.warm_phase.is_set:
                # Задан разогрев

                # Проверяем разогрев. К этому моменту система точно не в режиме охлаждения
                # и не в состоянии задержки начала охлаждения.
                # Значит либо:
                # 1. В режиме разогрева -> отправляем заявку в очередь
                # 2. Она пустая и была выклюбчена после охлаждения. Запускаем разогрев
                # 3. Не пустая и разогретая -> тогда оправляем на обслуживание
                if self.warm_phase.is_start:
                    # 1. В режиме разогрева -> отправляем заявку в очередь
                    self.send_task_to_queue()
                else:
                    if self.free_channels == self.n:
                        # 2. Она пустая и была выключена после охлаждения. Запускаем разогрев
                        self.start_warm()
                        # Отправляем заявку в очередь
                        self.send_task_to_queue()
                    else:
                        # 3. Не пустая и разогретая -> тогда оправляем на обслуживание
                        self.send_task_to_channel()

            else:
                # Без разогрева. Отправляем заявку в канал обслуживания
                self.send_task_to_channel()

    def start_warm(self):
        """
        Start WarmUp Period
        """
        self.warm_phase.is_start = True
        self.warm_phase.start_mom = self.ttek
        self.warm_phase.starts_times += 1
        self.warm_phase.end_time = self.ttek + self.warm_phase.dist.generate()

    def start_cold(self):
        """
        Start Cold Period
        """
        self.cold_phase.is_start = True
        self.cold_phase.start_mom = self.ttek
        self.cold_phase.starts_times += 1
        self.cold_phase.end_time = self.ttek + self.cold_phase.dist.generate()

    def start_cold_delay(self):
        """
        Start Cold Delay Period
        """
        self.cold_delay_phase.is_start = True
        self.cold_delay_phase.start_mom = self.ttek
        self.cold_delay_phase.starts_times += 1
        self.cold_delay_phase.end_time = self.ttek + \
            self.cold_delay_phase.dist.generate()

    def serving(self, c):
        """
        Дейтсвия по поступлению заявки на обслуживание
        с - номер канала
        """
        time_to_end = self.servers[c].time_to_end_service
        end_ts = self.servers[c].end_service()

        self.p[self.in_sys] += time_to_end - self.t_old

        self.ttek = time_to_end
        self.t_old = self.ttek
        self.served += 1
        self.total += 1
        self.free_channels += 1
        self.refresh_v_stat(self.ttek - end_ts.arr_time)
        # self.refresh_w_stat(end_ts.wait_time)

        self.in_sys -= 1

        # ПНЗ
        if self.queue.size() == 0 and self.free_channels == 1:
            if self.in_sys == self.n - 1:
                # Конец ПНЗ
                self.ppnz_moments += 1
                self.refresh_ppnz_stat(self.ttek - self.start_ppnz)

        # COLD
        if self.cold_phase.is_set:
            if self.queue.size() == 0 and self.free_channels == self.n:
                # Система стала пустой.
                # 1. Если задана задержка начала охлаждения - разыгрываем время ее окончания
                # 2. Если нет - запускаем охлаждение
                if self.cold_delay_phase.is_set:
                    self.start_cold_delay()
                else:
                    self.start_cold()

        if self.queue.size() != 0:
            self.send_head_of_queue_to_channel(c)

    def send_head_of_queue_to_channel(self, channel_num):
        """
        Send first Task (head of queue) to Channel
        """
        que_ts = self.queue.pop()

        if self.free_channels == 1:
            self.start_ppnz = self.ttek

        self.taked += 1
        que_ts.wait_time += self.ttek - que_ts.start_waiting_time
        self.refresh_w_stat(que_ts.wait_time)

        self.servers[channel_num].start_service(que_ts, self.ttek)
        self.free_channels -= 1

    def on_end_warming(self):
        """
        Job that has to be done after WarmUp Period Ends
        """

        self.p[self.in_sys] += self.warm_phase.end_time - self.t_old

        self.ttek = self.warm_phase.end_time
        self.t_old = self.ttek

        self.warm_phase.prob += self.ttek - self.warm_phase.start_mom

        self.warm_phase.is_start = False
        self.warm_phase.end_time = 1e16

        # Отправляем n заявок из очереди в каналы
        for i in range(self.n):
            if self.queue.size() != 0:
                self.send_head_of_queue_to_channel(i)

    def on_end_cold(self):
        """
        Job that has to be done after Cold Period Ends
        """
        self.p[self.in_sys] += self.cold_phase.end_time - self.t_old

        self.ttek = self.cold_phase.end_time
        self.t_old = self.ttek

        self.cold_phase.prob += self.ttek - self.cold_phase.start_mom

        self.cold_phase.is_start = False
        self.cold_phase.end_time = 1e16

        if self.warm_phase.is_set:
            if self.queue.size() != 0:
                # Запускаем разогрев только если в очереди скопились заявки.
                self.warm_after_cold_starts += 1
                self.start_warm()

        else:
            # Отправляем n заявок из очереди в каналы
            for i in range(self.n):
                if self.queue.size() != 0:
                    self.send_head_of_queue_to_channel(i)

    def on_end_cold_delay(self):
        """
        Job that has to be done after Cold Delay Period Ends
        """
        self.p[self.in_sys] += self.cold_delay_phase.end_time - self.t_old

        self.ttek = self.cold_delay_phase.end_time
        self.t_old = self.ttek

        self.cold_delay_phase.prob += self.ttek - self.cold_delay_phase.start_mom

        self.cold_delay_phase.is_start = False
        self.cold_delay_phase.end_time = 1e16

        # Запускаем процесс охлаждения
        self.start_cold()

    def run_one_step(self):
        """
        Run Open step of simulation
        """

        num_of_server_earlier = -1
        serv_earl = 1e16

        for c in range(self.n):
            if self.servers[c].time_to_end_service < serv_earl:
                serv_earl = self.servers[c].time_to_end_service
                num_of_server_earlier = c

        # Задан глобальный разогрев. Нужно отслеживать
        # в том числе момент окончания разогрева
        times = [serv_earl, self.arrival_time, self.warm_phase.end_time,
                 self.cold_phase.end_time, self.cold_delay_phase.end_time]
        min_time_num = np.argmin(times)
        if min_time_num == 0:
            # Обслуживание
            self.serving(num_of_server_earlier)
        elif min_time_num == 1:
            # Прибытие
            self.arrival()
        elif min_time_num == 2:
            # конец разогрева
            self.on_end_warming()
        elif min_time_num == 3:
            # конец охлаждения
            self.on_end_cold()
        else:
            # конец задержки начала охлаждения
            self.on_end_cold_delay()

    def run(self, total_served, is_real_served=False):
        """
        Run simulation process
        """
        start = time.process_time()

        if is_real_served:
            served_old = 0
            while self.served < total_served:
                self.run_one_step()
                if (self.served - served_old) % 5000 == 0:
                    sys.stderr.write(
                        f'\rStart simulation. Job served: {self.served}/{total_served}')
                    sys.stderr.flush()
                served_old = self.served
        else:
            print(Fore.GREEN + '\rStart simulation')
            print(Style.RESET_ALL)

            for i in tqdm(range(total_served)):
                self.run_one_step()

            print(Fore.GREEN + '\rSimulation is finished')
            print(Style.RESET_ALL)

        self.time_spent = time.process_time() - start

    def get_warmup_prob(self):
        """
        Returns probability of the system being in the warm-up phase
        """

        return self.warm_phase.prob / self.ttek

    def get_cold_prob(self):
        """
        Returns probability of the system being in the cold phase
        """

        return self.cold_phase.prob / self.ttek

    def get_cold_delay_prob(self):
        """
        Returns probability of the system being in the delay cold phase
        """

        return self.cold_delay_phase.prob / self.ttek

    def refresh_ppnz_stat(self, new_a):
        """
        Updating statistics of the busy period 
        """

        for i in range(3):
            self.ppnz[i] = self.ppnz[i] * (1.0 - (1.0 / self.ppnz_moments)) + \
                math.pow(new_a, i + 1) / self.ppnz_moments

    def refresh_v_stat(self, new_a):
        """
        Updating statistics of sojourn times
        """

        for i in range(3):
            self.v[i] = self.v[i] * (1.0 - (1.0 / self.served)) + \
                math.pow(new_a, i + 1) / self.served

    def refresh_w_stat(self, new_a):
        """
        Updating statistics of wait times
        """

        for i in range(3):
            self.w[i] = self.w[i] * (1.0 - (1.0 / self.taked)) + \
                math.pow(new_a, i + 1) / self.taked

    def get_p(self):
        """
        Возвращает список с вероятностями состояний СМО
        p[j] - вероятность того, что в СМО в случайный момент времени будет ровно j заявок
        """
        res = [0.0] * len(self.p)
        for j in range(0, self.num_of_states):
            res[j] = self.p[j] / self.ttek
        return res

    def __str__(self, is_short=False):

        res = "Queueing system " + self.source_types + \
            "/" + self.server_types + "/" + str(self.n)
        if self.buffer is not None:
            res += "/" + str(self.buffer)
        res += f"\nLoad: {self.calc_load():4.3f}\n"
        res += f"Current Time {self.ttek:8.3f}\n"
        res += f"Arrival Time: {self.arrival_time:8.3f}\n"

        res += "Sojourn moments:\n"
        for i in range(3):
            res += f"\t{self.v[i]:8.4f}"
        res += "\n"

        res += "Wait moments:\n"
        for i in range(3):
            res += f"\t{self.w[i]:8.4f}"
        res += "\n"

        if not is_short:
            res += "Stationary prob:\n"
            res += "\t"
            for i in range(10):
                res += f"{self.p[i] / self.ttek:6.5f}\t"
            res += "\n"
            res += f"Arrived: {self.arrived}\n"
            if self.buffer is not None:
                res += f"Dropped: {self.dropped}\n"
            res += f"Taken: {self.taked}\n"
            res += f"Served: {self.served}\n"
            res += f"In System: {self.in_sys}\n"
            res += "PPNZ moments:\n"
            for j in range(3):
                res += f"\t{self.ppnz[j]:8.4f}"
            res += "\n"
            for c in range(self.n):
                res += str(self.servers[c])
            res += f"\nQueue Count {self.queue.size()}\n"

        return res


class Task:
    """
    Заявка
    """
    id = 0

    def __init__(self, arr_time):
        """
        :param arr_time: Момент прибытия в СМО
        """
        self.arr_time = arr_time

        self.start_waiting_time = 0

        self.wait_time = 0

        Task.id += 1
        self.id = Task.id

    def __str__(self):
        res = f"Task #{self.id}\n"
        res += f"\tArrival moment: {self.arr_time:8.3f}"
        return res


class Server:
    """
    Канал обслуживания
    """
    id = 0

    def __init__(self, params, types, generator=None):
        """
        params - параметры распределения
        types -  тип распределения
        """
        self.dist = create_distribution(params, types, generator)
        self.time_to_end_service = 1e10
        self.is_free = True
        self.tsk_on_service = None
        Server.id += 1
        self.id = Server.id

        self.params_warm = None
        self.types_warm = None
        self.warm_phase = QsPhase("WarmUp", None)

    def set_warm(self, params, types, generator=None):
        """
        Set local warmup period distrubution on server
        """

        self.warm_phase.set_dist(create_distribution(params, types, generator))

    def start_service(self, ts: Task, ttek, is_warm=False):
        """
        Starts serving
        ttek - current time 
        is_warm - if warmUp needed
        """

        self.tsk_on_service = ts
        self.is_free = False
        if not is_warm:
            self.time_to_end_service = ttek + self.dist.generate()
        else:
            self.time_to_end_service = ttek + self.warm_phase.dist.generate()

    def end_service(self):
        """
        End service
        """
        self.time_to_end_service = 1e10
        self.is_free = True
        ts = self.tsk_on_service
        self.tsk_on_service = None
        return ts

    def __str__(self):
        res = f"\nServer # {self.id}\n"
        if self.is_free:
            res += "\tFree"
        else:
            res += "\tServing.. Time to end " + \
                f"{self.time_to_end_service:8.3f}\n"
            res += f"\tTask on service:\n\t{self.tsk_on_service}"
        return res
