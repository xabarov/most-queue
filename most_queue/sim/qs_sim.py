import numpy as np

import sim.rand_destribution as rd
import math
from tqdm import tqdm
import time
import sys

from colorama import init
from colorama import Fore, Style

init()


class QueueingSystemSimulator:
    """
    Имитационная модель СМО GI/G/n/r и GI/G/n
    """

    def __init__(self, num_of_channels, buffer=None, verbose=True, calc_next_event_time=False, cuda=False):
        """
        num_of_channels - количество каналов СМО
        buffer - максимальная длина очереди
        verbose - вывод комментариев в процессе ИМ
        calc_next_event_time - нужно ли осуществлять расчет времени для след события (используется для визуализации)
        cuda - нужно ли использовать ускорение GPU при генерации заявок. Прирост в скорости небольшой, поэтому
        по умолчанию cuda=False

        Для запуска ИМ необходимо:
        - вызвать конструктор с параметрами
        - задать вх поток с помощью метода set_sorces() экземпляра созданного класса SmoIm
        - задать распределение обслуживания с помощью метода set_servers() экземпляра созданного класса SmoIm
        - запустить ИМ с помощью метода run() экземпляра созданного класса SmoIm,
        которому нужно передать число требуемых к обслуживанию заявок

        """
        self.n = num_of_channels
        self.buffer = buffer
        self.verbose = verbose  # выводить ли текстовые сообщения о работе
        self.cuda = cuda  # использование CUDA при ИМ для генерации ПСЧ

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
        self.is_next_calc = calc_next_event_time

        self.queue = []  # очередь, класс заявок - Task

        self.servers = []  # каналы обслуживания, список с классами Server

        self.source = None
        self.source_params = None
        self.source_types = None
        self.server_params = None
        self.server_types = None

        self.is_set_source_params = False
        self.is_set_server_params = False

        # Warm-UP
        self.is_set_warm = False
        self.is_start_warm = False
        self.end_warm_time = 1e16
        self.warm_prob = 0
        self.warm_start_mom = 0
        self.warm_starts_times = 0
        self.warm_after_cold_starts = 0
        # Cold
        self.is_set_cold = False
        self.is_start_cold = False
        self.end_cold_time = 1e16
        self.cold_start_mom = 0
        self.cold_prob = 0
        self.cold_starts_times = 0
        # Delay
        self.is_set_cold_delay = False
        self.is_start_cold_delay = False
        self.end_cold_delay_time = 1e16
        self.cold_delay_start_mom = 0
        self.cold_delay_prob = 0
        self.cold_delay_starts_times = 0

        self.zero_wait_arrivals_num = 0

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

        if types == "M":
            self.warm_dist = rd.Exp_dist(params, generator=self.generator)
        elif types == "H":
            self.warm_dist = rd.H2_dist(params, generator=self.generator)
        elif types == "E":
            self.warm_dist = rd.Erlang_dist(params, generator=self.generator)
        elif types == "Gamma":
            self.warm_dist = rd.Gamma(params, generator=self.generator)
        elif types == "C":
            self.warm_dist = rd.Cox_dist(params, generator=self.generator)
        elif types == "Pa":
            self.warm_dist = rd.Pareto_dist(params, generator=self.generator)
        elif types == "Unifrorm":
            self.warm_dist = rd.Uniform_dist(params, generator=self.generator)
        elif types == "Norm":
            self.warm_dist = rd.Normal_dist(params, generator=self.generator)
        elif types == "D":
            self.warm_dist = rd.Det_dist(params)
        else:
            raise SetSmoException(
                "Неправильно задан тип распределения времени разогрева. Варианты М, Н, Е, С, Pa, Uniform, Norm, D")

        self.is_set_warm = True

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

        if types == "M":
            self.cold_dist = rd.Exp_dist(params, generator=self.generator)
        elif types == "H":
            self.cold_dist = rd.H2_dist(params, generator=self.generator)
        elif types == "E":
            self.cold_dist = rd.Erlang_dist(params, generator=self.generator)
        elif types == "Gamma":
            self.cold_dist = rd.Gamma(params, generator=self.generator)
        elif types == "C":
            self.cold_dist = rd.Cox_dist(params, generator=self.generator)
        elif types == "Pa":
            self.cold_dist = rd.Pareto_dist(params, generator=self.generator)
        elif types == "Unifrorm":
            self.cold_dist = rd.Uniform_dist(params, generator=self.generator)
        elif types == "Norm":
            self.cold_dist = rd.Normal_dist(params, generator=self.generator)
        elif types == "D":
            self.cold_dist = rd.Det_dist(params)
        else:
            raise SetSmoException(
                "Неправильно задан тип распределения времени охлаждения. Варианты М, Н, Е, С, Pa, Uniform, Norm, D")

        self.is_set_cold = True

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

        if not self.is_set_cold:
            raise SetSmoException(
                "Необходимо сперва задать время охлаждения. Используйте метод set_cold()")

        if types == "M":
            self.cold_delay_dist = rd.Exp_dist(params, generator=self.generator)
        elif types == "H":
            self.cold_delay_dist = rd.H2_dist(params, generator=self.generator)
        elif types == "E":
            self.cold_delay_dist = rd.Erlang_dist(params, generator=self.generator)
        elif types == "Gamma":
            self.cold_delay_dist = rd.Gamma(params, generator=self.generator)
        elif types == "C":
            self.cold_delay_dist = rd.Cox_dist(params, generator=self.generator)
        elif types == "Pa":
            self.cold_delay_dist = rd.Pareto_dist(params, generator=self.generator)
        elif types == "Unifrorm":
            self.cold_delay_dist = rd.Uniform_dist(params, generator=self.generator)
        elif types == "Norm":
            self.cold_delay_dist = rd.Normal_dist(params, generator=self.generator)
        elif types == "D":
            self.cold_delay_dist = rd.Det_dist(params)
        else:
            raise SetSmoException(
                "Неправильно задан тип распределения времени задержки начала охлаждения. Варианты М, Н, Е, С, Pa, Uniform, Norm, D")

        self.is_set_cold_delay = True

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

        if self.source_types == "M":
            self.source = rd.Exp_dist(self.source_params, generator=self.generator)
        elif self.source_types == "H":
            self.source = rd.H2_dist(self.source_params, generator=self.generator)
        elif self.source_types == "E":
            self.source = rd.Erlang_dist(self.source_params, generator=self.generator)
        elif self.source_types == "C":
            self.source = rd.Cox_dist(self.source_params, generator=self.generator)
        elif self.source_types == "Pa":
            self.source = rd.Pareto_dist(self.source_params, generator=self.generator)
        elif self.source_types == "Gamma":
            self.source = rd.Gamma(self.source_params, generator=self.generator)
        elif self.source_types == "Uniform":
            self.source = rd.Uniform_dist(self.source_params, generator=self.generator)
        elif self.source_types == "Norm":
            self.source = rd.Normal_dist(self.source_params, generator=self.generator)
        elif self.source_types == "D":
            self.source = rd.Det_dist(self.source_params)
        else:
            raise SetSmoException(
                "Неправильно задан тип распределения источника. Варианты М, Н, Е, С, Pa, Norm, Uniform")
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

        for i in range(self.n):
            self.servers.append(Server(self.server_params, self.server_types, generator=self.generator))

    def calc_load(self):

        """
        вычисляет коэффициент загрузки СМО
        """

        l = 0
        if self.source_types == "M":
            l = self.source_params
        elif self.source_types == "D":
            l = 1.00 / self.source_params
        elif self.source_types == "Uniform":
            l = 1.00 / self.source_params[0]
        elif self.source_types == "H":
            y1 = self.source_params[0]
            y2 = 1.0 - self.source_params[0]
            mu1 = self.source_params[1]
            mu2 = self.source_params[2]

            f1 = y1 / mu1 + y2 / mu2
            l = 1.0 / f1

        elif self.source_types == "E":
            r = self.source_params[0]
            mu = self.source_params[1]
            l = mu / r

        elif self.source_types == "Gamma":
            mu = self.source_params[0]
            alpha = self.source_params[1]
            l = mu / alpha

        elif self.source_types == "C":
            y1 = self.source_params[0]
            y2 = 1.0 - self.source_params[0]
            mu1 = self.source_params[1]
            mu2 = self.source_params[2]

            f1 = y2 / mu1 + y1 * (1.0 / mu1 + 1.0 / mu2)
            l = 1.0 / f1
        elif self.source_types == "Pa":
            if self.source_params[0] < 1:
                return None
            else:
                a = self.source_params[0]
                k = self.source_params[1]
                f1 = a * k / (a - 1)
                l = 1.0 / f1

        b1 = 0
        if self.server_types == "M":
            mu = self.server_params
            b1 = 1.0 / mu
        elif self.server_types == "D":
            b1 = self.server_params
        elif self.server_types == "Uniform":
            b1 = self.server_params[0]

        elif self.server_types == "H":
            y1 = self.server_params[0]
            y2 = 1.0 - self.server_params[0]
            mu1 = self.server_params[1]
            mu2 = self.server_params[2]

            b1 = y1 / mu1 + y2 / mu2

        elif self.server_types == "Gamma":
            mu = self.server_params[0]
            alpha = self.server_params[1]
            b1 = alpha / mu

        elif self.server_types == "E":
            r = self.server_params[0]
            mu = self.server_params[1]
            b1 = r / mu

        elif self.server_types == "C":
            y1 = self.server_params[0]
            y2 = 1.0 - self.server_params[0]
            mu1 = self.server_params[1]
            mu2 = self.server_params[2]

            b1 = y2 / mu1 + y1 * (1.0 / mu1 + 1.0 / mu2)
        elif self.server_types == "Pa":
            if self.server_params[0] < 1:
                return math.inf
            else:
                a = self.server_params[0]
                k = self.server_params[1]
                b1 = a * k / (a - 1)

        return l * b1 / self.n

    def send_task_to_channel(self, is_warm_start=False):
        # Отправляет заявку в канал обслуживания
        # is_warm_start- нужен ли разогрев
        for s in self.servers:
            if s.is_free:
                tsk = Task(self.ttek)
                tsk.wai_time = 0
                self.taked += 1
                self.refresh_w_stat(tsk.wai_time)
                self.zero_wait_arrivals_num += 1

                s.start_service(tsk, self.ttek, is_warm_start)
                self.free_channels -= 1

                # Проверям, не наступил ли ПНЗ:
                if self.free_channels == 0:
                    if self.in_sys == self.n:
                        self.start_ppnz = self.ttek

                break

    def send_task_to_queue(self):
        if self.buffer == None:  # не задана длина очереди, т.е бесконечная очередь
            new_tsk = Task(self.ttek)
            new_tsk.start_waiting_time = self.ttek
            self.queue.append(new_tsk)
        else:
            if len(self.queue) < self.buffer:
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
        if self.cuda:
            self.arrival_time = self.ttek + self.source_random_vars[self.tek_source_num]
            self.tek_source_num += 1
            if self.tek_source_num == len(self.source_random_vars):
                self.tek_source_num = 0
        else:
            self.arrival_time = self.ttek + self.source.generate()

        if self.free_channels == 0:
            self.send_task_to_queue()

        else:  # there are free channels:

            if self.is_set_cold:
                if self.is_start_cold:
                    # Еще не закончено охлаждение. В очередь
                    self.send_task_to_queue()
                    return

            if self.is_set_cold_delay:
                if self.is_start_cold_delay:
                    # Заявка пришла раньше окончания времени задержки начала охлаждения
                    self.is_start_cold_delay = False
                    self.end_cold_delay_time = 1e16
                    self.cold_delay_prob += self.ttek - self.cold_delay_start_mom
                    self.send_task_to_channel()
                    return

            if self.is_set_warm:
                # Задан разогрев

                # Проверяем разогрев. К этому моменту система точно не в режиме охлаждения
                # и не в состоянии задержки начала охлаждения.
                # Значит либо:
                # 1. В режиме разогрева -> отправляем заявку в очередь
                # 2. Она пустая и была выклюбчена после охлаждения. Запускаем разогрев
                # 3. Не пустая и разогретая -> тогда оправляем на обслуживание
                if self.is_start_warm:
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
        self.is_start_warm = True
        self.warm_start_mom = self.ttek
        self.warm_starts_times += 1
        self.end_warm_time = self.ttek + self.warm_dist.generate()

    def start_cold(self):
        self.is_start_cold = True
        self.cold_start_mom = self.ttek
        self.cold_starts_times += 1
        self.end_cold_time = self.ttek + self.cold_dist.generate()

    def start_cold_delay(self):
        self.is_start_cold_delay = True
        self.cold_delay_start_mom = self.ttek
        self.cold_delay_starts_times += 1
        self.end_cold_delay_time = self.ttek + self.cold_delay_dist.generate()

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
        if len(self.queue) == 0 and self.free_channels == 1:
            if self.in_sys == self.n - 1:
                # Конец ПНЗ
                self.ppnz_moments += 1
                self.refresh_ppnz_stat(self.ttek - self.start_ppnz)

        # COLD
        if self.is_set_cold:
            if len(self.queue) == 0 and self.free_channels == self.n:
                # Система стала пустой.
                # 1. Если задана задержка начала охлаждения - разыгрываем время ее окончания
                # 2. Если нет - запускаем охлаждение
                if self.is_set_cold_delay:
                    self.start_cold_delay()
                else:
                    self.start_cold()

        if len(self.queue) != 0:
            self.send_head_of_queue_to_channel(c)

    def send_head_of_queue_to_channel(self, channel_num):
        que_ts = self.queue.pop(0)

        if self.free_channels == 1:
            self.start_ppnz = self.ttek

        self.taked += 1
        que_ts.wait_time += self.ttek - que_ts.start_waiting_time
        self.refresh_w_stat(que_ts.wait_time)

        self.servers[channel_num].start_service(que_ts, self.ttek)
        self.free_channels -= 1

    def calc_next_event_time(self):

        serv_earl = 1e16

        for c in range(self.n):
            if self.servers[c].time_to_end_service < serv_earl:
                serv_earl = self.servers[c].time_to_end_service

        if self.arrival_time < serv_earl:
            self.time_to_next_event = self.arrival_time - self.ttek
        else:
            self.time_to_next_event = serv_earl - self.ttek

    def on_end_warming(self):

        self.p[self.in_sys] += self.end_warm_time - self.t_old

        self.ttek = self.end_warm_time
        self.t_old = self.ttek

        self.warm_prob += self.ttek - self.warm_start_mom

        self.is_start_warm = False
        self.end_warm_time = 1e16

        # Отправляем n заявок из очереди в каналы
        for i in range(self.n):
            if len(self.queue) != 0:
                self.send_head_of_queue_to_channel(i)

    def on_end_cold(self):
        self.p[self.in_sys] += self.end_cold_time - self.t_old

        self.ttek = self.end_cold_time
        self.t_old = self.ttek

        self.cold_prob += self.ttek - self.cold_start_mom

        self.is_start_cold = False
        self.end_cold_time = 1e16

        if self.is_set_warm:
            if len(self.queue) != 0:
                # Запускаем разогрев только если в очереди скопились заявки.
                self.warm_after_cold_starts += 1
                self.start_warm()

        else:
            # Отправляем n заявок из очереди в каналы
            for i in range(self.n):
                if len(self.queue) != 0:
                    self.send_head_of_queue_to_channel(i)

    def on_end_cold_delay(self):
        self.p[self.in_sys] += self.end_cold_delay_time - self.t_old

        self.ttek = self.end_cold_delay_time
        self.t_old = self.ttek

        self.cold_delay_prob += self.ttek - self.cold_delay_start_mom

        self.is_start_cold_delay = False
        self.end_cold_delay_time = 1e16

        # Запускаем процесс охлаждения
        self.start_cold()

    def run_one_step(self):

        num_of_server_earlier = -1
        serv_earl = 1e16

        for c in range(self.n):
            if self.servers[c].time_to_end_service < serv_earl:
                serv_earl = self.servers[c].time_to_end_service
                num_of_server_earlier = c

        # Задан глобальный разогрев. Нужно отслеживать в том числе момент окончания разогрева
        times = [serv_earl, self.arrival_time, self.end_warm_time, self.end_cold_time, self.end_cold_delay_time]
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

        if self.is_next_calc:
            self.calc_next_event_time()

    def run(self, total_served, is_real_served=False):
        start = time.process_time()
        if self.cuda:
            from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
            import numpy as np
            from numba import cuda
            threads_per_block = 512
            blocks = int(total_served / 512)
            rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=1)

            # source:
            source_out = np.zeros(threads_per_block * blocks, dtype=np.float32)
            if self.source_types == "M":
                rd.generate_m_jit[blocks, threads_per_block](rng_states, self.source_params, source_out)
            elif self.source_types == "E":
                rd.generate_e_jit[blocks, threads_per_block](rng_states, self.source_params[0], self.source_params[1],
                                                             source_out)
            elif self.source_types == "H":
                rd.generate_h2_jit[blocks, threads_per_block](rng_states, self.source_params[0], self.source_params[1],
                                                              self.source_params[2], source_out)
            else:
                for j in range(len(source_out)):
                    source_out[j] = self.source.generate()

            self.source_random_vars = source_out
            self.tek_source_num = 0

        # while (self.served < total_served):
        #     self.run_one_step()
        #     sys.stderr.write('\rStart simulation. Job served: %d/%d' % (self.served, total_served))
        #     sys.stderr.flush()
        if is_real_served:
            served_old = 0
            while self.served < total_served:
                self.run_one_step()
                if (self.served - served_old) % 5000 == 0:
                    sys.stderr.write('\rStart simulation. Job served: %d/%d' % (self.served, total_served))
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

        return self.warm_prob / self.ttek

    def get_cold_prob(self):

        return self.cold_prob / self.ttek

    def get_cold_delay_prob(self):

        return self.cold_delay_prob / self.ttek

    def refresh_ppnz_stat(self, new_a):

        for i in range(3):
            self.ppnz[i] = self.ppnz[i] * (1.0 - (1.0 / self.ppnz_moments)) + math.pow(new_a, i + 1) / self.ppnz_moments

    def refresh_v_stat(self, new_a):

        for i in range(3):
            self.v[i] = self.v[i] * (1.0 - (1.0 / self.served)) + math.pow(new_a, i + 1) / self.served

    def refresh_w_stat(self, new_a):

        for i in range(3):
            self.w[i] = self.w[i] * (1.0 - (1.0 / self.taked)) + math.pow(new_a, i + 1) / self.taked

        # for i in range(3):
        #     self.w[i] = self.w[i] * (1.0 - (1.0 / self.served)) + math.pow(new_a, i + 1) / self.served

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

        res = "Queueing system " + self.source_types + "/" + self.server_types + "/" + str(self.n)
        if self.buffer != None:
            res += "/" + str(self.buffer)
        res += "\n"
        res += "Load: " + "{0:4.3f}".format(self.calc_load()) + "\n"
        res += "Current Time " + "{0:8.3f}".format(self.ttek) + "\n"
        res += "Arrival Time: " + "{0:8.3f}".format(self.arrival_time) + "\n"

        res += "Sojourn moments:\n"
        for i in range(3):
            res += "\t" + "{0:8.4f}".format(self.v[i])
        res += "\n"

        res += "Wait moments:\n"
        for i in range(3):
            res += "\t" + "{0:8.4f}".format(self.w[i])
        res += "\n"

        if not is_short:
            res += "Stationary prob:\n"
            res += "\t"
            for i in range(10):
                res += "{0:6.5f}".format(self.p[i] / self.ttek) + "   "
            res += "\n"
            res += "Arrived: " + str(self.arrived) + "\n"
            if self.buffer != None:
                res += "Dropped: " + str(self.dropped) + "\n"
            res += "Taken: " + str(self.taked) + "\n"
            res += "Served: " + str(self.served) + "\n"
            res += "In System:" + str(self.in_sys) + "\n"
            res += "PPNZ moments:" + "\n"
            for j in range(3):
                res += "\t{0:8.4f}".format(self.ppnz[j]) + "    "
            res += "\n"
            for c in range(self.n):
                res += str(self.servers[c])
            res += "\nQueue Count " + str(len(self.queue)) + "\n"

        return res


class SetSmoException(Exception):

    def __str__(self, text):
        return text


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
        res = "Task #" + str(self.id) + "\n"
        res += "\tArrival moment: " + "{0:8.3f}".format(self.arr_time)
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
        if types == "M":
            self.dist = rd.Exp_dist(params, generator=generator)
        elif types == "H":
            self.dist = rd.H2_dist(params, generator=generator)
        elif types == "E":
            self.dist = rd.Erlang_dist(params, generator=generator)
        elif types == "C":
            self.dist = rd.Cox_dist(params, generator=generator)
        elif types == "Gamma":
            self.dist = rd.Gamma(params, generator=generator)
        elif types == "Pa":
            self.dist = rd.Pareto_dist(params, generator=generator)
        elif types == "Uniform":
            self.dist = rd.Uniform_dist(params, generator=generator)
        elif types == "Norm":
            self.dist = rd.Normal_dist(params, generator=generator)
        elif types == "D":
            self.dist = rd.Det_dist(params)
        else:
            raise SetSmoException(
                "Неправильно задан тип распределения сервера. Варианты М, Н, Е, С, Pa, Norm, Uniform, D")
        self.time_to_end_service = 1e10
        self.is_free = True
        self.tsk_on_service = None
        Server.id += 1
        self.id = Server.id

        self.params_warm = None
        self.types_warm = None
        self.warm_dist = None

    def set_warm(self, params, types, generator=None):

        if types == "M":
            self.warm_dist = rd.Exp_dist(params, generator=generator)
        elif types == "H":
            self.warm_dist = rd.H2_dist(params, generator=generator)
        elif types == "E":
            self.warm_dist = rd.Erlang_dist(params, generator=generator)
        elif types == "Gamma":
            self.warm_dist = rd.Gamma(params, generator=generator)
        elif types == "C":
            self.warm_dist = rd.Cox_dist(params, generator=generator)
        elif types == "Pa":
            self.warm_dist = rd.Pareto_dist(params, generator=generator)
        elif types == "Unifrorm":
            self.warm_dist = rd.Uniform_dist(params, generator=generator)
        elif types == "Norm":
            self.warm_dist = rd.Normal_dist(params, generator=generator)
        elif types == "D":
            self.warm_dist = rd.Det_dist(params)
        else:
            raise SetSmoException(
                "Неправильно задан тип распределения времени обсл с разогревом. Варианты М, Н, Е, С, Pa, Uniform, Norm, D")

    def start_service(self, ts, ttek, is_warm=False):

        self.tsk_on_service = ts
        self.is_free = False
        if not is_warm:
            self.time_to_end_service = ttek + self.dist.generate()
        else:
            self.time_to_end_service = ttek + self.warm_dist.generate()

    def end_service(self):
        self.time_to_end_service = 1e10
        self.is_free = True
        ts = self.tsk_on_service
        self.tsk_on_service = None
        return ts

    def __str__(self):
        res = "\nServer #" + str(self.id) + "\n"
        if self.is_free:
            res += "\tFree"
        else:
            res += "\tServing.. Time to end " + "{0:8.3f}".format(self.time_to_end_service) + "\n"
            res += "\tTask on service:\n"
            res += "\t" + str(self.tsk_on_service)
        return res



