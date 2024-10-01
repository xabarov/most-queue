from most_queue.sim.utils.distribution_utils import create_distribution
from most_queue.sim.qs_sim import QueueingSystemSimulator, Task


class QueueingFiniteSourceSim(QueueingSystemSimulator):
    """
    Имитационная модель СМО GI/G/n/r и GI/G/n с конечным числом источников заявок
    """

    def __init__(self, num_of_channels, m, buffer=None, verbose=True):
        """
        num_of_channels - количество каналов СМО
        m - число источников заявок, каждый источник заявок имеет одинаковый закон распределения интервалов поступления заявок в систему
        buffer - максимальная длина очереди
        verbose - вывод комментариев в процессе ИМ
        calc_next_event_time - нужно ли осуществлять расчет времени для след события (используется для визуализации)
        cuda - нужно ли использовать ускорение GPU при генерации заявок. Прирост в скорости небольшой, поэтому
        по умолчанию cuda=False

        Для запуска ИМ необходимо:
        - вызвать конструктор с параметрами
        - задать вх поток с помощью метода set_sorces() экземпляра созданного класса QueueingFiniteSourceSim
        - задать распределение обслуживания с помощью метода set_servers() экземпляра QueueingFiniteSourceSim
        - запустить ИМ с помощью метода run() экземпляра созданного класса QueueingFiniteSourceSim,
        которому нужно передать число требуемых к обслуживанию заявок

        """
        self.m = m
        self.sources_left = m  # сколько источников готовы посылать заявки

        super().__init__(num_of_channels, buffer=buffer, verbose=verbose)

        self.arrival_times = []
        self.arrived_num = -1
        self.p = [0.0] * (m+1)

    def set_sources(self, params, types):
        """
        Задает тип и параметры распределения интервала поступления заявок для каждого из источников
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

        self.arrival_times = [self.source.generate() for i in range(self.m)]

    def arrival(self):
        """
        Действия по прибытию заявки в СМО.
        """

        self.arrived += 1
        self.p[self.in_sys] += self.arrival_times[self.arrived_num] - self.t_old
        self.sources_left -= 1
        self.in_sys += 1
        self.ttek = self.arrival_times[self.arrived_num]
        self.t_old = self.ttek
        self.arrival_times[self.arrived_num] = 1e10

        if self.free_channels == 0:
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

        else:  # there are free channels:

            # check if its a warm phase:
            is_warm_start = False
            if len(self.queue) == 0 and self.free_channels == self.n and self.warm_phase.is_set:
                is_warm_start = True

            for s in self.servers:
                if s.is_free:
                    self.taked += 1
                    s.start_service(Task(self.ttek), self.ttek, is_warm_start)
                    self.free_channels -= 1

                    # Проверям, не наступил ли ПНЗ:
                    if self.free_channels == 0:
                        if self.in_sys == self.n:
                            self.start_ppnz = self.ttek
                    break

    def serving(self, c):
        """
        Дейтсвия по поступлению заявки на обслуживание
        с - номер канала
        """
        self.sources_left += 1

        time_to_end = self.servers[c].time_to_end_service
        end_ts = self.servers[c].end_service()

        self.p[self.in_sys] += time_to_end - self.t_old

        self.ttek = time_to_end

        for i, a in enumerate(self.arrival_times):
            if a > 9.9e9:
                self.arrival_times[i] = self.ttek + self.source.generate()
                break

        self.t_old = self.ttek
        self.served += 1
        self.total += 1
        self.free_channels += 1
        self.refresh_v_stat(self.ttek - end_ts.arr_time)
        self.refresh_w_stat(end_ts.wait_time)
        self.in_sys -= 1

        if len(self.queue) == 0 and self.free_channels == 1:
            if self.in_sys == self.n - 1:
                # Конец ПНЗ
                self.ppnz_moments += 1
                self.refresh_ppnz_stat(self.ttek - self.start_ppnz)

        if len(self.queue) != 0:

            que_ts = self.queue.pop()

            if self.free_channels == 1:
                self.start_ppnz = self.ttek

            self.taked += 1
            que_ts.wait_time += self.ttek - que_ts.start_waiting_time
            self.servers[c].start_service(que_ts, self.ttek)
            self.free_channels -= 1

    def calc_next_event_time(self):

        serv_earl = 1e10
        arr_earl = 1e10

        for c in range(self.n):
            if self.servers[c].time_to_end_service < serv_earl:
                serv_earl = self.servers[c].time_to_end_service

        for a in self.arrival_times:
            if a < arr_earl:
                arr_earl = a

        if arr_earl < serv_earl:
            self.time_to_next_event = arr_earl - self.ttek
        else:
            self.time_to_next_event = serv_earl - self.ttek

    def run_one_step(self):

        num_of_server_earlier = -1
        serv_earl = 1e10
        arr_earl = 1e10
        self.arrived_num = -1

        if self.sources_left != 0:
            for i, a in enumerate(self.arrival_times):
                if a < arr_earl:
                    arr_earl = a
                    self.arrived_num = i

        for c in range(self.n):
            if self.servers[c].time_to_end_service < serv_earl:
                serv_earl = self.servers[c].time_to_end_service
                num_of_server_earlier = c

        # Key moment:

        if arr_earl < serv_earl:
            self.arrival()
        else:
            self.serving(num_of_server_earlier)

    def get_p(self):
        """
        Возвращает список с вероятностями состояний СМО
        p[j] - вероятность того, что в СМО в случайный момент времени будет ровно j заявок
        """
        res = [0.0] * len(self.p)
        for j in range(0, self.m+1):
            res[j] = self.p[j] / self.ttek
        return res

    def __str__(self, is_short=False):

        res = "Queueing system " + self.source_types + \
            "/" + self.server_types + "/" + str(self.n)
        if self.buffer != None:
            res += "/" + str(self.buffer)
        res += "\n"
        res += "Load: " + "{0:4.3f}".format(self.calc_load()) + "\n"
        res += "Current Time " + "{0:8.3f}".format(self.ttek) + "\n"
        for i, a in enumerate(self.arrival_times):
            res += f"Arrival Time of Source {i + 1}: {a:8.3f}\n"

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
            res += "\nQueue Count " + str(self.queue.size()) + "\n"

        return res
