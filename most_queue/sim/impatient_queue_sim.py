from most_queue.sim.utils.distribution_utils import create_distribution
from most_queue.sim.qs_sim import QueueingSystemSimulator, Task


class ImpatientTask(Task):
    def __init__(self, arr_time, moment_to_leave):
        super().__init__(arr_time)
        self.moment_to_leave = moment_to_leave

    def __str__(self):
        return f'Task # {self.id}\nArrival moment: {self.arr_time:8.3f}\nMoment to leave: {self.moment_to_leave:8.3f}'


class ImpatientQueueSim(QueueingSystemSimulator):
    def __init__(self, num_of_channels, buffer=None, verbose=True):
        super().__init__(num_of_channels, buffer, verbose)

        self.impatience_params = None
        self.impatience_types = None
        
        self.impatience = None
        self.is_set_impatience_params = False

    def set_impatiens(self, params, types):
        """
        Задает тип и параметры распределения интервала нетерпения заявок.
        Вид распределения                   Тип[types]     Параметры [params]
        Экспоненциальное                      'М'             [mu]
        Гиперэкспоненциальное 2-го порядка    'Н'         [y1, mu1, mu2]
        Гамма-распределение                   'Gamma'       [mu, alpha]
        Эрланга                               'E'           [r, mu]
        Кокса 2-го порядка                    'C'         [y1, mu1, mu2]
        Парето                                'Pa'         [alpha, K]
        Детерминированное                      'D'         [b]
        Равномерное                         'Uniform'     [mean, half_interval]
        """
        self.impatience_params = params
        self.impatience_types = types

        self.is_set_impatience_params = True

        self.impatience = create_distribution(params, types, self.generator)

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

        moment_to_leave = self.ttek + self.impatience.generate()

        if self.free_channels == 0:
            if self.buffer == None:  # не задана длина очереди, т.е бесконечная очередь
                new_tsk = ImpatientTask(self.ttek, moment_to_leave)
                new_tsk.start_waiting_time = self.ttek
                self.queue.append(new_tsk)
            else:
                if self.queue.size() < self.buffer:
                    new_tsk = ImpatientTask(self.ttek, moment_to_leave)
                    new_tsk.start_waiting_time = self.ttek
                    self.queue.append(new_tsk)
                else:
                    self.dropped += 1
                    self.in_sys -= 1

        else:  # there are free channels:

            # check if its a warm phase:
            is_warm_start = False
            if self.queue.size() == 0 and self.free_channels == self.n and self.warm_phase.is_set:
                is_warm_start = True

            for s in self.servers:
                if s.is_free:
                    self.taked += 1
                    s.start_service(ImpatientTask(
                        self.ttek, moment_to_leave), self.ttek, is_warm_start)
                    self.free_channels -= 1

                    # Проверям, не наступил ли ПНЗ:
                    if self.free_channels == 0:
                        if self.in_sys == self.n:
                            self.start_ppnz = self.ttek
                    break

    def drop_task(self, num_of_task_in_queue, moment_to_leave_earlier):
        self.ttek = moment_to_leave_earlier
        new_queue = []
        for i, tsk in enumerate(self.queue.queue):
            if i != num_of_task_in_queue:
                new_queue.append(tsk)
            else:
                end_ts = self.queue.queue[i]

        self.queue.queue = new_queue
        self.in_sys -= 1
        self.dropped += 1
        self.served += 1
        self.refresh_v_stat(self.ttek - end_ts.arr_time)
        end_ts.wait_time = self.ttek - end_ts.arr_time
        self.refresh_w_stat(end_ts.wait_time)

    def run_one_step(self):

        num_of_server_earlier = -1
        serv_earl = 1e10

        for c in range(self.n):
            if self.servers[c].time_to_end_service < serv_earl:
                serv_earl = self.servers[c].time_to_end_service
                num_of_server_earlier = c

        num_of_task_earlier = -1
        moment_to_leave_earlier = 1e10
        for i, tsk in enumerate(self.queue.queue):
            if tsk.moment_to_leave < moment_to_leave_earlier:
                moment_to_leave_earlier = tsk.moment_to_leave
                num_of_task_earlier = i

        # Key moment:

        if self.arrival_time < serv_earl:
            if self.arrival_time < moment_to_leave_earlier:
                self.arrival()
            else:
                self.drop_task(num_of_task_earlier, moment_to_leave_earlier)
        else:
            if serv_earl < moment_to_leave_earlier:
                self.serving(num_of_server_earlier)
            else:
                self.drop_task(num_of_task_earlier, moment_to_leave_earlier)

    def __str__(self, is_short=False):
        super().__str__(is_short)
        num_of_task_earlier = -1
        moment_to_leave_earlier = 1e10
        for i, tsk in enumerate(self.queue.queue):
            if tsk.moment_to_leave < moment_to_leave_earlier:
                moment_to_leave_earlier = tsk.moment_to_leave
                num_of_task_earlier = i
        return f'Task {num_of_task_earlier} is earlier to leave at {moment_to_leave_earlier:8.3f}'
