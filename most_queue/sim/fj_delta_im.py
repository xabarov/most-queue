import rand_destribution as rd
from fj_im import SmoFJ as SmoFJ
from fj_im import SubTask as SubTask
from fj_im import Task as Task
from smo_im import Server
from most_queue.theory import sv_sum_calc


class ServerWarmUp(Server):
    """
    Канал обслуживания
    """
    id = 0

    def __init__(self, params, types, delta=None):
        """
        params - параметры распределения
        types -  тип распределения
        """
        Server.__init__(self, params, types)
        self.delta = delta

    def start_service(self, ts, ttek, isWarmUp=False):

        Server.start_service(self, ts, ttek)
        if isWarmUp:
            if not isinstance(self.delta, dict):
                self.time_to_end_service = ttek + self.dist.generate() + self.delta
            else:
                b = [0, 0, 0]
                if self.dist.type == 'M':
                    b = rd.Exp_dist.calc_theory_moments(self.dist.params)
                elif self.dist.type == 'H':
                    b = rd.H2_dist.calc_theory_moments(*self.dist.params)
                elif self.dist.type == 'C':
                    b = rd.Cox_dist.calc_theory_moments(*self.dist.params)
                elif self.dist.type == 'Pa':
                    b = rd.Pareto_dist.calc_theory_moments(*self.dist.params)
                elif self.dist.type == 'Gamma':
                    b = rd.Gamma.calc_theory_moments(*self.dist.params)

                f_summ = sv_sum_calc.get_moments(b, self.delta)
                # variance = f_summ[1] - math.pow(f_summ[0], 2)
                # coev = math.sqrt(variance)/f_summ[0]
                params = rd.Gamma.get_mu_alpha(f_summ)
                self.time_to_end_service = ttek + rd.Gamma.generate_static(*params)


class SubTaskDelta(SubTask):
    """
    Позадача
    """

    def __init__(self, arr_time, task_id):
        SubTask.__init__(self, arr_time, task_id)
        self.future_arr_time = 0


class SmoFJDelta(SmoFJ):
    """
    Имитационная модель СМО Fork-Join, Split-Join
    """

    def __init__(self, num_of_channels, num_of_parts, delta, is_SJ=False, buffer=None):
        """
        num_of_channels - количество каналов СМО
        buffer - максимальная длина очереди
        """
        SmoFJ.__init__(self, num_of_channels, num_of_parts, is_SJ, buffer)
        self.delta = delta
        self.subtask_arr_queue = []
        self.serv_task_id = -1
        self.first_subtask_arr_time = {}

    def task_arrival(self):
        """
            Действия по прибытию заявки в СМО.
        """

        self.arrived += 1
        self.p[self.in_sys] += self.arrival_time - self.t_old
        self.ttek = self.arrival_time
        self.t_old = self.ttek
        self.arrival_time = self.ttek + self.source.generate()
        is_dropped = False

        if self.buffer:  # ограниченная длина очереди
            if not self.is_SJ:
                if len(self.queue) + self.k - 1 > self.buffer + self.free_channels:
                    self.dropped += 1
                    is_dropped = True
            else:
                if self.free_channels == 0 and len(self.queue) + self.k - 1 > self.buffer:
                    self.dropped += 1
                    is_dropped = True

        if not is_dropped:
            self.served_subtask_in_task[Task.task_id] = 0
            t = Task(self.k, self.ttek)
            self.first_subtask_arr_time[t.id] = self.ttek

            self.in_sys += 1
            self.sub_task_in_sys += 1

            if not self.is_SJ:  # Fork-Join discipline

                if self.free_channels == 0:
                    self.queue.append(t.subtasks[0])
                else:  # there are free channels:
                    for s in self.servers:
                        if s.is_free:
                            s.start_service(t.subtasks[0], self.ttek)
                            self.free_channels -= 1
                            break

            else:  # Split-Join discipline

                if self.free_channels < self.k:
                    self.queue.append(t.subtasks[0])
                else:
                    self.servers[0].start_service(t.subtasks[0], self.ttek)
                    self.free_channels -= 1
                    self.serv_task_id = t.id

            for i in range(1, self.k):
                if not isinstance(self.delta, list):
                    t.subtasks[i].future_arr_time = self.ttek + i * self.delta
                else:
                    b_delta = sv_sum_calc.get_self_concolution(self.delta, i)
                    params_delta = rd.Gamma.get_mu_alpha(b_delta)
                    t.subtasks[i].future_arr_time = self.ttek + rd.Gamma.generate_static(*params_delta)
                self.subtask_arr_queue.append(t.subtasks[i])

    def subtask_arrival(self, subtask_num):

        subtsk = self.subtask_arr_queue.pop(subtask_num)
        self.p[self.in_sys] += subtsk.future_arr_time - self.t_old
        self.ttek = subtsk.future_arr_time
        self.t_old = self.ttek

        is_dropped = False

        if self.buffer:  # ограниченная длина очереди
            pass

        if not is_dropped:
            self.sub_task_in_sys += 1

            if not self.is_SJ:  # Fork-Join discipline

                if self.free_channels == 0:
                    self.queue.append(subtsk)
                else:  # there are free channels:
                    for s in self.servers:
                        if s.is_free:
                            s.start_service(subtsk)
                            self.free_channels -= 1
                            break

            else:  # Split-Join discipline

                if self.free_channels != 0 and self.serv_task_id == subtsk.task_id:
                    for free_c in self.servers:
                        if free_c.is_free:
                            free_c.start_service(subtsk, self.ttek)
                            self.free_channels -= 1
                            break
                else:
                    self.queue.append(subtsk)

    def serving(self, c):
        """
        Дейтсвия по поступлению заявки на обслуживание
        с - номер канала
        """
        time_to_end = self.servers[c].time_to_end_service
        self.p[self.in_sys] += time_to_end - self.t_old
        end_ts = self.servers[c].end_service()
        self.serv_task_id = end_ts.task_id
        self.ttek = time_to_end
        self.t_old = self.ttek
        self.served_subtask_in_task[end_ts.task_id] += 1
        self.total += 1
        self.free_channels += 1

        if not self.is_SJ:

            if self.served_subtask_in_task[end_ts.task_id] == self.k:
                self.served += 1
                self.refresh_v_stat(self.ttek - self.first_subtask_arr_time[end_ts.task_id])
                self.in_sys -= 1

            if len(self.queue) != 0:
                que_ts = self.queue.pop(0)
                self.servers[c].start_service(que_ts, self.ttek)
                self.free_channels -= 1

        else:
            if self.served_subtask_in_task[end_ts.task_id] == self.k:

                self.served += 1
                self.refresh_v_stat(self.ttek - end_ts.arr_time)
                self.in_sys -= 1

                if len(self.queue) != 0:

                    new_task_id = self.queue[0].task_id

                    brothers = [q for q in range(len(self.queue)) if self.queue[q].task_id == new_task_id]
                    for q in brothers:
                        que_ts = self.queue[q]
                        for serv in self.servers:
                            if serv.is_free:
                                serv.start_service(que_ts, self.ttek)
                                self.free_channels -= 1
                                break
                    self.queue = [q for q in self.queue if q.task_id != new_task_id]
            else:
                if len(self.queue) != 0:
                    brothers = [q for q in range(len(self.queue)) if self.queue[q].task_id == end_ts.task_id]
                    for q in brothers:
                        que_ts = self.queue[q]
                        for serv in self.servers:
                            if serv.is_free:
                                serv.start_service(que_ts, self.ttek)
                                self.free_channels -= 1
                                break
                    self.queue = [q for q in self.queue if q.task_id != end_ts.task_id]

    def run_one_step(self):

        num_of_server_earlier = -1
        serv_earl = 1e10

        for c in range(self.n):
            if self.servers[c].time_to_end_service < serv_earl:
                serv_earl = self.servers[c].time_to_end_service
                num_of_server_earlier = c

        subtask_arr_time_earl = 1e10
        num_of_subtask_earl = -1

        for c in range(len(self.subtask_arr_queue)):
            if self.subtask_arr_queue[c].future_arr_time < subtask_arr_time_earl:
                subtask_arr_time_earl = self.subtask_arr_queue[c].future_arr_time
                num_of_subtask_earl = c

        is_task_arr = False

        if subtask_arr_time_earl > self.arrival_time:
            is_task_arr = True

        arr_time = min(self.arrival_time, subtask_arr_time_earl)

        if arr_time < serv_earl:
            if is_task_arr:
                self.task_arrival()
            else:
                self.subtask_arrival(num_of_subtask_earl)
        else:
            self.serving(num_of_server_earlier)

    def __str__(self, is_short=False):

        res = "Queueing system " + self.source_types + "/" + self.server_types + "/" + str(self.n)
        if self.buffer != None:
            res += "/" + str(self.buffer)
        if self.is_SJ:
            res += '| Split-Join'
        else:
            res += '| Fork-Join'

        res += "\n"
        # res += "Load: " + "{0:4.3f}".format(self.calc_load()) + "\n"
        res += "Current Time " + "{0:8.3f}".format(self.ttek) + "\n"
        res += "Arrival Time: " + "{0:8.3f}".format(self.arrival_time) + "\n"

        res += "Sojourn moments:\n"
        for i in range(3):
            res += "\t" + "{0:8.4f}".format(self.v[i])
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
            res += "Served: " + str(self.served) + "\n"
            res += "In System:" + str(self.in_sys) + "\n"

            for c in range(self.n):
                res += str(self.servers[c])
            res += "\nQueue Count " + str(len(self.queue)) + "\n"

        return res


if __name__ == '__main__':

    from  most_queue.theory import mg1_calc
    from most_queue.theory import fj_calc
    from most_queue.theory import mg1_warm_calc

    n = 3
    l = 1.0
    b1 = 0.35
    coev = 1.2
    b1_delta = 0.1
    b_params = rd.H2_dist.get_params_by_mean_and_coev(b1, coev)

    delta_params = rd.H2_dist.get_params_by_mean_and_coev(b1_delta, coev)
    b_delta = rd.H2_dist.calc_theory_moments(*delta_params)
    b = rd.H2_dist.calc_theory_moments(*b_params, 4)

    smo = SmoFJDelta(n, n, b_delta, True)
    smo.set_sources(l, 'M')
    smo.set_servers(b_params, 'H')
    smo.run(100000)
    v_im = smo.v

    b_max_warm = fj_calc.getMaxMomentsDelta(n, b, 4, b_delta)
    b_max = fj_calc.getMaxMoments(n, b, 4)
    ro = l * b_max[0]
    v_ch = mg1_warm_calc.get_v(l, b_max, b_max_warm)

    print("\n")
    print("-" * 60)
    print("{:^60s}".format('СМО Split-Join c задержкой начала обслуживания'))
    print("-" * 60)
    print("Коэфф вариации времени обслуживания: ", coev)
    print("Среднее время задежки начала обслуживания: {:4.3f}".format(b1_delta))
    print("Коэфф вариации времени задержки: {:4.3f}".format(coev))
    print("Коэффициент загрузки: {:4.3f}".format(ro))
    print("Начальные моменты времени пребывания заявок в системе:")
    print("-" * 60)
    print("{0:^15s}|{1:^20s}|{2:^20s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 60)
    for j in range(min(len(v_ch), len(v_im))):
        print("{0:^16d}|{1:^20.5g}|{2:^20.5g}".format(j + 1, v_ch[j], v_im[j]))
    print("-" * 60)

    coev = 0.53
    b1 = 0.5

    b1_delta = 0.1
    delta_params = rd.Erlang_dist.get_params_by_mean_and_coev(b1_delta, coev)
    b_delta = rd.Erlang_dist.calc_theory_moments(*delta_params)

    b_params = rd.Erlang_dist.get_params_by_mean_and_coev(b1, coev)
    b = rd.Erlang_dist.calc_theory_moments(*b_params, 4)

    smo = SmoFJDelta(n, n, b_delta, True)
    smo.set_sources(l, 'M')
    smo.set_servers(b_params, 'E')
    smo.run(100000)
    v_im = smo.v

    b_max_warm = fj_calc.getMaxMomentsDelta(n, b, 4, b_delta)
    b_max = fj_calc.getMaxMoments(n, b, 4)
    ro = l * b_max[0]
    v_ch = mg1_warm_calc.get_v(l, b_max, b_max_warm)

    print("\n\nКоэфф вариации времени обслуживания: ", coev)
    print("Коэффициент загрузки: {:4.3f}".format(ro))
    print("Среднее время задежки начала обслуживания: {:4.3f}".format(b1_delta))
    print("Коэфф вариации времени задержки: {:4.3f}".format(coev))
    print("Начальные моменты времени пребывания заявок в системе:")
    print("-" * 60)
    print("{0:^15s}|{1:^20s}|{2:^20s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 60)
    for j in range(min(len(v_ch), len(v_im))):
        print("{0:^16d}|{1:^20.5g}|{2:^20.5g}".format(j + 1, v_ch[j], v_im[j]))
    print("-" * 60)
