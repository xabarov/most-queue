import rand_destribution as rd
import smo_im
import math
from tqdm import tqdm


class SubTask:
    """
    Позадача
    """
    sub_task_id = 0

    def __init__(self, arr_time, task_id):
        self.arr_time = arr_time
        self.task_id = task_id
        self.id = SubTask.sub_task_id
        SubTask.sub_task_id += 1

    def __str__(self):
        res = "\tSubTask #" + str(self.id) + " parent Task #" + str(self.task_id) + "\n"
        res += "\t\tArrival time: " + str(self.arr_time) + "\n"
        return res


class Task:
    """
    Задача, состоящая из subtask_num подзадач
    """
    task_id = 0

    def __init__(self, subtask_num, arr_time):
        self.subtask_num = subtask_num
        self.arr_time = arr_time
        self.subtasks = []
        for i in range(subtask_num):
            self.subtasks.append(SubTask(arr_time, Task.task_id))
        self.id = Task.task_id
        Task.task_id += 1

    def __str__(self):
        res = "\tTask #" + str(self.id) + "\n"
        res += "\t\tArrival time: " + str(self.arr_time) + "\n"
        return res


class SmoFJ(smo_im.SmoIm):
    """
    Имитационная модель СМО Fork-Join, Split-Join
    """

    def __init__(self, num_of_channels, k, is_SJ=False, is_Purge=False):
        """
        num_of_channels - количество каналов СМО
        buffer - максимальная длина очереди
        """
        smo_im.SmoIm.__init__(self, num_of_channels)
        self.k = k
        self.is_SJ = is_SJ
        self.is_Purge = is_Purge
        self.served_subtask_in_task = {}
        self.sub_task_in_sys = 0

        self.queues = []
        for i in range(num_of_channels):
            self.queues.append([])

    def calc_load(self):

        """
        вычисляет коэффициент загрузки СМО
        """

        pass

    def arrival(self):

        """
        Действия по прибытию заявки в СМО.
        """

        self.arrived += 1
        self.p[self.in_sys] += self.arrival_time - self.t_old
        self.ttek = self.arrival_time
        self.t_old = self.ttek
        self.arrival_time = self.ttek + self.source.generate()

        self.served_subtask_in_task[Task.task_id] = 0
        t = Task(self.n, self.ttek)
        self.in_sys += 1
        self.sub_task_in_sys += self.n

        if not self.is_SJ:  # Fork-Join discipline

            for i in range(self.n):
                if self.free_channels == 0:
                    self.queues[i].append(t.subtasks[i])
                else:  # there are free channels:
                    if self.servers[i].is_free:
                        self.servers[i].start_service(t.subtasks[i], self.ttek)
                        self.free_channels -= 1
                    else:
                        self.queues[i].append(t.subtasks[i])

        else:  # Split-Join discipline

            if self.free_channels < self.n:
                for i in range(self.n):
                    self.queue.append(t.subtasks[i])
            else:
                for i in range(self.n):
                    self.servers[i].start_service(t.subtasks[i], self.ttek)
                    self.free_channels -= 1

    def serving(self, c):
        """
        Дейтсвия по поступлению заявки на обслуживание
        с - номер канала
        """
        time_to_end = self.servers[c].time_to_end_service
        self.p[self.in_sys] += time_to_end - self.t_old
        end_ts = self.servers[c].end_service()
        self.ttek = time_to_end
        self.t_old = self.ttek
        self.served_subtask_in_task[end_ts.task_id] += 1
        self.total += 1
        self.free_channels += 1
        self.sub_task_in_sys -= 1

        if not self.is_SJ:

            if self.served_subtask_in_task[end_ts.task_id] == self.k:

                if self.is_Purge:
                    # найти все остальные подзадачи в СМО и выкинуть
                    task_id = end_ts.task_id
                    for i in range(self.n):
                        if self.servers[i].tsk_on_service.task_id == task_id:
                            self.servers[c].end_service()
                    for i in range(self.n):
                        for j in range(len(self.queues[i])):
                            if self.queues[i][j].task_id == task_id:
                                self.queues[i].pop(j)

                self.served += 1
                self.refresh_v_stat(self.ttek - end_ts.arr_time)
                self.in_sys -= 1

            if len(self.queues[c]) != 0:
                que_ts = self.queues[c].pop(0)
                self.servers[c].start_service(que_ts, self.ttek)
                self.free_channels -= 1

        else:
            if self.served_subtask_in_task[end_ts.task_id] == self.n:

                self.served += 1
                self.refresh_v_stat(self.ttek - end_ts.arr_time)
                self.in_sys -= 1

                if len(self.queue) != 0:
                    for i in range(self.n):
                        que_ts = self.queue.pop(0)
                        self.servers[i].start_service(que_ts, self.ttek)
                        self.free_channels -= 1

    def run_one_step(self):

        num_of_server_earlier = -1
        serv_earl = 1e10

        for c in range(self.n):
            if self.servers[c].time_to_end_service < serv_earl:
                serv_earl = self.servers[c].time_to_end_service
                num_of_server_earlier = c

        # Key moment:

        if self.arrival_time < serv_earl:
            self.arrival()
        else:
            self.serving(num_of_server_earlier)

    def run(self, total_served, verbose=True):

        if verbose:
            progress_bar = tqdm(range(total_served))
            for i in progress_bar:
                self.run_one_step()
                progress_bar.update()

            progress_bar.close()
        else:
            for i in range(total_served):
                self.run_one_step()

    def refresh_v_stat(self, new_a):
        for i in range(3):
            self.v[i] = self.v[i] * (1.0 - (1.0 / self.served)) + math.pow(new_a, i + 1) / self.served

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


class SetSmoException(Exception):

    def __str__(self, text):
        return text


if __name__ == '__main__':

    import mg1_calc
    import fj_calc

    n = 3
    l = 1.0
    b1 = 0.37
    coev = 1.5
    params = rd.H2_dist.get_params_by_mean_and_coev(b1, coev)
    b = rd.H2_dist.calc_theory_moments(*params, 4)

    smo = SmoFJ(n, n, True)
    smo.set_sources(l, 'M')
    smo.set_servers(params, 'H')
    smo.run(100000)
    v_im = smo.v

    b_max = fj_calc.getMaxMoments(n, b, 4)
    ro = l * b_max[0]
    v_ch = mg1_calc.get_v(l, b_max)

    print("\n")
    print("-" * 60)
    print("{:^60s}".format('СМО Split-Join'))
    print("-" * 60)
    print("Коэфф вариации времени обслуживания: ", coev)
    print("Коэффициент загрузки: {:4.3f}".format(ro))
    print("Начальные моменты времени пребывания заявок в системе:")
    print("-" * 60)
    print("{0:^15s}|{1:^20s}|{2:^20s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 60)
    for j in range(min(len(v_ch), len(v_im))):
        print("{0:^16d}|{1:^20.5g}|{2:^20.5g}".format(j + 1, v_ch[j], v_im[j]))
    print("-" * 60)

    coev = 0.8
    b1 = 0.5
    params = rd.Erlang_dist.get_params_by_mean_and_coev(b1, coev)
    b = rd.Erlang_dist.calc_theory_moments(*params, 4)

    smo = SmoFJ(n, n, True)
    smo.set_sources(l, 'M')
    smo.set_servers(params, 'E')
    smo.run(100000)
    v_im = smo.v

    b_max = fj_calc.getMaxMoments(n, b, 4)
    ro = l * b_max[0]
    v_ch = mg1_calc.get_v(l, b_max)

    print("\n\nКоэфф вариации времени обслуживания: ", coev)
    print("Коэффициент загрузки: {:4.3f}".format(ro))
    print("Начальные моменты времени пребывания заявок в системе:")
    print("-" * 60)
    print("{0:^15s}|{1:^20s}|{2:^20s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 60)
    for j in range(min(len(v_ch), len(v_im))):
        print("{0:^16d}|{1:^20.5g}|{2:^20.5g}".format(j + 1, v_ch[j], v_im[j]))
    print("-" * 60)
