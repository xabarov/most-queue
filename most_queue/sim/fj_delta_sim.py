"""
ForkJoin Queue with delta
"""
from most_queue.general_utils.conv import get_self_conv_moments
from most_queue.rand_distribution import Gamma
from most_queue.sim.fj_sim import ForkJoinSim
from most_queue.sim.utils.tasks import ForkJoinTask


class ForkJoinSimDelta(ForkJoinSim):
    """
    Имитационная модель СМО Fork-Join, Split-Join
    """

    def __init__(self, num_of_channels, num_of_parts, delta, is_SJ=False, buffer=None):
        """
        num_of_channels - количество каналов СМО
        buffer - максимальная длина очереди
        """
        ForkJoinSim.__init__(self, num_of_channels,
                             num_of_parts, is_SJ, buffer)
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
                if self.queue.size() + self.k - 1 > self.buffer + self.free_channels:
                    self.dropped += 1
                    is_dropped = True
            else:
                if self.free_channels == 0 and self.queue.size() + self.k - 1 > self.buffer:
                    self.dropped += 1
                    is_dropped = True

        if not is_dropped:
            self.served_subtask_in_task[ForkJoinTask.task_id] = 0
            t = ForkJoinTask(self.k, self.ttek)
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
                    b_delta = get_self_conv_moments(self.delta, i)
                    params_delta = Gamma.get_mu_alpha(b_delta)
                    t.subtasks[i].future_arr_time = self.ttek + \
                        Gamma.generate_static(*params_delta)
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
                self.refresh_v_stat(
                    self.ttek - self.first_subtask_arr_time[end_ts.task_id])
                self.in_sys -= 1

            if self.queue.size() != 0:
                que_ts = self.queue.pop()
                self.servers[c].start_service(que_ts, self.ttek)
                self.free_channels -= 1

        else:
            if self.served_subtask_in_task[end_ts.task_id] == self.k:

                self.served += 1
                self.refresh_v_stat(self.ttek - end_ts.arr_time)
                self.in_sys -= 1

                if self.queue.size() != 0:

                    new_task_id = self.queue.queue[0].task_id

                    brothers = [q for q in range(
                        self.queue.size()) if self.queue.queue[q].task_id == new_task_id]
                    for q in brothers:
                        que_ts = self.queue.queue[q]
                        for serv in self.servers:
                            if serv.is_free:
                                serv.start_service(que_ts, self.ttek)
                                self.free_channels -= 1
                                break
                    self.queue.queue = [
                        q for q in self.queue.queue if q.task_id != new_task_id]
            else:
                if self.queue.size() != 0:
                    brothers = [q for q in range(
                        self.queue.size()) if self.queue.queue[q].task_id == end_ts.task_id]
                    for q in brothers:
                        que_ts = self.queue.queue[q]
                        for serv in self.servers:
                            if serv.is_free:
                                serv.start_service(que_ts, self.ttek)
                                self.free_channels -= 1
                                break
                    self.queue.queue = [
                        q for q in self.queue.queue if q.task_id != end_ts.task_id]

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

        res = "Queueing system " + self.source_types + \
            "/" + self.server_types + "/" + str(self.n)
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
            res += "\nQueue Count " + str(self.queue.size()) + "\n"

        return res
