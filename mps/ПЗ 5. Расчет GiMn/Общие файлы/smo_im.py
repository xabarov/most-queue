import rand_destribution as rd
import math
from tqdm import tqdm


class SmoIm:
    """
    Имитационная модель СМО GI/G/n/r и GI/G/n
    """

    def __init__(self, num_of_channels, buffer=None, verbose=True):
        """
        num_of_channels - количество каналов СМО
        buffer - максимальная длина очереди
        """
        self.n = num_of_channels
        self.buffer = buffer
        self.verbose = verbose  # выводить ли текстовые сообщения о работе

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

        self.queue = []  # очередь, класс заявок - Task

        self.servers = []  # каналы обслуживания, список с классами Server

        self.source = None
        self.source_params = None
        self.source_types = None
        self.server_params = None
        self.server_types = None

        self.is_set_source_params = False
        self.is_set_server_params = False
        self.is_set_warm = False

    def set_warm(self, params, types):
        """
            Задает тип и параметры распределения времени обслуживания с разогревом
            Вид распределения                   Тип[types]     Параметры [params]
            Экспоненциальное                      'М'             [mu]
            Гиперэкспоненциальное 2-го порядка    'Н'         [y1, mu1, mu2]
            Эрланга                               'E'           [r, mu]
            Кокса 2-го порядка                    'C'         [y1, mu1, mu2]
            Парето                                'Pa'         [alpha, K]
            Детерминированное                      'D'         [b]
            Равномерное                         'Uniform'     [mean, half_interval]
        """
        for i in range(self.n):
            self.servers[i].set_warm(params, types)

    def set_sources(self, params, types):
        """
        Задает тип и параметры распределения интервала поступления заявок.
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
        self.source_params = params
        self.source_types = types

        self.is_set_source_params = True

        if self.source_types == "M":
            self.source = rd.Exp_dist(self.source_params)
        elif self.source_types == "H":
            self.source = rd.H2_dist(self.source_params)
        elif self.source_types == "E":
            self.source = rd.Erlang_dist(self.source_params)
        elif self.source_types == "C":
            self.source = rd.Cox_dist(self.source_params)
        elif self.source_types == "Pa":
            self.source = rd.Pareto_dist(self.source_params)
        elif self.source_types == "Gamma":
            self.source = rd.Gamma(self.source_params)
        elif self.source_types == "Uniform":
            self.source = rd.Uniform_dist(self.source_params)
        elif self.source_types == "D":
            self.source = rd.Det_dist(self.source_params)
        else:
            raise SetSmoException("Неправильно задан тип распределения источника. Варианты М, Н, Е, С, Pa, Uniform")
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
        """
        self.server_params = params
        self.server_types = types

        self.is_set_server_params = True

        for i in range(self.n):
            self.servers.append(Server(self.server_params, self.server_types))

    def calc_load(self):

        """
        вычисляет коэффициент загрузки СМО
        """

        l = 0
        if self.source_types == "M":
            l = self.source_params
        elif self.source_types == "D":
            l = 1.00/self.source_params
        elif self.source_types == "Uniform":
            l = 1.00/self.source_params[0]
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
            b1 = self.source_params
        elif self.server_types == "Uniform":
            b1 = self.source_params[0]

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
            if self.buffer == None:  # не задана длина очередиб т.е бесконечная очередь
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
            if len(self.queue) == 0 and self.free_channels == self.n and self.is_set_warm:
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
        time_to_end = self.servers[c].time_to_end_service
        end_ts = self.servers[c].end_service()
        self.p[self.in_sys] += time_to_end - self.t_old

        self.ttek = time_to_end
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

            que_ts = self.queue.pop(0)

            if self.free_channels == 1:
                self.start_ppnz = self.ttek

            self.taked += 1
            que_ts.wait_time += self.ttek - que_ts.start_waiting_time
            self.servers[c].start_service(que_ts, self.ttek)
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

    def run(self, total_served):
        if self.verbose:
            print("Start simulation. Please wait...")
        for i in tqdm(range(total_served)):
            self.run_one_step()

    def refresh_ppnz_stat(self, new_a):
        for i in range(3):
            self.ppnz[i] = self.ppnz[i] * (1.0 - (1.0 / self.ppnz_moments)) + math.pow(new_a, i + 1) / self.ppnz_moments

    def refresh_v_stat(self, new_a):
        for i in range(3):
            self.v[i] = self.v[i] * (1.0 - (1.0 / self.served)) + math.pow(new_a, i + 1) / self.served

    def refresh_w_stat(self, new_a):
        for i in range(3):
            self.w[i] = self.w[i] * (1.0 - (1.0 / self.taked)) + math.pow(new_a, i + 1) / self.taked

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

        self.start_waiting_time = -1

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

    def __init__(self, params, types):
        """
        params - параметры распределения
        types -  тип распределения
        """
        if types == "M":
            self.dist = rd.Exp_dist(params)
        elif types == "H":
            self.dist = rd.H2_dist(params)
        elif types == "E":
            self.dist = rd.Erlang_dist(params)
        elif types == "C":
            self.dist = rd.Cox_dist(params)
        elif types == "Gamma":
            self.dist = rd.Gamma(params)
        elif types == "Pa":
            self.dist = rd.Pareto_dist(params)
        elif types == "Uniform":
            self.dist = rd.Uniform_dist(params)
        elif types == "D":
            self.dist = rd.Det_dist(params)
        else:
            raise SetSmoException("Неправильно задан тип распределения сервера. Варианты М, Н, Е, С, Pa, Uniform, D")
        self.time_to_end_service = 1e10
        self.is_free = True
        self.tsk_on_service = None
        Server.id += 1
        self.id = Server.id

        self.params_warm = None
        self.types_warm = None
        self.warm_dist = None

    def set_warm(self, params, types):

        if types == "M":
            self.warm_dist = rd.Exp_dist(params)
        elif types == "H":
            self.warm_dist = rd.H2_dist(params)
        elif types == "E":
            self.warm_dist = rd.Erlang_dist(params)
        elif types == "Gamma":
            self.warm_dist = rd.Gamma(params)
        elif types == "C":
            self.warm_dist = rd.Cox_dist(params)
        elif types == "Pa":
            self.warm_dist = rd.Pareto_dist(params)
        elif types == "Unifrorm":
            self.warm_dist = rd.Uniform_dist(params)
        elif types == "D":
            self.warm_dist = rd.Det_dist(params)
        else:
            raise SetSmoException(
                "Неправильно задан тип распределения времени обсл с разогревом. Варианты М, Н, Е, С, Pa, Uniform, D")

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


if __name__ == '__main__':
    import mmnr_calc
    import m_d_n_calc

    n = 3
    l = 1.0
    r = 30
    ro = 0.8
    mu = l / (ro * n)
    smo = SmoIm(n, buffer=r)

    smo.set_sources(l, 'M')
    smo.set_servers(mu, 'M')

    smo.run(100000)

    w = mmnr_calc.M_M_n_formula.get_w(l, mu, n, r)

    w_im = smo.w

    print("\nЗначения начальных моментов времени ожидания заявок в системе:\n")

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 45)
    for j in range(3):
        print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(j + 1, w[j], w_im[j]))
    print("\n\nДанные ИМ::\n")
    print(smo)
    
    smo = SmoIm(n)

    smo.set_sources(l, 'M')
    smo.set_servers(1.0/mu, 'D')

    smo.run(100000)

    mdn = m_d_n_calc.M_D_n(l, 1/mu, n)
    p_ch = mdn.calc_p()
    p_im = smo.get_p()

    print("-" * 36)
    print("{0:^36s}".format("Вероятности состояний СМО M/D/{0:d}".format(n)))
    print("-" * 36)
    print("{0:^4s}|{1:^15s}|{2:^15s}".format("№", "Числ", "ИМ"))
    print("-" * 36)
    for i in range(11):
        print("{0:^4d}|{1:^15.5g}|{2:^15.5g}".format(i, p_ch[i], p_im[i]))
    print("-" * 36)
