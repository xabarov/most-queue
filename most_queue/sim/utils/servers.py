"""
QS channels or servers
"""
from most_queue.general_utils.conv import get_moments
from most_queue.rand_distribution import Cox_dist, Exp_dist, Gamma, H2_dist, Pareto_dist
from most_queue.sim.utils.distribution_utils import create_distribution
from most_queue.sim.utils.phase import QsPhase
from most_queue.sim.utils.tasks import Task


class Server:
    """
    QS channel
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
                    b = Exp_dist.calc_theory_moments(self.dist.params)
                elif self.dist.type == 'H':
                    b = H2_dist.calc_theory_moments(*self.dist.params)
                elif self.dist.type == 'C':
                    b = Cox_dist.calc_theory_moments(*self.dist.params)
                elif self.dist.type == 'Pa':
                    b = Pareto_dist.calc_theory_moments(*self.dist.params)
                elif self.dist.type == 'Gamma':
                    b = Gamma.calc_theory_moments(*self.dist.params)

                f_summ = get_moments(b, self.delta)
                # variance = f_summ[1] - math.pow(f_summ[0], 2)
                # coev = math.sqrt(variance)/f_summ[0]
                params = Gamma.get_mu_alpha(f_summ)
                self.time_to_end_service = ttek + \
                    Gamma.generate_static(*params)
