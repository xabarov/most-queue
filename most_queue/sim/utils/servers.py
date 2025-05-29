"""
QS channels or servers
"""
from colorama import Fore, Style

from most_queue.theory.utils.conv import conv_moments
from most_queue.rand_distribution import (
    CoxDistribution,
    ExpDistribution,
    GammaDistribution,
    H2Distribution,
    ParetoDistribution,
    NormalDistribution,
    ErlangDistribution
)
from most_queue.sim.utils.distribution_utils import create_distribution
from most_queue.sim.utils.phase import QsPhase
from most_queue.sim.utils.tasks import Task


class Server:
    """
    QS channel
    """
    id = 0

    def __init__(self, params, kendall_notation, generator=None):
        """
        params - параметры распределения
        types -  тип распределения
        """
        self.dist = create_distribution(params, kendall_notation, generator)
        self.time_to_end_service = 1e10
        self.is_free = True
        self.tsk_on_service = None
        Server.id += 1
        self.id = Server.id

        self.params_warm = None
        self.types_warm = None
        self.warm_phase = QsPhase("WarmUp", None)

    def set_warm(self, params, kendall_notation, generator=None):
        """
        Set local warmup period distrubution on server
        """

        self.warm_phase.set_dist(create_distribution(params, kendall_notation, generator))

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
        res = f"\n{Fore.GREEN}Server #{self.id}{Style.RESET_ALL}\n"
        if self.is_free:
            res += "\t" + Fore.CYAN + "Free" + Style.RESET_ALL
        else:
            res += "\t" + Fore.YELLOW + "Serving.." + \
                Style.RESET_ALL + f"{self.time_to_end_service:8.3f}\n"
            res += f"\t{Fore.MAGENTA}Task on service:{Style.RESET_ALL}\n\t{self.tsk_on_service}"
        return res


class ServerWarmUp(Server):
    """
    server with warm-up phase
    """
    id = 0

    def __init__(self, params, kendall_notation, delta=None):
        """
        params - server parameters
        types - server distribution types
        delta - warm-up time or dictionary with moments for different classes of tasks
        """
        Server.__init__(self, params, kendall_notation)
        self.delta = delta

    def start_service(self, ts, ttek, is_warm=False):
        """
        Start service of a task
        ts - time of start
        ttek - task service time
        is_warm_up - flag if it's warm-up phase or not
        """

        Server.start_service(self, ts, ttek)
        if is_warm:
            if not isinstance(self.delta, dict):
                self.time_to_end_service = ttek + self.dist.generate() + self.delta
            else:
                b = [0, 0, 0]
                if self.dist.type == 'M':
                    b = ExpDistribution.calc_theory_moments(self.dist.params)
                elif self.dist.type == 'H':
                    b = H2Distribution.calc_theory_moments(self.dist.params)
                elif self.dist.type == 'C':
                    b = CoxDistribution.calc_theory_moments(self.dist.params)
                elif self.dist.type == 'Pa':
                    b = ParetoDistribution.calc_theory_moments(
                        self.dist.params)
                elif self.dist.type == 'E':
                    b = ErlangDistribution.calc_theory_moments(
                        self.dist.params)
                elif self.dist.type == 'Gamma':
                    b = GammaDistribution.calc_theory_moments(
                        self.dist.params)
                elif self.dist.type == 'Normal':
                    b = NormalDistribution.calc_theory_moments(
                        self.dist.params)
                else:
                    raise ValueError("Unknown distribution type")

                f_summ = conv_moments(b, self.delta)
                # variance = f_summ[1] - math.pow(f_summ[0], 2)
                # coev = math.sqrt(variance)/f_summ[0]
                params = GammaDistribution.get_params(f_summ)
                self.time_to_end_service = ttek + \
                    GammaDistribution.generate_static(params)


class ServerPriority:
    """
    Priority server. Server can have different priorities and can be preempted or not.
    """
    id = 0

    def __init__(self, server_params: dict, prty_type, generator):
        """
        :param server_params: dict with keys: 'params' and 'type'   
        :param prty_type: priority type (str) -
            No  - no priorities, FIFO
            PR  - preemptive resume, with resuming interrupted request
            RS  - preemptive repeat with resampling, re-sampling duration for new service
            RW  - preemptive repeat without resampling, repeating service with previous duration
            NP  - non preemptive, relative priority
        :param generator: Generator object for random number generation.
        """
        self.dist = []  # distribution for each class of requests

        self.generator = generator
        for params in server_params:
            dist_type = params['type']
            params = params['params']
            self.dist.append(create_distribution(
                params, dist_type, self.generator))

        self.time_to_end_service = 1e10
        self.total_time_to_serve = 0
        self.prty_type = prty_type
        self.is_free = True
        self.class_on_service = None
        self.tsk_on_service = None
        Server.id += 1
        self.id = Server.id

    def start_service(self, ts, ttek, warm_up=None, is_network=False):
        """
        Start service for task.
        :param ts: Task object.
        :param ttek: Time of the end of the previous task.
        :param warm_up: WarmUp object. If not None, then it is used to generate
        the time of the end of the current task.
        :param is_network: If True, then it means that the task is generated by a network.
        :return: None.
        """

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
        """
        End service of task
        :return: task that was served

        """
        self.time_to_end_service = 1e10
        self.is_free = True
        ts = self.tsk_on_service
        self.tsk_on_service = None
        self.class_on_service = None
        self.total_time_to_serve = 0
        return ts

    def __str__(self):
        """
        Return string representation of server
        :param ttek: current time
        :return: string representation of server

        """
        res = f"\n{Fore.CYAN}Server #{self.id}{Style.RESET_ALL}\n"
        if self.is_free:
            res += f"{Fore.GREEN}\tFree\n{Style.RESET_ALL}"
        else:
            res += f"{Fore.YELLOW}\tServing...\n{Style.RESET_ALL}"
            res += f"\tTask on service:\t{self.tsk_on_service}\n"
        return res
