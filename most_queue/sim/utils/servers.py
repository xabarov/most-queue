"""
QS channels or servers
"""
from sim.utils.distribution_utils import create_distribution
from sim.utils.phase import QsPhase

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