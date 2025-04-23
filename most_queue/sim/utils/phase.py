"""
Simulation period
"""


class QsPhase:
    """
    Simulation period
    """

    def __init__(self, name: str = "", dist=None):
        """
        Args:
            name (str, optional): Name of simulation phase like Cold, WarmUP. Defaults to "".
            dist (_type_, optional): distribution object from random_distrubution 
                                     like ExpDistribution. Defaults to None.
        """
        self.name = name
        self.is_set = False
        self.is_start = False
        self.end_time = 1e16
        self.prob = 0
        self.start_mom = 0
        self.starts_times = 0
        self.after_cold_starts = 0
        self.dist = dist

    def set_dist(self, dist):
        """Set distribution object from random_distrubution like ExpDistribution

        Args:
            dist (_type_): _distribution object from random_distrubution like ExpDistribution
        """
        self.dist = dist
        self.is_set = True

    def start(self, ttek):
        """
        Start period
        ttek - current system time
        """

        self.is_start = True
        self.start_mom = ttek
        self.starts_times += 1
        self.end_time = ttek + self.dist.generate()

    def end(self, ttek):
        """
        Start period
        ttek - current system time
        """
        self.prob += ttek - self.start_mom

        self.is_start = False
        self.end_time = 1e16

    def get_prob(self, ttek):
        """
        Get probability of be in phase
        """
        return self.prob/ttek
