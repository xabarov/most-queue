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
                                     like Exp_dist. Defaults to None.
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
        """Set distribution object from random_distrubution like Exp_dist

        Args:
            dist (_type_): _distribution object from random_distrubution like Exp_dist
        """
        self.dist = dist
        self.is_set = True