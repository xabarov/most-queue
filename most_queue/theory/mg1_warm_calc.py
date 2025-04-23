"""
Class for calculating M/G/1 queue with warm-up.
"""
from most_queue.rand_distribution import GammaDistribution
from most_queue.theory.utils.diff5dots import diff5dots


class MG1WarmCalc:
    """
    Class for calculating M/G/1 queue with warm-up.
    """

    def __init__(self, l: float, b: list[float], b_warm: list[float]):
        """
        Initialize the MG1WarmCalc class with arrival rate l, 
        service time initial moments b, and warm-up service time moments b_warm.
        Parameters:
        l (float): Arrival rate.
        b (list[float]): Initial moments of service time distribution.
        b_warm (list[float]): Warm-up moments of service time distribution.
        """
        self.l = l
        self.b = b
        self.b_warm = b_warm

    def get_v(self) -> list[float]:
        """
        Calculate sourjourn moments for M/G/1 queue with warm-up.
        """
        tv = self.b_warm[0] / (1 - self.l * self.b[0])
        p0_star = 1 / (1 + self.l * tv)

        b_param = GammaDistribution.get_params(self.b)
        b_warm_param = GammaDistribution.get_params(self.b_warm)

        h = 0.0001
        steps = 5

        v_pls = []

        for c in range(1, steps):
            s = h * c
            chisl = p0_star * \
                ((1 - s / self.l) * GammaDistribution.get_pls(b_warm_param, s) -
                 GammaDistribution.get_pls(b_param, s))
            znam = 1 - s / self.l - GammaDistribution.get_pls(b_param, s)
            v_pls.append(chisl / znam)

        v = diff5dots(v_pls, h)
        v[0] = -v[0]
        v[2] = -v[2]

        return v
