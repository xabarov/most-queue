"""
Structures for queueing systems.
"""

import json
from dataclasses import asdict, dataclass, is_dataclass


@dataclass
class QueueResults:
    """
    Result of calculation for queueing system.
    """

    v: list[float] | None = None  # sojourn time rawmoments
    w: list[float] | None = None  # waiting time rawmoments
    p: list[float] | None = None  # probabilities of states
    pi: list[float] | None = None  # probabilities of states before arrival
    utilization: float | None = None  # utilization factor

    duration: float = 0.0  # calculation or simulation duration in seconds


@dataclass
class MulticlassResults:
    """
    Results of queue with multiple classes.
    """

    v: list[list[float]] | None = None  # sojourn time moments for each class
    w: list[list[float]] | None = None  # waiting time moments for each class
    p: list[float] | None = None  # probabilities of states (low priority jobs)

    utilization: float | None = None  # utilization factor

    duration: float = 0.0  # calculation or simulation duration in seconds


@dataclass
class PriorityResults(MulticlassResults):
    """
    Results of priority queue calculation
    """

    h: list[list[float]] | None = None  # raw moments of active time
    busy: list[list[float]] | None = None  # raw moments of busy period
    w_with_pr: list[list[float]] | None = None  # raw moments of waiting for service with interruptions


@dataclass
class VacationResults(QueueResults):
    """
    Result of queue with vacations.
    """

    warmup_prob: float = 0
    cold_prob: float = 0
    cold_delay_prob: float = 0
    servers_busy_probs: list[float] | None = None


@dataclass
class NetworkResults:
    """
    Data class to store network results.
    """

    v: list[float] | None = None  # raw moments of sojourn time distribution
    intensities: list[float] | None = None  # intensities of arrivals into nodes
    loads: list[float] | None = None  # nodes utilizations

    duration: float = 0.0  # calculation or simulation duration in seconds

    arrived: int = 0  # number of arrived jobs (for simulation)
    served: int = 0  # number of served jobs (for simulation)


@dataclass
class NetworkResultsPriority:
    """
    Data class to store results for network with priority discipline in nodes.
    """

    v: list[list[float]] | None = None  # raw moments of sojourn time distribution for each class
    intensities: list[list[float]] | None = None  # intensities of arrivals into nodes  for each class
    loads: list[float] | None = None  # nodes utilizations

    duration: float = 0.0  # calculation or simulation duration in seconds

    arrived: int = 0  # number of arrived jobs (for simulation)
    served: int = 0  # number of served jobs (for simulation)


@dataclass
class NegativeArrivalsResults(QueueResults):
    """
    Class for storing results related to negative arrivals in a queueing system.
     Contains lists of waiting times (w), sojourn times  (v),
     sojourn times of served jobs (v_served), sojourn times of broken jobs (v_broken),
     and probabilities (p).
    """

    v_served: list[float] | None = None
    v_broken: list[float] | None = None


@dataclass
class DependsOnChannelsResults:
    """
    Class for storing results that depend on the number of channels.
     Contains calculated and simulated results,
     as well as parameters such as the number of channels,
     utilization factor, and service time variation coefficient.
    """

    calc: list[NegativeArrivalsResults]
    sim: list[NegativeArrivalsResults]
    channels: list[int]
    utilization_factor: float
    service_time_variation_coef: float


@dataclass
class DependsOnUtilizationResults:
    """
    Class for storing results that depend on the utilization factor.
     Contains calculated and simulated results,
     as well as parameters such as the number of channels,
     utilization factor, and service time variation coefficient.
    """

    calc: list[NegativeArrivalsResults]
    sim: list[NegativeArrivalsResults]
    utilization_factor: list[float]
    channels: int
    service_time_variation_coef: float


@dataclass
class DependsOnVariationResults:
    """
    Class for storing results that depend on the service time variation coefficient.
     Contains calculated and simulated results,
     as well as parameters such as the number of channels,
     utilization factor, and service time variation coefficient.
    """

    calc: list[NegativeArrivalsResults]
    sim: list[NegativeArrivalsResults]
    service_time_variation_coef: list[float]
    channels: int
    utilization_factor: float


class DependsOnJSONEncoder(json.JSONEncoder):
    """
    Encoder for dataclasses to JSON.
    """

    def default(self, o):
        if is_dataclass(o):
            json_res = {
                "service_time_variation_coef": o.service_time_variation_coef,
                "channels": o.channels,
                "utilization_factor": o.utilization_factor,
                "calc": [asdict(r) for r in o.calc],
                "sim": [asdict(r) for r in o.sim],
            }
            return json_res
        return super().default(o)
