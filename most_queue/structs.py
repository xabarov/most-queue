"""
Structures for queueing systems.
"""

import json
from dataclasses import asdict, dataclass, is_dataclass


@dataclass
class AoIResults:
    """Age-of-Information results: time-average age and average peak age."""

    avg_aoi: float | None = None  # time-average age of information
    peak_aoi: float | None = None  # average peak age of information (PAoI)
    duration: float = 0.0


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
class ClosedNetworkResults(NetworkResults):
    """
    Results for a closed queueing network (MVA / Buzen convolution).

    v[0] holds the mean cycle time N / X (time for a job to complete one
    full cycle relative to the reference node with visit ratio e = 1).
    """

    throughput: float = 0.0  # X — throughput of the reference node (visit ratio 1)
    mean_jobs: list[float] | None = None  # L_i — mean number of jobs at each node
    v_node: list[float] | None = None  # W_i — mean sojourn time per visit at each node


@dataclass
class NetworkMeansResults(NetworkResults):
    """
    Mean-value results for open networks (Jackson, QNA, G-network).

    Only the mean network sojourn time v[0] is produced (for product-form
    solvers it is exact; higher moments are not available in closed form
    because of overtaking).
    """

    mean_jobs: list[float] | None = None  # L_i — mean number of jobs at each node
    v_node: list[float] | None = None  # mean sojourn time per visit at each node
    negative_intensities: list[float] | None = None  # G-networks: total negative rate per node


@dataclass
class BCMPNetworkResults:
    """
    Results for a multi-class BCMP network (open or closed).
    """

    v: list[list[float]] | None = None  # per class: [mean network sojourn / cycle time]
    intensities: list[list[float]] | None = None  # per class arrival rate at each node
    loads: list[float] | None = None  # total utilization per node
    mean_jobs: list[list[float]] | None = None  # per class mean jobs at each node
    v_node: list[list[float]] | None = None  # per class mean sojourn per visit at each node
    throughput: list[float] | None = None  # per class throughput (closed networks)
    duration: float = 0.0


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
    q: float | None = None


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


@dataclass
class DependsOnNegativeRateResults:
    """
    Class for storing results that depend on the negative arrivals rate delta.
    """

    calc: list[NegativeArrivalsResults]
    sim: list[NegativeArrivalsResults]
    negative_rate: list[float]
    channels: int
    utilization_factor: float
    service_time_variation_coef: float


class DependsOnJSONEncoder(json.JSONEncoder):
    """
    Encoder for dataclasses to JSON.
    """

    def default(self, o):
        if is_dataclass(o):
            json_res = {
                "channels": o.channels,
                "utilization_factor": o.utilization_factor,
                "service_time_variation_coef": getattr(o, "service_time_variation_coef", None),
                "calc": [asdict(r) for r in o.calc],
                "sim": [asdict(r) for r in o.sim],
            }
            if hasattr(o, "service_time_variation_coef") and isinstance(
                getattr(o, "service_time_variation_coef"), list
            ):
                json_res["service_time_variation_coef"] = o.service_time_variation_coef
            if hasattr(o, "negative_rate"):
                json_res["negative_rate"] = o.negative_rate
            return json_res
        # Numpy/complex scalars (e.g. from disaster theory with complex H2 params)
        try:
            import numpy as np

            if isinstance(o, (np.floating, np.integer)):
                return float(o) if isinstance(o, np.floating) else int(o)
            if isinstance(o, (np.complexfloating, complex)):
                return [float(np.real(o)), float(np.imag(o))]
        except (ImportError, TypeError):
            pass
        return super().default(o)
