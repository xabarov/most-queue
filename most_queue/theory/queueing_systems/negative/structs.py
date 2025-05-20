"""
Supporting structures for the queueing system with negative arrivals.
"""
import json
from dataclasses import asdict, dataclass, is_dataclass


@dataclass
class NegativeArrivalsResults:
    """
    Class for storing results related to negative arrivals in a queueing system.
     Contains lists of waiting times (w), sojourn times  (v),
     sojourn times of served jobs (v_served), sojourn times of broken jobs (v_broken),
     and probabilities (p).
    """
    w: list[float]

    v: list[float]
    v_served: list[float]
    v_broken: list[float]

    p: list[float]


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
            json_res = {'service_time_variation_coef': o.service_time_variation_coef,
                        'channels': o.channels, 'utilization_factor': o.utilization_factor,
                        'calc': [asdict(r) for r in o.calc], 'sim': [asdict(r) for r in o.sim]}
            return json_res
        return super().default(o)
