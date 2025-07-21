"""
Structs for priority queues
"""

from dataclasses import dataclass


@dataclass
class PriorityResults:
    """
    Results of priority queue calculation
    """

    v: list[list[float]] | None = None  # sojourn time moments for each class
    w: list[list[float]] | None = None  # waiting time moments for each class
    p: list[float] | None = None  # probabilities of states (low priority jobs)

    h: list[list[float]] | None = None  # initial moments of active time
    busy: list[list[float]] | None = None  # initial moments of busy period
    w_with_pr: list[list[float]] | None = None  # initial moments of waiting for service with interruptions

    utilization: float | None = None  # utilization factor
