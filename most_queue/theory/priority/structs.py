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
    utilization: float | None = None  # utilization factor
