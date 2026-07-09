"""
Queueing systems with negative customers (RCS and disasters).
"""

from most_queue.theory.negative.mg1_disasters import MG1Disasters
from most_queue.theory.negative.mg1_rcs import MG1NegativeCalcRCS
from most_queue.theory.negative.mgn_disaster import MGnNegativeDisasterCalc
from most_queue.theory.negative.mgn_rcs import MGnNegativeRCSCalc

__all__ = [
    "MG1Disasters",
    "MG1NegativeCalcRCS",
    "MGnNegativeDisasterCalc",
    "MGnNegativeRCSCalc",
]
