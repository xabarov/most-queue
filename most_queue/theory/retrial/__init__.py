"""
Retrial queues: blocked jobs join an orbit and retry after random delays.
"""

from most_queue.theory.retrial.mg1 import MG1RetrialCalc
from most_queue.theory.retrial.mm1 import MM1RetrialCalc

__all__ = ["MM1RetrialCalc", "MG1RetrialCalc"]
