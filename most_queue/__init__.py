"""
Most-Queue: A Python package for simulating and analyzing queueing systems.

This package provides both simulation and numerical calculation methods
for various types of queueing systems and networks.
"""

__version__ = "2.05"

# Import main simulation and theory classes for easy access
from most_queue.sim.base import QsSim
from most_queue.theory.base_queue import BaseQueue

__all__ = [
    "QsSim",
    "BaseQueue",
    "__version__",
]
