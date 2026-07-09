"""
Most-Queue: A Python package for simulating and analyzing queueing systems.

This package provides both simulation and numerical calculation methods
for various types of queueing systems and networks.
"""

from importlib.metadata import PackageNotFoundError, version

# Import main simulation and theory classes for easy access
from most_queue.sim.base import QsSim
from most_queue.theory.base_queue import BaseQueue

try:
    # single source of truth for the version: the installed package metadata
    __version__ = version("most-queue")
except PackageNotFoundError:  # running from a source checkout that is not installed
    __version__ = "0.0.0.dev0"

__all__ = [
    "QsSim",
    "BaseQueue",
    "__version__",
]
