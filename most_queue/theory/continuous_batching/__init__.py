"""
Queueing systems modeling continuous (iteration-level) batching, as used by
modern LLM-inference engines (vLLM/SGLang-style): unlike
``most_queue.theory.batch`` (atomic bulk-service -- a paket departs as one
group), here customers join and leave a shared, occupancy-capped pool
independently, with a per-customer completion rate that may depend on the
CURRENT occupancy of that pool (shared-compute/memory-bandwidth slowdown).
"""

from most_queue.theory.continuous_batching.occupancy_dependent import OccupancyDependentQueueCalc

__all__ = ["OccupancyDependentQueueCalc"]
