"""
Queueing systems modeling continuous (iteration-level) batching, as used by
modern LLM-inference engines (vLLM/SGLang-style): unlike
``most_queue.theory.batch`` (atomic bulk-service -- a paket departs as one
group), here customers join and leave a shared, occupancy-capped pool
independently, with a per-customer completion rate that may depend on the
CURRENT occupancy of that pool (shared-compute/memory-bandwidth slowdown).

The KV-memory stability calculator instead tracks lifetime resource area:
constant-duration slots and variable footprints determine conditional
stability bounds, not occupancy-dependent completion rates or delay moments.
The congestion calculator uses a saturated deterministic stage map and a
finite-input spectrum; it does not certify stochastic open-queue recurrence.
"""

from most_queue.theory.continuous_batching.core_aware_ttft import CoreAwareTtftCalc
from most_queue.theory.continuous_batching.infinite_stage_aware_ttft import InfiniteStageAwareTtftCalc
from most_queue.theory.continuous_batching.kv_cache_congestion import KvCacheCongestionCalc
from most_queue.theory.continuous_batching.kv_cache_stability import KvCacheStabilityCalc
from most_queue.theory.continuous_batching.occupancy_dependent import OccupancyDependentQueueCalc
from most_queue.theory.continuous_batching.occupancy_dependent_h2 import OccupancyDependentH2QueueCalc
from most_queue.theory.continuous_batching.phase_aware_stability import PhaseAwareStabilityCalc
from most_queue.theory.continuous_batching.phase_aware_ttft import PhaseAwareTtftCalc
from most_queue.theory.continuous_batching.prefill_decode import PrefillDecodeCalc
from most_queue.theory.continuous_batching.reduced_stage_aware_ttft import ReducedStageAwareTtftCalc
from most_queue.theory.continuous_batching.stage_aware_ttft import StageAwareTtftCalc

__all__ = [
    "OccupancyDependentQueueCalc",
    "OccupancyDependentH2QueueCalc",
    "PrefillDecodeCalc",
    "StageAwareTtftCalc",
    "ReducedStageAwareTtftCalc",
    "InfiniteStageAwareTtftCalc",
    "PhaseAwareStabilityCalc",
    "PhaseAwareTtftCalc",
    "CoreAwareTtftCalc",
    "KvCacheStabilityCalc",
    "KvCacheCongestionCalc",
]
