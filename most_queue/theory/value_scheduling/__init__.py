"""
Value-based scheduling under overload: the exact clairvoyant optimum.

When the offered load exceeds one, deadlines cannot all be met and the question
stops being "how long do jobs wait" and becomes "which jobs are worth running".
Each job carries an importance value, collected in full if it finishes by its
deadline and lost otherwise, and the figure of merit is the cumulative value
banked rather than any waiting time.

This package holds the off-line side: the best total value a scheduler that
knew the whole future could have obtained. The on-line algorithms it exists to
measure are in :mod:`most_queue.sim.value_scheduling`.
"""

from most_queue.theory.value_scheduling.offline import (
    ClairvoyantResult,
    ClairvoyantValueScheduler,
    ValueTask,
)

__all__ = [
    "ClairvoyantResult",
    "ClairvoyantValueScheduler",
    "ValueTask",
]
