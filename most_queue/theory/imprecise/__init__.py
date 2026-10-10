"""
Imprecise computation / controllable processing times.

A task is split into a MANDATORY part, which must run in full, and an OPTIONAL
part, which may be cut short -- the unexecuted remainder is the task's *error*.
Quality of the answer therefore becomes a control variable alongside the
schedule, rather than a fixed property of the job.

This is the real-time-systems formulation of a trade-off that classical
queueing theory does not express: under overload the system degrades the
ANSWERS rather than dropping jobs or missing deadlines.
"""

from most_queue.theory.imprecise.offline import (
    ImpreciseComputationScheduler,
    ImpreciseScheduleResult,
    ImpreciseTask,
    verify_schedule,
)

__all__ = [
    "ImpreciseComputationScheduler",
    "ImpreciseScheduleResult",
    "ImpreciseTask",
    "verify_schedule",
]
