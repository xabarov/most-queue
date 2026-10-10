"""
Service rate control for jobs whose value decays during service.

A finite batch of jobs is worked through by one server in discrete time. The
job at the head of the line loses value with every slot it fails to complete,
and is ejected for nothing once that value runs out, so the controller trades
the cost of running fast against the reward it still stands to collect.

Unlike the impatience models elsewhere in this library, the decay runs only
while a job is IN SERVICE, not while it waits -- the distinction that makes
this a control problem over a shrinking batch rather than a stationary queue.
"""

from most_queue.theory.decaying_value.service_rate_control import (
    DecayingValueRateControl,
    DecayingValueResult,
)

__all__ = [
    "DecayingValueRateControl",
    "DecayingValueResult",
]
