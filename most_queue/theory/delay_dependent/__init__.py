"""
Queues whose SERVICE rate depends on the delay the customer has already
experienced -- as opposed to on the number in system, the workload, or the
elapsed service time.

Empirically this is the "slowdown"/"speedup" effect: in health care, call
centres and telecommunication networks the time a customer spent waiting is
observed to change how long their service then takes.
"""

from most_queue.theory.delay_dependent.mmc_threshold_service import MMcDelayDependentServiceCalc

__all__ = ["MMcDelayDependentServiceCalc"]
