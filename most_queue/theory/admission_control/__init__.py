"""
Queueing systems with deadline-aware admission control: a job is either
admitted (and served FCFS, no reordering) or rejected outright at arrival,
based on whether its own random deadline is still feasible given the
current system state. Distinct from EDF (theory of service order) and from
the SLA layer (theory.utils.sla, a passive post-hoc metric on top of an
already-fixed discipline).
"""

from most_queue.theory.admission_control.mm1_deadline_admission import MM1DeadlineAdmissionControlCalc

__all__ = [
    "MM1DeadlineAdmissionControlCalc",
]
