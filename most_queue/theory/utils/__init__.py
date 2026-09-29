"""
Utility functions for queueing theory calculations.
"""

from most_queue.theory.utils.sla import (
    deadline_violation_prob,
    fit_from_moments,
    mm1_deadline_violation_prob,
    slo_quantile,
)
from most_queue.theory.utils.tail import upper_bound
from most_queue.theory.utils.timing import get_elapsed_time, measure_time, timed_method
from most_queue.theory.utils.validation import (
    validate_integer,
    validate_list_length,
    validate_list_not_empty,
    validate_non_negative,
    validate_positive,
    validate_positive_integer,
)

__all__ = [
    "timed_method",
    "measure_time",
    "get_elapsed_time",
    "validate_positive",
    "validate_non_negative",
    "validate_integer",
    "validate_positive_integer",
    "validate_list_not_empty",
    "validate_list_length",
    "upper_bound",
    "fit_from_moments",
    "deadline_violation_prob",
    "slo_quantile",
    "mm1_deadline_violation_prob",
]
