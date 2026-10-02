"""Select a service model using positive queue summaries from validation only.

The mean absolute log error is an explicitly chosen downstream objective,
not a proper distributional scoring rule. Callers own temporal separation,
cohort comparability and aggregation over Monte Carlo replications.
"""

from collections.abc import Mapping

import numpy as np


def _metrics(values):
    array = np.asarray(values)
    if array.dtype.kind not in "iuf" or array.size == 0 or not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError("queue metrics must be nonempty, finite, positive real values")
    return array.astype(float)


def queue_log_error(predicted, observed):
    """Mean |log(predicted)-log(observed)| with equal weights and exact shapes.

    Pass MC means when comparing expected queue summaries. Zero references are
    rejected, never stabilized with an implicit epsilon. Log differences avoid
    overflow in a ratio of valid finite metrics.
    """
    predicted, observed = _metrics(predicted), _metrics(observed)
    if predicted.shape != observed.shape:
        raise ValueError("predicted and observed shapes must match exactly")
    return float(np.mean(np.abs(np.log(predicted) - np.log(observed))))


def select_queue_model(predictions, observed, *, candidate_order):
    """Score supplied validation summaries; break exact ties in declared order.

    Predictions maps every candidate name to an array with the observed shape.
    This function neither fits a model nor reads held-out outcomes.
    """
    if isinstance(candidate_order, (str, bytes)):
        raise ValueError("candidate_order must be a sequence of unique names")
    order = tuple(candidate_order)
    if not order or any(not isinstance(name, str) or not name for name in order) or len(set(order)) != len(order):
        raise ValueError("candidate_order must contain unique nonempty names")
    if not isinstance(predictions, Mapping) or set(predictions) != set(order):
        raise ValueError("predictions must contain exactly the declared candidates")
    losses = {name: queue_log_error(predictions[name], observed) for name in order}
    return {"selected": min(order, key=losses.get), "losses": losses, "candidate_order": list(order)}
