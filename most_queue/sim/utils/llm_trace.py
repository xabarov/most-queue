"""Audited BurstGPT length-only replay and joint-mark permutation controls.

Source: https://github.com/HPMLL/BurstGPT (CC BY 4.0). Original elapsed
response times are never interpreted as TTFT. Conversation records are not
eligible: their causal/session structure cannot be reconstructed from v2 CSV.
"""

from collections import Counter
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

# The two explicit controls are part of the permutation contract.
# pylint: disable=too-many-arguments


@dataclass(frozen=True)
class LlmTrace:
    """Successful API cohort with an explicit, mutually exclusive row audit."""

    arrivals: np.ndarray
    marks: np.ndarray
    audit: dict


def read_burstgpt(path, model="ChatGPT"):
    """Read numeric lengths without sorting away invalid chronological order.

    Fail on malformed numeric fields; separately count excluded populations
    and zero-length outcomes. The eligible population is NOT all offered API
    demand. Original failures have unknown service needs and are not replayed.
    """
    data = pd.read_csv(path)
    columns = ["Timestamp", "Request tokens", "Response tokens", "Total tokens"]
    if not set(columns + ["Model", "Log Type"]).issubset(data.columns):
        raise ValueError("missing BurstGPT columns")
    values = data[columns].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values != np.floor(values)):
        raise ValueError("timestamps and token counts must be finite nonnegative integers")
    if np.any(np.diff(values[:, 0]) < 0):
        raise ValueError("source must be chronological")
    if np.any(values[:, 3] != values[:, 1] + values[:, 2]):
        raise ValueError("total tokens disagree with input plus output")
    counts, selected = Counter(records=len(data)), []
    for i, (name, log_type) in enumerate(zip(data.Model, data["Log Type"])):
        reason = (
            "other_model"
            if name != model
            else (
                "not_api"
                if log_type != "API log"
                else "zero_output" if values[i, 2] == 0 else "zero_input" if values[i, 1] == 0 else "eligible"
            )
        )
        counts[reason] += 1
        if reason == "eligible":
            selected.append(i)
    if not selected:
        raise ValueError("no eligible requests")
    chosen = values[selected]
    return LlmTrace(chosen[:, 0], chosen[:, 1:3].astype(int), dict(counts))


def permute_joint_marks(arrivals, marks, mode, seed, *, window=300, block_size=32):
    """Reorder WHOLE pairs; preserve timestamps and the empirical joint law.

    global changes serial dependence and time/mark alignment; local preserves
    the pair multiset in each fixed clock window; blocks preserves order inside
    chunks (including a short final chunk); shift preserves circular ordering
    but changes alignment. These are permutations, not literal iid sampling.
    """
    times = np.asarray(arrivals, dtype=float)
    pairs = np.asarray(marks)
    if times.ndim != 1 or not times.size or not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0):
        raise ValueError("arrivals must be finite, ordered and nonempty")
    if pairs.shape != (len(times), 2) or pairs.dtype.kind not in "iu" or np.any(pairs <= 0):
        raise ValueError("marks must be positive integer pairs")
    if not np.isfinite(window) or window <= 0:
        raise ValueError("window must be finite and positive")
    if isinstance(block_size, bool) or not isinstance(block_size, Integral) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(times))
    if mode == "global":
        rng.shuffle(indices)
    elif mode == "local":
        labels = np.floor(times / window)
        for label in np.unique(labels):
            positions = np.flatnonzero(labels == label)
            indices[positions] = rng.permutation(positions)
    elif mode == "blocks":
        blocks = [indices[i : i + block_size] for i in range(0, len(times), block_size)]
        indices = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
    elif mode == "shift":
        if len(times) > 1:
            indices = np.roll(indices, int(rng.integers(1, len(times))))
    elif mode != "observed":
        raise ValueError("unknown permutation mode")
    return pairs[indices].copy()
