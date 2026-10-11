"""KV-memory stability bounds of Nie, Si and Zhou, arXiv:2605.04595v1.

Sections 3--4 and Appendix A. Constant-duration slots, joint integer lengths,
chunked prefill, growing decode footprints. The upper bound is NECESSARY;
the lower bound is SUFFICIENT for maximal packing with free CPU swapping.
Neither is an exact capacity or a latency formula.
"""

import math
from collections.abc import Mapping
from numbers import Integral, Real

from most_queue.structs import KvCacheStabilityResults
from most_queue.theory.base_queue import BaseQueue


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_real(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'} and finite")
    return value


class KvCacheStabilityCalc(BaseQueue):
    """Resource-area certificates, conditional on the stated scheduler assumptions.

    A finite joint PMF preserves correlation and bounds the largest footprint.
    Input lengths must be exact multiples of chunk_size (no silent rounding).
    Set either policy assumption to False to suppress the stable certificate.
    The large-memory regime is diagnosed by delta, not an invented cutoff.
    """

    def __init__(self, memory_tokens, chunk_size=1, *, maximal_packing=True, zero_cost_swap=True):
        super().__init__(n=1)
        self.memory_tokens = _positive_integer(memory_tokens, "memory_tokens")
        self.chunk_size = _positive_integer(chunk_size, "chunk_size")
        if not isinstance(maximal_packing, bool) or not isinstance(zero_cost_swap, bool):
            raise ValueError("policy assumptions must be bool")
        self.maximal_packing = maximal_packing
        self.zero_cost_swap = zero_cost_swap
        self.l = None
        self.slot_time = None
        self.joint_pmf = None

    def set_sources(self, l):  # pylint: disable=arguments-differ
        """Set requests per time unit; iid slot arrivals need not be Poisson."""
        self.l = _finite_real(l, "arrival rate", allow_zero=True)
        self.is_sources_set = True

    def set_servers(self, joint_pmf, slot_time):  # pylint: disable=arguments-differ
        """Set {(input_tokens, decode_stages): probability} and constant slot time.

        Zero-probability entries do not contribute to the essential supremum.
        Every positive-probability request must fit at its largest stage.
        No decoding token is charged to the last prefill stage.
        """
        slot_time = _finite_real(slot_time, "slot_time")
        if not isinstance(joint_pmf, Mapping) or not joint_pmf:
            raise ValueError("joint_pmf must be a nonempty mapping")
        pmf = {}
        for lengths, probability in joint_pmf.items():
            if not isinstance(lengths, tuple) or len(lengths) != 2:
                raise ValueError("joint PMF keys must be (input_tokens, decode_stages)")
            s, o = (_positive_integer(x, "length") for x in lengths)
            if s % self.chunk_size:
                raise ValueError("input_tokens must be a multiple of chunk_size")
            probability = _finite_real(probability, "probability", allow_zero=True)
            if probability > 0:
                if s + o > self.memory_tokens:
                    raise ValueError("a supported request's peak footprint exceeds memory_tokens")
                pmf[s, o] = probability
        total = math.fsum(pmf.values())
        if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-10):
            raise ValueError("joint probabilities must sum to one")
        # Only correct floating-point summation, never normalize arbitrary weights.
        pmf = {key: p / total for key, p in pmf.items()}
        self.joint_pmf, self.slot_time = pmf, slot_time
        self.is_servers_set = True

    def memory_area(self, input_tokens, decode_stages):
        """Exact integer lifetime KV token-slots, not peak reserved memory."""
        s = _positive_integer(input_tokens, "input_tokens")
        o = _positive_integer(decode_stages, "decode_stages")
        if s % self.chunk_size:
            raise ValueError("input_tokens must be a multiple of chunk_size")
        chunks = s // self.chunk_size
        return self.chunk_size * chunks * (chunks + 1) // 2 + s * o + o * (o + 1) // 2

    def run(self):
        """Return strict stable/overloaded/inconclusive bounds, with no latency PMF."""
        self._check_if_servers_and_sources_set()
        start = self._measure_time()
        try:
            area = math.fsum(p * self.memory_area(s, o) for (s, o), p in self.joint_pmf.items())
            es = math.fsum(p * s for (s, _), p in self.joint_pmf.items())
            eo = math.fsum(p * o for (_, o), p in self.joint_pmf.items())
            cross = math.fsum(p * s * o for (s, o), p in self.joint_pmf.items())
            peak = max(s + o for s, o in self.joint_pmf)
            resource_time = self.slot_time * area
            upper = self.memory_tokens / resource_time
            lower = (self.memory_tokens - peak) / resource_time
            offered = self.l * resource_time
            delta = peak / self.memory_tokens
            covariance = cross - es * eo
            values = [area, es, eo, cross, covariance, resource_time, upper, lower, offered, delta]
            if not all(math.isfinite(x) for x in values) or resource_time <= 0 or upper <= 0:
                raise ValueError("resource-area arithmetic overflowed or underflowed; rescale units")
        except (OverflowError, ZeroDivisionError) as exc:
            raise ValueError("resource-area arithmetic overflowed or underflowed; rescale units") from exc
        assumptions = self.maximal_packing and self.zero_cost_swap
        status = "inconclusive"
        if self.l > upper:
            status = "overloaded"
        elif assumptions and self.l < lower:
            status = "stable_under_assumptions"
        diagnostics = [
            "Constant slot duration; iid arrivals/lengths; infinite queue.",
            "Lower certificate requires maximal next-stage packing and free swapping.",
            "delta quantifies the large-memory regime; no universal small-delta cutoff is claimed.",
        ]
        if peak == self.memory_tokens:
            diagnostics.append("delta=1: the sufficient interval is empty.")
        if not assumptions:
            diagnostics.append("Scheduler assumptions disabled: no stable certificate or sufficient replica count.")
        result = KvCacheStabilityResults(
            status=status,
            mean_memory_area=area,
            peak_tokens=peak,
            delta=delta,
            capacity_upper=upper,
            capacity_lower=lower if assumptions else None,
            utilization=offered / self.memory_tokens,
            expected_drift_upper=offered - (self.memory_tokens - peak) if assumptions else None,
            length_covariance=covariance,
            necessary_replicas=math.ceil(self.l / upper),
            sufficient_replicas=(
                (math.floor(self.l / lower) + 1 if self.l else 0) if assumptions and lower > 0 else None
            ),
            diagnostics=diagnostics,
        )
        self.ro = result.utilization
        self._set_duration(result, start)
        return result
