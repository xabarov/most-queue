"""
Graceful-degradation analysis for prediction-based (learning-augmented) M/G/1
scheduling.

Sweeps the prediction quality of SPJF (Shortest Predicted Job First) and reports
how the mean response time degrades as predictions get noisier, bracketed by the
size-aware optimum (SRPT), the perfect-prediction non-preemptive policy (SJF) and
a blind policy (FB/LAS). This reproduces the central theme of the SIGMETRICS 2025
survey "Queueing, Predictions, and LLMs": a size-based policy fed bad predictions
can perform *worse than a blind* policy — there is no free graceful-degradation
guarantee. The break-even noise level where SPJF loses to blind is reported.
"""

from dataclasses import dataclass, field

from most_queue.theory.srpt.mg1_fb import MG1FbCalc
from most_queue.theory.srpt.mg1_sjf import MG1SjfCalc
from most_queue.theory.srpt.mg1_spjf import MG1SpjfCalc
from most_queue.theory.srpt.mg1_srpt import MG1SrptCalc
from most_queue.theory.srpt.utils.predictor import LognormalNoisePredictor


@dataclass
class DegradationCurve:
    """Mean response time of SPJF vs prediction noise, with reference policies."""

    sigmas: list[float] = field(default_factory=list)
    spjf: list[float] = field(default_factory=list)  # SPJF mean response per sigma
    srpt: float = 0.0  # size-aware preemptive optimum (lower bound)
    sjf: float = 0.0  # perfect-prediction non-preemptive (sigma -> 0)
    blind_fb: float = 0.0  # blind FB/LAS (no size info)
    breakeven_sigma: float | None = None  # noise where SPJF first loses to blind


def prediction_degradation_curve(
    l: float,
    service_params,
    kendall_notation: str = "H",
    sigmas: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
) -> DegradationCurve:
    """
    Compute the SPJF mean-response degradation curve for an M/G/1 queue.

    :param l: arrival rate.
    :param service_params: service-time distribution params (e.g. H2Params).
    :param kendall_notation: distribution code passed to the size-based calculators.
    :param sigmas: log-normal prediction-noise levels (sigma=0 -> perfect = SJF).
    :returns: a DegradationCurve.
    """

    def mean_response(calc):
        calc.set_sources(l)
        calc.set_servers(service_params, kendall_notation)
        return float(calc.run().v[0])

    curve = DegradationCurve(sigmas=list(sigmas))
    curve.srpt = mean_response(MG1SrptCalc())
    curve.sjf = mean_response(MG1SjfCalc())
    curve.blind_fb = mean_response(MG1FbCalc())

    for sigma in sigmas:
        calc = MG1SpjfCalc()
        calc.set_sources(l)
        calc.set_servers(service_params, kendall_notation)
        calc.set_predictor(LognormalNoisePredictor(sigma))
        et = float(calc.run().v[0])
        curve.spjf.append(et)
        if curve.breakeven_sigma is None and et > curve.blind_fb:
            curve.breakeven_sigma = sigma

    return curve
