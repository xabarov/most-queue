"""Advance FCFS reservation with sequential iteration compute.

This specified approximation is not exact two-inflight vLLM execution. See
docs/research/serving-v2-model-audit.md and the vLLM v0.32.0 sources:
https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/core/sched/scheduler.py
https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/engine/core.py
https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/worker/gpu_model_runner.py
"""

from most_queue.sim.prefill_decode import PrefillDecodeSim


class AdvanceReservationSim(PrefillDecodeSim):
    """Reserve free active slots at arrival, and compute at iteration boundaries.

    Running actions remain frozen; reservation may precede first compute.
    Completed unchunked prefill emits the first token, then each decode emits
    one more. Shared statistics retain their definitions: first_decode_mean is
    zero because the first output timestamp equals prefill completion.
    A finite budget requires batch_cap * max(input lengths) <= token_budget.
    This conservative envelope prevents silent prefill chunking, including in
    mixed prefill/decode iterations. None permits an unlimited token budget.
    """

    _reserve_on_arrival = True
    _emit_on_prefill = True

    def _lengths(self, inputs, outputs):
        super()._lengths(inputs, outputs)
        if len(inputs) and self.token_budget is not None and self.batch_cap * max(inputs) > self.token_budget:
            raise ValueError("unchunked envelope requires batch_cap * max(input lengths) <= token_budget")
