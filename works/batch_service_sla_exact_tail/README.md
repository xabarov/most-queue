# Batch-service exact SLA tail: comparison artifacts

EPIC-066 adds an exact (not moment-fitted, not an upper bound) waiting-time
tail `P(W>D)` for bounded-window batch-service queues (`M/PH^[a,b]/1`, a=1),
and quantitatively compares it against (a) Inoue (2021)'s closed-form upper
bound for mean latency under an unbounded, deterministic batch model, and
(b) the library's own moment-fit SLA approximation.

[Method](../../docs/research/batch-service-sla-exact-tail-2026.md),
[epic](../../docs/epics/EPIC-066-batch-service-sla-exact-tail.md),
[results](../../docs/research/batch-service-sla-exact-tail-results-2026.md).

This is a pure synthetic numerical study: no external data, no randomness.
`comparison.json` is fully determined by the prescribed constants at the top
of `examples/batch_service_sla_exact_tail_comparison.py` (Inoue's own fitted
`alpha`/`tau0` for Tesla V100 ResNet50, a prescribed `lambda`, a prescribed
`b_max` sweep and deadline set) — identical code always reproduces it
byte-for-byte, so no hashing/manifest machinery is needed.

```bash
.venv/bin/python -m examples.batch_service_sla_exact_tail_comparison
```

Takes several minutes (the larger `b_max` cases solve the full phase-type
CTMC; the smaller ones need many matrix-exponential-action calls for their
long "batches ahead" chains). Key findings: a bounded `[1,b_max]` window can
push the true mean wait above Inoue's unbounded upper bound by up to ~10x
(CV=2.0, `b_max=2`); the moment-fit SLA approximation overestimates deep-tail
`P(W>D)` by orders of magnitude once `D` is several times the mean. Neither
finding is a claim about any specific production GPU/LLM system -- see the
results report for the full table and stated limitations.
