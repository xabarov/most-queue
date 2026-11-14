# Fresh validation of advance reservation, v2

Status: locally prepared; **new paid campaign awaits authorization**.
The completed calibration-v1 campaign remains closed. Its data are development
data for the new policy. No GPU is allocated for this preparation.

## Frozen model and question

Does the advance-reservation policy retain its development improvement on a
new A40 run, with no coefficient changes? The original `PrefillDecodeSim` is
the baseline. Both fits use the same 27 original isolated training requests.
[Model audit and development results](serving-v2-model-audit.md).
[Frozen coefficients](../../configs/runpod/reservation-v2-fit.json).

Keep Qwen3-8B revision, BF16, vLLM 0.32.0 image digest, eager execution, B=4,
token budget 18432, no prefix caching and no chunked prefill unchanged. Preserve
the engine's existing async scheduling. A40 only: a replacement GPU would need
a separately calibrated profile. The observational final-output trace wrapper
is unchanged; new per-iteration instrumentation is outside this cycle.

## Requests, separation and provenance

[Workload](../../configs/runpod/reservation-v2-workload.json),
[profile](../../configs/runpod/reservation-v2-profile.json), seed **2026111403**.
All requests are determined locally before allocation. All series drain before
the next series. Fresh requests must never be used to refit this version.

- 21 warmup requests, retained but excluded from timing evaluation.
- 9 isolated validation requests: input 128/1024/4096 by output 32/128/512.
- 108 burst requests in 36 series: the same 9 cells, concurrency 2/4, two repeats.
- Six open-loop 180-second series at rates 0.2/0.5/0.8 requests/s with seeds
  64101/64102. Counts are respectively **33, 28, 91, 91, 142, 167** (552 total).
- **690 requests, 63 series; no new training split.**

Save plan, raw requests/events, server timestamps, runtime, summary, series,
logs and hashes. Audit exact request identities/specifications and counts
against the local deterministic plan. Retain failures and not-offered requests;
an incomplete or failed campaign cannot pass the validation gates by dropping
them. Preserve failed attempts separately. A retry is allowed only for an
identified operational failure, not for an unfavorable accuracy result.

Replay uses exact queue-entry timestamps and actual token lengths; scheduling
and output timestamps are evaluation targets, never predictor inputs. Compare
all per-request arrays in original identity order. Client and server metric
bases remain separate. Save absolute errors, cohort means and quantiles; do
not present this sample as reliable p99 estimation or steady-state validation.

## Decisions fixed before collection

All accuracy targets below apply to **absolute relative error of cohort means**,
except the explicitly normalized queue and absolute SLO-fraction errors.
Each gate is reported individually, including failures; a summary cannot hide
weak small-prompt bursts. These empirical thresholds define this narrow profile,
not a universal precision guarantee.

| Family | Target for v2 |
|---|---|
| Isolated server TTFT | maximum error <=15% |
| Isolated server TPOT | maximum error <=5% |
| Burst server TTFT | median error <=15%, maximum <=35% |
| Burst client TTFT | median error <=15%, maximum <=40% |
| Burst server TPOT | maximum error <=10% |
| Open-loop client and server TTFT | each cohort error <=20% |
| Open-loop server TPOT | each cohort error <=5% |
| Open-loop queue wait | each error/max(observed mean, 50 ms) <=25%; absolute errors also reported |
| Primary joint client SLO: TTFT <=1 s AND TPOT <=35 ms | absolute fraction error <=0.10 in every open-loop cohort |
| Baseline comparison | burst median server-TTFT error and maximum server-TPOT error each <= half of v1 |

Always report all 12 combinations TTFT 0.25/0.5/1/2 s by TPOT 30/35/40 ms,
including false passes and false failures. Only the explicitly designated
1-s/35-ms combination gates validation. Paired evaluation uses client TTFT
and client TPOT, all offers in the denominator, and no IID confidence interval.

If a gate fails, report it and keep the new model experimental with that limit;
do not retune and call the same run held out. A future correction requires
another separately approved fresh campaign. Publication of code must not
imply that failed precision criteria passed.

## Cost and cleanup

Proposed limits: **$5 total, four aggregate GPU-hours, one GPU at a time,
at most two attempts**, compute <=$0.75/h, total <=$0.77/h, no persistent disk
or network volume. [Policy](../../configs/runpod/reservation-v2-policy.json)
remains `authorized=false` until this specific campaign is approved.

Fresh read-only catalog check: A40 Secure is $0.59/h, CUDA >=13.0 stock LOW;
80-GB container disk is budgeted at $0.011112/h. Expected 30–45 minutes costs
about **$0.30–0.45**. The conservative 4500-second work allocation plus a
900-second cleanup reserve budgets **$0.901668 per attempt** at this rate.
[Read-only preflight](serving-reservation-v2-preflight.json) currently rejects
only missing authorization and found no active Most-Queue pods. Recheck price,
stock, runtime and budget before any paid create.

Use the same user-selected key source through the ignored project `.env`;
credentials remain local. Persist campaign intent and exact owned pod ID, run
the already-tested detached local watchdog, export and terminate on completion,
failure, deadline or stall, and verify absence through a fresh provider read.
The local watchdog does not guarantee cleanup after complete host/network loss;
this supervised fallback must be accepted for this campaign. No pod stays up
while awaiting user input. No resources from other projects are changed.

New ledger (never reuse or reset the closed v1 ledger):
`~/.local/state/most-queue/runpod/serving-reservation-v2-2026-11-14/`.

After authorization, use the established controller with these exact paths:

```bash
.venv/bin/python tools/runpod_pilot.py preflight \
  --policy configs/runpod/reservation-v2-policy.json \
  --profile configs/runpod/reservation-v2-profile.json \
  --state-dir ~/.local/state/most-queue/runpod/serving-reservation-v2-2026-11-14
```

`run` uses the same arguments. Preparation is CPU-only. Validate and document
results after cleanup, then package the supported APIs and example for the next
release, with the measured limitations stated explicitly.
