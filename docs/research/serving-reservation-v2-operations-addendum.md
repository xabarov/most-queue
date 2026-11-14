# Reservation v2: operational interruptions and authorized extra allocation

The scientific [plan](serving-reservation-v2-plan.md), workload, both fitted
parameter sets and accuracy targets remain unchanged. This addendum records
infrastructure failures; neither trial was retried because of accuracy results.
No fit or accuracy evaluation was performed on the incomplete first trial.

## Completed allocations

| Attempt | Outcome | Accounted allocation | Estimated cost, including disk |
|---|---|---:|---:|
| 1, `85ithanoh2ot7r` | API response read failed during progress polling; partial export retained | 1256 s | $0.209722 |
| 2, `gh1squify1rwr8` | Readiness exec returned HTTP 404 before measurements | 8 s | $0.001336 |
| Total | Both exact pods confirmed absent by subsequent provider reads | **1264 s** | **$0.211058** |

The first client exception had no HTTP status; its exact transport/parser cause
was not retained. The live export is not atomic: 361 raw successful request
records, 362 engine records and a producer summary count of 362. The plan and
summary never finalized, and actual offered/in-flight counts at interruption
cannot be reconstructed exactly. Preserve this
[interrupted fixture](../../tests/fixtures/serving_measurements/runpod_qwen3_8b_a40_reservation_v2_interrupted_20261114/README.md)
unchanged and exclude it from accuracy/SLO validation. The
[completeness audit](serving-v2-interrupted-audit.json) correctly refuses it.
The second allocation produced no measurement fixture.

Provider billing queries for both pod IDs returned zero records at the time
of checking. Those empty responses do not mean free compute; costs above are
the conservative local ledger estimates, not a finalized provider invoice.

## Prepared controller corrections

1. Observational progress reads retry at most three times for transient API
   failures, with bounded backoff and watchdog/deadline checks. Mutating exec
   calls, creation and deletion do not acquire automatic retries. Fixed in
   commit `4511402`; the pilot/lifecycle/budget suites passed 211 tests.
2. Initial readiness allows one fixed 30-second visibility window for GET
   absence or exec404, with exact ownership checked on each iteration. It
   cannot reset stage/progress deadlines or suppress failures after that window.
   The current live Runpod exec-tool contract documents short-lived 404 after
   create, before container startup answers 409. The controller previously
   treated that early 404 as permanent. The combined operational suites now
   pass **220 tests**, including bounded failure and cleanup-guard cases.

The second correction is supported by the live `mcp__runpod__exec_pod` tool
description inspected on 2026-11-14; the
[REST v2 contract](https://api.runpod.io/v2/openapi.json) also documents readiness
409 and temporary exec 503. No server scheduling, measurement or model code
was changed. The scientific SHA256 manifest still matches all 18 files.

## Authorized amendment — one additional allocation

The original authorization covered at most **two allocations**. After both
operational interruptions and confirmed cleanup, the user explicitly approved
**one additional allocation** on 2026-11-14: “Разрешаю одну дополнительную попытку”.

This amendment permits **one additional allocation**, raising the aggregate maximum from 2
to 3 while retaining **$5 total, 14400 seconds total, one A40 at a time**, the
same hourly limits, no persistent storage and the accepted local watchdog.
The first two allocations remain charged to the same persistent ledger;
no campaign or account counters are reset. Expected extra cost is $0.30–0.45.
At the last observed $0.59/h compute plus $0.011112/h disk, the conservative
4500-second work window plus 900-second cleanup reserve projects an aggregate
$1.112726, below $5. Price and stock must be rechecked before create.

The authorization amendment was persisted before the third allocation; both
the private policy and campaign metadata now allow at most three allocations.
The original authorization, prior attempt evidence and ledger are retained.
Keep the existing
scientific plan, seed, 690 requests, frozen fits and gates unchanged. Completion
was required to include export, exact termination proof, completeness audit
and all gate outcomes before release publication; the final outcome follows.


## Final outcome

The approved third allocation, `e1tya23ne00ouo`, completed all 690 requests.
It ran for 2065 s, estimated $0.344805 including disk; the aggregate ledger is
**3329 s / $0.555862 / three allocations**, with no active pod IDs. Exact GET
and inventory absence for all three IDs was checked again after completion.
The campaign is closed and its private authorization is now false.
One transient observational read was recovered by the bounded probe retry.

The shell runner reported exit code 143 after durable success, verified export,
cleanup and accounting. Its cause was not established; success is substantiated
separately by the closed campaign record, matching full export SHA256/size,
provider absence and the complete 690-request audit. The shell code is retained
in the sanitized completion evidence rather than interpreted as a successful
process exit.

All 18 scientific hashes remain unchanged. The
[full result](serving-reservation-v2-results.md) records 13/14 criteria passed,
the failed isolated TTFT target and all diagnostic SLO errors. No additional
allocation or refitting followed those accuracy results. Provider billing reads
for all three IDs still returned no records; reported costs remain estimates.
