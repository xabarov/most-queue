# Real trace service calibration

EPIC-055 provides an audited SWF adapter and a reproducible comparison of existing
MSJ disciplines on one historical workload. It measures whether fitting service
times preserves queue metrics and policy choices. It is not a new scheduler or
a reconstruction of the production cluster.

## Source and scope

The source is the cleaned SDSC SP2 4.2 log described by the
[Parallel Workloads Archive](https://www.cs.huji.ac.il/labs/parallel/workload/l_sdsc_sp2/index.html).
Credit: SDSC, Victor Hazlewood; SWF conversion by Dror Feitelson. The public
research copy comes from DIR-LAB, pinned to commit
`cd433e3d62a01705f255bd48cb55b393862800e7`; the downloader verifies SHA-256
`64bc6c621c97fe10e74cfc6a8c568ba544fddd99c4f810473027dca99efc2ed1`.
Direct archive downloads returned HTML in this environment, so byte equivalence
to that unavailable download is not asserted. The source header and documented
version are consistent with the mirror; neither alone is an authenticity proof.

This historical 128-node workload is a manageable first validation case, not
evidence about present-day GPU serving. The original notice remains in the cached
file. The JOBLOG research/educational/non-profit usage conditions are separate
from Most-Queue's MIT license; consult the source page before reuse.
Raw jobs and user IDs are not committed, only aggregate results and hashes.

## Input contract

`most_queue.sim.utils.workload_trace.parse_swf(lines, capacity)` accepts standard
18-column records. Malformed/nonfinite data, noninteger identifiers or resource
counts, duplicate IDs, unsorted submissions, incompatible capacity and preemption
headers fail explicitly. No repair, sorting, rounding or silent deduplication occurs.

For the accepted cohort, status must be 1, runtime positive, wait nonnegative,
allocated processors within capacity, requested processors equal to allocation,
and predecessor absent (`-1`). Other valid records contribute to mutually exclusive
exclusion counters, in that order. Counts describe this adapter's filtering only;
the upstream cleaned trace has already removed selected bursts and early records.

The [SWF specification](https://www.cs.huji.ac.il/labs/parallel/workload/swf.html)
distinguishes submit, wait, runtime and CPU time. We use wall-clock runtime as
fixed rigid-job service and allocated processors as K. Historical wait is used
only for `completed_at = submit + wait + runtime`, never as simulated waiting.
Status 1 is a recorded outcome, not a guarantee of useful application success.

Cancellation and nonpositive durations are excluded, not repaired or treated as
complete latent service. This creates a retrospectively selected completed-job
population, omits its competitors and reduces offered work. Results must not be
advertised as full-cluster performance. The capacity is assumed constant at 128;
memory, topology, queue priorities, outages and interactive feedback are omitted.

## Chronological protocol

The cutoff is 60% of the raw submission-time range, including excluded records.
The full range is retrospective information used to define an offline experiment;
the protocol does not reproduce an online choice of the cutoff or predict future
arrival times. Held-out arrival/K sequences are conditioned on for evaluation.
Only eligible jobs whose recorded completion is strictly earlier enter fitting.
Pre-cutoff arrivals still unfinished at that time are excluded; outcomes after
the cutoff never enter the fit. This completed-history restriction can bias
against long pending jobs; survival estimation is not implemented in this epic.

After the cutoff take the first three nonoverlapping blocks, each 200 warmup plus
1000 measured jobs. Each block starts empty; all jobs are drained. Warmup does not
reconstruct the real backlog or establish stationarity. The same history serves
all blocks; these blocks are not independent statistical replications.

Submission gaps and exact resource needs are preserved; only durations change in
model variants. No test-mean matching, workload compression, winsorization or
Poissonization is applied. Finite `offered_work_ratio = sum(K*S)/(128*arrival_span)`
is reported for measured jobs, not called a stability parameter. Utilization and
idle-with-queue use the existing replay observation window, excluding drain.

## Service models and information boundary

Fit four fixed resource groups: K=1, 2–8, 9–32, 33–128. A group with fewer than
20 historical observations explicitly falls back to the pooled history. Actual
K remains unchanged in replay; grouping affects fitting and forecasts only.

`most_queue.random.service_calibration.ServiceCalibration.fit(samples)` supplies:

- `empirical`: inverse discrete CDF resampling, retaining only historical support.
- `exponential`: mean-matched Exp, CV²=1.
- `ph`: balanced-means H2 if historical CV²>1, matching mean and variance;
  otherwise mean-matched Erlang with nearest integer `1/CV²`, capped at 64 phases.
  Erlang's achieved CV² is recorded; deterministic data is approximated, not exact.
- `lognormal`: moment-matched mean and variance, not log-space maximum likelihood.

For H2, with `v=CV²`, choose
`p=(1+sqrt((v-1)/(v+1)))/2`, rates `2p/m` and `2(1-p)/m`.
Then `E[S]=m` and `E[S²]=m²(1+v)`. The inverse CDF is obtained by monotone bisection;
tests independently cross-check moments with the library PH implementation.
For lognormal, `sigma²=log(1+v)` and `mu=log(m)-sigma²/2`.

Each model is iid conditional on the coarse K group. It loses duration ordering,
within-group resource dependence, user/application structure and drift. Therefore
an error against observed replay combines fitting, dependence and nonstationarity;
it cannot be attributed solely to choosing Exp rather than G.

Seeds 55000–55007 provide common inverse-CDF uniforms across the four models.
Every discipline receives exactly the same generated workload for that model,
block and seed, certified by numerical fingerprints. No generated sample is
rescaled to match test work. Model population means match historical group means;
finite generated means and held-out means generally differ.

Policies are FCFS, FirstFit, MSF, Adaptive Quickswap, EASY and conservative.
Backfilling uses one fixed historical group p90 forecast under every service
model, including the observed control. Forecasts do not see service realizations.
This isolates workload modelling from forecast changes. There is no oracle mode,
online refit, kill-at-estimate or cost-free-preemption competitor here.

## Metrics and uncertainty

Primary metric: mean sojourn time T. Also report mean W, p95/p99 T, mean T weighted
by each measured job's K, four group means/p99s, utilization, idle-with-queue and
reservation violations. K weighting is not K*S weighting. Quantiles use NumPy's
default linear interpolation. A group absent from a measured block yields `null`,
never nonstandard JSON NaN. T includes all waiting and service until completion.

For each model/policy/block, record its Monte Carlo mean and exploratory 95% Student
t interval over eight generated workloads, relative error against the single
observed-duration replay, and paired contrasts to FCFS. These intervals quantify
conditional simulation variability, not uncertainty in history fitting, the real
population, rare-tail guarantees or multiple-comparison-adjusted significance.
P99 summaries average eight finite-cohort p99s, not pool all jobs into one p99.

For mean T and p99 separately, select the policy with the lowest model MC mean;
report its excess metric on observed replay relative to the best observed policy.
This is an offline decision-fidelity diagnostic, not an independent deployment
trial. Exact ties use declared policy order. Pair reversals exclude reference
ties within 1e-9 seconds; a model tie is not counted as a strict reversal.

## Reproduction

From the repository root, after reviewing source usage conditions:

Use Python 3.10+ and a development environment installed with `pip install -e '.[dev]'`;
see [environment setup](INFRASTRUCTURE.md). Commands below assume its Python is
available at `.venv/bin/python`. Keep the experiment, adapter and calibration
source files from the same repository revision as the result artifact; JSON
records the Python/NumPy/SciPy versions but does not pin the code revision itself.

```bash
.venv/bin/python -m examples.real_trace_calibration_experiment --download \
  --output works/real_trace_calibration/sdsc-sp2.json
```

The raw 5.6 MB file is cached under `.cache/real_trace/`, already git-ignored.
Every invocation verifies its hash. Without `--download`, missing data is an
error; an existing corrupt cache is never overwritten. Tests are offline and
contain only synthetic records. Result JSON includes config, versions, provenance,
fit parameters, exclusion/split audit, 594 scheduling rows, summaries and decisions.

To reproduce with a different output path and the same cached source:

```bash
.venv/bin/python -m examples.real_trace_calibration_experiment \
  --output /tmp/sdsc-sp2-replay.json
```

CLI `--jobs`, `--warmup`, `--blocks`, `--replications` allow smoke experiments;
their outputs carry their actual config and are not the prescribed full result.

See [results](research/real-trace-calibration-results-2026-10.md),
[roadmap](roadmaps/real-trace-calibration.md) and
[EPIC-055](epics/EPIC-055-real-trace-calibration.md).
