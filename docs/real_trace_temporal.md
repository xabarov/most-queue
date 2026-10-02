# Temporal history and dependent service replay

EPIC-056 follows the real-trace calibration study with prespecified workload
ablations. It separates an update of historical mean from an update of empirical
shape, adds finer conditioning on resource demand K, and compares independent
with dependent rank resampling. It does not add a scheduler or identify causal
effects in the production cluster.

## Data and rolling origins

The source, checksum, usage conditions, selected status=1 population and fixed
128-node assumption are unchanged from [EPIC-055](real_trace_calibration.md).
SDSC / Victor Hazlewood provided the data; Dror Feitelson converted SWF. The pinned
DIR-LAB mirror is used because the upstream download was unavailable. No raw jobs
or user identifiers are redistributed. Cancellation, initial production backlog,
priority queues, hardware heterogeneity and failures are still outside the model.

Use four origins at fractions 0.35, 0.50, 0.65 and 0.80 of the raw submission range.
Each takes the first 1200 eligible later arrivals, of which 200 warm up an initially
empty simulator and 1000 are measured. The implementation rejects overlapping or
incomplete test windows before any scheduling run. Every measured job is drained.
Warmup does not certify stationarity or reconstruct production backlog.

At each origin, training jobs must have original-log completion strictly before
the cutoff. `expanding` uses all such jobs. `recent` takes the last 4000 of them
in submission order, not the last 4000 completion events or a fixed number of days.
This changes recency AND sample size. Pending historical jobs are omitted, which
can bias the sample against long service; no survival correction is claimed.

Later origins may train on already-completed jobs from earlier evaluation periods.
The outcomes come from the original log, not the simulated counterfactual policy.
Origins are therefore not independent folds. Cutoffs depend retrospectively on
the full trace span. Evaluation conditions on future arrival/K sequences; it is
not a prospective prediction of these sequences or an online deployment trial.

## Conditional empirical distributions

`ConditionalEmpirical.fit(needs, services, exact=False, minimum=20)` in
`most_queue.random.trace_resampling` uses only supplied history. Coarse groups are
K=1, 2–8, 9–32, >=33. A group with fewer than 20 observations falls back to pooled
history. With `exact=True`, an exact-K distribution is used when it has at least
20 observations; otherwise use coarse, then pooled. Actual replay K is never rounded.

The chosen distribution and fallback level are inspectable with `distribution(K)`.
`quantiles(needs, uniforms)` applies inverse discrete CDFs with probabilities
strictly inside (0,1). Distributions are empirical rather than parametric in this
epic. In-sample midranks invert back to the original values, including ties.

Seven variants are fixed before outcomes:

| Variant | History and conditional law | Dependence |
|---|---|---|
| `expanding_coarse` | Full history, coarse K | Uniform iid |
| `recent_mean_only` | Expanding samples times recent/expanding coarse mean ratio | Uniform iid |
| `recent_coarse` | Recent history, coarse K | Uniform iid |
| `recent_exact` | Recent history, exact K with fallback | Uniform iid |
| `recent_rank_iid` | Same recent exact-K laws | iid historical conditional ranks |
| `recent_block20` | Same recent exact-K laws | Circular rank blocks of 20 |
| `recent_block60` | Same recent exact-K laws | Circular rank blocks of 60 |

The mean-only ratio is fitted exclusively from history. It keeps the standardized
shape and CV of the expanding distribution, while changing its mean to the recent
mean. Both means come from the actually selected distributions, including any
coarse/pooled fallback, rather than from an unavailable sparse group estimate.
It is not a rescaling to realized held-out service or workload. Comparing
recent_coarse to mean_only then changes shape while retaining the same fitted
group means. Finite generated sample means need not be equal.

## What the block generator preserves

For each recent historical job, let `F_K` be its fitted conditional ECDF and define
`u = (F_K(S-) + F_K(S))/2`. These midranks are a chronological tape of relative
durations. Ties get the same midrank. A pooled tape mixes ranks from different
conditional distributions; its finite marginal need not be exactly uniform.

Generate donor indices in equal-length circular blocks: uniformly sample a start
index, append L consecutive positions modulo history size, and truncate the final
block to the requested output length. This is the usual
[circular-block convention](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.CircularBlockBootstrap.html).
Our implementation is independent and does not add the `arch` package dependency.
No population bootstrap confidence-interval theorem is used here.

Map sampled ranks through the ECDF of each target job's K. Donor K and donor
arrival times are not transferred; real target arrivals and exact K stay fixed.
Thus rank blocks carry some temporal dependence of relative service sizes, not
the complete joint production process `(arrival, K, S, user, application)`.

Every output position has a uniform donor index, since adding a fixed offset
modulo n permutes all n possible starts. Therefore block20, block60 and rank-iid
have exactly the same donor-rank marginal distribution in their sampling laws.
Their finite realized histograms need not match. Rank-iid is the dependence
control; uniform-iid is a separate control for pooling/discretization effects.
Using uniform-iid alone would confound rank marginal and temporal dependence.

Circular end/start seams are artificial. So are adjacencies across jobs omitted
by status filtering, incomplete-history selection or upstream cleaning. The audit
counts gaps in the accepted-job sequence within recent history; it cannot recover
removed upstream jobs. Blocks of 20 are the primary dependence comparison and
60 a prespecified sensitivity, not a data-selected optimal block length.

## Common inputs and forecasting boundary

At each origin seeds 56000–56007 supply common uniforms. Starts for length L use
the uniforms at output positions 0,L,2L,... . The origin's cutoff identifies its
random stream independently of its position in the requested list.

All six disciplines from EPIC-055 see exactly the same workload for each
variant/seed. Backfilling uses an expanding-coarse historical p90, fixed across
ALL variants in the origin. Recent fitting therefore does not simultaneously
improve the scheduler's forecasts. No predictor sees held-out S, and there is no
kill-at-estimate or checkpoint/resume extension in this experiment.

There are `4 × (1 + 7 × 8) × 6 = 1368` scheduler runs: 24 observed-duration
controls and 1344 generated-model runs. Each origin is independently initialized
empty. Across variants, arrival times, K and forecasts are unchanged; generated
durations change deliberately and are fingerprinted.

## Metrics and interpretation

Primary metric is mean sojourn T; secondary metrics include mean W, p95/p99 T,
K-weighted mean T, per-group T/p99, utilization, idle-with-queue and reservation
violations. See EPIC-055 for the shared finite-cohort measurement conventions.
The mean of eight cohort p99s is not the p99 of a pooled sample.

For each origin/variant/policy the JSON reports conditional 95% Student t
intervals over eight independent generator seeds and error against observed replay.
These are exploratory Monte Carlo intervals, not population intervals or
multiple-comparison-adjusted tests. No uncertainty from fitting history or from
selection of completed jobs is included.

Each prespecified contrast compares the paired seed-wise absolute relative errors:
`delta = abs(T_new / T_observed - 1) - abs(T_base / T_observed - 1)`.

The six new/base pairs are mean_only/expanding_coarse, recent_coarse/mean_only,
recent_exact/recent_coarse, rank_iid/recent_exact, block20/rank_iid and
block60/rank_iid (names shortened here; JSON retains their `recent_` prefixes).
Negative values mean lower simulation error; the interval quantifies conditional
Monte Carlo variability. These intervals use a different statistic from the
descriptive summary `abs(mean(T_model) / T_observed - 1)`; cancellation of signed
errors can make their conclusions differ. All contrasts, not only favorable ones,
are retained. They characterize interventions on the workload generator, not
causal contributions of real-world drift, mixing or serial correlation.

The aggregate MAPE weights all four origins and six policies equally. It must not
be compared directly with EPIC-055's 60–69% as an improvement estimate: the test
periods differ. Policy-choice regret is calculated on each observed replay after
choosing a policy from model MC means; it is an offline fidelity diagnostic.

Service diagnostics record mean, CV², p99, offered work ratio, and lag-1/20 Pearson
correlation of log S. Conditional midrank lag correlations are also reported.
Zero variance or insufficient observations gives `null`, not NaN. ACF similarity
does not imply preservation of bursts, higher-order dependence or tail risk.

## Reproduction and code provenance

Use the development environment described in [infrastructure](INFRASTRUCTURE.md),
Python 3.10+ and `pip install -e '.[dev]'`. From the repository root:

```bash
.venv/bin/python -m examples.real_trace_temporal_experiment \
  --output-dir works/real_trace_temporal
```

If the EPIC-055 cache is absent, add `--download` after reviewing the source usage
conditions. Every run verifies the source SHA-256. Input stays in git-ignored
`.cache/real_trace`; synthetic offline tests require neither cache nor network.

Output includes `origin-0.json` through `origin-3.json` and `manifest.json`.
The manifest pins dataset identity, protocol, environment versions, byte hashes
of all origin artifacts, and ten implementation files including shared scheduler
dependencies. These hashes detect mismatched code but do not archive dependencies
or replace a repository revision/package lock. Keep files from the same code revision.

`--fractions`, `--jobs`, `--warmup`, `--replications` allow explicitly labelled smoke
runs; never merge them with the prescribed experiment. No hidden tuning is done.

See [EPIC-056](epics/EPIC-056-real-trace-temporal-dependence.md),
[results](research/real-trace-temporal-results-2026-10.md),
and [roadmap](roadmaps/real-trace-calibration.md).
