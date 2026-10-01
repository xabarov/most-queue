# PH multiserver jobs and general-service EASY simulation

[Model/API examples](models/msj.md) · [EPIC-047](epics/EPIC-047-msj-ph-backfilling.md)

## Scope and state

Classes arrive as independent Poisson streams with rates lambda_i > 0. Each job
holds need_i identical servers throughout its independent PH(alpha_i,T_i) service.
Correlated resource need and duration can be represented by the class mixture.
FCFS starts the longest fitting prefix and stops at the first blocked job.

The finite-chain state is an arrival-ordered tuple of (class, phase), where phase
-1 denotes waiting. At a start, branch on alpha_i; while running, T_i gives phase
transitions and -T_i*1 the completion rates. After a completion, admit the maximal
fitting waiting prefix, branching on its initial phases. Arrival-ordered running
jobs retain their order even though completions need not follow that order.

Enumeration stops at N jobs in the system; arrivals there are lost. This changes
the model, rather than imposing a mathematically exact boundary condition for the
infinite queue. The returned p_N is a convergence diagnostic. Check several N and
the chosen output metric; no universal error bound follows from p_N alone.

Solve pi Q = 0 and pi*1 = 1 using a sparse linear solve. Reject invalid stationary
probabilities or excessive residuals. Compute per-class number waiting Lq_i and
number present L_i from rewards. PASTA gives lambda_i,accepted = lambda_i*(1-p_N).
Then W_i=Lq_i/lambda_i,accepted and T_i=L_i/lambda_i,accepted. All classes have the
same blocking probability, so their admitted mixture equals their arrival mixture.
Mean occupied resource fraction is the reward E[sum_running need_i]/k.

The old `MsjExactCalc` divides by offered rates; the new finite-model calculator
deliberately uses admitted rates. Exp regression therefore accounts for this
factor, and converges to the old open approximation as boundary mass vanishes.
No existing calculator has been changed.

## Saturated throughput

The saturated state consists of a sorted multiset of running (class,phase) pairs
and the class of the blocked head. At a completion, start the head if possible,
then draw iid new classes with probabilities lambda_i/sum(lambda), and initial
PH phases, until another head blocks. Equivalent multisets are combined during
filling to avoid enumerating every permutation. Self-events contribute to the
completion reward even when they do not change the CTMC state.

The stationary completion rate X_sat is the FCFS stability threshold for this
model. `run()` requires sum(lambda)<X_sat. The weaker necessary resource condition
sum(lambda_i*need_i*E[S_i])/k<1 is not sufficient: on k=3, needs 2 and 3 can never
coexist, so equal unit-mean classes have X_sat=1 despite spare resource capacity.
This threshold says nothing about stability of a different scheduler such as EASY.

PH validation rejects complex parameters, negative probabilities/transitions,
positive row sums and nontransient matrices. Finite-state PH service has light
tails; matching a heavy-tailed distribution over a finite range is a separate
approximation and is not implemented implicitly here.

## EASY information and overrun policy

The input record stores both actual service and an explicit estimate. Only the
event engine accesses actual service to schedule its completion. The reservation
calculation receives occupied resources and estimated completion timestamps.

When the head is blocked, compute its earliest predicted start from the release
profile. A candidate must fit now and must not move that start later when added
to the profile. This handles both candidates finishing before the reservation and
long candidates fitting on capacity left over when the head starts. Recompute the
profile with all already accepted candidates; do not grant each one the same idle
resources independently. Simultaneous completions are processed before arrivals,
and all tied arrivals enter in their input order before dispatch.

An overrun makes that running job's release unknown. Do not peek at its scheduled
actual completion: suspend new backfills until no running job is overdue. Normal
FCFS admission remains allowed whenever the head fits. Jobs are not killed or
preempted. Count a violation if a head starts after its earliest recorded finite
reservation. This is an explicit conservative overrun rule, not an implementation
of the adaptive prediction correction from Tsafrir et al. Underestimation can
already have broken the promise; suspending later backfills cannot undo that.

## Measurement and evidence

Traces are generated before scheduling. This preserves identical arrivals, classes
and service requirements across policies and prevents different random-number
consumption at service starts from changing the experiment. A caller can also
provide non-Poisson or dependent inputs through trace replay.

Warm-up excludes an arrival-index prefix, not the fastest completed jobs. Every
remaining job is drained, so no measured response is censored. This is still a
finite-cohort experiment: stopping new arrivals may affect the last jobs under
overtaking policies. Use long traces and sensitivity to horizon/warm-up before
claiming stationary performance. Time averages exclude the draining interval.
Empirical p99 and raw moments are not guarantees; no confidence interval is attached
to an individual replay. For heavy tails even the existence of population moments
must be considered independently of finite sample moments.

Tests cover hand-computed schedules; actual-size hiding; safe/unsafe/long backfill;
underprediction; ties; cohort and occupancy accounting; invalid inputs; Exp
reductions; M/G/1 Pollaczek–Khinchine means; phase-type saturated throughput;
truncation convergence; and independent Erlang/Cox/H2 CTMC/DES comparisons.
The experiment module reports Student intervals over independent replications
and paired EASY-minus-FCFS differences, including per-run p99 differences.

## Sources and novelty boundary

- Anggraito, Olliaro, Marin, Ajmone Marsan, *The Multiserver Job Queuing Model with
  two job classes and Cox-2 service times*, Performance Evaluation 169, 2025,
  [DOI 10.1016/j.peva.2025.102486](https://doi.org/10.1016/j.peva.2025.102486).
  Its matrix-geometric algorithm and published numerical tables are not reproduced
  by this sequence-enumeration implementation.
- Grosof, Hong, Harchol-Balter, Scheller-Wolf, *The RESET and MARC techniques, with
  application to multiserver-job analysis*, Performance Evaluation 162, 2023,
  [DOI 10.1016/j.peva.2023.102378](https://doi.org/10.1016/j.peva.2023.102378).
  Saturated PH systems and correlated server need/duration are established tools.
- Tsafrir, Etsion, Feitelson, *Backfilling Using System-Generated Predictions Rather
  Than User Runtime Estimates*, IEEE TPDS, 2007,
  [DOI 10.1109/TPDS.2007.70606](https://doi.org/10.1109/TPDS.2007.70606).
  Runtime predictions must be distinguished from enforced kill-time limits.

This implementation is an engineering and reproducible-experiment contribution,
not a claim that PH-MSJ or EASY are new theories. Conservative backfilling,
ServerFilling/MSFQ, large-system approximations and certified tails remain outside
this epic; see the linked roadmap.
