# EPIC-096: фазозависимый сертификат устойчивости token queue

- **Статус:** done (research pilot)
- **Создан:** 2026-10-11
- **Предшественник:** [EPIC-095](EPIC-095-infinite-stage-aware-ttft.md).
- **Выбор автора:** явно выбран фазозависимый сертификат EPIC-096.

## Протокол до численного эксперимента

Сохраняется iid finite-support joint law, FIFO boundary admission, decode-first
и общий token budget. Active-only reachability из empty/одной первой заявки:
все fill=0..vacancies с iid admission words. Каждый путь реализуется при
waiting count0 выбором ровно fill Poisson arrivals. Поэтому полный phase set
насыщенной цепи строится без усечения queue count. Для h>=B admission всегда
заполняет все vacancies, и фазовая матрица P независима от h/rate.

Проверить все reachable closed SCCs, включая периодические; transient phases
учесть через absorption/harmonic drift. Class capacity=E[departures]/E[holding]
с правильным physical-time denominator. Corrector Poisson equation для
V(h,a)=h+b(a)+constant даёт row-wise drift. Acceptance of stability требует
строгой отрицательности **всех** фазовых row drifts. Проверить эту конечную
inequality точно рационально на binary input costs/rate и нормированной
binary probability law; stationary solves/critical capacity остаются numerical
diagnostics. Critical band1e-10 relative — unresolved, не stable.

Положительный drift на reachable closed class проверяется отдельно;
его escape implication объяснить, не выводить instability из finite DES.
Mean error control расширить через linear/quadratic exterior potentials и
явную bound на return из finite core. Arithmetic verification of all Poisson
tails/mean bounds остаётся отдельной задачей: exact drift check не делает
весь delay interval machine-verified.

Frozen original grid48: прежние B1/2,M2/4,laws2,costs2,serialized loads.1/.5/.8.
Primary criterion: получить phase stability certificates на всех24batched,
включая3EPIC-095refusals. Mean refinement N8/16/32/64/128/256,
width<=.001 relative плюс1e-8absolute для W/V/TTFT, прежние budgets
max_states20000/max_transition_words2000000. Все accuracy/budget refusals
публиковать; параметры/требования после прогона не менять.

Extended stability-only grid: все8 прежних B2 law/budget/cost combinations,
lambda/capacity=.6/.85/.95/1/1.05 (40cases). Expected stable for strictly
subcritical, positive-drift for supercritical, unresolved for critical.
Capacity-dependent rates — выбранный диагностический дизайн, не независимое
доказательство threshold; analytic B1/homogeneous cycles и exact row checks
нужны как отдельные arbiters.

Fresh independent infinite DES: mixed law,costs(.1,.02,.003),B2,M2/4,
saturated load.6/.85,50000jobs,warmup20%,3seeds96000..96002. Проверять delay
predictions и sampling uncertainty; finite cohort drain не stability proof.
Symbolic checks до production: periodic two-phase corrector, Poisson quadratic
increment и exterior area potential. Independent phase construction/SCC,
homogeneous threshold, stability/critical/supercritical, boundary return bounds,
old mean intervals и fresh token replay — обязательные tests.

## Задачи

- [x] Первоисточники, exact reachability proof и multi-class drift theorem.
- [x] Phase calculator и exact rational drift verifier с complexity guards.
- [x] Mean continuation potentials и проверка сохранения error control.
- [x] Frozen grids, negative cases и independent DES.
- [x] EN/RU documentation, dissertation sources/derivation/hypotheses/journal/archive.
- [x] Required fast/full tests, formatting/lint и local links.

Real-support compression, correlated marks, KV/evictions, GPU/SLO и
publication novelty не входят в автоматически заявляемые результаты.

## Результаты

[Полный вывод/атрибуция/проверки](../research/phase-aware-stability-validation-2026.md),
[JSON всех случаев](../research/phase-aware-stability-results-2026.json).
48/48 original bounds,24/24batched, все3 EPIC-095refusals сняты; selectedN<=32,
max701states. Extended40:24positive recurrence/8possible escape/8critical
unresolved. Fresh50000jobs×3seed×4configs:12 mean predictions в replicate
t95 intervals, highloadselectedN128/max2717states. Protocol unchanged.
Single-seed highload discrepancy6–7% сохранено; сверка использует предусмотренное
three-seed mean со старым3.5% tolerance, не расширенным post hoc.

27 new targeted tests passed,112 stage-aware family tests passed;
final fast tests-only suite3332passed,16existing warnings389.16s;
full tests-only suite3350passed,16existing warnings470.21s. Black/isort/diff clean;
affected library/examples pylint10/10, full-library9.99/10 с прежними unrelated
warnings.108private snapshots,7archive tests,439Markdown/2814local links/0missing.
Historical journal prefix/GL3/095JSON SHA preserved.

Critical boundary и mean rounding verification открыты, novelty не заявлена.
Следующий предложенный, не запущенный кандидат EPIC-097: tighter phase-specific
full-cycle continuation/finite-core fundamental matrix и новая near-critical
preregistration. Real-support/correlated/GPU/SLO требуют отдельного выбора.
