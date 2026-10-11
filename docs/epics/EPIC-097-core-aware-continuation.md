# EPIC-097: уточнение полного возврата и ошибок усечения

- **Статус:** done (research pilot; engineering target not met)
- **Создан:** 2026-10-11
- **Предшественник:** [EPIC-096](EPIC-096-phase-aware-stability.md).
- **Выбор автора:** явно выбран EPIC-097 после обзора направлений.

## Протокол до численного эксперимента

Модель и все physical costs неизменны: finite iid joint lengths, mark-blind
FIFO admission, decode-first и token budget. Exact rational drift остаётся
отдельным сертификатом; mean arithmetic floating, не verified rounding.

Три ablations: прежний global continuation; actual exit-phase shifts при
прежнем clearing-trial core bound; exit-phase shifts плюс finite-core solve.
Для последнего численная вспомогательная область h<=K, **K=8** до всех прогонов.
Она не буфер потерь. При выходе учитывать всю excursion до h<B и новый
полный возврат, не отражать границу и не отбрасывать overshoot.
Конечная система даёт затраты до idle/выхода и вероятность выхода; максимум
по restart core h<B замыкает неизвестные возвраты. Условия, доказательство,
contraction/mass/residual checks и явные отказы обязательны.

Сохранить original grid48 и добавить32 near-critical cases: B2, M2/4,
прежние2 laws/2 costs, lambda/capacity=.85/.90/.95/.98. Capacity берётся
из EPIC-096, это diagnostic design, не независимый capacity arbiter.
N8/16/32/64/128/256/512/1024; target widths W/V/TTFT <=.001 relative+1e-8
absolute. Budgets20000states/2000000transition words включают обе count
системы нового метода; saturated phase construction учитывается отдельно.
Сохранять все attempts/refusals, widths, state/word counts, elapsed time,
sparse matrix storage bytes. Memory metric не peak RSS.

Primary safety: интервалы нового метода вложены в global при общем N
(floating tolerance1e-10relative+1e-12absolute), B1 PK неизменен.
Engineering target: не менее16/32 near-critical cases имеют >=2x меньше
суммарных выбранных states, если оба метода bounded. Baseline refusal recovery
считается отдельно, не как бесконечное ускорение. Target может не пройти:
это результат пилота, не основание менять K/grid/budgets/precision после прогона.
Runtime — descriptive one-run, не повторённый statistical speedup benchmark.

Fresh independent token DES: mixed law, costs(.1,.02,.003), M2/4,
loads.85/.95,100000jobs×5seeds97000..97004,warmup20%. Все predictions/errors
и replicate t95 intervals публикуются. Это sampling diagnostic, не proof
stability/coverage; отсутствие bounded prediction не заменяется mid-run tuning.

## Задачи

- [x] Primary source and full-cycle finite-core derivation.
- [x] Exit-phase hook без изменения старых численных методов.
- [x] CoreAwareTtftCalc, bounds, combined budgets и отказ при плохом solve.
- [x] Independent moments/core/supersolution/PK/periodic/refinement tests.
- [x] Frozen original/near-critical ablations и fresh DES, все отказы.
- [x] README/models, отчёт/JSON, private synthesis/sources/journal/archive.
- [x] Required fast/full tests, formatting/lint и local links.

Real-support compression, dependent marks, critical recurrence, GPU/SLO tails
и полный verified mean arithmetic не входят в этот эпик. Publication novelty
не следует из реализации classical regenerative/Lyapunov machinery.

## Результаты

[Полный вывод и все ограничения](../research/core-aware-continuation-validation-2026.md),
[JSON всех739attempts](../research/core-aware-continuation-results-2026.json).
Original48/48 для всех ablations; near-critical global24/phase24/core28 из32.
4word-budget refusals recovered при mixed load.95; mixed.98 по-прежнему refuse.
Все219common-domain nesting checks passed. Target>=16/32 с>=2xtotalstates
**failed:1/32**.24paired cases:median1.726/range.871–3.441,16reductions/8increases.
Old original maxmainstates701→365, actualnew totalmax562 с auxiliary overhead.
Fresh12mean predictions внутри5replicate t95 intervals; high-load sampling
discrepancies8%/широкие intervals сохранены. Протокол не менялся.

25newtests/137familytests passed. Fast tests-directory3359passed/16warnings333.54s;
final full3377passed/16warnings518.71s. First full3376passed/1legacy stochastic
failure516.64s сохранён в отчёте;3isolated repeats и secondfull прошли без правок.
Black/isort/diff clean; affected pylint10/10, wholelibrary9.99/10 oldwarnings.
110private snapshots/SHA,7archive tests, local-link integrity checked.
Historical journal prefix/4JSON preserved; mean arithmetic floating.
Следующие phase-compression/structured-count/decision пилоты не запущены.
