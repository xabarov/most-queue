# EPIC-095: бесконечная стадийная очередь и контроль усечения

- **Статус:** done (infinite-model research pilot; analytic truncation bounds, no verified rounding)
- **Создан:** 2026-10-11
- **Предшественник:** [EPIC-094](EPIC-094-stage-aware-ttft-reduction.md).
- **Выбор автора:** «выполни следующий этап» после предложения infinite queue/error bound.

## Протокол до расчёта

Политика и iid joint marks прежние. Очередь бесконечна, arrivals не теряются.
Регенеративный цикл от пустой системы останавливается при первом выходе
waiting count за N. Внутри области используется настоящий substochastic
kernel, без отражения, переноса хвоста на границу или нормировки строк.
Пропущенные cycle time/areas ограничиваются сверху busy-period potential
вспомогательной M/G/1 с поклиентной верхней оценкой работы. Это достаточное,
не необходимое условие устойчивости. Отказ вне этой области публикуется;
его нельзя интерпретировать как неустойчивость реального batch scheduler.

Primary: интервалы W/V/mean TTFT, явные overshoot moments, cycle-exit
probability и bound на stationary boundary-count tail. Это не TTFT quantile
или SLO certificate. Отдельно указать floating-point residual и отсутствие
interval-arithmetic проверки округления: аналитическая оценка model truncation
не равна machine-verified enclosure.

Frozen grid: прежние48 случаев B1/2, M2/4, два joint laws, два costs,
serialized load .1/.5/.8; использовать сохранённые независимые DES только
как историческое сравнение, не повторные новые измерения. N=4,8,16,32,64;
первый N, удовлетворяющий relative interval width <=.001 для всех трёх means
(absolute allowance 1e-8). max_states=20000, max_transition_words=2000000.
Отказы по sufficient stability/budgets/accuracy не скрывать и не донастраивать.
Свежий независимый infinite token DES: три seeds95000..95002, 30000 jobs,
warmup20%, mixed law, costs=(.1,.02,.003), B2, M2/4, rate=1.3.
B1 PK, low load, kernel row mass+exit=1, killed-domain refinement и
overshoot moments against explicit positive Poisson series — unit checks.
Не требовать попадания stochastic sample mean в узкий deterministic interval:
sampling uncertainty оценивается отдельно.

## Задачи

- [x] Вывод domination, busy-period potentials и regenerative ratio bounds.
- [x] Infinite calculator, явные ограничения и независимые тесты.
- [x] Frozen grid, fresh DES и публикация всех отказов.
- [x] EN/RU catalogs/docs, dissertation sources/synthesis/journal/archive.
- [x] Fast/full tests, formatting/lint и local links.

Точная saturated stability boundary, phase-dependent Lyapunov refinement,
machine-verified rounding, correlated/real-support marks, GPU и SLO — отдельные
продолжения, не объявляются завершёнными в этом эпике. Они явно перенесены
в возможные следующие исследования roadmap, требующие выбора автора.

## Результаты

[Вывод и проверки](../research/infinite-stage-aware-ttft-validation-2026.md) ·
[JSON с каждым refinement и отказом](../research/infinite-stage-aware-ttft-results-2026.json).
45/48 results:24 B1 PK и21batched;3 sufficient-stability refusals, не
instability claims. Successful batched widths W/V/TTFT<=0.1%, N<=32,
max701states; waiting-tail upper<=6.1331e-6. Midpoints не exact closed forms.
Actual within-iteration waiting tail учитывается, не путается с TTFT quantiles.

Fresh token DES30000jobs/3seeds/M2/4 согласуется с prediction по W/V/TTFT
в пределах replicate t95 uncertainty. Historical finite K6 TTFT вне новых
infinite intervals в13/21batched случаях; это различие моделей, не ошибка
finite solver. Protocol не loosened; historical GL3 JSON hashes сохранены.

31 новых tests,85 stage-aware targeted. Fast `pytest -n auto -m 'not slow'`:
3311passed/18existing warnings/386.55s. Full `pytest -n auto`:3331passed/
18existing warnings/470.95s. Оба запуска из root включают misc/tests,
поэтому не сравнивать counts напрямую с прежним tests-only запуском.
Последний B1 null-diagnostic patch отдельно проверен85targeted и повторно31
new tests; повторный frozen experiment воспроизвёл результаты и отказные cases.
Black/isort/diff clean, library/example pylint10/10,7archive tests passed.

Диссертационные source cards/read depth/две HTML/SHA, proof/results,
hypotheses/plan/concept/lemma registry и append-only journal сохранены.
106 public research/roadmap snapshots с SHA256; private gitignore прежний.
Publication novelty и general real-support scalability не установлены.

Предпочтительный следующий математический кандидат: phase-dependent drift
и строгая обработка reachable closed phase classes, чтобы улучшить нынешнее
conservative sufficient condition. Не начат автоматически.
