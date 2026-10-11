# Стадийный расчёт среднего TTFT

[English](stage-aware-ttft.md) · [Каталог](../models.ru.md)

## Простыми словами

Запрос ждёт допуска, обрабатывает prompt по частям, затем генерирует токены.
Приход посреди итерации должен ждать её остаток: калькулятор учитывает порядок
и достигнутую стадию запросов, а не помещает типичный запрос в безусловную
среднюю занятость. Future output length не участвует в решениях планировщика.

![Допуск на границе и отдельный первый токен](../figures/prefill_decode.ru.png)

```python
from most_queue.theory.continuous_batching import StageAwareTtftCalc

calc = StageAwareTtftCalc(batch_cap=2, token_budget=4, queue_capacity=6)
calc.set_sources(0.5)
calc.set_servers([(1, 1, 0.6), (3, 2, 0.4)], alpha=0.1, beta=0.02, gamma=0.003)
result = calc.run()
print(result.ttft_mean, result.blocking_probability, result.states)
```

## Контракт и ограничения StageAware/ReducedStageAware

B=1 — **бесконечная FCFS M/G/1**, точная формула PK для full service плюс
время только до первого output token; `queue_capacity` в этом shortcut
игнорируется, строка `result.model` явно сообщает модель. B>1 — **конечная
очередь с потерями**: exact semi-Markov reward для заданной модели, лишь
диагностическая аппроксимация infinite token DES. Low loss/refinement не
сертификат ошибки и не доказательство устойчивости бесконечной системы.
Консервативная serialized load также не exact shared-batching threshold.

Допуск FIFO между итерациями, decode-first, далее FCFS prompt chunks в остаток
total token budget; первый decode — после отдельной prefill iteration. Приход
во время итерации теряется, если waiting buffer заполнен. Первое прибытие
в idle запускается сразу. `None` снимает budget, не active cap. Тип с полной
длиной — латентная primitive процесса, планировщик её для выбора не использует.
Независимые Poisson arrivals и iid joint marks обязательны; серийную зависимость
ГЛ2 эта модель не воспроизводит.

Поддерживаются только первые моменты W/V, средний TTFT, utilization, accepted
rate, blocking, residual, state count, время solver. `p/pi` не выдаются за
wall-clock/arrival distribution; ITL и SLO tails не рассчитаны. Finite support
и короткие длины нужны из-за взрыва пространства состояний; budgets приводят
к явному отказу, а не к незаметному coarsening. **Не компактный предиктор
для полного empirical support BurstGPT.** Нет GPU calibration, KV/cache/offload.

На48 frozen synthetic settings точность повысилась, но выбора конфигурации
это не улучшило; старый predictor тоже верно выбрал все24 cap pairs. Некоторые
heterogeneous solves уже медленнее одного короткого DES: same-clock ratios
solver/DES≈1.56–1.83 против≈.067–.071 для homogeneous laws. Поэтому ГЛ3 поддержана
в части точности, не в части компактности/публикационной готовности.

## Точное сокращение iid состояния

`ReducedStageAwareTtftCalc` использует тот же API и возвращает те же средние
конечной loss-модели. Waiting queue хранится как число: независимые joint
метки раскрываются только при FIFO admission. Декодеры сортируются, prefills
сохраняют относительный порядок. В примере достаточно заменить имя класса;
`canonical_decoders=False` отключает только decoder symmetry для абляции.

Редукция точна из пустого старта при iid метках и type-blind admission/loss.
Это не strong lumpability произвольного известного waiting word и не условный
прогноз по наблюдаемым prompt lengths в очереди. Коррелированные метки,
length-aware admission, cancellation и KV-based loss исключены. B1 сохраняет
infinite PK shortcut; B>1 остаётся finite-loss, без сертификата infinite bias.
Дополнительный result `transition_words` считает admission words, не arrival
words полной модели. Active phase space всё ещё комбинаторен.

Все48 прежних и8 доступных expanded случаев совпали по W/V/TTFT/flow/loss/
utilization до4.36e-14. Ещё8 full state-budget refusals сохранены отдельно;
reduced решает все16 expanded cases, до2339states. Полное равенство на
отказавших случаях численно не проверено. Обоснование и новые артефакты:
[точная редукция](../research/stage-aware-ttft-reduction-validation-2026.md).
Paired timing mixed laws:13.1–13.5x меньше states,18.2–27.2x быстрее full;
CPU ratios близки. Decoder sorting здесь немного медленнее waiting-only:
уменьшение states не гарантирует ускорения каждого варианта.

[Вывод и независимые проверки](../research/stage-aware-ttft-validation-2026.md),
[результаты](../research/stage-aware-ttft-results-2026.json).
Источники методов — classical renewal reward/Little/PK,
[Bekker2005](https://ir.cwi.nl/pub/14125),
[Zhang2009](https://repository.gatech.edu/bitstreams/7e8688af-edf1-4c43-a81f-edae8dc4d4c2/download).
Стоимость итераций — [Ramani–Tantawi](https://arxiv.org/abs/2609.20957v1).
Общий метод не заявлен новым; policy — явно определённый token contract Л3.

## Фазозависимая устойчивость и средние (EPIC-096)

Длинная очередь поддерживает полный пул, но стадии prefill/decode чередуются.
Проверяем превышение завершений над поступлениями во всех достижимых замкнутых
классах. Не предполагаются экспоненциальное обслуживание или неприводимость.
[Схема](prefill-decode.ru.md) и дисциплина прежние.

```python
from most_queue.theory.continuous_batching import PhaseAwareStabilityCalc, PhaseAwareTtftCalc
law = [(1, 1, 0.6), (3, 2, 0.4)]
calc = PhaseAwareStabilityCalc(batch_cap=2, token_budget=4)
calc.set_servers(law, alpha=0.1, beta=0.02, gamma=0.003)
calc.set_sources(3.0)
print(calc.run().classification)
means = PhaseAwareTtftCalc(batch_cap=2, token_budget=4, truncation=64)
means.set_servers(law, alpha=0.1, beta=0.02, gamma=0.003)
means.set_sources(3.0)
print(means.run().mean_bounds)
```

Сертификат: closed SCCs, transient count, class capacities (jobs/physical time),
corrector и точные рациональные row drift/margin; не delay/p/pi.
Статусы: `positive_recurrence`, `positive_drift_escape`,
`critical_or_numerically_unresolved`; relative1e-10 critical band unresolved.
Underflow/budgets дают явный отказ. Defaults20000states/2000000words;
active combinatorics остаётся. Вероятности точно нормируются рационально.

Phase-aware means требуют negative drift и используют exterior potentials +
finite-core clearing вместо `lambda E[D]<1`. `dominating_load` — diagnostic,
может превышать1. N не loss buffer; B1 сохраняет PK. `drift_verified` проверяет
только drift, `roundoff_certified=False` относится ко всей арифметике средних.
Проверяйте widths: core bounds могут быть грубыми/overflow. Ограничения
iid/finite-support/mark-blind/policy прежние; не GPU/SLO certificate.
48/48 original cases, все3 отказа сняты, N<=32. Fresh high-load N128,12 means
в three-seed t95 intervals. [Вывод](../research/phase-aware-stability-validation-2026.md) ·
[JSON](../research/phase-aware-stability-results-2026.json).

## Бесконечная очередь с оценкой ошибки усечения

```python
from most_queue.theory.continuous_batching import InfiniteStageAwareTtftCalc

calc = InfiniteStageAwareTtftCalc(batch_cap=2, token_budget=2, truncation=8)
calc.set_sources(1.3)
calc.set_servers([(1, 1, 0.6), (3, 2, 0.4)], alpha=0.1, beta=0.02, gamma=0.003)
r = calc.run()
print(r.mean_bounds["ttft"], r.waiting_tail_upper, r.dominating_load)
```

`truncation` — численная область waiting count, **не физический loss buffer**.
Все приходы обслуживаются. Регенеративная траектория прекращается при первом
выходе из области; оставшиеся time/areas до idle ограничиваются сверху через
busy period доминирующей M/G/1, с настоящими моментами Poisson overshoot.
Строки substochastic kernel не нормируются. B1 использует прежний exact PK;
truncation/tail/cycle/domination diagnostics у него `None`, не нулевой хвост.

Для B>1 необходимо для этого сертификата `lambda*E[D]<1`, где
`D=(alpha+beta)*(S+O)+gamma*(S+O)*(S+O+1)/2`.
Токеновые service credits доказывают достаточное условие устойчивости данной
дисциплины, не точную границу. Отказ калькулятора вне этой области не означает
неустойчивость системы. State/transition budgets остаются явными.

`mean_bounds`: пары `(lower,upper)` по `w`, `v`, `ttft`; стандартные means —
середины интервалов. Дополнительно: `utilization_bounds`, `cycle_exit_probability`,
cycle-time lower/remainder upper, `dominating_load`, `truncation`.
Унаследованный `stationary_residual` здесь — residual killed regenerative
linear solve, не balance бесконечного stationary vector.
Accepted rate=lambda, blocking0. `waiting_tail_upper` ограничивает вероятность
actual waiting count>N в стационарном физическом времени, включая bursts
внутри iteration; это **не** TTFT quantile/tail или SLO/KV certificate.
Higher moments и `p/pi` не выдаются.

Аналитические bounds ошибки модели вычисляются в **обычном floating point**:
`roundoff_certified=False`, residual не сертификат округления. Очень узкий
напечатанный интервал не доказывает соответствующее число значащих цифр.
Ограничения iid/mark-blind/finite support/synthetic costs сохраняются; real
GPU/empirical-support scalability не установлены.45/48 frozen cases решены:
24PK и21batched,3 sufficient-stability refusals сохранены. Successful batched
means достигают ширины <=0.1% при N<=32/701states.
[Вывод, ограничения и независимый DES](../research/infinite-stage-aware-ttft-validation-2026.md)
· [полные числа](../research/infinite-stage-aware-ttft-results-2026.json).
