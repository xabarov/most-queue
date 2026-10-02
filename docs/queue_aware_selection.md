# Выбор модели обслуживания по ранним ошибкам очереди

[EPIC-061](epics/EPIC-061-queue-aware-model-selection.md) сравнивает выбор
генератора S по CRPS распределения и по downstream-ошибкам mean/p99 T.
Тип модели выбирается на двух ранних replay-блоках и замораживается до двух
поздних. Параметры переобучаются только на доступной к каждому cutoff истории.
Это не новый scheduler и не модель совместной генерации arrival/K/context.

Временное разделение следует принципу обучения только на прошлом из
[rolling-origin evaluation](https://otexts.com/fpp3/tscv.html). Здесь вместо
обычного прогноза временного ряда оцениваются очереди на фиксированных arrivals;
это проектный эксперимент, не прямое применение чужой теоремы. CRPS и ECDF
реализованы ранее в [условной модели](feature_service.md). Новый queue loss
**не** является proper scoring rule полного распределения S.

## Область данных и временные границы

Источники, SHA-256, правила eligibility и ограничения лицензий неизменны:
[SDSC lifecycle](real_trace_lifecycle.md), [Kalos](modern_gpu_trace.md).
Raw остаётся вне git. Длительность SDSC — runtime, Kalos — end−start,
не exported duration. Калибровка только по completed; положительная наблюдаемая
занятость cancelled/failed — окружение, не восстановленный latent successful S.
CPU-only и неизвестные интервалы не превращаются в нулевые работы.

| Source | Validation origins | Test origins | Warmup в каждом блоке | Validation targets | Test targets | Capacity |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| SDSC SP2 | .60, .70 | .85, .90 | 200 | 400 | 1000 | 128 allocated processors |
| Acme/Kalos | .35, .50 | .65, .70 | 100 | 300 | 300 | 2416 requested GPU |

Cutoff=first_raw_submit+fraction×raw_submit_span, без округления. Cohort — первые
warmup+targets eligible completed с submit>=cutoff, ties в порядке исходного
файла. Выбранные блоки не перекрываются: первый submit следующего строго больше
последнего предыдущего. Недостаточный cohort или пересечение — ошибка, не
повод удалить jobs/уменьшить выборку. Истории и carry/environment могут повторяться.

Fit-история — последние 4000 в submit-order среди completed с end<cutoff,
либо вся доступная история, если её меньше. Equality end=cutoff не допускается.
Минимум отдельной CDF cell=20, pooled=2. Все записанные outcomes validation
cohort, carry-in и terminal companions должны иметь end<первого test cutoff
своего источника. Если нет, отклоняется весь протокол. Не отбираются только
быстрые завершения validation, чтобы выполнить эту проверку.

После проверки metadata выполняется последовательность:

`4 validation replay → validation JSON → selection.json обоих источников → 4 test replay`

Test-scoring также начинается только после записи selection.json. Поздний fit
может использовать уже завершённые validation и более ранние test outcomes,
но не незавершённые. Модель внутри блока не обновляется. Тип выбранного кандидата
остаётся неизменным на обоих test origins.

Эти трассы и часть периодов уже анализировались в EPIC-055–060. Разделение
времени защищает алгоритм выбора, но **не создаёт нового слепого holdout**.
Completed-only отбор, исходные длительности окружения и Kalos exported type
ретроспективны; online availability полного такого набора не заявляется.

## Кандидаты и общие ленты

SDSC: `coarse`, `request_bin`, `request_ratio`. Kalos: `coarse`, `type_coarse`,
`type_exact`. Это одновременно порядок exact ties. Формулы ECDF/fallback,
фиксированные request buckets и масштабирование S/request из
[EPIC-059](feature_service.md#три-кандидата-на-источник) не меняются.
Все кандидаты показаны на test независимо от выбора; выбранные методы — ссылки,
не дополнительные прогоны или четвёртая модель.

По восемь seeds: validation 61000–61007, test 62000–62007. Для origin и seed:
NumPy default_rng(SeedSequence([seed, source_index, *cutoff_words])), где source
index SDSC=0/Kalos=1, cutoff_words — little-endian float64 cutoff как два uint32.
`U=(integers(0,2**52)+0.5)/2**52`; для sorted support x размера n берётся
`x[ceil(U*n)-1]`, без интерполяции. Ratio support умножается на target request.
U общий между кандидатами, policies и cap-сценариями на одинаковых source IDs.
Случайных draws внутри worker нет. Target observed S не входит в prediction.

Генерируется только S новых выбранных completed, включая warmup. Приходы,
K и признаки фиксированы. Boundary — первый submit выбранной когорты;
carry: submit<boundary<end; running при start<=boundary, иначе waiting.
Carry сохраняет observed S и elapsed age. Все пригодные cancelled/failed arrivals
между boundary и last selected submit включены; другие новые completed вне
выбранного cohort не добавляются, даже при совпавшем последнем timestamp.

Forecast один для всех кандидатов/seeds/scenarios: linear p90 expanding-history
coarse CDF по исходному K, fitted на cutoff. Он не обновляется по generated
completions. Для running остаток обещания max(p90−age,0); истёкшее обещание
не раскрывает скрытый остаток S. Reservation memory при replay сбрасывается.

## Сценарии и метрики

Неизменённые FCFS, FirstFit, MSF, Adaptive Quickswap, EASY, Conservative.
Основные сценарии: SDSC `carry_cancelled`, Kalos `carry_terminal`, без новых cap.
Дополнительно на **двух SDSC test origins**, не validation: `requested_limit`.
Cap=request действует на новые arrivals, включая cancelled окружение, не на carry.
S>cap приводит к timed_out после cap занятых секунд; равенство S=cap не timeout,
None не ограничивает. Это service-clock counterfactual, не исходное время отмены.

8 основных ячеек × (1 observed+3×8 generated) × 6 policies=1200 расписаний;
2 дополнительных cap-ячейки дают ещё 300. Всего **1500**, из них 60 observed.
У Kalos nominal pool — назначенный контроль ожидаемой слабой различимости
policies; фактические ties измеряются, не предполагаются.

Targets не включают warmup; дренирование полное. W=start−submit,
terminal T=release−submit. Mean/p99 и weighted T считаются на фиксированных
targets, вес всегда исходный request K. JSON имя `node_weighted_mean_release_t`
историческое; `resource_unit` уточняет processors/GPU. p99 — NumPy linear quantile,
model mean p99 — среднее восьми quantiles, не quantile объединённых выборок.
Классы отчёта: K=1/2–8/9–32/>=33; пустой класс — count=0, latency=null.

Success/timeout считаются на всех targets, successful T — только на завершившихся
успешно, с явным числом успехов; при нуле null. Поэтому сокращение terminal T
в cap-сценарии нельзя считать улучшением успешного обслуживания. Utilization и
idle-with-queue — интегралы между первым и последним measured arrival без drain;
resource work ledger включает окружение, warmup, drain и только остатки running
после boundary. Requested-resource occupancy не является hardware utilization.

## Два способа выбрать модель

Для m, validation origin o, policy p, metric k из {mean T,p99 T} сначала:

`M[m,o,p,k] = mean по 8 seeds от metric(m,o,p,k,seed)`.

Контроль R[o,p,k] — **replay** с recorded S в той же ячейке, не историческое T/W.
`Q(m) = mean по 2×6×2 ячейкам от |log(M[m,o,p,k])−log(R[o,p,k])|`.
Все 24 ячейки имеют одинаковый вес. Q dimensionless, multiplicative error
с обеих сторон симметричен. Положительные finite T обязательны; shape mismatch,
пустой массив, нули и nonfinite отклоняются, без broadcasting или epsilon.
Log difference избегает overflow при делении больших на малые числа.

Queue-aware выбирает min Q. CRPS comparator выбирает min от среднего двух
origin-level mean job CRPS, рассчитанных для full predictive CDF на measured
targets без MC. Точные float ties — порядок кандидатов выше, без tolerance.
Fixed coarse — третий метод. Выбор не оптимизирует W с нулевым denominator,
success-rate cap или классовую fairness.

Leave-one-origin-out и leave-one-seed-out повторяют **только validation выбор**,
с тем же equal-weight loss. Это описательная чувствительность; нет majority vote,
перенастройки по диагностике или CI вероятности правильного выбора.

На test сохраняются Q от MC-средних для frozen choices и иной seed-wise contrast:
L_s(m)=mean по 6×2 ячейкам от |log(metric(m,s))−log(R)|.
Парная разность D_s=L_s(queue-selected)−L_s(comparator), comparator CRPS/coarse.
Отрицательное значение лучше queue. 95% CI — mean(D)±t_.975,7 sd(D)/sqrt(8).
**Mean seed loss не тождествен Q от MC means.** Если оба сравниваемых метода
выбрали одного и того же кандидата, разности и ширина CI нулевые.
Для cap это лишь ошибка terminal-метрик;
успешность должна читаться отдельно.

Сохраняются также прежние per-policy интервалы и absolute-error differences
каждого кандидата против coarse. Все CI условны на fit, сделанном выборе,
arrivals и окружении; не оценивают переобучение selector или независимые origins.
Policy chosen по минимуму MC mean соответствующего T; exact ties — порядок
policies. Regret=observed(chosen)/min_policy observed−1. Все reference ties
по isclose(rtol=1e-12,atol=1e-9) сохраняются. Zero regret при шести ties не
валидирует выбор production scheduler.

## Coverage и drift

Для каждого кандидата и четырёх coarse K-групп: число recent-history и measured
target jobs, fallback level counts, min/median/max **числа samples** выбранной
CDF. Это не число уникальных значений. Empty group даёт null размеров, не ноль.
Predictive p90 coverage из CDF-scoring — другая величина.

TV=.5 Σ|p_history(c)−p_target(c)| считается отдельно для coarse K-группы и context,
по union категорий. History — последние ≤4000 fit jobs, не expanding forecast;
targets без warmup. Context SDSC — фиксированный request bucket, None отдельная
категория; Kalos — exported type. Сохраняются counts и число targets с context,
не встречавшимся в history. Это marginal, не joint TV, не significance test,
не доказательство причины ошибки. Ни TV, ни support coverage не меняют выбор.

## API и повторение

```python
from most_queue.random.queue_selection import select_queue_model

result = select_queue_model(
    {"coarse": [[20, 200]], "conditional": [[11, 110]]},
    [[10, 100]], candidate_order=("coarse", "conditional"),
)
assert result["selected"] == "conditional"
```

```bash
.venv/bin/python -m examples.queue_aware_selection_experiment --download \
  --output-dir works/queue_aware_selection
.venv/bin/python -m examples.queue_aware_selection_experiment --workers 6 \
  --output-dir /tmp/most-queue-selection-repeat
.venv/bin/python -m works.queue_aware_selection.audit
```

Download opt-in и hash-checked, cache не перезаписывается. CLI фиксирует параметры;
`--workers` (default=8, последовательный=1) меняет только способ выполнения.
Manifest содержит hashes исходников, 19 implementation files и девяти outputs;
сам manifest — десятый JSON. Финальная сортировка/порядок не зависят от workers.

Независимый аудитор не импортирует runner, adapters, fit или новый queue loss.
Он переиспользует независимый raw parser EPIC-059, заново строит CDF/ленты,
проверяет temporal guards, coverage, выборы и интервалы. Все observed schedules
перезапускаются на тех же dispatchers, метрики и ресурсные интегралы считаются
из events; независимая реализация scheduler не заявляется.

[Артефакты](../works/queue_aware_selection/README.md),
[результаты](research/queue-aware-selection-results-2026-10.md).
