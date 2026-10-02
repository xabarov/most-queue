# Совместная генерация приходов и ресурсных признаков

[EPIC-062](epics/EPIC-062-joint-marked-arrivals.md) добавляет эмпирический
генератор `(Δ,K,context,request)` и проверяет его на SDSC SP2 и Acme/Kalos.
Δ — интервал от предыдущего прихода, K — одновременно занятый ресурс.
Длительность S генерируется отдельно по общей для вариантов coarse ECDF:
`P(Δ,K,context,request) × P(S|K)`. Context/request сохраняются вместе с K,
но здесь не влияют на S, runtime cap или scheduler. Это не полная совместная
модель S/признаков и не новый алгоритм диспетчеризации.

Временной контракт следует принципу обучения на прошлом из
[rolling-origin evaluation](https://otexts.com/fpp3/tscv.html). Круговые блоки
используют end-to-start wrap, как в описании
[circular block bootstrap](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.CircularBlockBootstrap.html).
Реализация своя, без зависимости от arch; это генератор нагрузки, не применение
теоремы о стационарности или доверительном интервале по исходной популяции.

## Популяция и история

Все сценарии **completed-only, empty start**. Источники, закреплённые версии,
SHA-256 и eligibility прежние: [SDSC](real_trace_calibration.md),
[Kalos](modern_gpu_trace.md). SDSC S=runtime, Kalos S=end−start, не duration.
Raw данные не включаются в git; SDSC имеет условия NPACI JOBLOG, Kalos —
CC-BY-4.0, ни один источник не становится MIT вместе с кодом проекта.

| Источник | Доли raw submit span | Warmup | Targets | Capacity |
| --- | --- | ---: | ---: | ---: |
| SDSC SP2 | .85, .90 | 200 | 1000 | 128 processors |
| Acme/Kalos | .65, .70 | 100 | 300 | 2416 requested GPU |

Cutoff=first_raw_submit+fraction×raw_submit_span, без округления. Cohort —
первые warmup+targets eligible completed с submit>=cutoff, ties в source order.
Первый submit следующего cohort строго больше последнего предыдущего; нехватка
данных или пересечение отклоняют конфигурацию, а не уменьшают counts.
Origins могут разделять историю; их статистическая независимость не предполагается.

Arrival history — **полностью разрешённый префикс** eligible completed с
submit<cutoff, заканчивающийся перед первым end>=cutoff. Весь последующий
суффикс исключается, в том числе уже завершённые jobs. Так Δ образуется из
действительно соседних arrivals этой выбранной популяции, без соединения времён
через поодиночке удалённые незавершённые работы. Первая строка нужна только как
предыдущий timestamp; N строк дают N−1 donor tuples. Recent берёт последние
4000 tuples (4001 prefix job), expanding — все. Нулевые интервалы сохраняются.
История короче 20 donors отклоняется; блок автоматически не укорачивается.

Это консервативное правило может сильно состарить поток: `prefix_audit` хранит
lag=cutoff−last_prefix_submit, размер исключённого suffix и число уже известных
outcomes в нём. Completed-label и Kalos exported type ретроспективны. Защита
от использования будущего S не делает весь информационный набор online-доступным.

Service fit имеет **отдельную** историю: последние 4000 всех completed с
end<cutoff, включая известные jobs после остановки arrival prefix. Coarse
K-группы: 1, 2–8, 9–32, >=33; минимум 20 samples, иначе pooled fallback
(минимум 2). Forecast один для всех вариантов: linear p90 expanding completed
history по тем же K-группам, замороженный до replay. Прогнозы не обновляются
по generated completions. Held-out S не участвует в fit или forecasts.

Нет carry-in, cancelled/failed окружения, runtime enforcement или подгонки
capacity. Историческое окружение нельзя автоматически перенести на новые
synthetic timestamps. Эти очереди нельзя численно сравнивать с EPIC-061 как
с одинаковыми начальными состояниями. Warmup не доказывает стационарности.

## Варианты и согласованные контроли

| Имя | Генерация | Что меняется относительно контроля |
| --- | --- | --- |
| observed | recorded arrivals/K/marks/S | общий empty-start reference |
| fixed_coarse | recorded arrivals/K/marks, coarse S | только S, обязательный baseline |
| recent_joint_iid | iid tuples из recent prefix, coarse S | joint law одного donor |
| recent_gap_independent | те же Δ, переставленные K/context/request bundles | связь Δ с bundle, не K с context |
| recent_joint_block20 | circular blocks 20 tuples, iid innovations S при K | порядок соседних tuples |
| recent_joint_shuffle20 | переставленные готовые block20 tuples, **включая S** | порядок при том же multiset/work |
| expanding_joint_iid | iid tuples всего prefix, прежний recent service fit | только окно arrival fit |

Context SDSC — фиксированный request bucket из [EPIC-059](feature_service.md),
вместе с исходным numeric request; Kalos — exported type и request=None.
Отсутствующий контекст/запрос — None, не нулевая длительность.

Перестановки раздельны внутри `[0,warmup)` и `[warmup,total)`. Первая строка
каждой части закреплена, остальные сортируются по независимым uniform keys
со stable ties. В gap-independent переносятся только целые mark bundles;
S затем строится по новому K с прежним positional U. Поэтому inventory marks
совпадает с joint_iid, а resource work не обязан. В block/shuffle сохраняется
точно весь inventory обеих частей, включая S, их work, первый measured arrival
и measured horizon. Меняется также warmup-order, следовательно начальная очередь
в момент измерения может отличаться. Это общий эффект порядка, а не перестановка
только targets из одинакового состояния. Shuffle — не независимая iid-выборка.
Wrap и границы блоков создают искусственные соседства.

`arrival_times = cumsum(Δ) − Δ[0]`. Первый job в нуле, zero gaps — batch без jitter.
Observed/fixed используют submit−first_cohort_submit. Все одновременные arrivals
попадают в очередь до dispatch; ties synthetic — generated position order,
observed — source order. Новые jobs имеют синтетические позиции, **не source IDs**.
Нет усечения до исторического last-submit, подгонки rate или масштаба времени.
Horizon и состав классов могут отличаться от reference. Изменение только
arrival-fit окна не означает одинаковых realized S или resource work при новом K.

## Ленты случайных величин и метрики

По восемь seeds 63000–63007. Для каждого origin:
`SeedSequence([seed, source_index, *cutoff_words]).spawn(4)`, SDSC=0/Kalos=1,
cutoff_words — little-endian float64 как два uint32. Четыре child default_rng
для donors, S, mark permutation, block permutation соответственно.
`U=(integers(0,2**52)+.5)/2**52`. Donor start=floor(U*n); block starts используют
позиции 0,20,40… с modulo wrap и усечением до total jobs. Coarse inverse ECDF:
sorted_support[ceil(U*n)−1]. Policies получают одну ленту; workers ничего не сэмплируют.

FCFS, FirstFit, MSF, Adaptive Quickswap, EASY, Conservative неизменны. Все jobs
drain. Targets исключают warmup; T/W в секундах, K-weighted T=ΣKT/ΣK. Для четырёх
K-групп counts/mean/p99 T; пустая группа count=0, mean/p99=null. P99 — linear
sample quantile в каждой репликации, затем среднее quantiles, не pooled quantile.
Utilization и idle-with-queue усредняются по времени от первого до последнего
target arrival, без drain. Последняя величина — доля доступного ресурса,
простаивающая при непустой очереди. Полная resource work=ΣK×S по всем jobs,
включая warmup; runner считает её по исполнению, аудитор сверяет с ΣKS. Единицы — processor-s
либо requested-GPU-s, не реально измеренные GPU cycles.

Diagnostics по targets: arrival span, mean/SCV/zero fraction gaps (без первого
target gap), lag-1 Δ и K, Pearson Δ↔K, coarse K/context/joint K-group×context TV
против observed targets, class counts, mean/p99 S, target и all-job work.
TV=.5Σ|p_generated−p_reference| по union категорий. SCV — population variance/mean².
Transition rate=(target_count−1)/span. Нулевой span даёт null rate/time averages,
но latency после drain определена; нулевой mean gap даёт null SCV. Корреляции
при недостатке samples или нулевой дисперсии null. Эти показатели не доказывают
устойчивости очереди или причинного объяснения её ошибки.

Главная описательная ошибка Q отдельно по origin — среднее 12 абсолютных
log-errors (6 policies×mean/p99 T), сначала MC mean восьми репликаций, затем
`abs(log(model)−log(observed))`. Seed-level L вычисляется до усреднения seeds.
Парные 95% t-CI по восьми L(a)−L(b), df=7: joint_iid−gap-independent,
block20−shuffle20, recent_iid−expanding_iid и каждый из пяти новых generators
против fixed_coarse. Отрицательный контраст лучше для левого варианта.
Это **не разность Q**, не CI по origins и не оценка production causal effect.
Нет поправки на множественные сравнения или выбора победителя после test.

Policy choice отдельно по MC mean mean T и p99 T; точные model ties — в порядке
policies выше. Все observed ties сохраняются, isclose(rtol=1e-12,atol=1e-9).
Regret=observed(chosen)/min(observed)−1. Нулевой regret при шести ties ничего
не подтверждает. Цели fixed и synthetic могут иметь разные class counts и
weighted denominators, поэтому сравнение не является job-by-job paired effect.

## API, повторение и проверка

```python
from most_queue.random.marked_arrivals import MarkedArrivalBootstrap, arrival_times

model = MarkedArrivalBootstrap.fit(
    arrivals=[0, 0, 4, 9], needs=[1, 2, 1, 4],
    contexts=["small", "wide", "small", "wide"],
)
records = model.sample([0.1, 0.9, 0.5, 0.2], block_length=2)
assert arrival_times(records).tolist() == [0.0, 4.0, 8.0, 13.0]
```

API строго проверяет поля, open-interval uniforms, chronological timestamps,
полноту permutations и переполнение cumulative times. Сам API не знает cutoff:
temporal eligibility и правильная adjacency — ответственность вызывающего кода.

```bash
.venv/bin/python -m examples.joint_marked_arrivals_experiment --download --output-dir works/joint_marked_arrivals
.venv/bin/python -m examples.joint_marked_arrivals_experiment --workers 6 --output-dir /tmp/most-queue-joint-repeat
.venv/bin/python -m works.joint_marked_arrivals.audit
```

Download opt-in, hash-checked, существующий cache не перезаписывается.
`protocol.json` с metadata всех origins записывается до первого replay.
Четыре origins×(1 observed+6 variants×8 seeds)×6 policies=**1176 schedules**,
из них 24 observed. Manifest хранит 20 implementation hashes и пять output
hashes; всего шесть JSON с manifest. `--workers` меняет выполнение, не протокол.

Аудитор не импортирует runner, adapters или новый generator. Он использует
независимый raw parser EPIC-059, заново строит prefix/CDF/PRNG/tuples, hashes
и diagnostics, проверяет matched inventories, losses/CI/decisions. Все 24
observed schedules пересчитываются теми же dispatchers, метрики и интегралы
из events проверяются отдельно. Второй независимый scheduler не заявляется.

[Артефакты](../works/joint_marked_arrivals/README.md),
[результаты](research/joint-marked-arrivals-results-2026-10.md).
