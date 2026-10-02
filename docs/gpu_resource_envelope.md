# Ресурсные ограничения и ёмкость GPU пула

[EPIC-060](epics/EPIC-060-gpu-resource-envelope.md) проверяет, как меняются очереди
и выбор дисциплины при уменьшении пула и эксклюзивном занятии узлов. Это
**предписанная sensitivity-модель**, не восстановленная effective capacity.
Ни один размер пула не выбирается по близости к историческому ожиданию.
Существующие шесть диспетчеров не меняются; меняются только их capacity и demands.

## Что известно из источника

Используется тот же [Acme/Kalos](modern_gpu_trace.md), commit
`f9fdf591b4876c2875a9e3d28adb1bda8120dfcb`, SHA-256
`7c5a7845da1d66448fa668342e4ecb622fd9a005118080660ec3ad2788d4c9bf`.
Shanghai AI Laboratory / InternLM; [CC-BY-4.0](https://github.com/InternLM/AcmeTrace/blob/f9fdf591b4876c2875a9e3d28adb1bda8120dfcb/LICENSE.txt),
не MIT пакета. Raw CSV остаётся в `.cache`, публикуются агрегаты и hashes.

В [схеме job trace](https://github.com/InternLM/AcmeTrace/blob/f9fdf591b4876c2875a9e3d28adb1bda8120dfcb/README.md)
есть requested GPU/node/CPU counts, memory configuration и timestamp/outcomes.
`queue` означает время ожидания, **не идентификатор очереди**. Hashed user не
свидетельствует о tenant quota. Для выбранного файла нет per-job node IDs,
quota/partition membership, событий available/unavailable или изменений квот.
Упомянутая авторами utilization telemetry не равнозначна доступному планировщику
ресурсу; многогигабайтные логи в этот эпик не загружаются.

Следовательно, ёмкость и квоты здесь не идентифицированы. Длительность ожидания
может включать неизвестные ограничения и задержки; из наличия wait нельзя
единственным образом вывести число доступных GPU. `node_num` используется
как запрос числа узлов, а не карта реально занятых физических машин.

## Два допущения о резервировании

| Модель | Demand диспетчера | Capacity | Смысл |
| --- | --- | --- | --- |
| `gpu_pool` | исходный requested GPU K | 8N GPU | Любые свободные единицы общего однородного пула совместимы |
| `exclusive_nodes` | записанный node_num | N nodes | Каждый запрошенный узел резервируется целиком, даже при K<8×node_num |

Обе модели используют прежнее допущение восьми GPU на узел. `ResourceRequest`
проверяет положительные целые K/node counts и K≤8×nodes. Число узлов **не**
заменяется ceil(K/8). Например, K=1 и nodes=2 означает резервирование двух узлов
в exclusive-сценарии. Ни sharing, ни topology, CPU/memory placement, сетевые
ограничения, неоднородные узлы или quota fairness не реконструируются.

Заранее задан N∈{302,226,151,113}: GPU inventories 2416,1808,1208,904.
Меньшие значения получены floor(302×[.75,.5,.375]), не оптимизацией по test W.
Шкала одинакова физически для обеих моделей, единицы scheduler capacity различны.
Модель не генерирует outages и не отнимает ресурс у уже начавшейся работы.

## Что означает невыполнимая ячейка

До replay проверяются все новые и waiting/running jobs. Если хотя бы один demand
больше capacity или сумма initial running demands больше capacity, вся ячейка
`infeasible`. В `envelope.json` остаются причина, oversized counts, максимальный
request и начальная занятость. Jobs не удаляются, K не уменьшается, начальное
состояние не сбрасывается. В результатах нет строк расписаний этой ячейки:
отсутствие результата не заменяется нулевой задержкой или нулевым успехом.

Суммарный demand waiting/arrivals может превышать capacity: это очередь, которая
дренируется последовательно. Feasibility означает возможность выполнения
фиксированных rigid reservations при этих условиях, не стационарную устойчивость.

Отдельный `observed_occupancy` анализирует **записанные** [start,end) интервалы
всех 19 902 пригодных GPU jobs. Одновременные releases/starts обрабатываются
атомарно; горизонт от первого accepted start до последнего accepted end, включая
промежуточные idle gaps. Выводятся peak, single-job maximum, интеграл работы,
время и excess work выше каждого capacity. Совместимость записанного пути
с размером пула не доказывает, что именно этот размер был доступен.

Возможен feasible counterfactual, несовместимый с recorded intervals: replay
сдвинет старты, оставив S. Поэтому две проверки нельзя объединять. Условие
initial-running feasibility всё равно сохраняется: занятые в boundary работы
не разрешено передвинуть задним числом.

## Когорты и начальное состояние

Формула cutoff: first_raw_submit+f×(last_raw_submit−first_raw_submit), float seconds
UTC без округления. Origins f=.35/.50/.70. Первые два используют первые 1200
eligible completed arrivals после cutoff: 200 warmup, 1000 targets. Последний —
первые 400: 100 warmup, 300 targets. Ties сохраняют порядок CSV. Блоки не
перекрываются между собой, но уже встречались в предыдущих эпиках — это не
новый blind holdout. По факту completed отбор ретроспективен.

Eligibility неизменно из [EPIC-058](modern_gpu_trace.md): положительный целый
GPU request, согласованные node requests, known positive end-start и terminal
outcome. CPU-only, отсутствующие окончания и нулевые интервалы исключаются с
аудитом; fractional GPU не округляются. Только COMPLETED учат successful S.

Boundary — первый selected submit. Carry jobs: submit<boundary<end; start≤boundary
означает running, иначе waiting. Running занимает исходный остаток S−age,
waiting — исходный S после нового старта. Все пригодные non-completed arrivals
в [boundary,last_selected_submit] также включены. На последнем timestamp
дополнительные completed за пределами fixed cohort не добавляются; terminal
competitors на обеих границах включены. Source IDs/order неизменны между ячейками.

Snapshot и будущие labels частичные и ретроспективные. Для running S/age всегда
записанные: sampled S≤age не возникает. Ресэмплируется только выбранная completed
arrival когорта, включая warmup. Окружение хранит записанное service-to-outcome;
отказ не превращается в success demand. Его абсолютный исторический release не
навязывается новым arrivals: после нового старта действует service clock.
Reservations/Adaptive phase начинаются заново. Runtime caps отсутствуют.

## Модель обслуживания и forecasts

Один baseline `recent_coarse`, без выбора новой модели. Train: только completed
end<cutoff, последние 4000 **общей истории** в submit-порядке, не 4000 на группу.
CDF S по исходным K-группам 1/2–8/9–32/≥33; если группа имеет <20 наблюдений,
pooled CDF этого же окна. Pooled требует минимум два. Ресурсная проекция не
перегруппировывает обучение по nodes. Никакого масштабирования service по capacity.
Инвариантность S при смене резервирования — допущение: влияние CPU/memory/I/O
contention на скорость исполнения здесь не моделируется.

Seeds 60000–60007. NumPy default_rng с SeedSequence([seed,*cutoff_words]), где
cutoff_words — два little-endian uint32 слова float64 cutoff. В submit-порядке
рисуется U=(randint[0,2^52)+0.5)/2^52, затем inverse ECDF x_[ceil(Un)−1].
Внутри variant/seed/origin лента S одна на все ресурсы, capacity и policies.
Независимы U; условные CDF могут быть различны, поэтому S не обязательно iid.

Forecast — линейный p90 expanding-history coarse CDF на cutoff, по **всей** completed истории end<cutoff,
с тем же minimum=20/fallback. Численные значения для исходного K заморожены
на весь replay во всех сценариях/seed; от generated completions не обновляются.
Для carry-running остаток обещания max(p90−age,0). Истёкшее обещание означает
неизвестный release, не знание фактического остатка.

Observed — replay с записанными S, **не исторические W/T**. FCFS, FirstFit, MSF,
Adaptive Quickswap, EASY и Conservative прежние. Проекция меняет видимый demand
для packing и reservations, не раскрывает S или labels. Изменение resource
granularity может менять действие эвристики: это часть данного сценария.
В event engine releases до dispatch обрабатываются вместе с arrivals того же
timestamp; исходные ties arrivals сохраняются.

Предварительный envelope audit: 24 resource cells, 20 feasible, четыре infeasible
(.50/.70 N=113, обе модели).
В каждой feasible cell (1 observed+8 generated tapes)×6 policies=54 расписания;
запланировано и выполнено **1080**. До первого replay `envelope.json` фиксирует всю матрицу.

## Метрики и единицы

Targets фиксированы; T=release−submit, W=start−submit. Mean/p99 считаются после
полного drain, без warmup jobs в знаменателе. Все targets completed, non-success
окружения учитываются отдельно. Weighted T=ΣK_i T_i/ΣK_i, вес всегда **исходный
GPU request**, не node demand. Групповые T/W/count также по исходному K.
p99 — NumPy linear quantile; итог модели — mean восьми отдельных p99,
не quantile объединённых реализаций.

Observation window: arrival первого target до arrival последнего target,
без drain. В этом окне независимо вычисляются:

- Reserved utilization: ∫reserved_units(t)dt / (scheduler_capacity×window).
- Requested-GPU utilization: ∫Σactive K_i dt / (8N×window).
- Idle-with-queue: ∫free_reserved_units(t)×1[waiting>0]dt / (scheduler_capacity×window).

Полный ledger от boundary до drain включает окружение, warmup и остатки
initial-running. Сохраняются reserved units-seconds по outcomes,
reserved GPU-equivalent time (×8 для nodes) и requested GPU-seconds.
Их разность — slack выбранного reservation-допущения, **не** измеренная загрузка
GPU/SM. Интегралы окна не сверяются с full-drain ledger как будто горизонты равны.

Ошибки recent_coarse — относительно observed той же resource/capacity ячейки.
Relative error=mean(model)/observed−1, при observed=0 null. Общий nominal GPU
reference используется **отдельно**, для resource contrasts в секундах:
metric(resource_cell,seed)−metric(gpu_pool,302,seed). Observed difference
детерминирован, без CI; model difference парный по seed. Resource scenarios
не считаются независимыми наблюдениями.

95% t-интервал mean±t_.975,7×sd/√8 условен на fit, истории, выбранных arrivals и
окружении. Не оценивает неопределённость истинной capacity. Policy chosen по
минимуму MC mean T либо mean p99 T; exact ties — порядок перечисленных policies.
Reference best содержит все isclose(rtol=1e-12,atol=1e-9) ties. Regret =
observed(chosen)/min_policy observed−1 внутри той же cell. Шесть reference ties
означают неразличающий контроль, а не шесть подтверждений правильного выбора.

## API и воспроизведение

```python
from most_queue.sim.utils.resource_envelope import ResourceRequest, assess_envelope

request = ResourceRequest(gpus=1, nodes=2)
assert request.demand("gpu_pool") == 1
assert request.demand("exclusive_nodes") == 2
assert not assess_envelope(1, [request.demand("exclusive_nodes")])["feasible"]
```

```bash
.venv/bin/python -m examples.gpu_resource_envelope_experiment --download \
  --output-dir works/gpu_resource_envelope
.venv/bin/python -m examples.gpu_resource_envelope_experiment \
  --output-dir /tmp/most-queue-resource-repeat
.venv/bin/python -m works.gpu_resource_envelope.audit
```

Download opt-in и hash/size-checked. [Manifest](../works/gpu_resource_envelope/manifest.json)
закрепляет source/config/environment/code/output hashes. CSV хранится вне git.
Независимые расписания выполняются в отдельных процессах (`--workers`, default=8),
без случайных draws внутри worker; порядок строк фиксирован. Последовательный
режим `--workers 1` имеет те же результаты, что проверяется offline-тестом.
Аудитор заново читает raw через отдельный independent parser EPIC-059, строит CDF,
проекцию, envelope и service tapes без experiment/adapter/fit helpers. Проверяет
все интервалы/choices, а observed schedules — повторным вызовом тех же
dispatchers с независимым вычислением T/W и интегралов по events. Независимой
реализации самого алгоритма планирования не заявляется.

[Результаты](research/gpu-resource-envelope-results-2026-10.md) отделены от протокола.
