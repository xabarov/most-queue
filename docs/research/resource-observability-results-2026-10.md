# Источники для очередей с квотами и меняющейся доступностью ресурсов

Дата завершения аудита: 2026-10-03 (Europe/Moscow).
[Эпик](../epics/EPIC-063-resource-observability-audit.md),
[методика](../resource_observability.md),
[агрегаты и воспроизведение](../../works/resource_observability/README.md).

**Решение:** для следующего ограниченного эксперимента выбрать Helios с дневными
GPU-counts по виртуальным кластерам (VC). Для точного production replay —
**no-go**: реальные hard quotas, внутридневные изменения, borrowing и размещение
не восстановлены. Полный аудит выявил работу при нулевых daily VC-counts, поэтому
механическая замена номинального пула на эти counts не воспроизводит исходную систему.

Исследованы шесть кандидатов. Raw-аудит выполнен только для Helios, на всех
четырёх опубликованных кластерах. Для остальных решение основано на закреплённой
документации и схемах: оно не удостоверяет качество строк или полноту данных.
Платные запросы, формы, внешняя переписка и загрузка многогигабайтных архивов
не выполнялись. В этом эпике нет scheduler runs или результатов сравнения policies.

## Матрица наблюдаемости

«Нет в схеме» ниже означает отсутствие в проверенной публикации, не доказательство
отсутствия внутреннего поля у оператора. Submission-time provenance запросов
требует отдельной проверки даже при наличии request column.

| Источник | Request и жизненный цикл | Квоты и доступность | Размещение | Допустимый следующий шаг |
| --- | --- | --- | --- | --- |
| Helios | GPU/CPU/node counts, submit/start/end, terminal state, VC | GPU-count каждого VC/day; нет точных enforcement/events | Node count без node IDs | Bounded daily-capacity сценарий; raw проверен |
| Philly | Submission и несколько execution attempts; GPU list — факт исполнения, не доказанный submission request | VC label без численных квот; per-minute NA у telemetry обозначает offline | Machine/GPU IDs каждой попытки | Attempt/placement audit, не online K-fit из будущего allocation |
| Alibaba PAI 2020 | Job/task/instance, planned fractional GPU, разные значения start_time | Static machine specs; metrics усреднены по жизни instance, не availability events | Instance→machine и sensor GPU | Многоуровневый adapter, не rigid job через один end−start |
| Alibaba GPU 2023 | Pod requests/QoS/phase, creation/scheduled/deletion | Static inventory; нет VC quota/event timeline | GPU-type requirements; опубликованные таблицы не дают фактический pod→node binding | Packing/fragmentation; не successful-service replay по deletion |
| Alibaba GPU 2026 | Hourly pods, fractional requests, execution-span summary | Hourly inventory, HP/LP; нет tenant quotas и точных arrival/start/end | Hosting server и ASW/rack domain по часу | Hourly resource-envelope, не точный event replay |
| Google 2019 | Collection/instance events, updates requests/constraints, missing flags | Machine ADD/REMOVE/UPDATE и capacity; alloc sets не равны всем tenant quotas | Machine IDs, attributes/constraints, switch | Отдельный CPU/RAM event adapter; не GPU-эквивалент |

Первичные источники для строк: [Helios schema](https://github.com/S-Lab-System-Group/HeliosData/blob/159f0caeec16600b9b6017862952a36aae01c43f/README.md),
[Philly schema](https://github.com/msr-fiddle/philly-traces/blob/29a1b87fa2d9ed80b83c9e3a37f3a88d382b031d/README.md),
[PAI schema](https://github.com/alibaba/clusterdata/blob/cb65b488983ce23efb09eb891f796c214e0a3fd3/cluster-trace-gpu-v2020/README.md),
[GPU 2023 schema](https://github.com/alibaba/clusterdata/blob/cb65b488983ce23efb09eb891f796c214e0a3fd3/cluster-trace-gpu-v2023/README.md),
[GPU 2026 schema](https://github.com/alibaba/clusterdata/blob/cb65b488983ce23efb09eb891f796c214e0a3fd3/cluster-trace-gpu-v2026/docs/schema.md),
[Google protobuf](https://github.com/google/cluster-data/blob/48b12446464b0422abcc18e0ec5b0b13e2f3a90c/clusterdata_trace_format_v3.proto).

Важные оговорки при переносе: у PAI `job.start_time` — submission, а
`task.start_time` — launch; плановый GPU задан в процентах. У Philly возможны
отсутствующие попытки и границы, незавершённый последний attempt; telemetry
пример содержит PDT/PST, а attempt clocks без зоны. У GPU 2023 варианты
`gpuspec` дополняют tasks type requirements, их нельзя без проверки считать
оригинальным журналом всех production constraints. Эти ограничения следуют
из соответствующих закреплённых схем выше, не из raw измерений данного эпика.

## Доступ и условия использования

| Источник | Проверенная версия | Условия и стоимость доступа |
| --- | --- | --- |
| Helios | `159f0caeec16600b9b6017862952a36aae01c43f` | [CC-BY-4.0](https://github.com/S-Lab-System-Group/HeliosData/blob/159f0caeec16600b9b6017862952a36aae01c43f/LICENSE.txt); официальный ZIP 36 437 672 bytes прочитан полностью |
| Philly | `29a1b87fa2d9ed80b83c9e3a37f3a88d382b031d` | [CC-BY-4.0](https://github.com/msr-fiddle/philly-traces/blob/29a1b87fa2d9ed80b83c9e3a37f3a88d382b031d/LICENSE); LFS-pointer содержит размер 1 055 988 361 bytes; архив не скачивался |
| Alibaba PAI 2020 | `cb65b488983ce23efb09eb891f796c214e0a3fd3` | Собственная [CC-BY-4.0](https://github.com/alibaba/clusterdata/blob/cb65b488983ce23efb09eb891f796c214e0a3fd3/cluster-trace-gpu-v2020/LICENSE); отдельные таблицы и опубликованные checksums, raw не проверялся |
| Alibaba GPU 2023/2026 | Тот же Alibaba commit | [Корневой README](https://github.com/alibaba/clusterdata/blob/cb65b488983ce23efb09eb891f796c214e0a3fd3/README.md) разрешает research/study; отдельная лицензия этих поддеревьев не найдена. Не переносить CC-BY PAI или MIT проекта автоматически |
| Google 2019 | `48b12446464b0422abcc18e0ec5b0b13e2f3a90c` | [CC-BY-4.0 и доступ](https://github.com/google/cluster-data/blob/48b12446464b0422abcc18e0ec5b0b13e2f3a90c/ClusterData2019.md); около 2.4 TiB compressed, опубликованный путь через BigQuery. Запросы и расходы не запускались |

Это запись найденных условий, не юридическое заключение. Attribution Helios:
SenseTime / S-Lab-System-Group, Hu et al.,
[SC 2021](https://doi.org/10.1145/3458817.3476223). Данные предоставлены as-is;
оператор не подтверждает наши выводы. Raw не включён в git; преобразования —
агрегирование, проверки, requested-work интегралы и hashes.

## Полный аудит Helios

SHA-256 архива:
`3d22a5f6c0ae669e2fcbfe4200fa9c48664507bc397c677bad8f085222c032ac`.
Счётчики ниже получены нашим кодом из восьми payloads, не переписаны из README.
3 362 981 row, из них 1 580 464 GPU>0 и 1 782 517 CPU-only. Job IDs уникальны
внутри трёх clusters; в Saturn один повтор ID у двух различающихся CPU-only
строк. Это не точная копия записи; дубликаты не удалялись.

| Cluster | Все rows | GPU>0 | Completed GPU с положительным S | Число VC в daily table | Daily total min…max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Earth | 872886 | 427148 | 313953 | 25 | 784…1232 |
| Saturn | 1753078 | 698896 | 406982 | 28 | 2056…2104 |
| Uranus | 490309 | 329117 | 199930 | 25 | 2072…2256 |
| Venus | 246708 | 125303 | 65912 | 25 | 968…1264 |

Во всех строках timestamps заполнены и submit<=start<end; duration=end−start,
queue=start−submit без расхождений. CPU/GPU/node counts валидны. Но заполненный
end не доказывает завершение: Uranus содержит **76 SUSPENDED и 2 RUNNING**, из них
53 и 1 GPU jobs. Эти 54 GPU rows исключены из terminal occupancy, но сохранены
в общем аудите; README упоминает лишь один SUSPENDED. Raw release имеет приоритет
над этой описательной оговоркой. У Venus два VC вне daily columns, оба встречаются
только в CPU-only rows; неизвестных VC среди GPU rows независимая сверка не нашла.

Каждая конфигурация содержит **181 последовательный день**, 2020-04-01…09-28;
пропусков дат и расхождений ΣVC с total нет. Число изменений total:
Earth/Saturn/Uranus/Venus = 12/7/8/5; отдельных VC-counts = 63/48/28/27.
Дневные строки не покрывают все clocks: приход/старт встречаются с марта,
последние arrivals — 09-27, некоторые starts — 09-29, ends доходят до 10-10.
Границы относятся к разным полям; это не шестимесячный интервал полных
submission и execution observations с известным initial state.

### Сопоставление с дневной конфигурацией

Для всех GPU rows берётся count того же VC на дату start. Нет подстановки
последней известной или будущей даты. «Выше count» включает нулевые значения.

| Cluster | Start-date сравнения | Нет start-date | Старт при count=0 | Request выше daily count |
| --- | ---: | ---: | ---: | ---: |
| Earth | 427112 | 36 | 6727 | 6789 |
| Saturn | 698643 | 253 | 3420 | 3420 |
| Uranus | 328919 | 198 | 2918 | 2918 |
| Venus | 125219 | 84 | 4394 | 4398 |

Суммарно **17 459** GPU rows стартуют при count=0 и **17 525** требуют больше
указанного на этот день. Это не приговор scheduler: ежедневная конфигурация
может не быть hard quota на каждую секунду. Варианты объяснения — intraday
transitions, borrowing, семантика принадлежности или качество экспорта — пока
гипотезы, не выявленные причины. Ужесточать counts до hard admission нельзя молча.

Для closed terminal GPU rows отдельно интегрирована requested occupancy при
гипотезе постоянного дневного count. Единицы work — GPU-s; VC-time суммирует
превышения по VC, не объединяет их на временной оси.

| Cluster | Полная work, GPU-s | Work на датах конфигурации | ΣVC excess, GPU-s | ΣVC excess, VC-s | VC с excess |
| --- | ---: | ---: | ---: | ---: | ---: |
| Earth | 11439202755 | 11292501629 | 86158112 | 2123911 | 16 |
| Saturn | 27652361263 | 27323001256 | 134120362 | 1393431 | 19 |
| Uranus | 26543219559 | 26033077864 | 263557525 | 2067820 | 11 |
| Venus | 12345612608 | 12135923931 | 282188402 | 5066804 | 13 |

При этом **aggregate-pool excess=0 во всех четырёх clusters** на опубликованных
датах. Согласованность общего пула не распространяется на изолированные VC.
Она не доказывает физическую utilization, отсутствие пропущенной нагрузки
или корректность входного журнала за пределами covered dates.

Completed GPU mean S по Earth/Saturn/Uranus/Venus = 3571.23/4334.95/7685.49/11942.44 s;
mean recorded W = 247.37/574.37/2427.18/1601.14 s. Положительное historical W
здесь есть, но это ещё не reference, различающий шесть policies Most-Queue:
контрфактические расписания не вычислялись. Нельзя объявлять nonzero W доказательством
валидности нового replay или использовать его для подбора capacity.

## Решение и следующие работы

Наша оценка ценности/сложности, а не вывод владельцев данных:

| Направление | Ценность | Сложность | Решение |
| --- | --- | --- | --- |
| Helios daily capacity и VC | Близко к текущему MSJ, компактные проверенные данные | Средняя: календарь, границы и nonterminal accounting | Следующий bounded эпик |
| Philly attempts/placement | Реальные node/GPU assignments и повторные исполнения | Высокая: большой архив, clocks, requests не отделены от allocation | Отдельная ветка после source audit |
| PAI instance-level fractional GPU | Совместные CPU/RAM/GPU и sharing | Высокая: multi-role DAG, fractional resources, heterogeneous service | Не втискивать в scalar rigid K |
| Alibaba 2023/2026 packing/inventory | Современные sharing/topology и temporal inventory | Средняя/высокая, иной estimand и условия данных | Самостоятельное исследование, не замена service trace |
| Google event-driven availability | Точные типы lifecycle/resource events в схеме | Высокая: CPU/RAM, missing flags, доступ/стоимость | При отдельном решении о BigQuery и scope |

Следующий [EPIC-064](../epics/EPIC-064-msj-capacity-calendar.md) должен добавить
отдельный opt-in календарь capacity и контракт VC scopes, затем выполнить
заранее заданную sensitivity-проверку Helios. При уменьшении ёмкости явно
определить судьбу уже запущенных работ; не убивать их для устранения превышения.
Недостижимые requests и выход за известный календарь — отдельные результаты,
не бесконечный drain и не незаметное удаление jobs. Сохранять фиксированный
общий пул и записанный S как контроли; не выбирать pool/boundary по observed W.

До production claims недостаёт: intraday quota events с effective timestamps,
borrowing/reservation rules, node placement и GPU types, snapshot backlog,
обновления request/VC и их submission-time provenance, attempt/preemption logs,
наблюдения о lost/censored jobs. Нельзя склеивать Helios/Philly/Alibaba IDs,
подставлять чужие квоты или превращать длину неуспешного execution в successful S.

## Проверки

Добавлены строгий аудитор и 34 offline-теста: календарные границы, half-open
интервалы, точные work/excess, отсутствующие даты и VC без imputation,
раздельные статусы, malformed schemas, duplicate IDs, SHA-256/cache opt-in,
allowlisted ZIP и агрегированные outputs без raw IDs. Повтор и независимая
сверка фиксируются в эпике после окончания; результаты suite также там.
