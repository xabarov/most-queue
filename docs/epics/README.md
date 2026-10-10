# Эпики Most-Queue

Папка с эпиками — крупными направлениями разработки. Каждый эпик — отдельный файл
`EPIC-NNN-<slug>.md` с последовательной нумерацией.

## Процесс

1. Новое направление → новый файл эпика по шаблону ниже, статус `proposed`.
2. Эпик декомпозируется на задачи (чек-лист внутри файла). Детальный технический план
   при необходимости выносится в `docs/roadmaps/` (взаимные ссылки обязательны).
3. В работе — статус `in progress`; по завершении — `done` + раздел «Результаты».
   Критерии готовности — [../DOD.md](../DOD.md).
4. Статусы задач отмечаются прямо в чек-листе эпика; история — через git.

## Шаблон эпика

```markdown
# EPIC-NNN: <Название>

- **Статус:** proposed | in progress | done | dropped
- **Создан:** YYYY-MM-DD
- **Roadmap:** ссылка на docs/roadmaps/... (если есть)

## Цель
Зачем это нужно и что изменится в библиотеке.

## Контекст
Текущее состояние, источники, ограничения.

## Задачи
- [ ] ...

## Критерии готовности (DoD эпика)
Специфичные для эпика критерии + общий DoD.

## Результаты
Заполняется по завершении.
```

## Реестр эпиков

| № | Эпик | Статус |
|---|------|--------|
| [EPIC-001](EPIC-001-queueing-models-gap-analysis.md) | Исследование: модели теории очередей vs реализованные | done |
| [EPIC-002](EPIC-002-wave1-exact-models.md) | Волна 1: точные базовые модели (Erlang B/C, M/G/∞, GI/G, vacations, PS/FB, breakdowns) | done |
| [EPIC-003](EPIC-003-qbd-map-ph.md) | QBD/MAP/PH-стек (матрично-аналитические методы) | done |
| [EPIC-004](EPIC-004-retrial-erlang-a.md) | Retrial-очереди и Erlang-A | done |
| [EPIC-005](EPIC-005-illustrated-catalog.md) | Иллюстрированный каталог моделей (простые описания + схемы) | done |
| [EPIC-006](EPIC-006-english-docs.md) | EN-центричная документация (перевод + двуязычные схемы) | done |
| [EPIC-007](EPIC-007-map-phase2.md) | MAP-стек, фаза 2 (MAP/M/c, MAP/PH/c, фиттинг, BMAP/M/1) | done |
| [EPIC-008](EPIC-008-map-phase3.md) | MAP-стек, фаза 3 (BMAP/PH/1 общего вида) | done |
| [EPIC-009](EPIC-009-rdr-priority.md) | RDR: многоканальные многоприоритетные СМО (оптимизация G-матрицы + RDR-A + точная CTMC + фикс сходимости + аудит статьи) | done |
| [EPIC-010](EPIC-010-age-of-information.md) | Age of Information (AoI/PAoI) — свежесть информации | done |
| [EPIC-011](EPIC-011-multiserver-job.md) | Multiserver-job (MSJ): заявка занимает k серверов одновременно | done |
| [EPIC-012](EPIC-012-bulk-service.md) | Bulk-service очереди (batching, LLM inference) | done |
| [EPIC-013](EPIC-013-predictions-scheduling.md) | Планирование с предсказаниями (learning-augmented) | done |
| [EPIC-014](EPIC-014-load-balancing.md) | Балансировка нагрузки / диспетчеризация (JSQ/power-of-d/JIQ, mean-field) | done |
| [EPIC-015](EPIC-015-polling.md) | Polling-системы (циклический сервер, switchover) | done |
| [EPIC-016](EPIC-016-time-varying.md) | Нестационарные очереди Mt/M/c (PSA/MOL) | done |
| [EPIC-017](EPIC-017-networks-exact-methods.md) | Сети МО: закрытые сети и точные методы (MVA/Бьюзен, Джексон, QNA, G-networks, BCMP) | done |
| [EPIC-018](EPIC-018-networks-wave2.md) | Сети МО, волна 2: блокировки, fork-join в сети, MAP-вход, transient, схемы каталога, туториалы | done |
| [EPIC-019](EPIC-019-unreliable-servers.md) | Ненадёжные приборы: M/M/c с отказами, machine repair, working breakdowns, катастрофы с ремонтом, retrial + отказы | done |
| [EPIC-020](EPIC-020-priority-wave2.md) | Приоритеты, волна 2: accumulating priority, нетерпение, MAP-вход, retrial, preemptive-repeat | done |
| [EPIC-021](EPIC-021-slo-deadline-queueing.md) | SLA/deadline-violation probability: SLO-калькуляторы поверх каталога моделей, LLM-serving TTFT | done |
| [EPIC-022](EPIC-022-fork-join-heavy-tail.md) | Fork-Join с тяжёлыми хвостами: точный max n·Pareto (CDF + моменты через Beta-функцию) | done |
| [EPIC-023](EPIC-023-machine-repair-heterogeneous.md) | Machine repair с двумя гетерогенными ремонтниками: точная CTMC (Krishnamoorthi 1963) | done |
| [EPIC-024](EPIC-024-queueing-inventory.md) | Queueing-inventory systems: M/M/1, (0,S)-политика, backorder, точный QBD | done |
| [EPIC-025](EPIC-025-priority-heterogeneous-servers.md) | M/M/2 с приоритетами и гетерогенными серверами: усечённая CTMC (Krishnamoorthi + приоритеты) | done |
| [EPIC-026](EPIC-026-queueing-inventory-lost-sales.md) | Queueing-inventory lost-sales: параметр политики на `MM1QueueingInventoryCalc` | done |
| [EPIC-027](EPIC-027-queueing-inventory-general-sS.md) | Queueing-inventory, общая политика (s,S): параметр `s` на `MM1QueueingInventoryCalc` | done |
| [EPIC-028](EPIC-028-queueing-inventory-multiserver.md) | Queueing-inventory, многоканальный случай (M/M/c): `MMcQueueingInventoryCalc`, точный QBD со сложенным граничным суперблоком | done |
| [EPIC-029](EPIC-029-edf-scheduling.md) | EDF scheduling discipline: DES-симулятор + закон сохранения работы (только при редком reneging) + композиция с SLA-слоем (EPIC-021) | done |
| [EPIC-030](EPIC-030-fork-join-dag-heterogeneous.md) | Fork-Join с гетерогенными ветвями и series-parallel DAG: `heterogeneous_max_moments`, `ForkJoinDAGCalc` | done |
| [EPIC-031](EPIC-031-fork-join-nk-heterogeneous.md) | (n,k)-Fork-Join поверх гетерогенных/DAG-ветвей: `pareto_kth_order_moments`, `heterogeneous_kth_order_moments`, `("parallel",...,k)` | done |
| [EPIC-032](EPIC-032-bulk-service-waiting-moments.md) | Bulk-service (LLM/GPU dynamic batching): точные моменты `N`/`W` поверх `BulkServiceMM1Calc` | done |
| [EPIC-033](EPIC-033-queueing-inventory-heterogeneous-servers.md) | Queueing-inventory с c=2 гетерогенными серверами: расщепление состояний + сложенный граничный суперблок QBD | done |
| [EPIC-034](EPIC-034-llm-serving-deadline-admission-control.md) | M/M/1 с точным admission control по дедлайну (Exp(θ)): level-crossing функциональное уравнение, степенные ряды | done |
| [EPIC-035](EPIC-035-bulk-service-general-erlang.md) | M/G^[a,b]/1 с общим (Erlang-подогнанным) обслуживанием батча: фазовое расширение CTMC | done |
| [EPIC-036](EPIC-036-bulk-service-general-h2.md) | M/H2^[a,b]/1 с H2-подогнанным обслуживанием батча (CV≥1): фазовое расширение CTMC | done |
| [EPIC-037](EPIC-037-bulk-service-auto-dispatch.md) | Единый Erlang/H2 auto-dispatch для bulk-service батча по CV: `fit_bulk_service_calc` | done |
| [EPIC-038](EPIC-038-queueing-inventory-heterogeneous-servers-general-c.md) | Queueing-inventory с общим числом c гетерогенных серверов: расщепление на подмножества + сложенный граничный суперблок QBD | done |
| [EPIC-039](EPIC-039-queueing-inventory-heterogeneous-servers-h2-service.md) | Queueing-inventory, c гетерогенных серверов с H2-подогнанным обслуживанием (своё H2 на каждый сервер) | done |
| [EPIC-040](EPIC-040-queueing-inventory-phase-type-replenishment.md) | Queueing-inventory с фазовым (Erlang) временем пополнения склада | done |
| [EPIC-041](EPIC-041-queueing-inventory-heterogeneous-erlang-service.md) | Queueing-inventory, c гетерогенных серверов с Erlang-подогнанным обслуживанием (комплемент EPIC-039) | done |
| [EPIC-042](EPIC-042-bulk-service-batch-size-dependent-phase-params.md) | Bulk-service с batch-size-зависимыми параметрами Erlang/H2 | done |
| [EPIC-043](EPIC-043-phase-type-exact-moments.md) | Точные моменты (не только среднее) для фазово-расширенных CTMC | done |
| [EPIC-044](EPIC-044-non-exponential-machine-repair-msj.md) | Неэкспоненциальное время в machine repair / multiserver-job | proposed |
| [EPIC-045](EPIC-045-batch-arrival-priority-impatience.md) | Композиция batch arrival + priority + impatience | proposed |
| [EPIC-046](EPIC-046-time-limited-service-discipline.md) | Time-limited (T-policy/таймер) дисциплина обслуживания | proposed |
| [EPIC-047](EPIC-047-msj-ph-backfilling.md) | MSJ: PH-обслуживание, насыщенный порог, FCFS/EASY и ошибки прогнозов | done |
| [EPIC-048](EPIC-048-msj-conservative-controlled-load.md) | MSJ: conservative backfilling и сравнение при одинаковой ресурсной нагрузке | done |
| [EPIC-049](EPIC-049-msj-runtime-prediction-calibration.md) | MSJ: прогноз длительности по признакам и split-conformal калибровка | done |
| [EPIC-050](EPIC-050-msj-group-runtime-calibration.md) | MSJ: калибровка прогнозов по ресурсным классам и цена защиты широких заявок | done |
| [EPIC-051](EPIC-051-msj-age-residual-runtime.md) | MSJ: прогноз остатка по возрасту и цензурированная история | done |
| [EPIC-052](EPIC-052-msj-packing-baselines.md) | MSJ: prediction-free packing, Quickswap и ServerFilling | done |
| [EPIC-053](EPIC-053-msj-checkpoint-cost.md) | MSJ: стоимость checkpoint/resume и полезная загрузка | done |
| [EPIC-054](EPIC-054-msj-protected-service.md) | MSJ: минимальный полезный интервал и независимый выбор длительности защиты | done |
| [EPIC-055](EPIC-055-real-trace-calibration.md) | Реальная SWF-трасса: аудит, временная калибровка обслуживания и ошибка выбора дисциплины | done |
| [EPIC-056](EPIC-056-real-trace-temporal-dependence.md) | Реальная трасса: давность истории, точный K и блочная генерация длительностей | done |
| [EPIC-057](EPIC-057-real-trace-initial-state-cancellation.md) | Реальная трасса: начальное состояние, отменённая нагрузка и runtime limits | done |
| [EPIC-058](EPIC-058-modern-gpu-trace-validation.md) | Современная GPU-трасса Acme/Kalos: аудит времени, перенос калибровки и неуспешная нагрузка | done |
| [EPIC-059](EPIC-059-feature-conditional-service.md) | Условное обслуживание по requested time/type: ранний выбор по CRPS и поздний lifecycle replay | done |
| [EPIC-060](EPIC-060-gpu-resource-envelope.md) | GPU resource envelope: размер пула, эксклюзивные узлы и явно невыполнимые сценарии | done |
| [EPIC-061](EPIC-061-queue-aware-model-selection.md) | Выбор модели по ранним mean/p99 очереди, CRPS comparator и отдельная временная проверка | done |
| [EPIC-062](EPIC-062-joint-marked-arrivals.md) | Совместные gap/K/context arrivals, matched controls зависимости и аудит устарелости истории | done |
| [EPIC-063](EPIC-063-resource-observability-audit.md) | Наблюдаемость ресурсов: шесть источников, полный аудит Helios и границы трактовки дневных VC-counts | done |
| [EPIC-064](EPIC-064-msj-capacity-calendar.md) | Opt-in календарь MSJ capacity и явно модельные Helios VC-сценарии | done |
| [EPIC-065](EPIC-065-availability-aware-arrival-history.md) | Availability-aware arrival history без completed-prefix staleness | done |
| [EPIC-066](EPIC-066-batch-service-sla-exact-tail.md) | Точная вероятность нарушения SLA (хвост W) для batch-service очередей, статья | done |
| [EPIC-067](EPIC-067-bulk-service-idle-refill.md) | Idle-refill: точные W-моменты и хвост bulk-service при `a > 1` | done |
| [EPIC-068](EPIC-068-bulk-service-impatience.md) | Нетерпеливые заявки (impatience/reneging) в batch-service | done |
| [EPIC-069](EPIC-069-bulk-service-multiserver.md) | Bulk-service с `c>1` независимыми серверами (общая очередь, несколько GPU-реплик) | done |
| [EPIC-070](EPIC-070-bulk-service-multiserver-impatience.md) | Нетерпеливые заявки в многоканальном bulk-service (EPIC-068 × EPIC-069) | done |
| [EPIC-071](EPIC-071-bulk-service-h2-impatience.md) | Нетерпеливые заявки для H2-обслуживания (EPIC-036/042 × EPIC-068) | done |
| [EPIC-072](EPIC-072-bulk-service-multiserver-phase-type.md) | Многоканальный bulk-service с batch-size-зависимым обслуживанием | done |
| [EPIC-073](EPIC-073-occupancy-dependent-continuous-batching.md) | Occupancy-dependent обслуживание с потолком занятости (ядро continuous batching LLM-serving) | done |
| [EPIC-074](EPIC-074-occupancy-dependent-two-branch-service.md) | Occupancy-модулированное двухветвевое обслуживание с потолком занятости (неоднородные длины вывода) | done |
| [EPIC-075](EPIC-075-queueing-inventory-waiting-distribution.md) | Распределение (моменты и хвост) времени ожидания во всех 7 queueing-inventory классах — Р1 серии догоняющих работ; попутно найден и исправлен реальный дефект `E[W]` при `c > 1` | done |
| [EPIC-076](EPIC-076-mmc-delay-dependent-service.md) | M/M/c с интенсивностью обслуживания, зависящей от испытанного ожидания (D'Auria и др., EJOR 2022) — Р2 серии догоняющих работ | done |
| [EPIC-077](EPIC-077-mg1-ps-sojourn-moments.md) | Старшие моменты условного времени пребывания в M/G/1-PS (Яшков, arXiv:math/0512281) — Р3 серии догоняющих работ; закрыт собственный задокументированный пробел `MG1PSCalc` | done |

Направления EPIC-010…013 (первая волна) и EPIC-014…016 (вторая волна) выбраны по обзору трендов
сообщества: [../research/queueing-trends-2026.md](../research/queueing-trends-2026.md);
EPIC-017 и EPIC-018 — по обзору сетей:
[../research/queueing-networks-2026.md](../research/queueing-networks-2026.md);
EPIC-019 — по обзору ненадёжных приборов:
[../research/unreliable-queues-2026.md](../research/unreliable-queues-2026.md);
EPIC-020 — по обзору приоритетов:
[../research/priority-queues-2026.md](../research/priority-queues-2026.md);
EPIC-021 — по обзору SLA/deadline-aware очередей:
[../research/sla-deadline-queueing-2026.md](../research/sla-deadline-queueing-2026.md);
EPIC-022 — по обзору fork-join с тяжёлыми хвостами:
[../research/fork-join-heavy-tail-2026.md](../research/fork-join-heavy-tail-2026.md);
EPIC-023 — по обзору machine repair с гетерогенными ремонтниками:
[../research/machine-repair-heterogeneous-2026.md](../research/machine-repair-heterogeneous-2026.md);
EPIC-024 — по обзору queueing-inventory систем:
[../research/queueing-inventory-2026.md](../research/queueing-inventory-2026.md);
EPIC-025 — по обзору приоритетов с гетерогенными серверами:
[../research/priority-heterogeneous-servers-2026.md](../research/priority-heterogeneous-servers-2026.md);
EPIC-026 — по обзору queueing-inventory lost-sales:
[../research/queueing-inventory-lost-sales-2026.md](../research/queueing-inventory-lost-sales-2026.md);
EPIC-027 — по обзору queueing-inventory общей политики (s,S):
[../research/queueing-inventory-general-sS-2026.md](../research/queueing-inventory-general-sS-2026.md);
EPIC-028 — по обзору многоканального queueing-inventory:
[../research/queueing-inventory-multiserver-2026.md](../research/queueing-inventory-multiserver-2026.md);
EPIC-029 — по обзору EDF-планирования:
[../research/edf-scheduling-2026.md](../research/edf-scheduling-2026.md);
EPIC-030 — по обзору fork-join с гетерогенными ветвями и DAG:
[../research/fork-join-dag-heterogeneous-2026.md](../research/fork-join-dag-heterogeneous-2026.md);
EPIC-031 — по обзору (n,k)-fork-join поверх гетерогенных/DAG-ветвей:
[../research/fork-join-nk-heterogeneous-2026.md](../research/fork-join-nk-heterogeneous-2026.md);
EPIC-032 — по обзору bulk-service и трендов теории очередей 2025-2026:
[../research/bulk-service-waiting-moments-2026.md](../research/bulk-service-waiting-moments-2026.md);
EPIC-033 — по обзору queueing-inventory с гетерогенными серверами:
[../research/queueing-inventory-heterogeneous-servers-2026.md](../research/queueing-inventory-heterogeneous-servers-2026.md);
EPIC-034 — по обзору LLM-serving SLO/deadline admission control:
[../research/llm-serving-deadline-admission-control-2026.md](../research/llm-serving-deadline-admission-control-2026.md);
EPIC-035 — по обзору bulk-service с общим обслуживанием батча:
[../research/bulk-service-general-erlang-2026.md](../research/bulk-service-general-erlang-2026.md);
EPIC-036 — по обзору H2-подгонки bulk-service:
[../research/bulk-service-general-h2-2026.md](../research/bulk-service-general-h2-2026.md);
EPIC-037 — по обзору auto-dispatch bulk-service:
[../research/bulk-service-auto-dispatch-2026.md](../research/bulk-service-auto-dispatch-2026.md);
EPIC-038 — по обзору queueing-inventory с общим c гетерогенных серверов:
[../research/queueing-inventory-heterogeneous-servers-general-c-2026.md](../research/queueing-inventory-heterogeneous-servers-general-c-2026.md);
EPIC-039 — по обзору H2-подогнанного обслуживания у гетерогенных серверов:
[../research/queueing-inventory-heterogeneous-servers-h2-service-2026.md](../research/queueing-inventory-heterogeneous-servers-h2-service-2026.md);
EPIC-040 — по обзору фазового пополнения склада:
[../research/queueing-inventory-phase-type-replenishment-2026.md](../research/queueing-inventory-phase-type-replenishment-2026.md);
EPIC-041 — по обзору Erlang-обслуживания у гетерогенных серверов:
[../research/queueing-inventory-heterogeneous-servers-erlang-service-2026.md](../research/queueing-inventory-heterogeneous-servers-erlang-service-2026.md);
EPIC-042 — по обзору batch-size-зависимых параметров bulk-service:
[../research/bulk-service-batch-size-dependent-phase-params-2026.md](../research/bulk-service-batch-size-dependent-phase-params-2026.md).
