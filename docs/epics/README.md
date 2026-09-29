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
[../research/llm-serving-deadline-admission-control-2026.md](../research/llm-serving-deadline-admission-control-2026.md).
