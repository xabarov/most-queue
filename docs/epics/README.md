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

Направления EPIC-010…013 (первая волна) и EPIC-014…016 (вторая волна) выбраны по обзору трендов
сообщества: [../research/queueing-trends-2026.md](../research/queueing-trends-2026.md);
EPIC-017 и EPIC-018 — по обзору сетей:
[../research/queueing-networks-2026.md](../research/queueing-networks-2026.md);
EPIC-019 — по обзору ненадёжных приборов:
[../research/unreliable-queues-2026.md](../research/unreliable-queues-2026.md);
EPIC-020 — по обзору приоритетов:
[../research/priority-queues-2026.md](../research/priority-queues-2026.md).
