# Queueing-inventory lost-sales: обзор литературы и gap-анализ (2026-09)

- **Дата:** 2026-09-29
- **Источники:** OpenAlex, Crossref (скилл `lit-search`: M/M/1 queue inventory lost sales exact
  analysis) + инвентаризация кода
- **Эпик по итогам:** [EPIC-026](../epics/EPIC-026-queueing-inventory-lost-sales.md)

Прямое продолжение EPIC-024 (backorder-вариант queueing-inventory) — вариант с потерянными
продажами, отложенный в резерв при выборе EPIC-024 и реализуемый сейчас.

## 1. Что есть в most_queue сейчас

`MM1QueueingInventoryCalc` (EPIC-024) — M/M/1, политика `(0,S)`, **backorder**: при опустошении
склада (`i=0`) обслуживание блокируется, но приходящие заявки всё равно встают в очередь и ждут
пополнения. Точное решение через `QBDSolver`.

**Пробел:** вариант **lost sales** — заявка, пришедшая при `i=0`, **теряется** (не встаёт в
очередь), а не ждёт. Более реалистичен для розничной торговли/e-commerce (клиент, увидев
«нет в наличии», уходит, а не встаёт виртуально в очередь).

## 2. Литература

- Saffari M., Haji R., Hassanzadeh F., *The M/M/1 queue with inventory, lost sale, and general
  lead times*, Queueing Systems, 2013, doi:10.1007/s11134-012-9337-3 — 87 цит., флагманская
  работа направления (уже отмечена в research-доке EPIC-024 как кандидат следующей волны).
- *The M/M/1 queue with a production-inventory system and lost sales*, Applied Mathematics and
  Computation, 2014, doi:10.1016/j.amc.2014.02.033 — 27 цит., прямое совпадение постановки
  (M/M/1 + inventory + lost sales).
- *A production–inventory system with a Markovian service queue and lost sales*, Journal of the
  Korean Statistical Society, 2016, doi:10.1016/j.jkss.2015.05.002 — 29 цит.
- *Exact analysis of (R,s,S) inventory control systems with lost sales and zero lead time*, Naval
  Research Logistics, 2019, doi:10.1002/nav.21833 — 10 цит. (без очереди/сервера, чистое
  управление запасами — контекст, не прямой первоисточник).

## 3. Что реализуемо точно

Ключевое наблюдение: разница между backorder (EPIC-024) и lost-sales — **только** в одном
переходе. В backorder приход при `i=0` увеличивает `n` (заявка встаёт в очередь); в lost-sales
приход при `i=0` **не меняет состояние** (заявка потеряна, не входит в CTMC вовсе — «потерянное»
событие не является переходом). Всё остальное (обслуживание при `i≥1`, пополнение при `i=0`,
дисциплина `(0,S)`) идентично.

В терминах QBD-блоков (см. `docs/roadmaps/queueing_inventory_roadmap.md`): матрицы `A0` (уровень
вверх) и `B01` (граница, уровень 0 → 1) в backorder-варианте — `λ·I_{S+1}` (приход всегда поднимает
уровень, фаза не меняется); в lost-sales-варианте — та же матрица с **обнулённой строкой фазы 0**
(`diag(0, λ, ..., λ)`), плюс диагональ `A1[0,0]`/`B00[0,0]` теряет слагаемое `λ` (превращается в
`-θ` вместо `-(λ+θ)`, т.к. «потерянное» событие не расходует времени системы и не является
исходящим переходом). Обслуживание (`A2`), пополнение (`A1[0,S]`, `B00[0,S]`) и граница
(`B10`) — без изменений.

Это позволяет реализовать вариант **не отдельным классом**, а параметром политики
(`policy="backorder"|"lost_sales"`) на уже существующем `MM1QueueingInventoryCalc` — минимальное,
локальное изменение `_build_solver()`, без дублирования 95% логики (метрики, решение через
`QBDSolver`, `V=W+1/μ`).

## 4. Gap-анализ и решение

| Направление | Активность | В most-queue | Реализуемо точно? |
|---|---|---|---|
| M/M/1 queueing-inventory, `(0,S)`, lost sales | подтверждена (87+27+29 цит. на трёх первоисточниках) | нет (только backorder) | да — минимальное изменение двух QBD-блоков в существующем классе |

**Решение:** добавить `policy` параметр к `MM1QueueingInventoryCalc` — см.
[EPIC-026](../epics/EPIC-026-queueing-inventory-lost-sales.md). Общая `(s,S)`-политика и
многоканальный случай остаются в резерве (см. research-доку EPIC-024).
