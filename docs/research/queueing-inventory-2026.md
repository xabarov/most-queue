# Queueing-inventory systems: обзор литературы и gap-анализ (2026-09)

- **Дата:** 2026-09-29
- **Источники:** OpenAlex, Crossref (скилл `lit-search`: M/M/1 queueing-inventory lead time,
  (s,S)/backorder matrix-geometric, Schwarz-Daduna) + инвентаризация кода
- **Эпик по итогам:** [EPIC-024](../epics/EPIC-024-queueing-inventory.md)

Направление — единственный пункт из резерва `sla-deadline-queueing-2026.md`/
`queueing-trends-2026.md`, подтверждённый активным трижды подряд при повторных обзорах, но ещё не
реализованный (в отличие от fork-join heavy-tail и machine repair heterogeneous, закрытых в
EPIC-022/023).

## 1. Что это и почему в most_queue этого нет

**Queueing-inventory system** — очередь, где для обслуживания заявки нужна единица со склада: если
склад пуст, обслуживание невозможно (клиенты либо ждут — backorder, либо теряются — lost sales).
Склад пополняется по политике `(s,S)` (или проще — `(0,S)`: заказ на `S` единиц оформляется сразу,
как только запас падает до 0) за случайное время поставки (`Exp(θ)` — «positive lead time»).

Это **не** то же самое, что уже есть в библиотеке:
- Отпуска/N-policy (`vacations/`) — прибор простаивает по своей собственной логике, не из-за
  внешнего ресурса.
- Ненадёжные приборы (`reliability/`) — прибор ломается, не «заканчивается расходник».
- Отрицательные заявки (`negative/`) — убирают заявки из очереди, не блокируют обслуживание.

В most_queue queueing-inventory отсутствует полностью — ни в теории, ни в симуляции.

## 2. Литература

Основополагающие (школа Schwarz–Daduna, серия статей в Queueing Systems / Math. Methods of OR):

- Schwarz M., Daduna H., *M/M/1 Queueing systems with inventory*, Queueing Systems, 2006,
  doi:10.1007/s11134-006-8710-5 — **152 цит.**, флагманская работа направления.
- Schwarz M., Wichelhaus C., Daduna H., *Queueing systems with inventory management with random
  lead times and with backordering*, Mathematical Methods of Operations Research, 2006,
  doi:10.1007/s00186-006-0085-1 — 64 цит. — **прямой первоисточник для EPIC-024** (backorder,
  random lead time — ровно та постановка, что реализуема через QBD).
- Saffari M., Haji R., Hassanzadeh F., *The M/M/1 queue with inventory, lost sale, and general lead
  times*, Queueing Systems, 2013, doi:10.1007/s11134-012-9337-3 — 87 цит. — lost-sales вариант
  (кандидат на следующую волну).
- Krishnamoorthy A. и др., *Analysis of a Multiserver Queueing-Inventory System*, Advances in
  Operations Research, 2015, doi:10.1155/2015/747328 — многоканальное расширение (резерв).

Свежая активность (2022–2026), подтверждающая живость темы:
- *Stability of queueing-inventory systems with customers of different priorities*, Annals of OR,
  2022, doi:10.1007/s10479-022-05140-1 — стык с приоритетным стеком most_queue.
- *Single-Server Queuing-Inventory Systems with Negative Customers and Catastrophes in the
  Warehouse*, Mathematics, 2023, doi:10.3390/math11102380 — стык с фирменной темой negative
  customers.
- *Queueing-Inventory Systems with Catastrophes under Various Replenishment Policies*, Mathematics,
  2023, doi:10.3390/math11234854.
- *On the Control Policy of a Queuing–Inventory System with Variable Inventory Replenishment
  Speed*, Mathematics, 2024, doi:10.3390/math12020194.
- *A stochastic queueing-inventory system with renewal demands and positive lead time*, European
  J. of Industrial Engineering, 2020, doi:10.1504/ejie.2020.108600.
- *Inventory Control Model on Matrix Geometric Method for Intermittently Obtainable Server with
  Balking*, Baghdad Science Journal, 2026 — подтверждает, что matrix-geometric/QBD остаётся
  стандартной техникой направления и сейчас.

## 3. Что реально реализуемо

Базовая модель Schwarz–Daduna (backorder, `(0,S)`-политика, положительное время обслуживания,
экспоненциальный lead time) — это **ровно QBD-процесс**: уровень `n` = число заявок в системе
(неограничено), фаза `i ∈ {0,...,S}` = запас на складе. Переходы:

- приход (ставка `λ`): `(n,i) → (n+1,i)` — вверх, фаза не меняется;
- завершение обслуживания (ставка `μ`, только если `n≥1` **и** `i≥1` — без запаса обслуживание
  блокировано): `(n,i) → (n-1,i-1)` — вниз, фаза уменьшается на 1 (расход единицы склада);
- пополнение (ставка `θ`, только если `i=0`): `(n,0) → (n,S)` — тот же уровень, фаза скачком в S.

Это **точно** ложится на уже существующий в библиотеке `theory/matrix/qbd.py::QBDSolver`
(logarithmic reduction, тот же солвер, что использует флагманский MAP/PH-стек) — не нужна новая
численная машинерия, только новая матричная структура блоков `A0/A1/A2/B00/B01/B10`. Полный вывод
блоков — в [roadmap](../roadmaps/queueing_inventory_roadmap.md).

**Lost-sales вариант** (Saffari–Haji–Hassanzadeh 2013) — тоже QBD-совместим, но с другой структурой
переходов при `i=0` (приход не увеличивает `n`, а теряется) — естественное расширение после
backorder-варианта, в резерве.

**Многоканальный (`c` серверов) вариант** и **catastrophes/negative customers в складе** —Reserve,
требуют более сложной фазовой структуры (или композиции с уже существующим negative-стеком).

## 4. Gap-анализ и решение

| Направление | Активность | В most-queue | Реализуемо точно? |
|---|---|---|---|
| M/M/1 queueing-inventory, `(0,S)`, backorder, случайный lead time | подтверждена трижды, 64–152 цит. на первоисточниках | нет | да — прямое переиспользование `QBDSolver` |
| M/M/1 queueing-inventory, lost sales | активна (87 цит.) | нет | да, но другая структура при `i=0` — резерв |
| Многоканальный queueing-inventory | активна (19 цит. + 2015+) | нет | резерв — фазовое пространство растёт |
| Queueing-inventory + negative customers/catastrophes | активна (2023, 2×) | нет | резерв — стык с фирменной темой, но отдельная композиция |
| Queueing-inventory + приоритеты | активна (2022, AOR) | нет | резерв — стык с приоритетным стеком |

**Решение:** реализовать базовую M/M/1 backorder-модель `(0,S)` — см.
[EPIC-024](../epics/EPIC-024-queueing-inventory.md). Остальное — в резерве, в порядке убывания
готовности инфраструктуры (lost sales проще всего расширить следующим).
