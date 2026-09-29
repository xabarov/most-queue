# Machine repair с гетерогенными ремонтниками: обзор литературы и gap-анализ (2026-09)

- **Дата:** 2026-09-29
- **Источники:** OpenAlex, Crossref (скилл `lit-search`: machine repair two heterogeneous
  repairmen, M/M/2 heterogeneous servers exact) + инвентаризация кода
- **Эпик по итогам:** [EPIC-023](../epics/EPIC-023-machine-repair-heterogeneous.md)

Продолжение серии пост-EPIC-021/022 обзоров. Направление — из резерва
[`unreliable-queues-2026.md`](unreliable-queues-2026.md) и переподтверждено в
[`sla-deadline-queueing-2026.md`](sla-deadline-queueing-2026.md) §D («A Comprehensive Survey on
Machine Repair Problems with Standby Systems», 2026 — свежий survey, сигнал живой темы).

## 1. Что есть в most_queue сейчас

`MachineRepairCalc` (`theory/reliability/machine_repair.py`, из EPIC-019) — M машин, S тёплых
резервов (`xi_s <= xi` — от cold до hot standby), **R гомогенных** ремонтников (`eta` одна на
всех). Точное решение — birth-death CTMC (`death(j) = eta * min(j, R)`), т.к. при одинаковой
скорости ремонта не важно, какой именно ремонтник занят — состояние полностью описывается числом
неисправных `j`.

**Пробел:** при **разных** скоростях ремонта (`eta_a ≠ eta_b`) чистый birth-death по `j` уже не
описывает систему точно — состояние должно знать, *какой* ремонтник занят, когда `j=1` (иначе
неизвестно, с какой скоростью идёт ремонт). Это тот же классический эффект, что у M/M/2 с
гетерогенными серверами (Krishnamoorthi, Operations Research 1963, doi:10.1287/opre.11.3.321,
35 цит.) — только применённый к конечно-источниковой (finite-source) системе.

## 2. Литература

- Krishnamoorthi B., *On Poisson Queue with Two Heterogeneous Servers*, Operations Research, 1963,
  doi:10.1287/opre.11.3.321 — основополагающая работа: точная техника разбиения состояния `j=1` по
  тому, какой из двух серверов занят (иначе система не марковская по одному лишь `j`).
- *Performance analysis and optimization of a machine repair problem with warm spares and two
  heterogeneous repairmen*, Optimization and Engineering, 2012, doi:10.1007/s11081-012-9195-1 —
  прямое применение той же техники к machine repair (конечный источник + тёплый резерв + 2
  гетерогенных ремонтника). Ближайший первоисточник для EPIC-023.
- *An optimum approach of profit analysis on the machine repair system with heterogeneous
  repairmen*, Applied Mathematics and Computation, 2015, doi:10.1016/j.amc.2014.12.026.
- *Computational analysis of machine repair problem with unreliable multi-repairmen*, Computers &
  Operations Research, 2013, doi:10.1016/j.cor.2012.10.004 — 31 цит., самая цитируемая недавняя
  работа направления (произвольное R, но приближённо/численно, не точная малая CTMC).
- *A survey of the machine interference problem*, European Journal of Operational Research, 2006,
  doi:10.1016/j.ejor.2006.02.036 — 166 цит., обзорный контекст.
- *Waiting-Time Asymptotics for the M/G/2 Queue with Heterogeneous Servers*, Queueing Systems,
  2002, doi:10.1023/a:1017913826973 — смежная тема (не finite-source), подтверждает, что техника
  живёт и за пределами чистого machine repair.
- *A Comprehensive Survey on Machine Repair Problems with Standby Systems*, 2026,
  doi:10.4038/sljas.v27i1.8226 — свежий survey, подтверждает активность темы в 2026.

## 3. Что реально реализуемо

Для **ровно двух** гетерогенных ремонтников (`eta_a`, `eta_b`, `eta_a >= eta_b` — быстрый
назначается первым) состояние `j=1` расщепляется на два: «занят A» и «занят B» (второе достижимо
только из `j=2` при завершении A первым). Для `j=0` и `j>=2` состояние по-прежнему однозначно
описывается числом `j`, т.к. при `j>=2` оба ремонтника заняты одновременно, а при `j=0` — оба
свободны. Итоговая цепь — **не birth-death, а малая точная CTMC** (`M+S+2` состояния вместо
`M+S+1`), решаемая уже готовой утилитой `theory/reliability/utils.py::ctmc_stationary` (тот же
инструмент, что используют `priority/impatience.py`, `priority/map_ph_priority.py`).

При `eta_a = eta_b` цепь сворачивается обратно в существующий гомогенный `MachineRepairCalc` —
точный regression-тест на редукцию.

**R > 2 гетерогенных ремонтника** — комбинаторный взрыв состояний (нужно знать, какое именно
подмножество ремонтников занято, не только сколько) — не входит в эту волну, резерв на будущее
(возможно, для R=3 отдельным точным расширением, или приближением для произвольного R по образцу
Computers & OR 2013).

## 4. Gap-анализ и решение

| Направление | Активность | В most-queue | Реализуемо точно? |
|---|---|---|---|
| Machine repair, 2 гетерогенных ремонтника (тёплый резерв) | подтверждена дважды (2026 survey + первоисточник 2012) | нет (только гомогенные) | да — малая точная CTMC, расширяет EPIC-019 |
| Machine repair, R>2 гетерогенных ремонтников | активна (C&OR 2013, 31 цит.) | нет | резерв — комбинаторный взрыв состояний, нужна отдельная техника (приближение) |

**Решение:** реализовать точный случай R=2 гетерогенных ремонтника — см.
[EPIC-023](../epics/EPIC-023-machine-repair-heterogeneous.md). R>2 — в резерве.
