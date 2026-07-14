# Очереди с ненадёжными приборами: обзор литературы и gap-анализ (2026)

- **Дата:** 2026-07-15
- **Источники:** OpenAlex, Crossref, arXiv (скилл lit-search: запросы по server
  breakdowns/repairs, working breakdowns, machine repair problem, многоканальным
  системам с отказами, катастрофам с восстановлением, retrial + unreliable)
- **Эпик по итогам:** [EPIC-019](../epics/EPIC-019-unreliable-servers.md)

Назначение: зафиксировать ландшафт публикаций по СМО с ненадёжными приборами и
разрывы с библиотекой — «куда копать» в теме надёжности.

## 1. Что есть в most_queue сейчас

| Модуль | Модель |
|---|---|
| `theory/vacations/mg1_unreliable.py` → `MG1UnreliableCalc` | **M/G/1 с поломками и ремонтами** (Avi-Itzhak–Naor 1963): прибор ломается только под нагрузкой (Poisson ξ), ремонт общего вида, прерванная заявка дообслуживается; точное сведение к M/G/1 через completion time |
| `sim/unreliable.py` → `UnreliableQueueSim` | парный симулятор |
| Смежное: negative-стек (`theory/negative/*`) | DISASTER-модели — катастрофа мгновенно очищает систему, но прибор **сразу исправен** (нет фазы ремонта) |
| Смежное: retrial-стек (`theory/retrial/*`) | орбита без отказов прибора |
| Смежное: Engset (`theory/closed/engset.py`) | конечный источник, но прибор надёжен |

**Ключевой пробел:** надёжность представлена единственной одноканальной моделью.
Нет многоканальных систем с отказами, нет working breakdowns, нет machine repair
problem (классика теории надёжности), нет фазы ремонта после катастрофы, нет
отказов в retrial-моделях — при том, что строительные блоки (CTMC/QBD-утилиты,
negative- и retrial-стеки, closed-модели) в библиотеке уже есть.

## 2. Ландшафт публикаций (по направлениям)

### 2.1 Классика: поломки и ремонты

- Avi-Itzhak B., Naor P., Some Queuing Problems with the Service Station Subject
  to Breakdown, Operations Research, 1963 — **реализовано** (M/G/1).
- Mitrany I.L., Avi-Itzhak B., A Many-Server Queue with Service Interruptions,
  Management Science, 1968 — **M/M/N с отказами серверов**, матрично-аналитическое
  решение.
- Neuts M.F., Lucantoni D.M., A Markovian Queue with N Servers Subject to
  Breakdowns and Repairs, Management Science, 1979 — M/M/N с ограниченным числом
  ремонтников, matrix-geometric.
- Gaver D.P., A Waiting Line with Interrupted Service, Including Priorities,
  JRSS-B, 1962 — прерывания как приоритеты.

### 2.2 Working breakdowns (прибор работает медленнее, а не стоит)

- Kalidass K., Kasturi R., A queue with working breakdowns, Computers &
  Industrial Engineering, 2012 — **M/M/1 working breakdowns**, основополагающая.
- Markovian queue optimisation analysis with an unreliable server subject to
  working breakdowns and impatient customers, Int. J. Systems Science, 2013,
  doi:10.1080/00207721.2013.859326 — 36 цит.
- Modelling and optimisation of a two-server queue with multiple vacations and
  working breakdowns, IJPR, 2019, doi:10.1080/00207543.2019.1624856.
- Активная ветка 2015–2025: retrial + working breakdowns (MCA 2017,
  doi:10.3390/mca22010015), негативные заявки + working breakdowns (IEEE Access
  2019, doi:10.1109/access.2019.2950268), дискретное время (JSSI 2017).

### 2.3 Machine repair problem / машинная интерференция (закрытая классика)

- Palm C. (1947) / Benson–Cox (1951) — машинная интерференция, конечный парк.
- Computational analysis of machine repair problem with unreliable multi-repairmen,
  Computers & OR, 2013, doi:10.1016/j.cor.2012.10.004 — 30 цит.
- Vacation model for Markov machine repair problem with two heterogeneous
  unreliable servers, JIEI, 2017, doi:10.1007/s40092-017-0214-x.
- A time-shared machine repair problem with mixed spares under N-policy, JIEI,
  2016, doi:10.1007/s40092-015-0136-4 — запасные машины (spares).
- The Two Repairmen Problem: A Finite Source M/G/2 Queue, SIAM J. Appl. Math.,
  1987, doi:10.1137/0147024.

### 2.4 Катастрофы с фазой восстановления

- Towsley D., Tripathi S.K., A single server priority queue with server failures
  and queue flushing, Operations Research Letters, 1991 — сброс очереди + ремонт.
- Transient analysis of an M/G/1 retrial queue subject to disasters and server
  failures, EJOR, 2007, doi:10.1016/j.ejor.2007.04.054 — 67 цит.
- Analysis of a Discrete-Time Queueing Model with Disasters, Mathematics, 2021,
  doi:10.3390/math9243283.
- В библиотеке катастрофы (DISASTER) — с мгновенным восстановлением; фаза ремонта
  меняет поведение качественно (накопление очереди за время ремонта).

### 2.5 Retrial + ненадёжный прибор

- Artalejo J.R., New results in retrial queueing systems with breakdown of the
  servers, Statistica Neerlandica, 1994.
- Wang J., Cao J., Li Q., Reliability analysis of the retrial queue with server
  breakdowns and repairs, Queueing Systems, 2001 — базовая M/M/1 retrial +
  breakdowns.
- A BMAP/G/1 Retrial Queue with a Server Subject to Breakdowns and Repairs,
  Annals of OR, 2006, doi:10.1007/s10479-006-5301-0 — 42 цит.
- Analysis of a Retrial Queue With Two-Type Breakdowns and Delayed Repairs,
  IEEE Access, 2020, doi:10.1109/access.2020.3023191.
- Optimization of retrial queue with unreliable servers subject to imperfect
  coverage and reboot delay, QTQM, 2022, doi:10.1080/16843703.2021.2020952 —
  32 цит., живой прикладной фронт (облачные/беспроводные системы).

### 2.6 Комбинации с отпусками и резервированием

- A queueing model with server breakdowns, repairs, vacations, and backup server,
  Operations Research Perspectives, 2019, doi:10.1016/j.orp.2019.100131 — 66 цит.,
  самая цитируемая свежая работа направления: резервный прибор на время ремонта.
- Maximum entropy analysis of Mˣ/M/1 with vacations and breakdowns, C&IE, 2007.

## 3. Вывод: топ-5 кандидатов на реализацию

| # | Что | Опора | Почему | Оценка |
|---|-----|-------|--------|--------|
| 1 | **M/M/c с отказами и ремонтами серверов** | Mitrany–Avi-Itzhak 1968; Neuts–Lucantoni 1979 | Самый цитируемый пробел: надёжность есть только для одного прибора; CTMC/QBD-инфраструктура готова | средняя |
| 2 | **Machine repair problem** (конечный парк M машин, R ремонтников, тёплый резерв S) | Palm 1947; C&OR 2013 | Классика теории надёжности, целиком отсутствует; точное конечное birth-death решение; дополняет Engset | малая-средняя |
| 3 | **M/M/1 working breakdowns** (при поломке скорость μ_d < μ, не остановка) | Kalidass–Kasturi 2012 | Горячая ветка 2012–2025 (облака: деградация вместо отказа); QBD с 2 фазами | малая-средняя |
| 4 | **M/M/1 катастрофы + фаза ремонта** | Towsley–Tripathi 1991; EJOR 2007 | Прямое усиление фирменного negative-стека: сейчас после DISASTER прибор мгновенно исправен | малая-средняя |
| 5 | **M/M/1 retrial + ненадёжный прибор** | Wang–Cao–Li 2001; Artalejo 1994 | Синергия с retrial-стеком; прикладной фронт (reboot delay, imperfect coverage) | средняя |
| 6 | Резерв: backup server (ORP 2019), working breakdowns + vacations, MRP c неоднородными ремонтниками, BMAP/G/1 retrial + breakdowns | — | следующая волна | — |

Пункты 1–5 реализованы в [EPIC-019](../epics/EPIC-019-unreliable-servers.md) (done,
2026-07-15); п. 6 — резерв.
