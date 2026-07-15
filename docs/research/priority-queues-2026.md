# Приоритетные СМО: обзор литературы и gap-анализ (2026)

- **Дата:** 2026-07-15
- **Источники:** OpenAlex, Crossref, arXiv (скилл lit-search: accumulating priority,
  приоритеты + нетерпение, MAP/PH-приоритеты, retrial + приоритеты, динамические
  приоритеты, cμ-оптимальность) + инвентаризация кода
- **Эпик по итогам:** [EPIC-020](../epics/EPIC-020-priority-wave2.md)

Назначение: зафиксировать, что уже умеет приоритетный стек (он большой), какие
направления горячи в литературе, и вывести топ-5 кандидатов на реализацию.

## 1. Что есть в most_queue сейчас (сильный стек, но весь — статические классы)

| Модель | Класс | Метод | Ограничения |
|---|---|---|---|
| M/G/1 NP | `MG1NonPreemptiveCalc` | точный (R-факторы + ЛСТ) | моменты w/v, любое число классов |
| M/G/1 PR | `MG1PreemptiveCalc` | точный (busy periods) | моменты, любое число классов |
| M/G/n PR/NP | `MGnInvarApproximation` | инвариантная аппроксимация | любое n и классы; приближённо |
| M/PH/n PR, 2 кл. | `MPhNPrty` | Такахаси–Таками + Cox ПНЗ | низший класс — только v1 |
| M/M/n PR, 2 кл. | `MMnPR2ClsBusyApprox` | Такахаси–Таками | 2 класса, exp |
| M/M/2 PR, 3 кл. | `MM2BusyApprox3Classes` | Такахаси–Таками | n=2, 3 класса, exp |
| M/M/k PR, m кл. | `RDRAPriorityCalc` (RDR-A) | агрегирование + рекурсия | средние низших классов |
| M/PH/k PR, m кл. | `RDRAPriorityPH` | RDR-A + PH | средние |
| M/M/k PR, m кл. | `MMkPriorityExact` | точная CTMC (+ дисперсия §2.4) | эталон, малые m |
| M/PH/PH/k PR, 2 кл. | `MPhPhK2Class` | точная CTMC | 2 класса |
| Сети с приоритетами | `OpenNetworkCalcPriorities` | декомпозиция + инварианты | Poisson per class |
| Симулятор | `PriorityQueueSimulator` | No/NP/PR/**RS/RW** | RS/RW — только в симе |

Наследие EPIC-009 (RDR, done): точные эталоны, дисперсия отклика, аудит статьи.

**Ключевые пробелы (по инвентаризации):**

1. Все приоритеты **статические** — нет accumulating/dynamic/delay-dependent.
2. Нет приоритетов с **нетерпением** (reneging/deadline) — Erlang-A бесклассовый.
3. Весь приоритетный вход — **пуассоновский**: MAP/BMAP с приоритетами нет, хотя
   MAP/PH-стек — флагман библиотеки.
4. **Preemptive-repeat (RS/RW)** есть только в симуляторе — аналитики нет.
5. Retrial-стек бесприоритетный (а в EPIC-019 появился ещё и retrial + отказы).
6. Мелочи: старшие моменты низших классов у многоканальных решателей — заглушки;
   точной многоканальной NP-модели нет; примеры в `docs/priorities.md` расходятся с
   фактическими сигнатурами классов.

## 2. Ландшафт публикаций (по направлениям)

### 2.1 Accumulating priority (APQ) — самое горячее направление

Приоритет заявки растёт со временем ожидания (линейно с классовым коэффициентом) —
современная формализация delay-dependent приоритетов Клейнрока; стандарт де-факто
для медицинского триажа (KPI «доля принятых за T минут»).

- Stanford D.A., Taylor P., Ziedins I., Waiting time distributions in the accumulating
  priority queue, Queueing Systems, 2013, doi:10.1007/s11134-013-9382-6 — **93 цит.**,
  M/G/1 APQ, распределения ожидания по классам.
- Sharif A.B., Stanford D.A. и др., A multi-class multi-server accumulating priority
  queue with application to health care, ORHC, 2014, doi:10.1016/j.orhc.2014.01.002.
- Multi-server accumulating priority queues with heterogeneous servers, EJOR, 2016,
  doi:10.1016/j.ejor.2016.02.010 — 40 цит.
- Waiting Time Distributions in the Preemptive APQ, MCAP, 2015,
  doi:10.1007/s11009-015-9476-1.
- Nonlinear APQ with Equivalent Linear Proxies, Operations Research, 2017,
  doi:10.1287/opre.2017.1613.
- APQ versus pure priority queues for managing patients in emergency departments,
  ORHC, 2019, doi:10.1016/j.orhc.2019.100224 — прикладное сравнение.
- Asymptotics of waiting time distributions in the APQ, Queueing Systems, 2022,
  doi:10.1007/s11134-022-09839-7 — направление живо.

### 2.2 Приоритеты + нетерпение (call-центры, SLA)

- Iravani F., Balcıoğlu B., On priority queues with impatient customers, Queueing
  Systems, 2008, doi:10.1007/s11134-008-9069-6 — 57 цит.
- Choi B.D. и др., M/M/1 Queue with Impatient Customers of Higher Priority, Queueing
  Systems, 2001, doi:10.1023/a:1010820112080 — 54 цит.
- Atar R., Mandelbaum A., Reiman M., Scheduling a multi-class queue with many
  exponential servers, Ann. Appl. Prob., 2004, doi:10.1214/105051604000000233 —
  149 цит., асимптотика many-server.
- Workload-Dependent Dynamic Priority for the Multiclass Queue with Reneging, Math.
  of OR, 2018, doi:10.1287/moor.2017.0869 — стык с динамическими приоритетами.

### 2.3 MAP/PH-вход с приоритетами (стык с флагманским стеком)

- Takine T., Sengupta B. и др.: A nonpreemptive priority MAP/G/1 queue with two
  classes, JORSJ, 1996, doi:10.15807/jorsj.39.266 — 40 цит.
- Horváth G. и др., Efficient analysis of the queue length moments of the
  MMAP/MAP/1 preemptive priority queue, Performance Evaluation, 2012,
  doi:10.1016/j.peva.2012.08.003.
- Geometric tail of low-priority queue in a nonpreemptive MAP/PH/1, Queueing
  Systems, 2011, doi:10.1007/s11134-011-9221-6.
- Klimenok, Dudin и др., A Priority Queue with Many Customer Types, Correlated
  Arrivals and Changing Priorities, Mathematics, 2020, doi:10.3390/math8081292 —
  школа Дудина активно публикует QBD-решения таких моделей.

### 2.4 Retrial + приоритеты

- A preemptive priority retrial queue with two classes of customers and general
  retrial times, Operational Research, 2015, doi:10.1007/s12351-015-0175-z — 30 цит.
- Double orbit finite retrial queues with priority customers and service
  interruptions, AMC, 2015, doi:10.1016/j.amc.2014.12.066 — 31 цит.
- Retrial G-queue with priority and unreliable server under Bernoulli vacation,
  C&IE, 2012, doi:10.1016/j.cie.2012.08.015 — 52 цит. (стык с EPIC-019).

### 2.5 Preemptive-repeat и классика прерываний

- Gaver D.P., A Waiting Line with Interrupted Service, Including Priorities, JRSS-B,
  1962 — completion-time техника для repeat-дисциплин.
- Avi-Itzhak B. (1963) — та же техника, уже используется в `MG1UnreliableCalc`.
- RS/RW-дисциплины реализованы в симуляторе библиотеки — аналитики нет нигде в
  open source (проверено по конкурентам в EPIC-001).

### 2.6 Оптимальность и обучение (контекст, не кандидаты в ядро)

- Generalized cμ-rule (van Mieghem 1995; moderate deviations — QS 2017,
  doi:10.1007/s11134-017-9523-4) — правило назначения приоритетов.
- Due-date scheduling (Operations Research 2003, doi:10.1287/opre.51.1.113.12793).
- Learning-Augmented Priority Queues (arXiv 2406.04793, 2024) — уже в списке
  источников trends-2026; стык с EPIC-013 (predictions).

## 3. Вывод: топ-5 кандидатов на реализацию

| # | Что | Опора | Почему | Оценка |
|---|-----|-------|--------|--------|
| 1 | **M/G/1 accumulating priority (APQ)** + M/M/c APQ (средние) | Stanford–Taylor–Ziedins 2013; ORHC 2014 | Самое цитируемое живое направление приоритетов; закрывает пробел «только статические классы»; healthcare/SLA-триаж | средняя |
| 2 | **M/M/n + M с приоритетами и нетерпением** (2 класса, PR/NP, per-class θ) | QS 2008; QS 2001; Atar et al. 2004 | Call-центры/SLA: приоритет без нетерпения нереалистичен; CTMC-инфраструктура из EPIC-019 готова | средняя |
| 3 | **MAP/PH/1 с приоритетами** (2 класса, NP и PR) | JORSJ 1996; Perf Eval 2012; Dudin 2020 | Стыкует два флагманских стека (MAP/PH и приоритеты); QBD-утилиты есть | средняя-большая |
| 4 | **M/M/1 retrial с приоритетами** (2 класса: приоритетные ждут, обычные — на орбиту) | Operational Research 2015; Artalejo | Синергия с retrial-стеком и EPIC-019 (retrial+отказы); усечённая CTMC | малая-средняя |
| 5 | **M/G/1 preemptive-repeat (RS/RW) — аналитика** | Gaver 1962; Avi-Itzhak–Naor 1963 | Дисциплины уже в симуляторе — теории нет; completion-time техника в репо отработана на `MG1UnreliableCalc` | малая-средняя |
| 6 | Резерв: точная многоканальная NP CTMC; полные моменты низших классов; heterogeneous servers; mixed PR+NP; EDF/deadline; learning-augmented priorities (стык с EPIC-013); generalized cμ как оптимизационный хелпер | — | следующая волна | — |

Пункты 1–5 оформлены как [EPIC-020](../epics/EPIC-020-priority-wave2.md); п. 6 — резерв.
Попутно в эпик включены мелкие долги стека: рассинхрон примеров в `docs/priorities.md` с
фактическими сигнатурами и отсутствие экспорта `MMkPriorityExact`/`RDRAPriorityCalc`/
`RDRAPriorityPH`/`MPhPhK2Class` из `theory/priority/__init__.py`.
