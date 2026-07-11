# Тренды queueing theory (2024–2026) и приоритеты most-queue

Аналитический обзор активности научного сообщества по теории очередей и сопоставление с текущим
покрытием библиотеки. Цель — выбрать направления, максимально полезные сообществу и попадающие в
нишу most-queue.

*Составлен: 2026-07 (обзор литературы + позиционирование среди open-source).*

## Ниша most-queue

Точные **аналитические** калькуляторы **с начальными моментами** (не только средними) в паре с
дискретно-событийной симуляцией и кросс-валидацией. Модели, которых нет в других open-source
пакетах: Такахаси–Таками для многоканальных фазовых систем, size-based дисциплины (SRPT/SJF/PSJF/
SPJF), отрицательные заявки, MAP/PH/BMAP-стек, RDR-приоритеты.

Конкурентный ландшафт:
- **Ciw** — дискретно-событийная симуляция открытых сетей (Python), без аналитики.
- **LINE** — аналитика + симуляция, но MATLAB-центричная (Python-порт ограничен); фокус на
  расширенных сетях/MVA.
- **queuesim / OpenQTSim** — базовые M/M/c, учебные.

Вывод: most-queue занимает нишу «точные моменты + редкие модели + воспроизводимость». Новые
направления стоит выбирать так, чтобы усиливать именно её.

## Активные направления сообщества

### 1. Multiserver-job (MSJ) модель 🔥🔥🔥
Заявка требует **нескольких серверов одновременно** (облачные/GPU-джобы, запрашивающие k ядер) —
принципиально отличается от M/M/c (где заявка занимает один сервер). Самая активная датацентр-тема
группы Harchol-Balter / Grosof / Scully / Scheller-Wolf: устойчивость через saturated product-form
системы, точный/приближённый анализ времени отклика для 2 классов, heavy-traffic оптимальное
планирование, stochastic recurrence equation. **Open-source реализаций нет.** Ложится на QBD/CTMC.

### 2. Age of Information (AoI) 🔥🔥🔥
Метрика свежести информации: средний и пиковый возраст (PAoI). Огромная область (IoT, 5G/6G,
networked control, сенсорные сети). Closed-form для M/M/1, M/G/1, G/G/1, LCFS и preemptive-LCFS,
tandem-очередей. Считается из очередных примитивов (время пребывания, интервалы отправлений),
которые в библиотеке **уже есть**. Идеально ложится на нишу «моменты + sim». **В most-queue нет.**

### 3. Queueing + Predictions / Learning-augmented scheduling 🔥🔥
SIGMETRICS 2025 tutorial и survey «Queueing, Predictions, and LLMs: Challenges and Open Problems».
Планирование с предсказаниями размеров (ML-оценки), graceful degradation (робастность к ошибке
предсказания). most-queue **уже имеет SPJF** (планирование по предсказанному размеру) — прямая база
для расширения.

### 4. LLM inference serving / bulk-service 🔥🔥
Обслуживание LLM-инференса как M/G/1 с **пакетным обслуживанием** (bulk service): время обработки
батча зависит от размера батча и максимальной длины токенов; multi-bin batching; стабильность по
KV-cache. most-queue имеет пакетный **приход** (Mˣ/M/1), но не пакетное **обслуживание**
(M/G^[b]/1, Mˣ/M/c с групповым сервисом).

### 5. Power-of-d-choices / JSQ(d) / mean-field 🔥🔥
Балансировка нагрузки диспетчером, fluid/mean-field пределы, sub-Halfin–Whitt режимы. Активно, но
более «теоретично» (ODE предельных режимов, а не калькуляторы моментов) — слабее ложится на нишу.

### 6. Queueing-inventory systems 🔥
Совмещение очередей и управления запасами, game-theoretic анализ, случайные среды. Активно в
отдельных сообществах, ниша умеренная.

## Gap-анализ

| Направление | Активность | В most-queue | Fit ниши | Усилия | Эпик |
|---|---|---|---|---|---|
| Age of Information | 🔥🔥🔥 | нет | очень высокий | низко-средние | [EPIC-010](../epics/EPIC-010-age-of-information.md) |
| Multiserver-job (MSJ) | 🔥🔥🔥 | нет | высокий | средне-высокие | [EPIC-011](../epics/EPIC-011-multiserver-job.md) |
| Bulk-service (LLM) | 🔥🔥 | частично | высокий | средние | [EPIC-012](../epics/EPIC-012-bulk-service.md) |
| Predictions-scheduling | 🔥🔥 | да (SPJF) | высокий | низкие | [EPIC-013](../epics/EPIC-013-predictions-scheduling.md) |
| Power-of-d / mean-field | 🔥🔥 | нет | средний | средние | (кандидат на будущее) |
| Queueing-inventory | 🔥 | нет | средний | средние | (кандидат на будущее) |

## Решение

К реализации приняты четыре направления (эпики EPIC-010…013). Рекомендуемый порядок:
1. **AoI** (быстрая победа, большой охват, идеальный fit) →
2. **Predictions-scheduling** (низкие усилия, есть база SPJF) →
3. **Bulk-service** (практично, дополняет batch-приход) →
4. **MSJ** (нишевый флагман уровня RDR, наибольшая новизна).

Power-of-d и queueing-inventory оставлены кандидатами на будущее.

## Обновление — вторая волна (2026-07)

Первая волна **реализована и выпущена в v2.9**: AoI ([EPIC-010](../epics/EPIC-010-age-of-information.md)),
MSJ ([EPIC-011](../epics/EPIC-011-multiserver-job.md)), bulk-service ([EPIC-012](../epics/EPIC-012-bulk-service.md)),
predictions-degradation ([EPIC-013](../epics/EPIC-013-predictions-scheduling.md)), плюс полный RDR-стек
([EPIC-009](../epics/EPIC-009-rdr-priority.md)). Повторный обзор литературы выявил следующие
активные направления, ещё не покрытые калькуляторами (симуляционный power-of-d уже есть в туториале
`power_of_two_choices`, но аналитической модели нет).

### A. Балансировка нагрузки / диспетчеризация (mean-field) 🔥🔥🔥
JSQ, **JSQ(d) / power-of-d**, **JIQ (join-the-idle-queue)**, I1F. Fluid/mean-field пределы дают долю
серверов с ≥k заявками (для power-of-d — классическая двойная экспонента `s_k = ρ^{(d^k−1)/(d−1)}`),
среднее время отклика, свойства нечувствительности и «zero-waiting» режимы. Очень активно (крупные
ЦОД), нет open-source аналитики. Реально через фиксированную точку / ODE; дополняет уже готовый
power-of-d симулятор в туториале.

### B. Polling-системы (циклический сервер) 🔥🔥
Один сервер циклически обходит Q очередей с временами переключения (switchover); дисциплины
exhaustive / gated / k-limited. Известны средние времена ожидания по очередям и **псевдо-закон
сохранения** (взвешенная сумма ожиданий). Классика, до сих пор активна (token-ring, USB/Bluetooth
polling, IoT-шлюзы, обслуживание/ТОиР). Нет в OSS; хорошо ложится на нишу моментов.

### C. Нестационарные очереди Mt/M/c 🔥🔥
Зависящая от времени интенсивность λ(t) (суточные профили). Аппроксимации **PSA (pointwise
stationary)** и **MOL (modified offered load)** для нестационарной вероятности блокировки/задержки.
Практично для планирования штата/ёмкости; аппроксимации реализуемы и валидируемы симуляцией.

### D. Расширения fork-join / параллелизма 🔥🔥
Гетерогенные серверы, (k,l)-fork-join, тяжёлые хвосты, DAG/task-graph (датацентр-параллелизм,
speedup-функции). У most-queue есть базовый Fork-Join; точный анализ при k>2 — только границы/
аппроксимации.

### E. Queueing-inventory 🔥
Совмещение очередей и запасов (по-прежнему кандидат; ниша умеренная).

### Обновлённый gap-анализ

| Направление | Активность | В most-queue | Fit ниши | Усилия | Кандидат-эпик |
|---|---|---|---|---|---|
| Load balancing (JSQ/power-of-d/JIQ, mean-field) | 🔥🔥🔥 | только симулятор в туториале | высокий | средние | EPIC-014 |
| Polling-системы (циклический сервер) | 🔥🔥 | нет | высокий | средние | EPIC-015 |
| Нестационарные Mt/M/c (PSA/MOL) | 🔥🔥 | нет | средне-высокий | низко-средние | EPIC-016 |
| Fork-join / параллелизм (extend) | 🔥🔥 | частично | средний | средне-высокие | (будущее) |
| Queueing-inventory | 🔥 | нет | средний | средние | (будущее) |

**Рекомендация (порядок):** (1) **Load balancing mean-field** — самый горячий, завершает уже начатую
в туториале тему реальным калькулятором; (2) **Polling** — классика с чёткими формулами и широким
применением; (3) **Нестационарные Mt/M/c** — практичный и низкозатратный. Fork-join-расширения и
queueing-inventory — на будущее.

## Источники

- Sohn & Maguluri (ред.), *Queueing, Predictions, and Large Language Models: Challenges and Open
  Problems*, Stochastic Systems, 2025. <https://pubsonline.informs.org/doi/10.1287/stsy.2025.0106>
  (arXiv <https://arxiv.org/pdf/2503.07545>)
- Grosof, Harchol-Balter, Scheller-Wolf, *The Multiserver-Job Queueing Model*, QUESTA 2022.
  <https://www.cs.cmu.edu/~harchol/Papers/QUESTA22.pdf>
- Grosof et al., *Optimal Scheduling in the Multiserver-job Model under Heavy Traffic*, ACM
  SIGMETRICS. <https://dl.acm.org/doi/10.1145/3570612>
- *The saturated Multiserver Job Queuing Model with two classes of jobs: Exact and approximate
  results*, Performance Evaluation. <https://www.sciencedirect.com/science/article/abs/pii/S0166531623000408>
- *Multiserver-job Response Time under Multilevel Scaling*, 2025. <https://arxiv.org/pdf/2505.04754>
- *A Queueing Theoretic Perspective on Low-Latency LLM Inference with Variable Token Length*, 2024.
  <https://arxiv.org/abs/2407.05347>
- *Multi-Bin Batching for Increasing LLM Inference Throughput*, 2024. <https://arxiv.org/pdf/2412.04504>
- Yates et al., *Age of Information: An Introduction and Survey*. <https://user.eng.umd.edu/~ulukus/papers/journal/aoi-survey.pdf>
- *Age of Information in Unreliable Tandem Queues*, IEEE ISIT 2024. <https://arxiv.org/html/2506.09245>
- *Learning-Augmented Priority Queues*, 2024. <https://arxiv.org/pdf/2406.04793>
- Queueing Systems (Springer) — scope/тренды журнала. <https://link.springer.com/journal/11134>
- Ciw (Python DES). <https://github.com/CiwPython/Ciw> · LINE. <https://line-solver.sourceforge.net/doc/LINE-python.pdf>

**Вторая волна (2026-07):**
- *Mean-field analysis of load balancing principles in large scale systems*, 2023–2025.
  <https://arxiv.org/pdf/2307.04360>
- *Large-System Insensitivity of Zero-Waiting Load Balancing Algorithms*.
  <https://arxiv.org/pdf/2202.07971>
- Lu et al., *Join-Idle-Queue: A Novel Load Balancing Algorithm for Dynamically Scalable Web
  Services*, Performance Evaluation 2011. <https://www.microsoft.com/en-us/research/wp-content/uploads/2011/10/idleq.pdf>
- *Sparse Mean Field Load Balancing in Large Localized Queueing Systems*, 2024. <https://arxiv.org/html/2312.12973>
- Boxma & Groenevelt, *Pseudo-conservation laws in cyclic-service systems*, J. Appl. Prob.
  <https://www.cambridge.org/core/journals/journal-of-applied-probability/article/abs/pseudoconservation-laws-in-cyclicservice-systems/78058F3397B13F0AE0389BD035598786>
- Takagi, *Queuing analysis of polling models*, ACM Computing Surveys. <https://dl.acm.org/doi/10.1145/62058.62059>
- *Stability of Polling Systems for a Large Class of Markovian Switching Policies*, 2025. <https://www.arxiv.org/pdf/2504.13315>
- *Stabilizing the virtual response time in single-server PS queues with slowly time-varying arrival
  rates*. <https://arxiv.org/pdf/1811.01611> (Mt-очереди, PSA/MOL)
- *Open problems in queueing theory inspired by datacenter computing*, QUESTA.
  <https://link.springer.com/article/10.1007/s11134-020-09684-6>
