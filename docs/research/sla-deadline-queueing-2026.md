# SLA / deadline-aware очереди: обзор литературы и gap-анализ (2026-09)

- **Дата:** 2026-09-29
- **Источники:** OpenAlex, Crossref, arXiv (скилл `lit-search`: LLM inference serving,
  queueing-inventory, layered queueing networks, fork-join heavy-tail, energy/carbon-aware
  scheduling, learning-augmented scheduling, EDF/deadline queueing, GPU-cluster SLO scheduling,
  machine repair heterogeneous repairmen, accumulating priority) + инвентаризация кода
- **Эпик по итогам:** [EPIC-021](../epics/EPIC-021-slo-deadline-queueing.md)

Назначение: проверить, что изменилось в трендах теории очередей после `queueing-trends-2026.md`
(2026-07) и резервных пунктов `priority-queues-2026.md` / `unreliable-queues-2026.md` /
`queueing-networks-2026.md`, и выбрать следующее направление реализации. Все 20 эпиков
(EPIC-001…020) на момент обзора закрыты — покрыты AoI, MSJ, bulk-service, predictions-scheduling,
load balancing, polling, нестационарные Mt/M/c, сети волна 1-2, ненадёжные приборы, приоритеты
волна 2.

## 1. Что уже позволяет сделать код (важно для gap-анализа)

most-queue уже умеет **моменты → параметры распределения → хвостовая вероятность**, просто это
нигде не собрано в единый публичный слой:

- `most_queue.random.utils.fit`: `fit_h2` / `fit_h2_clx` (H2 по 2-3 моментам, требует cv ≥ 1),
  `fit_gamma` (по 2-3 моментам, любой cv > 0), `fit_erlang` (целый r, cv ≤ 1), `fit_cox`,
  `fit_weibull`, `fit_pareto_moments`.
- `most_queue.random.distributions.*Distribution.get_tail(params, t)` — уже реализовано для
  Weibull, Uniform, H2, Pareto, Erlang; для Gamma есть `get_cdf` (хвост = `1 - cdf`).
- `most_queue.theory.srpt.utils.load_below.upper_bound(cdf_fn, p)` — бисекция «найти x, при
  котором хвост < p» — по сути уже квантильный поиск, но привязан к SRPT-модулю.
- `most_queue.sim.base.QsSim.refresh_w_stat/refresh_v_stat` — паттерн онлайн-накопления
  статистики без хранения сырых выборок; естественное место добавить онлайн-счётчики превышения
  дедлайна для кросс-валидации sim vs theory.

Иными словами, направление «вероятность нарушения дедлайна / SLO-квантиль» почти не требует новой
математики — это склейка уже существующих строительных блоков в один переиспользуемый слой поверх
**любого** калькулятора, возвращающего моменты ожидания/пребывания (`QueueResults`,
`MulticlassResults`, `PriorityResults`).

## 2. Активные направления сообщества (2025-2026)

### A. SLO/deadline-aware LLM inference serving 🔥🔥🔥

Взрывной рост именно в 2025-2026: throttLL'eM (HPCA 2025, 14 цит., energy-aware GPU throttling),
**QUARTZ: Quantile-Aware Routing and Queueing for TTFT SLOs in LLM Serving** (ACL Findings 2026),
**Hermes: Efficient Serving of LLM Applications with Probabilistic Demand Modeling** (ACM TACO
2026), «Tool-Augmented LLM Serving Under Firm Deadlines: A Queueing-Control Approach» (2026),
«Optimal Scheduling Algorithms for LLM Inference: Theory and Practice» (ACM SIGMETRICS/POMACS
2025), TailGuard → «A Tail Latency SLO Guaranteed Task Scheduling Scheme for User-Facing Services»
(IEEE TPDS 2025). Общий знаменатель: не просто среднее время ответа, а **вероятность/квантиль
превышения дедлайна** (TTFT — time-to-first-token) под управлением диспетчера. Это прямой
наследник отложенного пункта «EDF/deadline scheduling» из `priority-queues-2026.md` (§2.6, п.6
резерва), но с гораздо более сильным прикладным фронтом, чем классический real-time-systems EDF
(который сам по себе почти не развивается — см. ниже).

### B. Queueing-inventory systems 🔥🔥 (подтверждение резерва)

Стабильно жива: «Analysis of junior servers approaching a senior server in the multi-server
queueing-inventory system» (Sci Reports 2025), «A Queueing Inventory System with Two Classes of
Customers» (2025), «Analysis of stochastic queueing-inventory system with idle-time preparatory
work» (2026). Остаётся кандидатом, но ниша прикладная (склад+очередь), а не «чистая СМО».

### C. Fork-join: heavy-tailed / extreme-value хвосты 🔥🔥 (подтверждение резерва)

«Extreme values for the waiting time in large fork-join queues», Queueing Systems 2025 (прямое
продолжение «Fork–join and redundancy systems with heavy-tailed job sizes», Queueing Systems
2022). Прямое расширение уже реализованного Fork-Join модуля.

### D. Machine repair: warm standby / гетерогенные ремонтники / retrial-MRP 🔥🔥 (подтверждение резерва)

«Multi-Objective Optimization of a Multi-Server Retrial Machine Repair System with Orbital
Search», Computation 2026; «A Comprehensive Survey on Machine Repair Problems with Standby
Systems», 2026 (свежий survey — сигнал живой темы). Расширяет EPIC-019.

### E. Layered Queueing Networks (LQN) — деприоритезировано

Активность в основном исторична (IEEE TSE 2009/2013, ACM workshops 2005-2012); свежих (2024-2026)
работ почти не нашлось. Оставляем в дальнем резерве networks-2026.

### F. Energy/carbon-aware scheduling — деприоритезировано для ядра

Активно (несколько статей 2025-2026: carbon-aware AI datacenter scheduling, green Kubernetes
scheduling), но почти все работы — оптимизационные/ML/системные, а не аналитические калькуляторы
моментов. Плохой fit ниши most-queue (точные моменты + DES-кросс-валидация). Не исключает
tutorial-уровня применения поверх уже готовых моделей.

### G. Learning-augmented scheduling — уже покрыто

«The Cost of Accurate Predictions in Learning-Augmented Scheduling» (RTCSA 2025), «Dynamic
scheduling with convex delay costs revisited» (Queueing Systems 2025) — тема жива, но уже закрыта
в EPIC-013 (predictions-degradation); новых открытых задач, не покрытых `degradation.py`, не
найдено.

### H. Классический EDF (real-time systems) — не поднимать отдельно

Литература почти вся из 1990-2015 (schedulability analysis для sporadic/DAG-tasks); собственной
новой активности в 2025-2026 не нашлось. Ценность появляется только в сочетании с LLM-serving-SLO
углом (пункт A) — то есть не как отдельная real-time-systems тема, а как «deadline-violation
probability» слой поверх существующих моделей.

## 3. Gap-анализ

| Направление | Активность | В most-queue | Fit ниши | Усилия |
|---|---|---|---|---|
| SLO/deadline-violation probability (+ LLM-serving TTFT) | 🔥🔥🔥 | нет (есть все строительные блоки: fit_*, get_tail, load_below) | очень высокий | низко-средние |
| Queueing-inventory | 🔥🔥 | нет | средний | средние |
| Fork-join heavy-tailed / extreme value | 🔥🔥 | частично (базовый Fork-Join есть) | высокий | средние |
| Machine repair: standby/heterog. repairmen/retrial | 🔥🔥 | частично (EPIC-019) | средне-высокий | малые-средние |
| Layered Queueing Networks | 🔥 (историческая) | нет | средний | большие |
| Energy/carbon-aware scheduling | 🔥🔥 | нет | низкий (не моменты-калькулятор) | средние, но плохой fit |

## 4. Решение

К реализации выбрано направление **A — SLO/deadline-violation probability**, детальный план —
[EPIC-021](../epics/EPIC-021-slo-deadline-queueing.md) +
[roadmap](../roadmaps/slo_deadline_roadmap.md). Причины:

1. Самый горячий прикладной фронт 2025-2026 (LLM inference serving TTFT SLO), при этом
   математически — это не RL/ML, а вероятность/квантиль хвоста времени ожидания, что прямо
   попадает в нишу «точные моменты + воспроизводимость».
2. Почти вся нужная инфраструктура уже в репозитории (`fit_h2/fit_gamma/fit_erlang`, `get_tail`,
   `upper_bound`-бисекция, паттерн `refresh_w_stat` для sim) — низкие усилия, высокая отдача.
3. Применим сразу ко **всем** существующим калькуляторам, возвращающим моменты (M/G/1, M/G/n,
   MAP/PH-стек, приоритеты, bulk-service, retrial) — не новый изолированный модуль, а
   горизонтальный слой поверх готового каталога моделей.
4. Естественно стыкуется с уже реализованными EPIC-012 (bulk-service = батч-инференс) и EPIC-013
   (predictions) для готового «LLM-serving SLO» композитного примера.

Резерв на будущее (без изменений в приоритете): queueing-inventory, fork-join heavy-tail, machine
repair standby/гетерогенные ремонтники, LQN.

## 5. Источники

**SLO/deadline/LLM-serving:**
- *throttLL'eM: Predictive GPU Throttling for Energy Efficient LLM Inference Serving*, HPCA 2025,
  doi:10.1109/hpca61900.2025.00103.
- *QUARTZ: Quantile-Aware Routing and Queueing for TTFT SLOs in LLM Serving*, ACL Findings 2026,
  doi:10.18653/v1/2026.findings-acl.1888.
- *Hermes: Efficient Serving of LLM Applications with Probabilistic Demand Modeling*, ACM TACO
  2026, doi:10.1145/3803390.
- *Tool-Augmented LLM Serving Under Firm Deadlines: A Queueing-Control Approach*, 2026,
  doi:10.2139/ssrn.6438169.
- *A Control-Oriented Survey of Load Balancing for LLM Inference Serving*, 2026,
  doi:10.2139/ssrn.6516666.
- *Optimal Scheduling Algorithms for LLM Inference: Theory and Practice*, Proc. ACM Meas. Anal.
  Comput. Syst. 2025, doi:10.1145/3771574.
- *A Tail Latency SLO Guaranteed Task Scheduling Scheme for User-Facing Services*, IEEE TPDS 2025,
  doi:10.1109/tpds.2025.3542638 (продолжение TailGuard, ICDCS 2023, doi:10.1109/icdcs57875.2023.00042).
- *LASSY: A Latency-Aware SLOs-Sufficing Scheduling System for the Cloud/Edge Continuum*, CCGrid
  2025, doi:10.1109/ccgrid64434.2025.00029.
- Real-time queues in heavy traffic with earliest-deadline-first queue discipline, Ann. Appl.
  Prob. 2001, doi:10.1214/aoap/1015345295 — классическая опора EDF-в-очередях.
- Уже в `queueing-trends-2026.md`: *A Queueing Theoretic Perspective on Low-Latency LLM Inference
  with Variable Token Length*, arXiv:2407.05347; *Multi-Bin Batching for Increasing LLM Inference
  Throughput*, arXiv:2412.04504.

**Резервные направления (подтверждение активности):**
- *Analysis of junior servers approaching a senior server in the multi-server queueing-inventory
  system*, Sci. Reports 2025, doi:10.1038/s41598-025-99748-5.
- *Extreme values for the waiting time in large fork-join queues*, Queueing Systems 2025,
  doi:10.1007/s11134-025-09937-2.
- *Multi-Objective Optimization of a Multi-Server Retrial Machine Repair System with Orbital
  Search*, Computation 2026, doi:10.3390/computation14070153.
- *A Comprehensive Survey on Machine Repair Problems with Standby Systems*, 2026,
  doi:10.4038/sljas.v27i1.8226.
