# Сети массового обслуживания: обзор литературы и gap-анализ (2026)

- **Дата:** 2026-07-13
- **Источники:** OpenAlex, Crossref, arXiv (скилл lit-search: запросы по декомпозиции,
  MVA/закрытым сетям, fork-join, блокировкам, G-networks, LQN, RL-управлению,
  нестационарным сетям)
- **Эпик по итогам:** [EPIC-017](../epics/EPIC-017-networks-exact-methods.md)

Назначение документа: зафиксировать ландшафт публикаций по сетям МО и разрывы
с текущими возможностями библиотеки — чтобы помнить, куда копать дальше.

## 1. Что есть в most_queue сейчас

Все аналитические сетевые модели — **открытые** сети с пуассоновским внешним потоком,
решаемые одной и той же приближённой декомпозицией (уравнения баланса потоков →
поузловой расчёт Такахаси–Таками → Gamma-аппроксимация ПЛС времени пребывания →
обращение матрицы `(I−N·Q)⁻¹`):

| Модуль | Модель |
|---|---|
| `theory/networks/open_network.py` | открытая сеть, один класс (M/G/n узлы) |
| `theory/networks/open_network_prty.py` | мультикласс с приоритетами (PR/RS/RW/NP) |
| `theory/networks/negative_network.py` | сеть с отрицательными заявками (DISASTER/RCS) |
| `theory/networks/opt/transition*.py` | оптимизация матрицы маршрутизации (Рыжиков 2019) |
| `sim/networks/*` | имитационные аналоги + RCH/RCE, кастомные события |

Смежные, но НЕ интегрированные с сетевым модулем: `theory/load_balancing.py`
(JSQ/JSQ(d)/JIQ, mean-field, EPIC-014), `theory/polling.py` (EPIC-015),
`theory/fork_join/` (изолированный fork-join, не внутри сети).

**Ключевые пробелы** (подтверждены [models_gap_analysis](../models_gap_analysis.md)
и EPIC-001): нет закрытых сетей вообще (ни Gordon–Newell, ни MVA, ни свёртки Бьюзена),
нет BCMP, нет точного product-form даже для марковской открытой сети (Джексон), нет QNA
(непуассоновские внутренние потоки), нет сетей с блокировками/конечными буферами,
негативные сети — только один позитивный класс и аналитика отстаёт от симулятора
(RCH/RCE есть только в симуляторе).

## 2. Ландшафт публикаций (по направлениям)

### 2.1 Классика product-form и закрытые сети (фундамент, максимальная цитируемость)

- Baskett F., Chandy K.M., Muntz R.R., Palacios F.G., Open, Closed, and Mixed Networks
  of Queues with Different Classes of Customers, Journal of the ACM, 1975,
  doi:10.1145/321879.321887 — **BCMP**, 2447 цит.
- Buzen J.P., Computational algorithms for closed queueing networks with exponential
  servers, Communications of the ACM, 1973, doi:10.1145/362342.362345 — свёртка /
  нормализационная константа, 868 цит.
- Reiser M., Lavenberg S.S., Mean-Value Analysis of Closed Multichain Queuing Networks,
  Journal of the ACM, 1980, doi:10.1145/322186.322195 — **MVA**, 1162 цит.
- Denning P.J., Buzen J.P., The Operational Analysis of Queueing Network Models,
  ACM Computing Surveys, 1978, doi:10.1145/356733.356735 — операционный анализ, 581 цит.
- Reiser M., Mean-value analysis and convolution method for queue-dependent servers in
  closed queueing networks, Performance Evaluation, 1981, doi:10.1016/0166-5316(81)90040-7.
- Approximate Mean Value Analysis for Closed Queuing Networks with Multiple-Server
  Stations, 2007 — approx-MVA для многоканальных станций; семейство Schweitzer/Bard.

Open-source ориентир по покрытию: queueing package для GNU Octave (arXiv:2209.04220) —
MVA, свёртка, марковские цепи.

### 2.2 Декомпозиционные аппроксимации открытых GI/G-сетей (развитие нашего текущего метода)

- Whitt W. (QNA-линия): Towards better multi-class parametric-decomposition approximations
  for open queueing networks, Annals of Operations Research, 1994, doi:10.1007/bf02024659;
  Variability Functions for Parametric-Decomposition Approximations of Queueing Networks,
  Management Science, 1995, doi:10.1287/mnsc.41.10.1704.
- Kim S., Modeling Cross Correlation in Three-Moment Four-Parameter Decomposition
  Approximation of Queueing Networks, Operations Research, 2011, doi:10.1287/opre.1100.0893.
- Whitt W., You W., A Robust Queueing Network Analyzer Based on Indices of Dispersion,
  2019, doi:10.48550/arxiv.2003.11174 — **RQNA**, современное переиздание QNA через
  индексы дисперсии.
- Bandi C., Bertsimas D., Youssef N., Performance Analysis of Queueing Networks via Robust
  Optimization, arXiv:1009.3948, 2010 — robust-optimization альтернатива.

### 2.3 Сети с блокировками / конечные буферы

- Balsamo S., de Nitto Personé V., Onvural R., Analysis of Queueing Networks with Blocking,
  Kluwer, 2001, doi:10.1007/978-1-4757-3345-7 — монография-справочник.
- Perros H., Queueing networks with blocking (обзор), ACM SIGMETRICS PER, 1984,
  doi:10.1145/1041823.1041824.
- Dallery Y., Frein Y., On Decomposition Methods for Tandem Queueing Networks with
  Blocking, Operations Research, 1993, doi:10.1287/opre.41.2.386.
- Brandwajn A., Jow Y.-L., An Approximation Method for Tandem Queues with Blocking,
  Operations Research, 1988, doi:10.1287/opre.36.1.73.

### 2.4 G-networks (отрицательные заявки — специализация репо)

- Gelenbe E., Product-form queueing networks with negative and positive customers,
  Journal of Applied Probability, 1991, doi:10.2307/3214499 — 374 цит., **точный
  product-form**.
- Gelenbe E., G-networks with multiple classes of negative and positive customers,
  Theoretical Computer Science, 1996, doi:10.1016/0304-3975(95)00018-6.
- Gelenbe E., Fourneau J.-M., G-networks with resets, Performance Evaluation, 2002,
  doi:10.1016/s0166-5316(02)00127-x.
- Chao X., Miyazawa M., Pinedo M., Queueing Networks: Customers, Signals and Product Form
  Solutions, Wiley, 1999 — монография.

### 2.5 Fork-join в сетях

- Response times in M/M/s fork-join networks, Advances in Applied Probability, 2004,
  doi:10.1239/aap/1093962238.
- Efficient Response Time Approximations for Multiclass Fork and Join Queues in Open and
  Closed Queuing Networks, IEEE TPDS, 2013, doi:10.1109/tpds.2013.70 — fork-join **внутри**
  открытых/закрытых сетей, мультикласс.
- Approximate analysis of finite fork/join queueing networks, Computers & IE, 1997,
  doi:10.1016/s0360-8352(97)00010-7.
- Кривулин Н., Algebraic modelling and performance evaluation of acyclic fork-join
  queueing networks ((max,+)-алгебра), arXiv:1212.4648, 2012.

### 2.6 Layered Queueing Networks (клиент-серверные программные системы)

- Franks G., Al-Omari T., Woodside M. и др., Enhanced Modeling and Solution of Layered
  Queueing Networks, IEEE TSE, 2009, doi:10.1109/tse.2008.74.
- Tribastone M., A fluid model for layered queueing networks, IEEE TSE, 2013,
  doi:10.1109/tse.2012.66.

### 2.7 Нестационарные и transitory-сети

- Liu Y., Whitt W., The Gt/GI/st+GI many-server fluid queue, Queueing Systems, 2012,
  doi:10.1007/s11134-012-9291-0.
- Honnappa H., Jain R., Ward A., Transitory Queueing Networks, arXiv:1708.05921, 2017.
- Transient behaviour of time-varying tandem queueing networks, OPSEARCH, 2024,
  doi:10.1007/s12597-024-00790-0.

### 2.8 Управление сетями и ML (горячий фронт; скорее вне ядра библиотеки)

- Dai J.G., Gluzman M., Queueing Network Controls via Deep Reinforcement Learning,
  Stochastic Systems, 2022, doi:10.1287/stsy.2021.0081.
- Reinforcement learning in queues, Queueing Systems, 2022, doi:10.1007/s11134-022-09844-w.
- NeuraliNQ: a neural network method for the transient performance analysis in
  non-Markovian Queues, Queueing Systems, 2025, doi:10.1007/s11134-025-09952-3.
- Gamarnik D., Katz D., On deciding stability of multiclass queueing networks under buffer
  priority scheduling policies (недецидируемость устойчивости), arXiv:0708.1034.

## 3. Вывод: кандидаты на реализацию (ранжировано)

| # | Что | Опора | Почему | Оценка |
|---|-----|-------|--------|--------|
| 1 | **Закрытые сети, один класс: MVA + свёртка Бьюзена** (+ approx-MVA Schweitzer/Bard, многоканальные станции) | Reiser–Lavenberg 1980; Buzen 1973 | Самый цитируемый пробел; точный метод; `should` в gap-analysis; README уже обещает | средняя |
| 2 | **Точный Джексон product-form** для марковской открытой сети | Jackson 1957/1963 | Дёшево; точный эталон для валидации текущей декомпозиции | малая |
| 3 | **QNA/RQNA: двухмоментное распространение вариабельности внутренних потоков** | Whitt 1983/1994; Whitt–You 2019 | Прямой апгрейд декомпозиции: снимает допущение пуассоновских внутренних потоков; двухмоментная узловая инфраструктура уже есть | средняя |
| 4 | **G-network точный product-form (Gelenbe)** | Gelenbe 1991/1996 | Синергия с фирменной темой (negative customers); точный эталон + мультикласс сигналов | средняя |
| 5 | **BCMP / мультиклассовый MVA** (open/closed/mixed; FCFS/PS/LCFS-PR/IS) | BCMP 1975; Reiser–Lavenberg 1980 | Логичное завершение п. 1–2; `could` в gap-analysis | большая |
| 6 | Тандемы/сети с блокировками (конечные буферы) | Dallery–Frein 1993; Balsamo 2001 | Производственные линии — частый практический запрос | средняя-большая |
| 7 | Fork-join внутри маршрутизируемой сети, (k,l)-fork-join | TPDS 2013; AAP 2004 | «Будущее» в trends-2026 §D; сейчас fork-join изолирован | большая |
| 8 | Transient/time-varying сети (PSA поверх декомпозиции), LQN, RL-управление | OPSEARCH 2024; TSE 2009; Dai–Gluzman 2022 | После 1–6; RL — скорее tutorial, не ядро | — |

Пункты 1–5 реализованы в [EPIC-017](../epics/EPIC-017-networks-exact-methods.md) (done,
2026-07-13); пункты 6–8 — резерв на следующие волны («куда копать дальше»), плюс
перенесённые из эпика мелочи: RQNA-поправки, мультикласс сигналов G-сетей, mixed BCMP,
многоканальные станции в мультичейн-MVA.
