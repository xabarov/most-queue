У меня достаточно материала, чтобы собрать полноценный обзор. Теперь я организую информацию в структурированный ответ по направлениям.

# Современная теория очередей: направления исследований и ключевые работы

Теория очередей (queueing theory) переживает ренессанс благодаря пересечению с машинным обучением, облачными вычислениями, LLM-системами и новыми математическими методами. Ниже — подробный обзор основных направлений на 2024–2026 гг.

---

## 🧭 Обзор: как выглядит область сегодня

Классическая «труба» Erlang → Kendall → Kleinrock → Neuts сегодня ветвится на несколько живых направлений. Условно их можно разделить на **три слоя**:

1. **Математический слой** — диффузионные приближения, heavy-traffic, Stein's method, BAR (basic adjoint relationship), матрично-аналитические методы.
2. **Алгоритмический слой** — scheduling с предсказаниями (learning-augmented algorithms), reinforcement learning для управления очередями, load balancing.
3. **Прикладной слой** — LLM-инференс, ЦОДы, ride-hailing, call-центры, здравоохранение, телеком (MAP/PH, retrial).

<details>
<summary><b>📊 Карта связей направлений (кликните, чтобы развернуть)</b></summary>

```
              ┌─── Stein's method / BAR ──── Heavy-traffic limits
  МАТЕМАТИКА ─┼─── Matrix-Analytic (QBD, LDQBD)
              └─── Mean-field, rank-based diffusions
                           │
                           ▼
              ┌─── Learning-augmented scheduling (SPRPT, SPJF)
  АЛГОРИТМЫ ──┼─── RL для маршрутизации / диспетчеризации  
              └─── SRPT в multi-server, fork-join
                           │
                           ▼
              ┌─── LLM inference (KV-cache, prefill/decode)
  ПРИЛОЖЕНИЯ ─┼─── Cloud / datacenters (tail latency)
              ├─── Ride-hailing, matching
              └─── Retrial queues (телеком, call-центры)
```
</details>

---

## 1. 🤖 Scheduling с предсказаниями и LLM-системы

Это, пожалуй, **самое горячее направление** 2023–2026 гг. Основной вопрос: как использовать ML-предсказания размера задач (job size) в классических дисциплинах типа SRPT?

### Ключевые работы

| Работа | Ключевая идея | Метод |
|---|---|---|
| **Mitzenmacher & Shahout (2025)** "Queueing, Predictions, and LLMs: Challenges and Open Problems", *Stochastic Systems* | Обзор use predictions в очередях + новые задачи планирования LLM-инференса (KV-cache, prefill/decode, preemption cost) | SOAP-методология, M/G/1 с двумерным распределением $g(x,y)$ истинного/предсказанного размера |
| **Mitzenmacher (2020)** | Ввёл варианты SPRPT, SPJF, PSPJF — size-based дисциплины на предсказаниях | Аналитические формулы среднего response time с плотностью $g(x,y)$ |
| **Scully et al. (2022)** SOAP framework | Универсальный способ анализа политик, где ранг задачи зависит от её типа и накопленного обслуживания | Rank-based analysis M/G/1 |
| **Salman et al. (2023)** | Scheduling с дедлайнами при наличии предсказаний | Learning-augmented algorithms |

**Почему это важно:** в LLM-системах (Llama, DeepSeek) инференс-запросы имеют переменную длину, динамическую память KV-cache и дорогое прерывание. Стандартные M/G/1-модели туда не ложатся — открыто море задач.

### Открытые вопросы (из Mitzenmacher–Shahout):
- Как учитывать **стоимость** preemption (KV-cache → GPU/CPU/recompute)?
- Маршрутизация между LLM разного размера (cost–quality trade-off).
- Планирование в **compound AI systems** с вызовами внешних API.
- «Размер» запроса в reasoning-системах — это число токенов или число шагов рассуждения?

---

## 2. 🎰 Reinforcement Learning для управления очередями

### Ключевые работы

<details open>
<summary><b>Подробный список (развернуть/свернуть)</b></summary>

**1. Liu, Xie, Modiano (MIT, 2019+)** — "Reinforcement Learning for Optimal Control of Queueing Systems"
- **Проблема:** классический UCRL/PSRL работают только на конечных пространствах состояний, а у очередей буфер неограничен.
- **Решение:** алгоритм **PDGRL** (Piecewise Decaying ε-Greedy RL) — применяет model-based RL на конечном подмножестве $\{Q : Q_{\max} \le U\}$, а за его пределами — стабилизирующую политику $\pi_0$ (типа MaxWeight).
- **Результат:** средний backlog $\to \rho^*$ с ошибкой $O(\text{poly}(U)/\exp(\text{poly}(U)))$.

**2. Jali, Qu, Wang, Joshi (AISTATS 2024)** — "Efficient RL for Routing Jobs in Heterogeneous Queueing Systems"
- **Проблема:** оптимальная пороговая политика известна только для 2 серверов (один быстрый, один медленный); state space экспоненциально велик.
- **Решение:** алгоритм **ACHQ** — policy gradient с **low-dimensional soft threshold parameterization**, использующей структуру очереди.
- **Гарантии:** сходимость к stationary point в общем случае; к approximate global optimum для 2 серверов. Улучшение response time на ~30% против greedy-политики «routing to fastest».

**3. Kim & Vojnovic (2024, v4 2025)** — "Learning to Schedule in Parallel-Server Queues with Stochastic Bilinear Rewards"
- Bandit-алгоритм для scheduling с билинейной структурой награды (feature-based). 
- Регрет и mean holding cost $\tilde{O}(\sqrt{T})$ при сохранении стабильности очереди.

**4. Kanikanti et al. (2025) + Keller (2026)** — Deep Q-learning + optimal queueing для cloud scheduling
- Интеграция queue length в состояние и reward deep Q-learning агента.
- Сравнение с SARSA, actor-critic, DDPG в задачах microgrid/cloud.

**5. Efrosinin, Vishnevsky, Stepanova, Sztrik (2025)** — "Use Cases of ML in Queueing Theory Based on a GI/G/K System", *Mathematics*
- Обзор применения ML (регрессия, классификация) для GI/G/K, где аналитика невозможна.
- Симуляция → обучающая выборка → предсказание mean sojourn time.

</details>

---

## 3. 📐 Heavy-traffic, диффузионные приближения и Stein's method

Это «классика», переживающая технический ренессанс. Главный прорыв — **basic adjoint relationship (BAR)** и **generator comparison** Штейна, которые дают **неасимптотические** оценки ошибки.

### Ключевые работы

**1. Braverman & Scully (2024/2025)** — "Diffusion approximation error for queueing systems with general primitives"
- **Охват:** G/G/1, G/M/∞, JSQ, tandem.
- **Главная находка:** ошибка диффузионного приближения раскладывается на **interior** (легко ограничить через первые 3 момента в метрике Вассерштейна) и **boundary** (сложные, требуют model-specific анализа).
- **Метод:** расширение generator-approach Штейна на piecewise-deterministic Markov processes (PDMP) через BAR вместо инфинитезимального генератора + формула Palm-инверсии.
- **Значение:** позволяет *калибровать модели по данным*, где оценить точное распределение нереально, а моменты — можно.

**2. Guang, Xu, Dai (2024)** — "Steady-State Convergence of Continuous-Time Routing System... in Heavy Traffic"
- **Результат:** для JSQ и Power-of-two-choices (Po2) при непрерывном времени и общих распределениях доказали, что $(rZ_1^{(r)}, \ldots, rZ_J^{(r)}) \Rightarrow (Z^*, \ldots, Z^*)$ — **state-space collapse** на линию $e=(1,\ldots,1)$.
- **Главное ослабление:** вместо «bounded support» требуется только $(2+\delta_0)$-й момент.
- **Метод:** BAR + Palm-версия.

**3. Banerjee, Budhiraja, Estevez (2024)** — "Load Balancing in Parallel Queues and Rank-based Diffusions"
- Новая схема **MSBLB / MJSQ** (marginal JSQ): большинство задач маршрутизируется случайно, но доля $O(\sqrt{n})$ — по JSQ.
- **Limit:** *constrained rank-based diffusion* типа Atlas model (известна из математических финансов).
- **Trade-off:** коммуникационная стоимость снижается в $\sqrt{n}$ раз vs JSQ, но steady-state total queue length растёт лишь на константу.

**4. Alwan & Ata (2026)** — "Heavy Traffic Diffusion Limit for a Closed Queueing Network with Single-Server and Infinite-Server Stations"
- Мотивация: **ride-hailing** (Uber, Lyft): single-server = зоны города, infinite-server = время в пути.
- Двухуровневая probabilistic routing, multidimensional reflected Brownian motion как предел.
- Метод: continuous mapping + нелинейное regulator mapping.

**5. Dębicki, Kriukov, Mandjes (2026)** — "Lévy-driven queuing networks in multi-scale light and heavy traffic"
- Сеть с upper-triangular routing, управляемая Lévy-процессом.
- Асимптотическое **decoupling** некоторых workloads при специальном скейлинге service rates.

---

## 4. 🏎 SRPT и scheduling в multi-server системах

Классическая SRPT в M/G/1 известна 50+ лет, но для M/G/k и систем с abandonment — открытая область.

### Ключевые работы

| Работа | Модель | Результат |
|---|---|---|
| **Grosof, Scully, Harchol-Balter (2018, Performance Eval.)** | M/G/k, SRPT | Первая stochastic оценка $E[T]$; asymptotic optimality в heavy traffic |
| **Dong & Ibrahim (2023)** "SRPT in Many-Server Queues with Impatient Customers" | M/GI/s+GI (с отказами) | State-space collapse: asymptotically работает как **two-class priority** (short — без ожидания, long — уходят без обслуживания). SRPT максимизирует throughput среди всех политик в overloaded regime |
| **Gieroba & Kruk (2023)** | Single-server SRPT, multi-class | Diffusion limits для multi-class SRPT с FIFO tie-breaking; пропорции workload между классами |
| **Chen & Dong (2020)** | GI/GI/1 heavy traffic | Two-class priority threshold rule асимптотически сравним с SRPT |

**Инсайты:**
- В many-server overloaded regime SRPT в $M/GI/s+GI$ **нечувствительна к распределению patience beyond mean** — в отличие от FCFS.
- Если hazard rate patience не убывает — SRPT бьёт FCFS и LCFS. Если убывает — может проигрывать FCFS по waiting time.

---

## 5. 🌿 Fork-Join очереди и tail latency в ЦОДах

Fork-Join недоступен для точного аналитического решения даже для M/M/1 узлов — поэтому упор делается на **эмпирические приближения для tail latency**.

### Ключевые работы

**1. Nguyen, Alesawi, Che, Jiang (2020, IEEE TPDS)** — "Black-Box Fork-Join Latency Prediction: ForkTail & ForkMean"
- **Идея:** не моделировать каждый узел как M/G/1, а смотреть только на *измеримые* mean и variance task response time (black-box).
- **База:** CLT для G/G/m queues в heavy traffic.
- **Формула для $p$-го перцентиля (однородный случай):**
$$x_p = -\beta \log\!\left(1 - (p/100)^{1/(k\alpha)}\right)$$
где $\alpha, \beta$ — параметры generalized exponential, вычисляемые из $E[T], V[T]$.
- **Точность:** <20% ошибка при $\rho = 80\%$, <15% при $\rho = 90\%$.

**2. Alesawi et al. (2019+)** — Tail Latency Prediction в консолидированных ЦОДах
- Target + background приложения, Fork-Join c смесью распределений.
- Closed-form для $p$-го перцентиля target-приложения; ошибки <10% при $\rho \ge 75\%$.

**3. Enganti, Rosenkrantz et al. (2022)** — ForkMV: mean-and-variance для Fork-Join networks с разными fanout degree и heavy-tailed service times.

---

## 6. 🔁 Retrial queues и matrix-analytic methods

Retrial — неочевидно живое направление, особенно в телеком-приложениях (cellular, CSMA, call-центры).

### Ключевые работы

**1. Vishnevsky, Klimenok, Semenova, Dang (2025, AIMS Math.)** — "Retrial tandem queueing system with correlated arrivals"
- **Модель:** MAP → PH-servers (tandem), finite buffers, общая orbit.
- **Метод:** для 2 узлов — полный аналитический anaisys (ergodicity condition, stationary distribution); для произвольного числа — **гибрид queueing + simulation + ML** для предсказания mean sojourn time.
- **Приложение:** linear-topology телеком-сети с retransmission.

**2. Lu (2024, Appl. Math.)** — "Exact Tail Asymptotics for a Queueing System with Retrial Orbit and Batch Service"
- Censoring technique + matrix analysis method + Karamata Tauberian theorem → точный тейл-асимптотик стационарного распределения.

**3. Sanga & Vankudothu (2025, RAIRO-OR)** — $M/M/1/K/WV$ retrial с dual-phase, F-policy, balking
- Matrix-analytical method для steady-state.
- Bi-variate cost function, оптимизация через **quasi-Newton + genetic algorithm** + валидация через **ANFIS** (adaptive neuro-fuzzy inference system).

**4. Kawanishi & Ino (2025, Mathematics, Editor's Choice)** — "Upper and Lower Bounds of Performance Metrics in Hybrid Systems with Setup Time"
- Гибридные системы с виртуальными машинами (setup time).
- **LDQBD** (level-dependent QBD) + stochastic bounding approach вместо дорогих matrix-analytic вычислений — верхние/нижние границы на стационарное распределение.

**5. Peshkova, Morozov, Pagano (2025, Mathematics)** — "Splitting-Based Regenerations for Accelerated Simulation of Queues"
- Расширение splitting на GI/M/1; увеличивает число regeneration cycles → быстрее оценка stationary metrics.

---

## 7. 🏥 Call-центры, staffing, prescriptive analytics

Более прикладное направление, где теория очередей сливается с ML и исследованиями операций.

- **Forecasting call arrivals** через temporal memory networks + gradient boosting (Expert Systems w/ Applications, 2023) — бустинг обгоняет deep learning при малых сезонных данных.
- **Transient staffing** (Int. J. Production Research, 2024) — динамическое расписание через transient Erlang-A, уменьшение over-staffing.
- **Prescriptive multi-shift staffing** (EJOR, 2023) — fluid + stationary approximations + ML для тяжелохвостых arrivals.

---

## 8. 💰 Pricing, congestion, game-theoretic queueing

**Laih (2025, Mathematics)** — "Queuing Pricing with Time-Varying and Step Tolls"
- Bottleneck model с time-varying и n-step tolls.
- Оптимальный time-varying toll **полностью убирает очередь**; оптимальный $n$-step toll создаёт структуру из $3n+2$ поведенческих групп.
- Применение: congestion pricing, dynamic tolling.

---

## 🎯 Главные тренды (summary)

<details open>
<summary><b>Развернуть ключевые выводы</b></summary>

1. **ML × Queueing** — это не «применим нейронки», а глубокая интеграция: предсказания размеров задач, RL для маршрутизации, ANFIS для валидации, симуляции как источник обучающих выборок для аналитически неразрешимых систем (GI/G/K, tandem retrial).

2. **LLM-scheduling** становится отдельной ветвью с новыми математическими задачами: KV-cache как «дополнительное состояние», preemption с переменной стоимостью, multi-phase jobs (prefill + decode).

3. **Методологически** доминируют:
 - **BAR (basic adjoint relationship)** для heavy-traffic в непрерывном времени
 - **Stein's method / generator comparison** для неасимптотических bound'ов
 - **Matrix-analytic methods** (QBD, LDQBD) для структурированных Markov-цепей
 - **Rank-based diffusions** (Atlas model) для рандомизированного load balancing

4. **Приложения**, подпитывающие теорию: ЦОДы (tail latency, fork-join), ride-hailing (closed networks), LLM-inference, здравоохранение (abandonment), 5G/telecom (MAP, PH, retrial).

5. **Открытые проблемы:** boundary terms в Stein's method для multi-clock PDMP; оптимальные политики в LLM-системах с memory-constrained preemption; RL-алгоритмы со строгими гарантиями стабильности для сетей очередей.

</details>

---

