# Reinforcement Learning в теории очередей: детальный разбор

Давайте я сначала соберу актуальные статьи с деталями по методам, архитектурам и экспериментам.

Отлично, у меня уже много материала. Давайте соберу дополнительную информацию по PPO в очередях, по Dai-Gluzman (важная работа), а также по применению RL к call-центрам и matching.

Отлично, у меня накопилось уже достаточно материала. Теперь составлю детальный, хорошо структурированный ответ.

# Reinforcement Learning в теории очередей: детальный разбор

RL в queueing — это нетривиальная задача, потому что классические RL-алгоритмы рассчитаны на **конечные** state spaces и **ограниченные** rewards, а в очередях state space **бесконечный** (длина очереди), rewards **неограничены** (holding cost растёт с длиной), и оптимизируется **long-run average cost**, а не discounted. Ниже — главные работы с подробным разбором.

---

## 🎯 Общая структура: как задача очереди превращается в MDP

Типичная формализация:

- **State** $s_t$ — вектор длин очередей $(Q_1, \ldots, Q_m)$, иногда + тип прибывшей задачи, остаточное время обслуживания, featur'ы сервера.
- **Action** $a_t$ — routing decision (в какой сервер), scheduling (какой класс обслуживать), admission (принять/отклонить), staffing (сколько агентов).
- **Reward** $r_t = -\sum_i c_i Q_i$ (negative holding cost) или $-E[\text{waiting time}]$.
- **Objective:** $\min_\pi \lim_{T\to\infty} \tfrac{1}{T}\mathbb{E}_\pi\!\left[\sum_{t=0}^{T-1} c(s_t, a_t)\right]$ — long-run average cost.
- **Uniformization** (для непрерывного времени с экспоненциальными распределениями) — приводит CTMDP к дискретному MDP.

---

## 1. 🏛 Классика современной эры: Dai & Gluzman (2020/2022) — PPO для queueing networks

> **J.G. Dai, M. Gluzman.** "Queueing Network Controls via Deep Reinforcement Learning", *Stochastic Systems* (2022), arXiv:2008.01644.

### Проблема
PPO (Schulman et al., 2017) был разработан для роботики/игр (конечные rewards, discounted return). В очередях:
1. State space $\mathbb{N}^m$ — бесконечный.
2. Holding cost не ограничен сверху.
3. Стандартная оценка relative value function $v^\pi(s)$ имеет **огромную дисперсию**.

### Метод: PPO + три техники снижения дисперсии

**1. Discounted approximation.** Вместо relative value function $v^\pi$ используется $v_\gamma^\pi$ с $\gamma$ близким к 1.

**2. Regenerative simulation.** Выбирается состояние $s^* = \mathbf{0}$ (пустая сеть) как **регенеративная точка**. Траектория разбивается на циклы между последовательными посещениями $\mathbf{0}$:
$$v_\gamma^\pi(s) \approx \mathbb{E}\!\left[\sum_{t=0}^{\tau^*-1} \gamma^t c(s_t, a_t) \,\Big|\, s_0 = s\right]$$
где $\tau^*$ — время возврата в $s^*$. Это резко снижает variance, потому что исключает «длинный хвост» вклада из будущих циклов.

**3. Approximating Martingale-Process Method (AMPM).** Добавляется control variate в виде мартингейла от текущего ценовой функции, что дополнительно уменьшает variance оценки advantage.

### Валидация

| Модель | Классическая эвристика | PPO vs. эвристика |
|---|---|---|
| **Parallel-server (N-model)** Harrison, 1998 | Threshold policy | Near-optimal, сравнимо с value iteration |
| **Criss-cross network** | MaxWeight, Longest-Queue | Лучше на 5–20% по holding cost |
| **Six-class reentrant line** (Kumar–Seidman) | FBFS, LBFS | Стабильный (FBFS/LBFS нестабильны при некоторых $\rho$) |
| **8-class multiclass** | Heuristic Lu–Kumar | 10–25% лучше |

Код открыт: [`github.com/mark-gluzman/NmodelPPO`](https://github.com/mark-gluzman/NmodelPPO) и [`MulticlassQueuingNetworkPolicyOptimization`](https://github.com/mark-gluzman/MulticlassQueuingNetworkPolicyOptimization).

### Значение
Первая работа, которая **систематически** адаптировала современный deep RL к полноценным queueing networks с теоретической проработкой policy improvement bounds для infinite-state average-reward MDP.

---

## 2. 🎛 Jali, Qu, Wang, Joshi (AISTATS 2024) — ACHQ для routing в гетерогенных серверах

> **N. Jali, G. Qu, W. Wang, G. Joshi.** "Efficient Reinforcement Learning for Routing Jobs in Heterogeneous Queueing Systems", AISTATS 2024, arXiv:2402.01147.

### Проблема
Центральная очередь $\to$ $N$ серверов с **разными** service rates $\mu_1 > \mu_2 > \ldots > \mu_N$. Для $N=2$ (один быстрый, один медленный) оптимальна пороговая политика: отсылать на медленный только при $Q \ge K^*$. Для $N \ge 3$ **структура оптимума неизвестна**.

Наивный deep RL не работает: state space $\mathbb{N} \times \{0,1\}^N$ растёт экспоненциально по $N$, и число iterations для сходимости PPO/DQN становится непрактичным.

### Метод: **ACHQ** (Actor-Critic for Heterogeneous Queues)

Ключевая идея — **параметризация политики через мягкие пороги**:
$$\pi_\theta(a_k = 1 \mid Q, \text{idle set}) = \sigma\!\big(\alpha_k (Q - \theta_k)\big)$$
где $\theta_k$ — порог для сервера $k$, $\alpha_k$ — «крутизна». Вместо параметризации политики сотнями-тысячами весов нейросети — **всего $2N$ параметров**.

Алгоритм — **policy gradient** с этой параметризацией, использующий structure of the underlying MDP.

### Теоретические гарантии

- **Общий случай:** сходимость к stationary point градиента.
- **$N = 2$:** сходимость к approximate global optimum (несмотря на малоразмерную параметризацию).

### Валидация
Симуляция M/M/N с гетерогенными $\mu_i$:
- Baseline: "greedy to fastest available" — классическая эвристика.
- **Результат:** ACHQ снижает expected response time **на ~30%**.
- Для $N=2$ ACHQ сходится к known optimal threshold.

---

## 3. 🏗 Grosof, Maguluri, Srikant (2024, v3 2025) — NPG convergence для infinite-state queueing MDP

> **I. Grosof, S.T. Maguluri, R. Srikant.** "Convergence of Natural Policy Gradient for a Family of Infinite-State Queueing MDPs", arXiv:2402.05274.

### Проблема
Natural Policy Gradient (NPG) — основа TRPO, PPO, natural actor-critic. Вся существующая теория сходимости — **только для finite-state** MDP. Для очередей это не работает, потому что стандартные допущения (bounded value function, bounded diameter) рушатся.

### Ключевое открытие: инициализация — это всё

**Теорема 1.** Для класса Generalized Switch networks, если NPG **инициализировать MaxWeight policy** $\pi_0$, то:
$$\mathbb{E}[\text{mean queue length}]^{\pi_t} - \text{OPT} \le O(1/\sqrt{T})$$
Это **первый** convergence rate для NPG в infinite-state average-reward MDP.

### Метод
Два ключевых хода:
1. **MaxWeight имеет low growth rate** relative value function $v^{\pi_0}(Q) = O(\|Q\|^2)$ — что критично для ограничения второго момента.
2. Показывают, что $v^{\pi_t}$ меняется «мягко» между итерациями $\Rightarrow$ state-dependent step size позволяет удержать рост.

### Валидация
Теоретическая работа; экспериментально показано на switch scheduling и N-model.

### Значение
Это объясняет, почему **warm start** (инициализация разумной эвристикой вроде MaxWeight, $c\mu$-rule, JSQ) в практических RL-симуляциях критически важен и не просто эмпирический трюк.

---

## 4. 📊 Weber, Busic, Zhu (ICML 2024) — regret bounds для admission control

> **L. Weber, A. Busic, J. Zhu.** "Reinforcement Learning and Regret Bounds for Admission Control", ICML 2024, PMLR 235:52403–52427.

### Проблема
**Admission control в $M/M/c/S$** очереди с $m$ классами задач. Каждый класс имеет reward $R_i$ за принятие и holding cost $c_i$. Решение: принимать или отклонять каждую прибывающую задачу.

Для обычного RL нижняя граница regret'а $\Omega(\sqrt{D X A T})$, где $D$ = диаметр MDP, $X$ = размер state space. В очередях $D$ и $X$ **экспоненциальны по buffer size $S$**, поэтому стандартные гарантии бесполезны.

### Метод
Алгоритм на базе **UCRL2** (upper-confidence reinforcement learning), использующий **структуру проблемы**:
- Monotonicity оптимальной политики (пороговая структура по классам).
- Конкретные bounds на transition dynamics $M/M/c/S$.

### Результат
**Finite-server:** $\mathbb{E}[\text{Regret}(T)] = O(S \log T + \sqrt{m T \log T})$ — **полиномиально** по $S$ вместо экспоненциально.

**Infinite-server:** зависимость от $S$ **исчезает** — $O(\sqrt{m T \log T})$.

### Валидация
Численно на $M/M/c/S$ с разными $m$, $c$, $S$; алгоритм сравнивается с UCRL2 и Q-learning — и значительно быстрее сходится.

---

## 5. 🤝 Sheldon & Casale (2025/26) — two-sided queues

> **M. Sheldon, G. Casale.** "Reinforcement Learning for Admission Control in Two-Sided Queueing Systems", ICLR 2026 submission.

### Проблема
**Two-sided queue** — модель двусторонних рынков (ride-hailing, Uber/Lyft; матч «работник–клиент»; спотовые биржи). Прибывают два типа «сущностей» (машины и пассажиры); остаются в очереди, пока не сматчатся. Надо контролировать admission, когда неизвестны arrival rates.

### Метод
Алгоритм с **regret bound, не зависящим от диаметра MDP**:
$$\tilde{O}(\kappa^3 S^{1.5} \sqrt{T} + \kappa^{2.5} S^{1.5} \sqrt{N T})$$
где $N$ — число типов, $\kappa$ — отношение upper/lower bounds на rate.

### Валидация
Симуляционное исследование; значительно превосходит UCRL2 и обычные RL-алгоритмы на средних state spaces.

---

## 6. ☁️ RLTune (SoCC 2025) — DL workloads на GPU-кластерах

> **S. Dongare et al.** "Hybrid Learning and Optimization-Based Dynamic Scheduling for DL Workloads on Heterogeneous GPU Clusters", ACM SoCC 2025.

### Проблема
Современные ЦОДы для DL-тренинга имеют **разные GPU** (A100, V100, H100, P100); job profiling невозможен (модели проприетарные, workloads разнообразны). Традиционные scheduler'ы (Slurm, Gavel, Sia) либо ad-hoc, либо требуют per-job profiling.

### Архитектура: RL + MILP

**Дизайн:**
1. **RL-агент** (на базе PPO/DQN-like) — выполняет **dynamic prioritization**: даёт каждому job priority score на основе состояния очереди и user metadata (**без** performance profiling).
2. **MILP-солвер** — на основе приоритетов решает **multi-resource allocation** (GPU, CPU, memory) по узлам с учётом heterogeneity.

**Features агента:** queue length, resource availability, user-submitted metadata (e.g. requested GPUs, time limit).

**Reward:** комбинация (negative JCT) + (GPU utilization) + (−queueing delay).

### Валидация: реальные production traces
- **Microsoft Philly** trace
- **Helios** trace
- **Alibaba PAI** MLaaS trace

### Результаты vs state-of-the-art (Slurm, Helios QSSF, Gavel, Sia)
- GPU utilization **+20%**
- Queueing delay **−81%**
- Job completion time **−70%**

### Значение
Один из немногих примеров, где RL-scheduler для очередей **задеплоен** в реальном Slurm-кластере — а не только симуляция.

---

## 7. 📞 Call Centers: два подхода

### 7a. Kumwilaisak et al. (IEEE Access 2022) — DNN-прогноз + Q-learning staffing

> Реальный thai call center **TTRS** (Thai Telecommunication Relay Service).

**Архитектура двухуровневая:**
1. **LSTM + DNN** прогнозирует интенсивность звонков $\lambda(t)$ на следующие 30-минутные интервалы (нелинейные паттерны: сезонность, тип сервиса — video/text/SMS).
2. Прогнозы $\lambda(t)$ подаются в **Erlang-A** (M/M/c + abandonment) → вычисляется ожидаемая waiting time и abandonment rate.
3. **Q-learning-агент** выбирает действие = {когда начинается смена, сколько агентов в ней} чтобы максимизировать reward = (−waiting time) − (−abandonment rate) − (labor cost).

**Валидация:** реальные данные TTRS за 1 год; побил experienced human supervisors и предыдущие DSS-схемы по QoS и среднему waiting time.

### 7b. Li & Karunarathne (2025) — VI vs PPO для SBR

> "Optimising Call Centre Operations using Reinforcement Learning: VI vs PPO", arXiv:2507.18398.

**Setup:** 2 агента, 2 типа инкуайров, Skills-Based Routing. Poisson arrivals, exponential service/abandonment. State $(n_0, n_1, \tau)$, action $\{0,1\}$.

**Методы:**
- **Value Iteration** (model-based, требует full knowledge transition dynamics).
- **PPO** (model-free, обучается через DES-симулятор в OpenAI Gym).

**Reward:** $-158$ за "assign to full queue" / "assign to busy while other idle", иначе $-\mathbb{E}[\text{waiting time}]$.

**Результат:** PPO > VI > random policy по reward, waiting time и idle time. Цена: на порядок больше времени обучения.

---

## 8. 🩺 Healthcare Scheduling — Liu et al. (2024)

> **X. Liu et al.** "Reinforcement Learning for Patient Scheduling with Combinatorial Optimisation", SGAI 2024, LNCS 15447.

### Проблема
GP appointment scheduling (NHS Scotland) и hospital scheduling (Anhui Medical Univ.). Сложность: combinatorial — выбор, когда и кому назначать visits разной длины.

### Метод: **EDQN** — DQN с early stopping

**State encoding:** "Tetris-like" 2D-grid:
- **Position Layer** — занятость слотов времени.
- **Booking Layer** — текущее booking (длина слота).

**Action:** выбор слота и GP для нового appointment.

**Early stopping в replay buffer:** если после $k$ последовательных negative rewards — прерывается эпизод, чтобы **балансировать exploration/exploitation** и не переполнять buffer плохими примерами.

### Валидация
- NHS Scotland: 4 типа consultations (face-to-face, home visit, telephone, video) с разной продолжительностью.
- Hospital: реальные appointment data.
- EDQN сходится быстрее и даёт более высокое utilization, чем обычный DQN и greedy.

---

## 9. 🚗 Ride-hailing: Gluzman's thesis (arXiv:2205.02119)

> **M. Gluzman.** "Processing Network Controls via Deep Reinforcement Learning", PhD thesis, 2022.

**Задача:** **driver repositioning** в ride-sharing. Город разбит на зоны; машины после поездки оказываются в разных зонах, и нужно решать — ждать в зоне или ехать в пустую соседнюю (где может быть спрос).

**Модель:** BCMP-сеть + MDP со state = (# drivers в каждой зоне), action = matrix of relocations.

**Метод:** PPO + regenerative simulation + AMPM (те же три техники variance reduction из Dai–Gluzman).

**Baseline:** fluid-optimal repositioning (классика из operations research).

**Результат:** PPO улучшает throughput (completed trips) на 5–15% при разных $\rho$.

---

## 10. 📐 Liu, Xie, Modiano (MIT, 2019, заложили базу) — PDGRL

> Ранняя, но важная работа. "Reinforcement Learning for Optimal Control of Queueing Systems".

**Проблема:** UCRL2, PSRL рассчитаны на конечный state space.

**Метод: PDGRL** (Piecewise Decaying $\epsilon$-greedy RL). Алгоритм model-based:
- Внутри "truncation box" $\{Q : Q_{\max} \le U\}$ — стандартное RL.
- За пределами — стабилизирующая политика $\pi_0$ (обычно MaxWeight).

**Теорема:** средний backlog сходится к оптимуму $\rho^*$ с ошибкой $O(\text{poly}(U) / \exp(\text{poly}(U)))$.

**Валидация:** switch scheduling, routing к гетерогенным серверам — сходится к известной оптимальной threshold-политике.

Это задало **парадигму warm-start стабилизирующей политикой**, которая потом появилась у Grosof et al. с MaxWeight.

---

## 🧪 Сводная таблица: модели × методы × валидация

<details open>
<summary><b>Развернуть таблицу</b></summary>

| Работа | Модель очереди | RL-метод | Warm start | Валидация |
|---|---|---|---|---|
| Dai–Gluzman (2022) | Multiclass networks (criss-cross, N-model, reentrant) | PPO + regenerative sim + AMPM | — | Симуляция vs VI/heuristics |
| Jali et al. (2024) | M/M/N heterogeneous | Policy gradient (ACHQ, soft threshold) | — | Симуляция vs "greedy to fastest" |
| Grosof et al. (2024) | Generalized Switch | NPG (теория) | **MaxWeight** | Теорема + switch scheduling |
| Weber et al. (2024) | M/M/c/S admission | UCRL2-variant | Structure of MDP | Теория + числ. эксперим. |
| Sheldon–Casale (2026) | Two-sided queue (ride-hailing) | Custom UCRL-like | — | Симуляция |
| RLTune (2025) | DL GPU cluster scheduling | RL + MILP гибрид | — | Philly/Helios/Alibaba traces + реальный Slurm |
| Kumwilaisak (2022) | Call center M/M/c+M (Erlang-A) | Q-learning + LSTM forecast | Erlang-A defaults | Реальные TTRS data |
| Li–Karunarathne (2025) | Skills-Based Routing M/M/2 | PPO, VI | — | DES в OpenAI Gym |
| Liu et al. (2024) | Patient scheduling (combinatorial) | **EDQN** (DQN + early stop) | — | NHS Scotland, Anhui data |
| Gluzman (2022) | Ride-hailing repositioning | PPO + regenerative sim | Fluid policy | BCMP simulation |
| Liu–Xie–Modiano (2019) | General queueing network | **PDGRL** (UCRL+truncation) | MaxWeight outside box | Switch + heterogeneous routing |

</details>

---

## 🔑 Ключевые практические уроки из этих работ

<details open>
<summary><b>Что работает и что нет</b></summary>

1. **Warm start стабилизирующей эвристикой (MaxWeight, $c\mu$, JSQ) резко ускоряет обучение** и часто необходим для теоретической сходимости (Grosof et al., Liu–Xie–Modiano).

2. **Variance reduction — это не опция, а необходимость.** Для long-run average cost naive PPO не сходится. Решения:
 - Regenerative simulation (выбор «пустого» состояния как regeneration point).
 - AMPM (approximating martingale).
 - Discounted approximation с $\gamma$ близким к 1.

3. **Structured parameterization бьёт полный deep RL на структурированных задачах.** ACHQ с $2N$ параметрами бьёт DQN с тысячами весов на routing.

4. **Infinite state space — главное препятствие.** Две стратегии:
 - Truncation + стабилизирующая политика снаружи (PDGRL).
 - Структурные assumptions на MDP + специальный алгоритм (Weber, Sheldon).

5. **Реальные production traces или DES-симуляторы** — стандарт валидации. OpenAI Gym + discrete event simulation стал фактическим каркасом.

6. **Reward design критичен.** Penalty constants (как $-158$ у Li–Karunarathne) часто нужно калибровать под конкретную модель. Reward = $-\sum c_i Q_i$ работает хорошо для holding cost; для fairness/tail нужны другие формы.

7. **Гибриды (RL + что-то ещё) — тренд 2025.** RLTune (RL + MILP), Kumwilaisak (LSTM + Erlang-A + Q-learning) — чистый deep RL часто недостаточен для продакшена.

</details>

---
