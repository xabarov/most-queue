У меня теперь достаточно материала для полноценного глубокого обзора. Составляю подробный ответ.

# Scheduling с предсказаниями и LLM-системы: подробный обзор

Это направление буквально взорвалось в 2023–2026 гг. Исторически оно выросло из двух источников:
1. **Классическая queueing theory** (Shrage, Kleinrock 1960s) — SRPT, SJF, PSJF для M/G/1.
2. **Learning-augmented algorithms** (Lykouris–Vassilvitskii ICML 2018, Kraska et al. SIGMOD 2018) — парадигма «алгоритм + предсказание от ML».

В 2023 году эти две линии неожиданно встретились в одной точке — **LLM inference serving**, где:
- запросы имеют **неизвестную длину** (число генерируемых токенов);
- есть **растущая memory footprint** (KV-cache);
- preemption **не бесплатна** — нужно сохранять/перевычислять KV-cache;
- есть **прекрасный источник предсказаний** — сам LLM может оценить, сколько токенов он сгенерирует.

Разобьём обзор на **два уровня**:
- **A. Теоретическая основа:** scheduling с предсказаниями в классических queueing models.
- **B. Приложение к LLM:** scheduling-системы для inference serving.

---

## 🏛 A. Теоретическая основа: predictions в M/G/1 и M/G/k

### A.1 Mitzenmacher (2020/2021) — базовая модель SPRPT/SPJF/PSPJF

> **M. Mitzenmacher.** "Scheduling with Predictions and the Price of Misprediction", ITCS 2020.
> **M. Mitzenmacher.** "Queues with Small Advice", SIAM ACDA 2021.

### Модель
В стандартной M/G/1 вводится **двумерная плотность** $g(x, y)$ — совместное распределение истинного размера $x$ и предсказанного размера $y$:
$$g: \mathbb{R}_+ \times \mathbb{R}_+ \to \mathbb{R}_+, \quad \iint g(x,y)\, dx\, dy = 1$$

Маргинальные плотности:
- $f_s(x) = \int_0^\infty g(x, y)\, dy$ — истинное распределение размеров (как обычно в M/G/1);
- $f_p(y) = \int_0^\infty g(x, y)\, dx$ — распределение предсказаний.

### Политики
- **SPJF** (Shortest Predicted Job First) — non-preemptive, по $y$.
- **SPRPT** (Shortest Predicted Remaining Processing Time) — preemptive, по $y - a$ (где $a$ — уже обслуженное время).
- **PSPJF** — preemptive shortest predicted job first.

### Ключевая формула для SPJF

Для задания предсказанного размера $y$:
$$E[W'(y)] = \frac{\rho\, E[S^2]}{2 E[S] (1 - \rho'_y)^2}$$
где $\rho'_y = \lambda \int_0^y \int_0^\infty x\, g(x, t)\, dx\, dt$ — «rate of inflow of work» для задач с **предсказанным** размером ≤ $y$.

Сравните с классической SJF:
$$E[W(x)] = \frac{\rho\, E[S^2]}{2 E[S] (1 - \rho_x)^2}, \quad \rho_x = \lambda \int_0^x t f_s(t)\, dt$$

### Price of Misprediction
Отношение expected waiting time SPJF к SJF (с идеальной информацией):
$$\text{PoM} = \frac{\int_0^\infty f_p(y) / (1 - \rho'_y)^2\, dy}{\int_0^\infty f_s(x) / (1 - \rho_x)^2\, dx}$$

### Численные результаты (из табл. 1 в Mitzenmacher 2020)

Предсказание для задачи размером $x$ — экспоненциальное с mean $x$ (очень шумное!):

| $\lambda$ | FIFO | SJF | SPJF | PSJF | PSPJF | SRPT | SPRPT |
|---|---|---|---|---|---|---|---|
| 0.5 | 2.000 | 1.713 | **1.795** | 1.531 | 1.664 | 1.425 | **1.653** |
| 0.8 | 5.000 | 2.882 | **3.376** | 2.659 | 3.194 | 2.353 | **3.117** |
| 0.95 | 20.000 | 6.264 | **8.654** | 6.265 | 8.617 | 5.541 | **8.322** |
| 0.99 | 100.000 | 18.45 | **29.05** | 18.96 | 29.38 | 17.63 | **28.73** |

**Главный вывод:** даже с очень шумными предсказаниями, SPRPT на порядок лучше FIFO при высокой загрузке ($\rho = 0.99$: 28.7 vs 100).

### A.2 Queues with Small Advice — 1-битные подсказки

В той же линии работ — что если доступна только **одна бита** информации: "задача короткая или длинная"?

**Политика с threshold $T$:**
- предсказано короткое ($y \le T$) → в начало очереди (high-priority);
- предсказано длинное ($y > T$) → в конец (low-priority).

**Результаты:** даже с одним битом достигаются 50–80% выигрыша SPRPT. Для Weibull-распределений формулы involve Meijer G-function и modified Bessel functions.

### A.3 SOAP framework (Scully, Harchol-Balter, Scheller-Wolf)

> **Z. Scully, M. Harchol-Balter, A. Scheller-Wolf.** "SOAP: One Clean Analysis of All Age-Based Scheduling Policies", SIGMETRICS 2018.

**Идея:** задача имеет **тип** $\tau$ (например, пара (истинный размер, предсказанный размер)) и **возраст** $a$ (уже полученное время обслуживания). Ранг задачи:
$$r(\tau, a)$$
Приоритет отдаётся задаче с **минимальным рангом**. SOAP даёт универсальную формулу E[T] для любой rank-функции.

**Применение к предсказаниям:**
- SPRPT: $r((x, y), a) = y - a$
- SPJF: $r((x, y), a) = y$
- Gittins index: $r$ — решение задачи оптимальной остановки.

**Ограничение:** rank не может зависеть от **числа задач в очереди** или состояния других задач.

### A.4 Uniform Bounds: Scully, Grosof, Mitzenmacher (2022)

> **Z. Scully, I. Grosof, M. Mitzenmacher.** "Uniform Bounds for Scheduling with Job Size Estimates", ITCS 2022.

**Проблема SPRPT:** если предсказание меньше реального размера, предсказанный remaining processing time становится 0 — и «большая» задача застревает наверху очереди.

**Модель:** $(\beta, \alpha)$-bounded noise: истинный размер $s$, предсказание $z \in [\beta s, \alpha s]$.

**Критерии качества политики:**
- **Consistency:** $E[T^P] / E[T^{\text{SRPT}}] \to 1$ при $\beta, \alpha \to 1$.
- **Graceful degradation:** $E[T^P] / E[T^{\text{SRPT}}] \le G(\alpha / \beta)$ для всех $\beta, \alpha$.
- **Robustness:** bounded для любых предсказаний.

**Главная теорема:** существует политика, $1$-consistent и $3.5$-graceful. **Constant robustness невозможна** при arbitrary predictions.

**Политика:** hybrid rank, которая при большом возрасте $a$ переходит с predicted remaining time на некоторое увеличивающееся значение, не давая задаче «застрять».

### A.5 SkipPredict — учёт стоимости предсказаний

> **R. Shahout, M. Mitzenmacher.** "SkipPredict: When to Invest in Predictions for Scheduling", NeurIPS 2024.

**Проблема:** предыдущие работы предполагают, что предсказания **бесплатны**. На практике получение предсказания стоит времени/ресурсов.

### Две модели стоимости
1. **External cost model:** предсказание делает внешний сервис, не отнимающий service time у основной очереди, но добавляющий **фиксированную стоимость $c$** к total cost на задачу.
2. **Server time cost model:** предсказание выполняется на **том же сервере**, что обрабатывает задачи. Нужно также scheduling предсказаний.

### Политика SkipPredict (двухэтапная)

1. **Cheap prediction (1 бит):** short/long. Дешёвое.
2. **Predicted short** → high priority, FIFO.
3. **Predicted long** → дополнительно получают **expensive prediction** (точный размер) и обслуживаются по SPRPT.

```
          ┌─→ predicted short (high prio, FIFO)
Arrival ──┤
          └─→ predicted long ──(2nd prediction)─→ SPRPT queue (low prio)
```

**Анализ:** через SOAP framework с mean response time выводятся формулы в обоих cost models. **Вывод:** SkipPredict бьёт и SPRPT (который тратится на дорогие предсказания для всех), и 1bit (который не использует детальную информацию).

### A.6 Shortest-Job-First в Many-Server с impatience

> **J. Dong, R. Ibrahim.** "Shortest-Job-First Scheduling in Many-Server Queues with Impatient Customers and Noisy Service-Time Estimates", *Operations Research* 2024.

**Модель:** $M/GI/s + GI$ — много серверов, нетерпеливые клиенты, шумные оценки service-time.

**Результат:**
1. Даже с шумными предсказаниями SJF асимптотически оптимальна в overloaded regime.
2. **Two-class priority rule** (просто split по threshold) даёт тот же асимптотический performance, что и полноценный SJF. То есть на практике не нужен SRPT — достаточно двух классов.

**Методология:** fluid-limit для $s \to \infty$ (many-server heavy-traffic) + two-class threshold analysis.

---

## 🤖 B. LLM inference scheduling: практическая революция

**Основная проблема:** state-of-the-art LLM-систем (vLLM, Orca, DeepSeek, Llama.cpp) по умолчанию используют **FCFS** — что приводит к head-of-line blocking. Короткий запрос ждёт за длинным.

**Но:** LLM inference имеет уникальные особенности, не покрытые стандартной теорией:

1. **Prefill + Decode фазы.** Prefill (обработка промпта) — compute-bound. Decode (генерация токенов один за другим) — memory-bandwidth-bound.
2. **KV-cache** растёт **линейно** с числом сгенерированных токенов. Невозможно держать неограниченно много активных запросов.
3. **Iteration-level scheduling** (Orca, Yu et al. OSDI 2022): можно добавлять/удалять запросы **между токенами**, а не между целыми запросами.
4. **Preemption cost:** чтобы вытеснить запрос, надо либо (a) держать KV-cache в GPU-памяти (блокируя других), (b) выгрузить в CPU (время), (c) пересчитать с нуля (GPU-время).
5. **Output length неизвестен** — именно здесь предсказания становятся критическими.

### B.1 Mitzenmacher & Shahout (2025) — программный манифест области

> **M. Mitzenmacher, R. Shahout.** "Queueing, Predictions, and LLMs: Challenges and Open Problems", arXiv:2503.07545, SIGMETRICS 2025 tutorial. *Stochastic Systems*.

Не «решение», а **программа исследований**. Выделяют три вида LLM-систем с разными scheduling-challenges:

#### Тип 1: Single LLM instance
- Prefill/decode scheduling.
- Memory-aware preemption.
- Cost of predictions (как в SkipPredict).

#### Тип 2: Compound AI systems
LLM делают **внешние API calls** (retrieval, tools, код). Задача: куда отнести **API delay** — это часть service time? Как управлять KV-cache во время ожидания API?

#### Тип 3: Reasoning LLM systems
DeepSeek-R1, o1, o3: LLM генерирует **reasoning steps** перед финальным ответом. «Размер» запроса теперь — не число токенов, а **число reasoning steps** до convergence. Можно **досрочно остановить**, если ответ уже уверенный.

### Открытые проблемы из этой статьи
1. Реалистичные модели двумерного распределения $(s, \hat{s})$ для LLM.
2. Predictions that **degrade over time** (изначальный prompt-based prediction vs refined prediction после $k$ токенов).
3. **Ordering requests with prompt sharing** — если несколько запросов имеют одинаковый prefix, KV-cache можно шарить (prefix caching). Как планировать?
4. Scheduling в multi-server setup (sharded LLM).
5. Predictions для **reasoning step count**.

### B.2 TRAIL / "Don't Stop Me Now" (ICLR 2025) — embedding-based predictions

> **R. Shahout, E. Malach, C. Liu, W. Jiang, M. Yu, M. Mitzenmacher.** "Don't Stop Me Now: Embedding Based Scheduling for LLMs", ICLR 2025.

### Проблема
Ранние работы предсказывали длину ответа **только по промпту** (S³ с fine-tuned BERT, NIPS 2023). Точность падает для запросов с большой вариативностью.

### Идея TRAIL
**Probing:** использовать сам LLM в качестве предсказателя.
1. Начальный prediction из BERT по промпту.
2. После каждой iteration (каждого сгенерированного токена) брать **embedding из промежуточного слоя** $\ell$ transformer'а.
3. Подать его в **линейный классификатор**, который предсказывает `remaining length`, разбитый на $k$ бинов.

$$p^{(t)} = \text{Linear}(u^{(t)})\!, \quad p^{(t)}_j \in [0,1], \quad B_j = \{b_j, \ldots, b_{j+1}-1\}$$

**Observation:** MAE достигает минимума на среднем слое $\ell \approx 11$ из 32 (для Llama-7B).

### SRPT with limited preemption

Ключевая инновация — не позволять бесконечный preemption:
- **Early tokens** (малый KV-cache): preemption разрешена.
- **Later tokens** (большой KV-cache): preemption **запрещена**.

Это контролируется параметром — максимальным числом preemptions $k_{\max}$ на запрос.

**Теоретический результат:** для M/G/1 с limited preemption выведена closed-form formula для E[T], расширяющая SOAP.

### Валидация
- **Датасет:** Alpaca (52K instructions для LLM).
- **Baseline:** vLLM (state-of-the-art serving system).
- **Результаты TRAIL:**
 - MAE предсказания длины: **2.66× ниже**, чем BERT-based S³.
 - **Mean latency:** 1.66–2.01× ниже, чем vLLM.
 - **Time to first token (TTFT):** 1.76–24.07× ниже (!).

### B.3 Efficient LLM Scheduling by Learning to Rank (NeurIPS 2024)

> **Y. Fu, S. Zhu, R. Su, A. Qiao, I. Stoica, H. Zhang.** "Efficient LLM Scheduling by Learning to Rank", NeurIPS 2024.

### Главная идея
**Зачем предсказывать точную длину?** Для SJF-подобных политик достаточно **относительного порядка** (ranking). Это намного легче предсказывать.

### Метод
- Обучается **ranking model** (pairwise loss) на парах (prompt A, prompt B) → какой запрос короче.
- Scheduler использует этот ranking вместо точных предсказаний.

### Результат
- Превосходит approaches с point predictions.
- Интегрирован в SOTA LLM-serving стэк.

### Значение
Сдвигает парадигму от **regression** (предсказывать число) к **learning-to-rank** — подход, который давно доминирует в information retrieval.

### B.4 Proxy-model based prediction (SSJF)

> **H. Qiu, W. Mao et al.** "Efficient Interactive LLM Serving with Proxy Model-based Sequence Length Prediction", arXiv 2024.

**Идея:** маленькая легковесная LLM (например, distilled) предсказывает длину ответа большой LLM. Это «speculative» (спекулятивная) SJF.

**Результаты:** значительное снижение JCT (job completion time) и рост throughput в трёх режимах: без batching, с dynamic batching, с continuous batching.

### B.5 Online scheduling with KV Cache constraints (MIT, Microsoft, 2025/26)

> **P. Jaillet, J. Jiang, K. Mellou, M. Molinaro, C. Podimata, Z. Zhou.** "Online Scheduling for LLM Inference with KV Cache Constraints", arXiv:2502.07115 (v5, Jan 2026).

### Постановка
Формальная **competitive analysis** модель. Запросы прибывают online. У каждого:
- размер промпта $s_i$
- длина output $o_i$ (предсказано)

KV-cache usage для $j$-го выходного токена запроса $i$: $s_i + j$. Memory constraint: $M$.

Цель: минимизировать total latency.

### Теоретические результаты
1. **Hindsight optimal benchmark** сформулирован как integer program (минимизация latency при полном знании будущего). Это «gold standard».
2. **Невозможность.** Нет детерминированного online algorithm с constant competitive ratio против adversarial arrivals.
3. **Алгоритм MC-SF** (Memory-Constrained Shortest First):
 - приоритет **partially-completed requests** (чтобы очистить память);
 - при заполнении batch — максимизировать размер batch с учётом **anticipated future memory usage**.
 - **constant competitive ratio** при stochastic arrivals (определённые условия).

### Валидация
- **Синтетика:** 200 instances, average ratio к hindsight optimal = **1.005** (при all-at-once arrivals), **1.047** (при online arrivals).
- **Реальные данные:** LMSYS-Chat-1M (conversations из Vicuna demo + Chatbot Arena, 210,000+ IP addresses), симуляция **Llama2-70B на A100 GPU**. Значительно обходит benchmark parameterized policies.

### B.6 Throughput-Optimal Scheduling for LLM Agents (Li, Dai, Peng 2025)

> **Y. Li, J. Dai, T. Peng.** "Throughput-Optimal Scheduling Algorithms for LLM Inference and AI Agents", arXiv:2504.07347, April 2025.

### Цель
Не latency, а **throughput** (stability) — когда систему **не переполнит**. Это классическая queueing-theoretic постановка Dai (автор важных работ 1990-х по fluid limits).

### Главные теоремы
1. **Single LLM engine:** широкий класс **work-conserving policies** достигает максимальный throughput.
2. **Network of LLM agents (multi-class workflow):** work-conserving **недостаточно**. Нужны более изощрённые политики (аналогично Kumar–Seidman, Lu–Kumar для reentrant lines).

### Практическая проверка real-world систем
Авторы **проанализировали существующие serving systems**:

| System | Throughput-optimal? |
|---|---|
| **Orca** (Yu et al., OSDI 2022) | ✓ Да |
| **Sarathi-Serve** | ✓ Да |
| **FasterTransformer** (NVIDIA) | ✗ Нет |
| **Vanilla vLLM** | ✗ Нет (не maximally stable) |

Это **практически важный** результат: некоторые популярные системы нестабильны при высоких нагрузках.

### B.7 Fluid-guided online scheduling (Hong, Ibrahim et al. 2025)

> "Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints", arXiv 2025.

Использует **fluid limit** (deterministic ODE-approximation системы при heavy traffic) как ориентир. Scheduler решает дискретные задачи так, чтобы следовать fluid-optimal траектории.

### B.8 Adaptively Robust Inference under Prediction Uncertainty (2025)

Авторы предлагают политики, которые **адаптируются** к качеству предсказаний: когда предсказания надёжны — работают как SRPT; когда шумные — как FCFS. Похоже на consistency/robustness подход из learning-augmented algorithms.

### B.9 Semi-Clairvoyant Speculative Decoding (2025)

> "Semi-Clairvoyant Scheduling of Speculative Decoding Requests to Minimize LLM Inference Latency", 2025.

**Speculative decoding** — техника, где маленький draft-model генерирует предположения, а большой LLM их верифицирует (акцептирует/отклоняет). **Количество принятых токенов рандомно** — классический сценарий для semi-clairvoyant scheduling (знаем распределение, не точное значение).

### B.10 DUCHESS / Intra-request branch orchestration (Jiang, Shahout et al. 2025)

> Tree-of-thoughts / reasoning branching в LLM: запрос порождает **несколько веток рассуждений**, часть из которых будет отброшена. Scheduler решает, какие ветки развивать. Предсказания используются для оценки «перспективности» веток.

---

## 📊 Сводная таблица работ

<details open>
<summary><b>Подробный обзор работ (развернуть)</b></summary>

| № | Работа | Класс | Метод | Валидация |
|---|---|---|---|---|
| A.1 | Mitzenmacher (2020) | M/G/1 teor | SPJF/SPRPT/PSPJF, $g(x,y)$ | Числ. M/G/1 |
| A.2 | Mitzenmacher (2021) "Small Advice" | M/G/1 teor | 1-bit threshold, PoM | Weibull, exp |
| A.3 | Scully–Harchol-Balter (2018) SOAP | M/G/1 framework | Rank-based analysis | Все age-based |
| A.4 | Scully–Grosof–Mitzenmacher (2022) | M/G/1 bounds | $(\beta, \alpha)$-bounded noise, hybrid rank | Теорема |
| A.5 | **SkipPredict** (2024 NeurIPS) | M/G/1 + cost | Two-stage predictions, SOAP | Числ. M/G/1 |
| A.6 | Dong–Ibrahim (2024 OR) | M/GI/s+GI | Noisy SJF, two-class asymptotic | Fluid limit |
| B.1 | Mitzenmacher–Shahout (2025) | LLM survey | Open problems, модели | Программный |
| B.2 | **TRAIL** (ICLR 2025) | Single LLM | Embedding probing + SRPT limited preemption | Alpaca + Llama |
| B.3 | Learning-to-Rank (NeurIPS 2024) | Single LLM | Pairwise ranking | SOTA stack |
| B.4 | SSJF (Qiu et al. 2024) | Single LLM | Proxy-model prediction | Realistic traces |
| B.5 | **Jaillet et al.** (2025/26) | Single LLM | MC-SF, competitive ratio, IP benchmark | LMSYS + Llama2-70B |
| B.6 | **Li–Dai–Peng** (2025) | LLM agents | Throughput stability, work-conserving | Orca/vLLM/Sarathi |
| B.7 | Fluid-Guided (2025) | Single LLM | Fluid ODE + memory | Симуляция |
| B.8 | Adaptive Robust (2025) | Single LLM | Consistency/robustness trade-off | Синтетика |
| B.9 | Semi-Clairvoyant Speculative | Speculative dec. | Distributional info | Real decoding |
| B.10 | DUCHESS (2025) | Reasoning LLM | Branch orchestration + predictions | Math reasoning |

</details>

---

## 🎯 Как связаны теория и практика: mapping

<details open>
<summary><b>Развернуть диаграмму связей</b></summary>

```
КЛАССИЧЕСКАЯ M/G/1 TEORY              LLM SYSTEMS APPLICATIONS
────────────────────────              ────────────────────────
SRPT / SJF ──────────────────────→ vLLM FCFS → SRPT-like (TRAIL)
SPRPT / SPJF ────────────────────→ Size-based LLM scheduling (Fu et al.)
Two-class priority (Dong-Ibrahim) ─→ Short/long queues in serving
Price of misprediction (Mitzenm.) ─→ Degradation with poor length predictions
SOAP framework ──────────────────→ Closed-form E[T] for TRAIL
$(\beta,\alpha)$-bounded noise ─────→ Robustness to LLM predictor errors
SkipPredict (cost of preds) ────→ "Should we predict length for every job?"
Hindsight-optimal IP ───────────→ Microsoft MC-SF benchmark
Work-conserving policies ────────→ Li-Dai-Peng throughput-optimality
Fluid limits ────────────────────→ Fluid-guided LLM scheduling
```

</details>

---

## 🔑 Ключевые инсайты

<details open>
<summary><b>Главные уроки из всего корпуса работ</b></summary>

1. **Даже очень шумные предсказания резко улучшают FIFO.** Это главный численный результат Mitzenmacher (2020): предсказания с огромной дисперсией всё ещё в 3–5× лучше FIFO при $\rho \to 1$.

2. **Learning-to-rank > regression** для scheduling (Fu et al. NeurIPS 2024). Предсказывать относительный порядок легче, чем абсолютный размер.

3. **SOAP framework — это швейцарский нож** для анализа prediction-based policies. Любая политика с rank $r(\text{type}, \text{age})$ анализируется унифицированно. Но он не умеет учитывать **состояние очереди**.

4. **KV-cache фундаментально меняет задачу.** В классической SRPT preemption бесплатна; в LLM — нет. Это требует **новых теорем** (TRAIL с limited preemption, MC-SF с memory constraints).

5. **Prompt-only prediction недостаточна.** Лучшие системы (TRAIL) предсказывают длину **итеративно** во время генерации — используя уже сгенерированные токены как дополнительный сигнал. Это новый тип **dynamic predictions** без аналога в классике.

6. **Не все SOTA-системы throughput-optimal.** vLLM и FasterTransformer могут быть нестабильны при высокой нагрузке, что показано в Li–Dai–Peng (2025). Это практически значимо для production.

7. **Стоимость предсказания** (SkipPredict) — недоисследованная область. В LLM она особенно релевантна: запустить целый BERT для предсказания длины может занять >5% от времени ответа.

8. **Reasoning LLMs — следующий фронтир.** «Размер» теперь не token count, а **depth of reasoning**. Scheduling для tree-of-thoughts, branching — практически новое поле.

9. **Prefix caching / prompt sharing** (когда много запросов имеют одинаковое начало) — открытая проблема scheduling: как приоритизировать запросы, чтобы максимизировать cache hit rate?

</details>

---

## 📚 Ключевые точки входа для углубления

1. **Survey-manifesto:** Mitzenmacher & Shahout, "Queueing, Predictions, and LLMs" — [arXiv:2503.07545](https://arxiv.org/abs/2503.07545). **Главный текст области.**
2. **SIGMETRICS 2025 tutorial slides:** [sigmetrics.org/sigmetrics2025/tutorials/Track 2A.pdf](http://www.sigmetrics.org/sigmetrics2025/tutorials/Track%202A.pdf) — доступный tutorial с примерами.
3. **TRAIL:** "Don't Stop Me Now", ICLR 2025 — [OpenReview](https://openreview.net/forum?id=tHlXpW-NKt). Лучший system paper с reproducible experiments.
4. **Mathematical paper:** Jaillet et al. "Online Scheduling for LLM Inference with KV Cache Constraints" — [arXiv:2502.07115](https://arxiv.org/abs/2502.07115). IP benchmark + competitive ratios.
5. **Paper repository:** [algorithms-with-predictions.github.io](https://algorithms-with-predictions.github.io/) — regularly updated list всех learning-augmented algorithms works.
6. **Throughput theory:** Li–Dai–Peng, [arXiv:2504.07347](https://arxiv.org/abs/2504.07347).

