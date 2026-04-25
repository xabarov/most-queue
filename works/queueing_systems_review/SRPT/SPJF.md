Продолжу получать полный текст статьи для детального разбора.У меня теперь есть практически весь содержательный материал статьи. Делаю полноценный глубокий разбор.

# Mitzenmacher & Shahout, «Queueing, Predictions, and LLMs: Challenges and Open Problems» — детальный разбор

**Выходные данные:** arXiv:2503.07545, март 2025; опубликовано в *Stochastic Systems* (INFORMS), 2025. Авторы — Michael Mitzenmacher (Harvard, один из основателей области «algorithms with predictions») и Rana Shahout (Harvard, автор TRAIL, ICLR 2025).

**Жанр:** это не research paper с одним результатом, а **программа исследований** (research manifesto): систематический каталог open problems на стыке queueing theory, learning-augmented algorithms и LLM systems. Именно поэтому статья уже стала референсом для всей области.

---

## 📐 Структура статьи

<details open>
<summary><b>Разделы и их логика</b></summary>

| § | Название | Цель |
|---|---|---|
| 1 | Introduction | Motivate: почему LLM-scheduling — новая задача |
| 2 | Background on Queueing | Краткий ликбез по M/G/1, SRPT, SOAP, price of anarchy/misprediction |
| 3 | Scheduling with Predictions | Формальная модель $g(x,y)$ и результаты для M/G/1 с предсказаниями |
| 4 | Scheduling LLMs | Где классика ломается: KV-cache, preemption cost, iteration-level |
| 5 | Open Problems for Single LLM | 8+ подробно сформулированных проблем |
| 6 | Compound AI Systems | Когда LLM делает API calls — новый класс моделей |
| 7 | Reasoning LLM Systems | DeepSeek-R1, o1: когда «размер» становится «глубиной» |
| 8 | Conclusion | Призыв к действию для queueing community |

</details>

---

## 1. 🎯 Центральный тезис статьи

**Авторы утверждают:** LLM serving — это **новый класс queueing systems**, который:

1. **Не описывается** классическими моделями (M/G/1, G/G/k и их расширениями).
2. **Естественным образом** производит предсказания размеров задач (LLM сам может оценить длину ответа).
3. **Обладает уникальными ограничениями**, ранее не изучавшимися в queueing theory (KV-cache, iteration-level preemption, shared prefixes).
4. **Практически важен** — сотни миллиардов долларов инвестиций в LLM-инфраструктуру.

Цитата из введения:

> *"While LLMs produce natural predictions of job sizes, they also impose constraints that have been little studied in queueing theory. Our goal is to convince the queueing community that LLM scheduling deserves their attention, and to catalogue the open problems that await solution."*

---

## 2. 📚 Background: что авторы считают фундаментом

### 2.1 Минимальная queueing-база

Статья предполагает знание M/G/1. Ключевые факты, которые всплывают везде:

**Формула Поллачека–Хинчина** для M/G/1 с FIFO:
$$E[W^{\text{FIFO}}] = \frac{\lambda E[S^2]}{2(1-\rho)}$$

**SRPT-формула** (Shrage 1968):
$$E[W^{\text{SRPT}}(x)] = \frac{\lambda \int_0^x t^2 f(t)\, dt + x^2 (1 - F(x))}{2(1 - \rho_x)^2} + \int_0^x \frac{dt}{1 - \rho_t}$$
где $\rho_x = \lambda \int_0^x t f(t)\, dt$ — «load» от задач размером ≤ $x$.

**Ключевой факт:** SRPT **оптимальна** (минимизирует $E[T]$) среди всех политик для M/G/1.

### 2.2 SOAP framework (Scully–Harchol-Balter–Scheller-Wolf 2018)

Здесь авторы даже включают **новый педагогический результат** — упрощённый вывод SOAP-формулы для частного случая **non-preemptive size-based scheduling with noisy predictions** (раздел 3 статьи).

**Идея SOAP.** Задача имеет **тип** $\tau$ (например, $(x, y)$ — истинный и предсказанный размеры). **Ранг** задачи как функция типа и возраста $a$: $r(\tau, a)$. Политика обслуживает задачу с минимальным рангом.

**Для каждой фиксированной «tagged job»** с типом $\tau$ рассматриваем три события:
- **Old jobs:** задачи, прибывшие раньше, с рангом $< r(\tau, 0)$.
- **New jobs:** задачи, прибывающие во время ожидания, с меньшим рангом.
- **Recurse jobs:** задачи, чей ранг становится меньше в процессе обслуживания tagged job.

Каждая из этих категорий даёт вклад в waiting time, вычисляемый через свой «effective load».

### 2.3 Price of Misprediction (формально)

$$\text{PoM}(g) = \frac{E[T]^{\text{policy with predictions from } g}}{E[T]^{\text{SRPT with perfect info}}}$$

Авторы подчёркивают: этот показатель **зависит только от $g(x, y)$** — совместного распределения. Не нужна «точность» предсказателя в ML-смысле (accuracy, R², etc.) — нужно именно всё $g$.

**Важный контринтуитивный факт:** предсказатель с низкой MAE может давать плохой PoM, если у него систематические ошибки **на длинных задачах** (где они дороже всего в SRPT).

---

## 3. 🧮 Раздел 3: Scheduling with Predictions — формальный каркас

### 3.1 Ключевая модель

Запросы прибывают по Poisson($\lambda$). Каждый имеет **истинный размер** $x$ и **предсказанный** $y$, совместно распределённые с плотностью $g(x, y)$.

**Четыре политики:**

| Политика | Preemptive? | Rank |
|---|---|---|
| **SPJF** | нет | $y$ |
| **PSPJF** | да, preempt-resume | $y$ |
| **SPRPT** | да, по remaining | $y - a$ (где $a$ — полученное обслуживание) |
| **Gittins** | по индексу | Gittins index на $g(\cdot \mid y)$ |

### 3.2 Вклад Mitzenmacher 2020 — ключевые формулы

**Для SPJF** (non-preemptive по предсказанию):
$$E[W^{\text{SPJF}}(y)] = \frac{\lambda \int \!\!\int t^2 g(t, z)\, dt\, dz}{2(1 - \rho'_y)^2}$$
где:
$$\rho'_y = \lambda \int_0^y \int_0^\infty t\, g(t, z)\, dt\, dz$$
— «эффективная загрузка» от задач с **предсказанием** ≤ $y$.

**Интерпретация:** политика группирует задачи по предсказаниям, а не по истинным размерам. Это «смешивает» истинно-длинные и истинно-короткие задачи в одну очередь — и чем точнее предсказания, тем меньше смешивание.

### 3.3 Численный эксперимент (ключевая таблица статьи)

Сценарий: $S \sim \exp(1)$ (истинные размеры), **предсказание $Y \mid S = x \sim \exp(x)$** — крайне шумный предсказатель (coefficient of variation = 1).

| $\rho$ | FIFO | SJF | **SPJF** | PSJF | **PSPJF** | SRPT | **SPRPT** |
|---|---|---|---|---|---|---|---|
| 0.5 | 2.000 | 1.713 | 1.795 | 1.531 | 1.664 | 1.425 | 1.653 |
| 0.8 | 5.000 | 2.882 | 3.376 | 2.659 | 3.194 | 2.353 | 3.117 |
| 0.9 | 10.000 | 4.462 | 5.527 | 4.130 | 5.285 | 3.642 | 5.131 |
| 0.95 | 20.000 | 6.264 | 8.654 | 6.265 | 8.617 | 5.541 | 8.322 |
| 0.99 | 100.000 | 18.45 | 29.05 | 18.96 | 29.38 | 17.63 | 28.73 |

**Что из этого следует (ключевые уроки статьи):**

1. **SPRPT даёт 3.5× выигрыш** над FIFO при $\rho = 0.99$, даже с ужасными предсказаниями.
2. PSPJF ≈ SPJF, SPRPT ≈ SPRPT — разница preemptive/non-preemptive **уменьшается** при noisy predictions (потому что preemption срабатывает «на мусор»).
3. SJF с **идеальной** информацией (1.713 → 18.45) **не сильно лучше** SPJF с шумом — при $\rho = 0.5$ разница всего 5%, при $\rho = 0.99$ — 57%.

---

## 4. 🤖 Раздел 4: где классика ломается на LLM

Это центральный концептуальный раздел. Авторы перечисляют **пять фундаментальных отличий** LLM serving от классического queueing:

### 4.1 Iteration-level scheduling (Orca, Yu et al. OSDI 2022)

В LLM-inference **каждый токен — отдельная «iteration»**. Scheduler может менять набор активных запросов **между iterations**. Это создаёт `batch` на каждом шаге.

**Следствие:** preemption происходит не как «пауза», а как **выбрасывание запроса из batch'а** на следующей итерации. Это ближе к **processor sharing с изменяемым множителем**, чем к стандартной M/G/1.

### 4.2 KV-cache и memory constraint

**Физика проблемы.** LLM-decoder на каждом шаге использует сохранённые ключи/значения attention для всех предыдущих токенов (`Key-Value cache`). Для запроса с $n$ сгенерированными токенами:
$$\text{KV memory} = n \cdot 2 \cdot L \cdot d \cdot \text{bytes}$$
где $L$ — число слоёв, $d$ — hidden dimension.

**Для Llama2-70B:** один токен KV = 0.625 MB. Запрос на 2000 токенов = 1.25 GB. GPU A100 (80 GB) поддерживает одновременно ~60 активных запросов.

**Два следствия:**
1. Жёсткое **bin-packing** constraint: нельзя запустить больше запросов, чем влезает KV в GPU memory.
2. **Растущая занятость:** чем дольше запрос обрабатывается, тем больше его memory footprint.

### 4.3 Preemption cost (не бесплатна!)

Три варианта обработки preempted request:
1. **Keep KV in GPU** — занимает память, блокирует других.
2. **Swap to CPU** — время на PCIe transfer (~10-100 ms).
3. **Recompute on resume** — полностью пересчитать prefill заново (компьют-дорого).

**Теоретическая проблема:** в классической SRPT preempt бесплатен, поэтому SRPT **оптимальна**. В LLM — **нет**. Оптимальная политика должна учитывать стоимость preemption.

### 4.4 Разнородность фаз: Prefill vs Decode

**Prefill** (обработка промпта) — **compute-bound**, параллельный по токенам промпта, $O(n_{\text{prompt}}^2)$ FLOPs (attention).
**Decode** — **memory-bandwidth-bound**, последовательный по одному токену, $O(1)$ FLOPs на токен, но требует чтения всей KV-cache.

**Следствие:** на одном GPU prefill и decode конкурируют за разные ресурсы. Scheduling должен их **раздельно управлять** (disaggregated serving — Splitwise, DistServe).

### 4.5 Предсказания есть и они богатые

**Источники предсказаний длины ответа:**
- Маленькая proxy-модель (BERT, distilled LLM).
- Embedding из промежуточного слоя самого LLM после первого токена (TRAIL).
- Сам промпт (например, «summarize in 3 sentences» → short).
- Ranking-based (learning to rank, Fu et al. NeurIPS 2024).

**Важно:** предсказания **улучшаются со временем** — после первых 10 токенов мы знаем о задаче больше. Это **dynamic predictions**, нехарактерные для классики.

---

## 5. 🔬 Раздел 5: восемь open problems для single LLM

### Problem 5.1: Двумерное распределение $g(x, y)$ для реальных LLM

**Постановка.** Какое эмпирическое распределение $g(x, y)$ наблюдается на реальных traces (LMSYS, ShareGPT, Alpaca)?

**Почему важно.** Все формулы из раздела 3 зависят от $g$. Но никто не опубликовал карту $g$ для современных LLM-предсказателей.

**Суб-вопросы:**
- Условное распределение $y \mid x$ — нормальное? Логнормальное? Heavy-tailed?
- Есть ли систематический bias (LLM переоценивает длинные ответы)?
- Зависит ли $g$ от типа задачи (code, summarization, chat)?

### Problem 5.2: Memory-aware SRPT

**Постановка.** Как выглядит оптимальная политика, если:
$$\min_\pi E[T] \quad \text{s.t.} \quad \sum_{i \in \text{active}} m_i(t) \le M, \ \forall t$$
где $m_i(t)$ — memory usage запроса $i$ в момент $t$.

**Почему сложно.** SRPT **не уважает memory constraints**. Может начать слишком много длинных запросов, упереться в memory wall и стать хуже FIFO.

**Авторы предлагают гипотезу:** оптимальная политика — **thresholded SRPT**, где новый запрос берётся, только если его memory **не превысит порог**.

### Problem 5.3: Стоимость preemption формально

**Постановка.** Расширить SOAP так, чтобы rank-функция зависела от **preemption cost function** $c(\tau, a)$.

**Конкретная модель:** политика назначает задаче приоритет, но решение о preemption требует уплатить $c$. Оптимально ли всегда preempt? Когда игнорировать предсказания и дать задаче завершиться?

**Намёк на решение** (из TRAIL): **limited preemption** — после $k$ preemptions запрос «защищён» от дальнейшего вытеснения. TRAIL показала, что $k = 3$ даёт near-SRPT performance при значительно сниженной preemption overhead.

### Problem 5.4: Prefill/Decode совместное scheduling

**Постановка.** У нас два «queue» (prefill и decode), делящих один сервер. Классическая **two-class priority** модель, но с **взаимной interaction**:
- decode требует данные, созданные prefill;
- prefill новых запросов тормозит decode активных.

**Математически:** это вариант **priority queueing with server sharing** и фазовыми переходами. Открытый вопрос: найти оптимальную политику приоритета между ними.

### Problem 5.5: Динамические предсказания

**Постановка.** Предсказание $y_t$ обновляется после каждого токена. Предыдущие модели SOAP полагают rank функцией **только типа и возраста**. Здесь ранг — **стохастический процесс**, зависящий от наблюдений.

**Сравнение с Bayesian scheduling.** Это близко к классической задаче **Gittins index** для восстановления Bayesian posterior, но с более богатой структурой observations (целые embedding-вектора, а не просто service time).

**Технический вызов:** расширить SOAP-framework так, чтобы rank мог зависеть от accumulated history, не теряя **tractability**.

### Problem 5.6: Prompt sharing и prefix caching

**Постановка.** Многие LLM-запросы имеют **общий префикс** (system prompt, few-shot examples). Если два запроса имеют общий префикс длины $p$, их prefill можно **переиспользовать** — сэкономить compute.

**Scheduling-вопрос:** как группировать запросы, чтобы максимизировать prefix cache hit rate?

**Связь с классикой:** это вариант **batching** из теории очередей (**MX/G/1** — batch Poisson arrivals), но с дополнительной структурой — группировка зависит от **контента** запросов.

**Гипотеза авторов:** возможно, оптимально **delay** некоторые запросы, чтобы они попали в batch с общим prefix, — нарушая work-conserving принцип.

### Problem 5.7: Multi-server LLM scheduling

**Постановка.** Большие LLM распределены по нескольким GPU (tensor-parallel) или нескольким nodes (pipeline-parallel). Каждая шарда имеет свою очередь запросов и свою memory.

**Вопросы:**
- Как делать JSQ / Po2 в multi-server LLM, учитывая memory footprint?
- Когда мигрировать запрос между репликами (дорого — копировать KV)?
- Оптимальное **replication factor** как функция нагрузки.

**Связь с классикой:** это **parallel-server heterogeneous queues** (как в Jali et al. AISTATS 2024), но с **memory-affinity** ограничением.

### Problem 5.8: Fairness в LLM scheduling

**Постановка.** SRPT хорошо минимизирует mean latency, но **плохо обходится с длинными запросами** (starvation). В LLM это критично: длинный запрос — это, возможно, важный reasoning call.

**Задача:** найти политики, оптимизирующие **Pareto-frontier** (mean latency, tail latency, fairness).

**Намёки:** RSRPT (regularized SRPT), **age-based boosting** (приоритет растёт с возрастом).

---

## 6. 🔗 Раздел 6: Compound AI Systems

Это более амбициозный раздел — про **системы с внешними вызовами** (RAG, agents, tool use).

### 6.1 Модель

Запрос не моноблочный, а **цепочка операций**:
$$\text{prompt} \to \text{LLM}_1 \to \text{API call} \to \text{LLM}_2 \to \text{retrieval} \to \text{LLM}_3 \to \text{answer}$$

Каждый этап имеет своё время. **External API calls** (OpenAI, Google Search, DB queries) добавляют задержки, **не занимающие** серверные ресурсы.

### 6.2 Queueing-класс

Это **BCMP network** или **Jackson network** с особенностями:
- некоторые «станции» — инфраструктура с infinite servers (API);
- между станциями — **предсказуемые** (но переменные) задержки;
- KV-cache должен **переживать** API calls или пересчитываться.

### 6.3 Open problems

1. **Тратить ли KV memory во время API wait?** Если API returns в 200 мс, может, выгодно освободить GPU memory и принять новые запросы.
2. **Предсказывать duration API calls.** Классическая задача, но с новым источником данных.
3. **Scheduling для цепочек разной длины.** Запрос с 5 LLM-calls vs один монолитный — разный treatment.
4. **Speculative execution.** Начать следующий LLM-call, не дожидаясь API, на основе **предсказания** API response.

### 6.4 Связь с классической теорией

Авторы явно проводят параллель с **reentrant lines** (Kumar–Seidman, Lu–Kumar 1990s). Та же структура многопроходной обработки, та же нестабильность при наивном FIFO, тот же простор для умных политик.

---

## 7. 🧠 Раздел 7: Reasoning LLM Systems

Самая амбициозная часть статьи. **o1, DeepSeek-R1, Claude thinking** — LLM, которые генерируют много reasoning tokens перед ответом.

### 7.1 Что нового

**«Size» = число reasoning steps** до convergence, не число токенов.

**Предсказания тут богаче:** можно оценить, насколько модель «уверена» после $k$ steps, и решить — stop early или continue.

### 7.2 Key insight

**Можно динамически варьировать compute per request!** В классике service time — экзогенный. В reasoning LLM — **контролируемая переменная**:
- Greedy decoding: 1 sample, быстро.
- Self-consistency: $k$ samples, агрегация — в $k$ раз дороже.
- Tree-of-thoughts: ветвление — непредсказуемый compute.

### 7.3 Новый класс задач: scheduling + quality

**Постановка:** оптимизировать **не только latency**, но **trade-off латентности и качества**:
$$\min_\pi E[T] + \alpha \cdot E[\text{quality loss}]$$
где quality loss зависит от compute, выделенного запросу.

**Это совершенно новый класс задач** — в классике quality не зависит от scheduling. В reasoning LLM — зависит напрямую.

### 7.4 Open problems

1. **Stopping rules.** Когда прекратить reasoning — предсказать convergence.
2. **Dynamic compute allocation.** Больше reasoning для «важных» запросов, меньше — для простых.
3. **Branch pruning в ToT.** Какие ветки развивать (DUCHESS подход).
4. **Speculative reasoning.** Параллельно считать несколько reasoning paths, выбрать лучшую.

---

## 8. 🎯 Методологический вклад статьи

### 8.1 Педагогический: упрощённый вывод SOAP для noisy predictions

В разделе 3 авторы дают **self-contained вывод формулы E[T] для SPJF** — без необходимости знать полный SOAP framework. Это делает материал доступным для queueing researchers без deep ML background.

### 8.2 Таксономический: классификация LLM-систем

До этой статьи термин «LLM scheduling» относился то к single-instance, то к multi-agent, то к reasoning. Авторы **формально разделяют три класса**, каждый со своими scheduling challenges.

### 8.3 Катализирующий: создание research agenda

Цитата из заключения:

> *"We believe that the queueing theory community has much to contribute... The problems we have outlined are natural extensions of classical queueing theory, but with new constraints that make them both challenging and practically important."*

Эффект: статья уже цитируется в 20+ follow-up работах (Jaillet et al. 2025, Li–Dai–Peng 2025, Hong et al. 2025), которые **явно берут** open problems отсюда.

---

## 9. 📊 Что статья **не** делает (честный критический взгляд)

<details open>
<summary><b>Ограничения и пробелы (развернуть)</b></summary>

1. **Нет новых теорем.** Это survey/manifesto, не research paper в строгом смысле. Для кого-то это минус, для кого-то — достоинство.

2. **Мало эмпирических данных.** Авторы не публикуют собственные измерения $g(x, y)$ для реальных предсказателей — хотя это могло бы стать вкладом.

3. **Слабая связь с networking/OS community.** LLM serving активно обсуждается в OSDI/SOSP/MLSys сообществах (Orca, vLLM, Splitwise, Sarathi) — авторы упоминают эти работы, но не погружаются в system-level детали (paged attention, chunked prefill, speculative decoding).

4. **Не обсуждаются energy costs.** Power consumption — важный scheduling objective в data centers, игнорируется.

5. **Нет обсуждения privacy / isolation.** Multi-tenant LLM serving имеет side-channel concerns.

6. **Reasoning section недооценён.** В конце 2025 и начале 2026 reasoning models (o3, DeepSeek-R1, Claude 4.5 sonnet thinking) стали центральной темой. Статье стоило бы посвятить им отдельный фундаментальный раздел.

</details>

---

## 10. 🔑 Почему эта статья стала «главным текстом» области

<details open>
<summary><b>Пять причин (развернуть)</b></summary>

1. **Timing.** Март 2025 — момент, когда LLM serving стал коммерчески огромным, но теория существенно отставала. Статья появилась в правильный момент.

2. **Authority.** Mitzenmacher — один из основателей learning-augmented algorithms (ITCS 2018 с Vassilvitskii). Shahout — автор TRAIL, одного из немногих work, где theory встречается с реальным LLM-serving.

3. **Scope.** Статья покрывает три уровня: чистая math (section 3), systems (section 4), future research (5–7). Редкое сочетание.

4. **Open problems well-posed.** Каждая из 8 проблем в section 5 — **операциональна**: ясна постановка, ясна связь с классикой, есть намёки на подходы. Это делает статью продуктивной для аспирантов/постдоков.

5. **Bridge-building.** Явно адресована **queueing community** (опубликована в *Stochastic Systems*), но обсуждает ML/systems концепты. Это создаёт мост между двумя сообществами, ранее мало коммуницировавшими.

</details>

---

## 11. 📚 Как читать статью с максимальной пользой

**Рекомендованный порядок:**

1. **Секции 1–2** (intro + queueing background) — быстро, для контекста.
2. **Секция 3** — **внимательно с карандашом**. Здесь единственная техническая часть. Попробуйте **самостоятельно** вывести формулу E[W^SPJF].
3. **Секция 4** — для понимания **почему LLM — новая задача**.
4. **Секция 5** — **выберите 1–2 проблемы** из 8 и продумайте, как вы бы их решали.
5. **Секции 6–7** — для расширения кругозора.

**Follow-up reading по порядку важности:**

1. **TRAIL** (ICLR 2025) — единственный полностью проработанный пример проблемы 5.3 + 5.5.
2. **Jaillet et al.** (arXiv:2502.07115) — формальная атака на проблему 5.2.
3. **Li–Dai–Peng** (arXiv:2504.07347) — throughput-optimal treatment, косвенно проблема 5.4.
4. **SOAP** (SIGMETRICS 2018) — для глубокого понимания математического аппарата.
5. **Orca** (OSDI 2022) — system-level фундамент для section 4.

---

## 12. 🎓 Исследовательские возможности

<details open>
<summary><b>Какие из проблем наиболее tractable для диссертации/research project (развернуть)</b></summary>

**Lower-hanging fruit (6–12 месяцев):**
- **Problem 5.1** — эмпирический сбор $g(x, y)$ для разных предсказателей. Дешёвые ресурсы, публикабельно, фундаментально полезно.
- **Problem 5.3** (limited preemption) — extension TRAIL на другие cost models.

**Средняя сложность (1–2 года):**
- **Problem 5.2** (memory-aware SRPT) — требует нетривиальной теории, но сжато.
- **Problem 5.5** (dynamic predictions) — новый SOAP framework.

**Амбициозные (PhD thesis):**
- **Problem 5.6** (prompt sharing) — новый класс batching policies, связь с combinatorial optimization.
- **Problem 7.x** (reasoning scheduling) — совершенно новая область, нужен междисциплинарный подход.

**Industrial-lab оптимально:**
- **Problem 5.7** (multi-server) — доступ к реальным ЦОД-трейсам критичен. NVIDIA, Anthropic, OpenAI Research — идеальные площадки.

</details>

---

## 13. ✍️ Итоговая оценка

**Сильные стороны:**
- Задаёт direction для области на 5+ лет вперёд.
- Хорошо связывает классику и modern systems.
- Open problems **операциональны**.
- Читаема для обеих аудиторий (queueing + ML).

**Слабые стороны:**
- Нет собственных теорем или экспериментов.
- Reasoning section недопроработан.
- Energy/sustainability ignored.
- System-level details сокращены.

**Практический совет:** если вы начинаете research в этой области — **прочитать обязательно**. Если уже в ней — **читать как каталог** для поиска open problems, соответствующих вашим навыкам.

---

Если хотите, могу:
- **Вывести подробно** формулу E[W^SPJF] из section 3 с пояснениями каждого шага.
- **Разобрать TRAIL** как «первый существенный прогресс» по проблемам 5.3 + 5.5 из этой статьи.
- **Сравнить MC-SF (Jaillet et al.)** с проблемой 5.2 — насколько они её закрывают.
- **Обсудить**, как можно расширить SOAP framework под problem 5.5 (dynamic predictions) — возможный research contribution.

Куда углубляться?