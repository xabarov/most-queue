# Точная вероятность нарушения SLA для batch-service очередей — обзор литературы (2026)

## Цель

Связать три уже существующих, но до сих пор не соединённых куска: (1) EPIC-032/043 — точные
моменты `W` для `M/PH^[a,b]/1` (a=1) через смесь гипоэкспоненциальных слагаемых
(`BulkServiceMM1Calc.get_w`/`BulkServiceErlangCalc`/`BulkServiceH2Calc`); (2) EPIC-021 —
SLA-слой `most_queue.theory.utils.sla`, который для ЛЮБОГО калькулятора считает `P(W>D)`
только через ПОДГОНКУ 2-3 моментов к H2/Gamma (приближение; единственный точный якорь —
`mm1_deadline_violation_prob`, который сам не покрывает batch-service); (3) практика
GPU/LLM-инференса с динамическим батчингом, где именно `P(latency > SLA)` — целевая метрика,
не среднее.

## Прямой конкурент

Inoue Y., *Queueing Analysis of GPU-Based Inference Servers with Dynamic Batching: A Closed-Form
Characterization*, Performance Evaluation 147, 2021, doi:10.1016/j.peva.2020.102183
(arXiv:1912.06322, открыт). Прочитан полностью. Модель: `b_max=∞` (сервер забирает ВСЕ ожидающие
заявки при освобождении — не `[a,b]`-окно), время обслуживания батча — детерминированно-линейная
функция размера (`αb+τ₀`, подогнано по медиане измерений, никакой variance). Результат — **верхняя
оценка СРЕДНЕЙ задержки** (Theorem 2), не точный момент и не хвост/CDF. Автор explicitly:
«batch-size dependent processing times make the system dynamics complicated and it is difficult
to obtain a closed-form formula even in the M/M/1 model» и указывает, что конечный `b_max`
обрабатывается только алгоритмически через матрично-аналитический аппарат Neuts (M.F. Neuts,
*Structured Stochastic Matrices of M/G/1 Type and Their Applications*, 1989) — без closed form.
Автор сам показывает (Fig. 8 статьи): при реалистичном (небольшом) `b_max` эта оценка систематически
расходится с точным значением. Статья имеет 24 цитирования за 5 лет — заметна, но не переоткрывает
нашу нишу (точный хвост, не только среднее; ограниченное окно `[a,b]`; произвольный CV через
фазовую подгонку, не только детерминированно-линейное время).

## Углублённая проверка пересечения «точное + время ожидания + batch-size-dependent + непрерывное время»

Отдельно, прицельно проверена многолетняя (1970–2026), очень активная школа continuous-time
bulk-service с batch-size-зависимым обслуживанием (Chaudhry, Gupta, Samanta, Banik, Banerjee,
Pradhan, Barbhuiya, Chakravarthy и соавторы — десятки статей, Performance Evaluation / Computers &
OR / Queueing Systems / OPSEARCH / QTQM / JISPS). Устойчивая картина по каждой найденной работе:

- Pradhan, Gupta, Samanta, *M/Gʸᵣ/1* (OPSEARCH, 2015) — batch-size-dependent, continuous-time,
  ТОЧНО, но только длина очереди (departure/random epoch), не время ожидания.
- Banerjee, Gupta, Chakravarthy, *Computers & OR* 60 (2015) 138–149 — MAP-вход,
  batch-size-dependent, явно phase-type сервис — распределения длины очереди/содержимого сервера
  «в разные моменты», не время ожидания (по всем найденным вторичным описаниям).
- Gupta и др., *On M/G^(a,b)/1/N queue with batch size- and queue length-dependent service*
  (Springer, 2018, book chapter) — явно phase-type, явно batch-size-dependent — снова joint
  queue-length/server-content distributions, не время ожидания.
- Pradhan, Nandy, Gupta, *QTQM* 22(4) (2024) 683–726 — batch-size-dependent + group-arrival +
  vacation — снова joint queue/server-content PGF at departure epoch.
- **Контрольная проверка той же школы в 2025:** Banik, Chaudhry, Barik и др., *On the Heuristic
  Computational Procedures of the Virtual Waiting-Time Distribution... MAP/R^(a,b)/1/N*, J. Indian
  Soc. Probab. Stat. 26 (2025) 585–630 — время ожидания ЕСТЬ в заголовке, но явно названо
  **приближением** («Approximation to the phase-wise virtual waiting-time distribution»), и сервис
  здесь НЕ batch-size-dependent (R-type, фиксированный класс). Если та же группа в 2025 году не
  получает точное время ожидания даже без batch-size-зависимости, это сильный аргумент, что точное
  время ожидания **с** batch-size-зависимостью остаётся открытым, а не тихо решённым где-то ещё.

Единственная найденная явная пара «время ожидания + batch-size-dependent service» —
Claeys, Steyaert, Walraevens, Laevens, Bruneel, *Computers & OR* 40(5) (2013) 1497–1505 — но это
**дискретное время** (телеком/ATM-школа) и явное **приближение** (авторы сами пишут
«deduces approximations»), с отдельным timer-механизмом не из нашей модели.

**Вывод:** пересечение «точное + continuous-time + время ожидания (не только длина очереди) +
batch-size-dependent service» не закрыто ни в основной (Chaudhry/Gupta-школа), ни в
телеком-школе (Bruneel/Claeys), включая самые свежие (2024-2025) публикации тех же групп.

## Смежная литература (проверено — не дублирует)

- Прецедент паywalled-уровня по admission control (Das, Jenkins & Sengupta, 2013) и соседние
  admission-control/impatience работы (2008 MOR asymptotically optimal admission control; 2017
  POM capacity-random-impatient; 2026 Queueing Systems "The power of letting go" — Shmelev &
  Zychlinski, fluid-асимптотика индексных политик rejection+abandonment) — другой механизм
  (control/optimization, не точный transform-анализ конкретной дисциплины), другая постановка.
- Обширная (десятки работ, 2012–2026: Performance Evaluation, Queueing Systems, QTQM, 4OR,
  Methodology and Computing in Applied Probability) линия «batch-size-dependent service bulk
  queues» — почти всегда discrete-time, finite-buffer, без GPU/LLM-мотивации и почти без
  фазово-типового (Erlang/H2, произвольный CV) времени обслуживания; ни одна из найденных не
  даёт ТОЧНУЮ CDF/хвост `W` для непрерывного `M/PH^[a,b]/1`.
- 2024–2026 LLM-serving SLO/TTFT статьи (QUARTZ, *Formal schedulability analysis for LLM
  inference: TTFT and TBT deadline guarantees*, *Tool-Augmented LLM Serving Under Firm
  Deadlines*) — алгоритмические/эвристические или response-time-analysis (real-time systems
  style, worst-case, не стохастический точный хвост), подтверждают практическую актуальность
  метрики `P(latency>SLA)`, но не дают того же результата.

## Что уже есть и что добавляется

`BulkServiceMM1Calc.get_w()` (a=1) раскладывает `W` по состояниям стационарного распределения
`pi(i,j)`: если сервер простаивает — `W=0`; если занят — `W` = остаток текущего батча
(`Exp(mu(i))`, по марковости) `+` `j // b` полных батчей размера `b` впереди (каждый `Exp(mu(b))`,
независимо) — ТОЧНАЯ гипоэкспоненциальная сумма. Сейчас из неё берутся только сырые моменты
(`conv_moments`). У гипоэкспоненциального распределения (сумма независимых, не обязательно
одинаковых экспонент) есть ТОЧНАЯ замкнутая CDF/хвост (классическая формула через разложение на
простые дроби при различных ставках, вырождающаяся в Erlang-формулу при совпадении ставок).
Следовательно: **смешивая эти точные per-state хвосты по `pi`, получаем ТОЧНУЮ (не подогнанную)
`P(W>D)` для `M/PH^[a,b]/1`, a=1** — ровно тот случай, который EPIC-021 исключил как mean-only,
и который теперь можно сделать СИЛЬНЕЕ стандартного приближённого SLA-слоя, а не просто
«догнать» его до уровня остальных калькуляторов.

## Научная новизна и позиционирование

1. Точная (не bound, не fitted) `P(W>D)`/CDF для реалистичной дисциплины `[a,b]`-окна —
   сильнее результата Inoue (2021) по глубине (хвост, не только среднее) и по реализму
   (ограниченный батч, произвольный CV через Erlang/H2-подгонку, не детерминированно-линейное
   время).
2. Первый точный (не приближённый через fit) анклав SLA-слоя most-queue за пределами M/M/1 —
   закрывает явно зафиксированный пробел EPIC-021 (`docs/research/sla-deadline-queueing-2026.md`,
   «bulk-service исключён, mean-only») новым, более сильным результатом, а не просто подключением
   старого приближённого механизма.
3. Численное сравнение точного результата с (а) приближением Inoue (на какую величину их bound
   расходится с истинным `P(W>D)` при реалистичном `b_max`), (б) приближением EPIC-021
   (`fit_from_moments`) на том же самом кейсе — честная количественная демонстрация цены
   приближений, не просто новая формула без контекста.

## Резерв (не в этом эпике)

- `a>1` — тот же «idle-refill» разрыв декомпозиции, что уже остановил точные моменты в EPIC-032
  (см. `docs/research/bulk-service-waiting-moments-2026.md`); для хвоста понадобится то же самое
  решение, что и для моментов (отдельная, более сложная задача).
- Переменная длина генерации токенов внутри батча (continuous batching, добавление заявок в уже
  идущий батч) — ни Inoue (2021), ни найденная bulk-service литература этого не рассматривают;
  требует отдельного обзора литературы (LLM-continuous-batching queueing модели) перед оценкой
  трактуемости — не хватает уверенности, чтобы включать в контракт прямо сейчас.
