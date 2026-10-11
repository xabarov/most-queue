# Л8: гетерогенное групповое обслуживание — воспроизведение и независимые проверки

Дата: 2026-10-11. [EPIC-091](../epics/EPIC-091-heterogeneous-bulk-service.md),
[API EN](../models/heterogeneous-bulk-service.md)/[RU](../models/heterogeneous-bulk-service.ru.md),
[JSON](heterogeneous-bulk-service-results-2026.json).
Л8 выбран пользователем после source gate Л7; Л7 остаётся waiting_source.

## Первоисточник и покрытие

Jan-Kees van Ommeren, Niek Baer, Nishant Mishra, Debjit Roy,
*Batch service systems with heterogeneous servers*, Queueing Systems
95:251–269 (2020), [DOI](https://doi.org/10.1007/s11134-020-09654-y),
[publisher PDF](https://link.springer.com/content/pdf/10.1007/s11134-020-09654-y.pdf).
Online publication 2020-06-11, CC-BY4.0; metadata сверены у издателя/Crossref.
Прочитаны все §§1–6 и references, 19 PDF страниц; формулы и таблицы
дополнительно осмотрены визуально на pp263,266,267.
PDF536765bytes и pdftotext сохранены локально в приватной диссертационной папке.
PDF SHA256 `00694b64aeb3f770d98bcdb4327c6fe20542d758fb91b6659410a636d3e68e0b`;
text `11713d1db8d753b4693a9f48c992f5f973f52e23c2c87cee49181f0131740cbf`.
Эти приватные копии не входят в Git/push.

**Не новая теория.** FP/BP decomposition, PH waiting и PH sojourn — результаты
авторов (Theorems1–2, Remark1). Библиотечный вклад — API, sparse implementation,
независимые арбитры и диагностики. General PH-service лишь обсуждается во введении,
а не решается этими формулами. Прежний [EPIC-069](../epics/EPIC-069-bulk-service-multiserver.md)
покрывает identical bulk servers, но не разные T/B/μ и эту PH-декомпозицию.

### Проверка предшественников

Локально прочитан Рыжиков, `CH7w.TEX` §7.12: busy-count state и геометрический
all-busy tail уже есть для heterogeneous single-job service; это не пакетная новизна.
В существующих карточках не найдена отдельная реализация данного контракта.

- Специализированная диссертация: [Dieter Claeys, *Analysis of queueing models with batch service*, UGent2011](https://biblio.ugent.be/publication/1955129).
  Прочитаны полное TOC и §§1.5–1.6, PDF/text сохранены приватно. Single-server
  discrete-time general size-dependent service, threshold/timer, delay moments/tails
  и correlated arrivals: важный сосед, не доказательство той же multi-type модели.
  Главы2–6 целиком не аудированы.
- Классическая ТМО: [Goswami–Samanta2009](https://doi.org/10.1016/j.cie.2008.08.009),
  publisher abstract/introduction/section previews: two heterogeneous geometric
  batch servers, waiting distribution; полный proof не прочитан.
  Arora1964, DOI10.1287/opre.12.2.286 найден в bibliography основной статьи;
  полные условия не прочитаны.
- Real-time/scheduling: [Efficient Online Scheduling for Deadline-Sensitive Batch Computing, Microsoft Research](https://www.microsoft.com/en-us/research/publication/efficient-online-scheduling-for-deadline-sensitive-batch-computing/):
  только авторский abstract, deadline-aware cluster batch jobs, не эта PH-модель.
- AoI: [Bhati–Pillai–Vaze2021](https://arxiv.org/abs/2109.05752), arXiv abstract:
  heterogeneous exponential servers, separate queues/random routing, average AoI;
  proofs не читались, queue/metric contract иной.
- ML systems: [Llumnix, OSDI2024](https://www.usenix.org/conference/osdi24/presentation/sun-biao),
  USENIX abstract; [Choi et al., ATC2022](https://www.usenix.org/conference/atc22/presentation/choi-seungbeom),
  abstract/BibTeX. Rescheduling и spatio-temporal sharing — не fixed exponential
  shared-queue contract. Их полные системы/измерения здесь не реплицированы.

Проверка по механизму с четырёх сторон не устанавливает отсутствие аналогов
неэкспоненциального расширения или новой свободной ниши. Л8 — согласованная
реализация известной модели, не автоматический переход к новой гипотезе.

## Математический контракт

Poisson λ; Nσ приборов типа σ, независимые batch Exp(μσ), 1≤Tσ≤Bσ.
Shared FCFS, пакет min(Q,Bσ), новый запрос не входит в работающий пакет.
Приоритет свободного типа — increasing T, при равенстве input order.
Устойчивость λ<ΣNσμσBσ; численный API guard ρ<1−1e−10 отделён от теоремы.
Гетерогенный набор не заменяется одним усреднённым μ.

All-busy Q имеет точную conditional PMF (1−α)α^q. Корень:

```text
λ = Σσ Nσ μσ Σ(k=1..Bσ) α^k,   0<α<1
D = Σσ Nσ μσ (1−α^Tσ),   E[BP]=1/D
BP exit to (N−eσ,q): rate Nσ μσ (1−α) α^q, q=0,...,Tσ−1
E[Q] = Σfree π_state q + P(BP) α/(1−α)
E[W] = E[Q]/λ
```

FP finite states include actual busy counts and q<lowest-idle threshold.
An aggregate BP with these exit rates preserves its exact mean duration and
exit distribution; it does not assert exponential physical BP duration.
Sparse stationary balance gives actual FP masses and P(BP).
Exact queue PMF adds the unbounded geometric term; no queue truncation in production.

Waiting PH: busy states ell=0,...,max(T)−1; free states (σ,m,ell) with
1≤m, m+ell<Tσ, m includes the tagged request. Arrival advances ell or starts
the tagged batch. Higher-priority completion either frees a better idle type,
absorbs the tagged request, or removes Bτ jobs ahead while **σ remains idle**.
In busy phases, geometric memorylessness removes the explicit number ahead.
Type absorption matrix R preserves dependence between W and eventual service type.
Adding service phases gives block generator [[A,R],[0,−diag(μ)]] for V.
Moments k!β(−A)^−k1, strict tail βexp(A t)1, LST atom+β(sI−A)^−1r.

Diagnostics compare mean PH waiting with Little, customer class flows with
PH absorption shares, total flow with λ, and batch starts with μ times busy
counts. They are numerical consistency checks, not certified rounding-error bounds.
Inherited p/pi remain None: Q-PMF is not total-population PMF.

## Расхождения с печатным текстом: не скрывать

| Место | Печатный вариант | Использованное правило и арбитр |
|---|---|---|
| Eq12, p264 | N1μ1 внутри суммы по σ | Nσμσ, сумма физических completion rates; direct CTMC/heterogeneous M/M/2 |
| E[W], p263 | BP mean-queue factor (1−α)/α | α/(1−α), среднее geometric PMF; Little + independent tagged chain |
| Theorem2, p266: busy→free | N1μ1 для каждого σ | Nσμσ; actual-type completion in physical oracle |
| Theorem2: m>Bτ | новый idle label τ | σ остаётся idle, τ сразу снова занят; физический tagged replay/CTMC |
| Theorem2 state boundary | m+ell=Tσ включена | transient только m+ell<Tσ; boundary immediate absorption, W=0 отдельный atom |
| Table2 equal thresholds | upper-right B-priority, lower-left A-priority | численные queue values соответствуют обратным приоритетам; Table3 labels согласованы |

Это проверенные расхождения с печатной математикой/подписями, не заключение
о реализации авторского ПО: оно не доступно и не аудировано. Eq9 LST помогает
разрешить повторённый индикатор n≤B в соседней ненумерованной рекурсии;
буквальная такая рекурсия не использована как самостоятельный эталон.

Для (TA,TB)=(4,4):

| Приоритет по §2 | Physical CTMC/калькулятор E[Q] | Table2 по подписи | P(W=0), согласуется с Table3 |
|---|---:|---:|---:|
| A | 2.23641674084 | 2.389 | 0.20465484050 |
| B | 2.38912742111 | 2.236 | 0.19525161842 |

Все 40 non-diagonal settings совпадают с округлением Table2; все 10 diagonal
priority variants совпадают после **явного** обмена Table2 значений, не Table3.
Максимальная абсолютная разность с rounding-adjusted Table2 —0.000496828;
с буквальной Table2 —0.153127421. JSON хранит оба варианта/разности.
Minimum E[Q] при (4,6):2.09417998633, max P(W=0) при (4,4), A-priority:
0.20465484050. Обе оптимальные настройки совпали с заключением статьи;
лучшие средняя очередь и нулевое ожидание требуют разных порогов.

## Независимые арбитры

`examples/heterogeneous_bulk_service_oracle.py` не импортирует production helpers.
Stationary physical CTMC хранит actual busy counts, Q=0..cap, rejecting at cap.
Tagged chain хранит actual counts, число впереди и позади; нет geometric ansatz,
BP/FP aggregate или сжатого PH. Число позади ограничивается max(T)−1 без
изменения threshold eligibility. Начальная Palm PMF из capped chain приближённая,
дальнейший tagged process не ограничивает число будущих приходов.

9 refinement runs: две source priorities и один three-type case, cap30/70/110.
Для A-priority boundary mass уменьшается 2.73e−4→7.68e−8→2.16e−11;
queue error −.010307→−6.19e−6→−2.67e−9;
max absolute W moment error (orders1–3) .01902→3.52e−5→3.09e−8.
Для B-priority cap110: queue error−3.22e−9, max W-moment error3.73e−8.
Показатель boundary mass не объявляется доказанной границей ошибки момента;
roundoff-level negative mass около1e−16 в three-type oracle оставлена в JSON явно.

Unit tests дополнительно проверяют Erlang C (включая heavy/light load),
identical-type regrouping, старый BulkServiceMultiserverCalc,
threshold-sort permutation/type-index mapping, PH row balance, LST derivative,
waiting atom, tail monotonicity, cache reset, ресурсные/аргументные guards.

## Независимый DES: без completed-only bias

Индивидуальные приборы и FCFS arrival timestamps, отдельные seeded streams;
no production kernel import. 10 settings×4 seeds81000..81003,
horizon12000, warmup1000. 4 source settings + 6 three-type settings,
load .2/.6/.85, immediate/threshold disciplines; synthetic, не GPU measurements.
Means — queue/population exposure divided by λ observation time.
Tails W/V at0,1,3 use all arrivals [1000,6500); available follow-up5500
exceeds every tail argument. Unfinished requests contribute known tail indicators,
не исключаются. Моменты только завершённых не выдаются.

Replication check: abs difference≤4.5SE+.02, не строгая simultaneous confidence
guarantee. Максимальная доля error/allowance —0.474105; все 40 runs сохраняют
arrived=completed+unfinished и обе area identities, max population-area
residual5.58e−9. Unit hand replay отдельно проверяет ожидание до **запуска**,
одновременное завершение пакета, остаточные exposure и полностью unfinished cohort.

## Воспроизведение

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m examples.heterogeneous_bulk_service_experiment
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/pytest \
  tests/units/test_heterogeneous_bulk_service.py \
  tests/test_heterogeneous_bulk_service_vs_sim.py -q
```

JSON excludes execution duration. Source table data copied with attribution
under the article's CC-BY4.0; derived numerical values and priority-label
corrections are identified as this reproduction, not verbatim author outputs.
Повторный запуск побайтно совпал: JSON SHA256
`4e5139154ee707f012cd1f4ef3a335a1a165dc5c3797fc4f5acac585ac821ae8`.

Целевые проверки:68passed/4.05s; отдельная slow-группа3passed/4.53s.
Full regression:3213passed/485.83s,16 прежних warnings, без reruns.
Fast regression:3199passed/389.71s,16 прежних warnings, без reruns.
Новые production/oracle/experiment modules pylint10/10; пакет9.99/10 с
прежними сообщениями. Black/isort/pre-commit, четыре исполняемых EN/RU
snippets и визуальный осмотр обеих новых схем прошли.
Архив95снимков/SHA256,7sync tests и локальные Markdown-ссылки проверены.
Journal prefix59772bytes/SHA256
`404522d9b15e4c9347e3443c3440ea7fd1c7b1cc7f739ce37641ec05fd5f7deb`
сохранён дословно; новая запись только добавлена.

## Границы и вывод для диссертации

Точное waiting/sojourn distribution для exponential heterogeneous batch service
**уже существует**. Не заявлять PH waiting или разные T/B/μ новой теорией.
Полезный базис синтеза: отдельно выбирать mean queue, mass at zero и tail SLO,
приоритет действительно меняет результаты. Source grid search не решает
allocation Л7 и не доказывает глобальную оптимальность произвольного control.
Не переносить результат на неэкспоненциальный GPU service, continuous admission,
растущий KV или TTFT/ITL. Для расширения нужны phase/residual-service states
и full-text проверка ближайших предшественников, а не усреднение μ.

Материалы/карточки/содержательные выводы и append-only journal сохраняются
приватно; source/report/JSON snapshots сверяются SHA256. Commit/push этой
папки не резервируют. Научная новизна нового направления не установлена.
