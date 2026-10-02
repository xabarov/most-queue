# EPIC-062: Совместная генерация приходов и ресурсных признаков

- **Статус:** done
- **Создан:** 2026-10-02
- **Завершён:** 2026-10-02
- **Roadmap:** [Реальные трассы](../roadmaps/real-trace-calibration.md)

## Цель и границы

Проверить генерацию интервала до прихода Δ вместе с mark=(K,context,request),
раздельно оценивая связь Δ↔mark, порядок соседних jobs и окно истории arrivals.
Сохранить fixed-arrival coarse baseline; не выбирать вариант после test.
S во всех генераторах берётся из одной recent coarse CDF по сгенерированному K;
context/request здесь сохраняются для диагностики, но не меняют S или scheduler.
Это факторизация P(Δ,K,context,request)×P(S|K), не полная зависимость S от признаков.

Все варианты — **completed-only, empty start**, с общим числом warmup/targets.
Старое историческое окружение не переносится на изменённые synthetic timestamps.
Нет carry-in, cancelled/failed конкурентов, caps, latent successful S, изменения
capacity или диспетчера. Результаты не сравниваются численно с lifecycle очередями
EPIC-061 как будто начальные состояния и популяции одинаковы. Warmup не является
доказательством стационарности. Источники и периоды уже изучались: не blind test.

## Протокол до новых replay

| Source | Origins raw submit span | Warmup | Targets | Capacity |
| --- | --- | ---: | ---: | ---: |
| SDSC | .85, .90 | 200 | 1000 | 128 processors |
| Kalos | .65, .70 | 100 | 300 | 2416 requested GPU |

Observed cohort — первые warmup+targets eligible completed с submit>=cutoff,
source-order ties. Выбранные arrival blocks не пересекаются; insufficient/overlap
отклоняют протокол, а не меняют counts. Все generated jobs — **синтетические
позиции**, не те же source IDs; классовый состав и weighted denominator могут
отличаться от observed. Fixed_coarse сохраняет observed K; matched pairs сохраняют
mark inventory каждой части. Их исключения указаны ниже.
Все policies одного variant/seed получают одну ленту. Нет обрезки потока по
историческому last-submit, подгонки его rate или rescaling горизонта.

Для arrival donors берётся полностью разрешённый **префикс** eligible completed
с submit<cutoff: остановка перед первой работой с end>=cutoff. Все следующие
completed исключаются из arrival history, даже если уже завершились. Поэтому
Δ_i=submit_i−submit_(i−1) — реальная соседняя пара в выбранной популяции, без
мостиков через удалённые незавершённые jobs. Первый prefix job даёт только
предшествующий timestamp; donor tuples — строки 2..N. Recent — последние 4000
таких tuples, expanding — все. Нулевые Δ сохраняются. Prefix lag, omitted suffix
и число уже известных, но исключённых outcomes аудируются; это возможная
устарелость arrival fit, не online-доступность будущего completed-label.

S-fit отдельно: последние 4000 всех completed с end<cutoff (как прежний coarse),
minimum=20 для K=1/2–8/9–32/>=33, pooled>=2. Forecast — общий expanding-coarse
linear p90 всех completed end<cutoff; заморожен во всех вариантах.
Нет fit по held-out S или перестройки выбора по результатам test.

| Вариант | Arrival/mark generator | Контроль |
| --- | --- | --- |
| observed | записанные arrival,K,context,request,S | completed-only empty reference |
| fixed_coarse | записанные arrival/K/marks; recent coarse S | обязательный baseline |
| recent_joint_iid | iid donor tuples (Δ,K,context,request) | joint one-job relation |
| recent_gap_independent | те же Δ, mark bundles переставлены внутри warmup/targets | ослабляет Δ↔mark, сохраняет K↔context/request |
| recent_joint_block20 | circular blocks 20 donor tuples | серийный генератор |
| recent_joint_shuffle20 | перестановка готовых block20 tuples **вместе с S** | тот же inventory/work/horizon, иной порядок |
| expanding_joint_iid | iid tuples из полного prefix, S-fit прежний recent | изменение только arrival-history window |

Перестановки отдельно в [0,warmup) и [warmup,total), первый элемент каждой
части закреплён. Остальные сортируются по независимым U-ключам (stable ties).
Это сохраняет inventory каждой части, первый measured arrival и его horizon;
shuffle не iid, а matched multiset control. В gap-independent перемещается
mark bundle, S затем генерируется по новому K; полная работа там не обязана
совпадать с joint_iid. В block/shuffle совпадает точно. Context=request bucket
SDSC вместе с исходным numeric request, либо exported type Kalos и request=None.
Block/shuffle меняет порядок и warmup, и targets: состояние очереди к началу
измерения может различаться. Это общий эффект порядка, не изолированная перестановка
targets из одного состояния. Wrap/block boundaries создают искусственные соседства.

Arrivals=cumsum(Δ)−Δ_0, так что первый job в 0. Zero Δ — batch, без jitter;
observed/fixed timestamps=submit−first_cohort_submit. Synthetic arrival ties
обрабатываются в порядке сгенерированных позиций, observed ties — source-order.
Если весь measured span нулевой, time averages/rate=null, latency после drain
всё равно считается. Empty/invalid history, слишком короткая для block20, ошибка,
без автоматического уменьшения блока. Нет часовой/суточной привязки или Hawkes fit.

Seeds 63000–63007. SeedSequence([seed, source_index, *cutoff_words]).spawn(4),
source_index SDSC=0/Kalos=1, cutoff_words — little-endian float64 как два uint32.
Каждый child используется в отдельном default_rng, U=(integers(0,2**52)+.5)/2**52.
Четыре независимые ленты в порядке: donors, S, mark-permutation, block-permutation.
Donor start floor(U*n),
block starts используют U в позициях 0,20,40...; end-to-start wrap. Coarse S через
inverse ECDF ceil(U*n)−1. Все варианты используют общие U своего назначения.
Четыре origins×(1 observed+6×8)×6 policies = **1176 расписаний**.

## Метрики, контрасты и приёмка

FCFS/FirstFit/MSF/Adaptive Quickswap/EASY/Conservative неизменны, все jobs drain.
Mean/p99 T, mean W, K-weighted T; counts/T/p99 четырёх K-групп.
Для пустой группы count=0, mean/p99=null. Pearson correlations при нулевой
дисперсии или недостаточном размере null; SCV при нулевом mean gap null.
Time averages от первого measured arrival до последнего, не включая drain;
resource work по всем jobs и отдельно targets. Rate/work-per-horizon описательны,
не доказательство стационарной устойчивости. Arrival horizon различается между
генераторами, кроме matched block/shuffle и joint/gap-independent controls.

Диагностика: mean/SCV/zero fraction gaps (исключая первый measured gap), lag-1 Δ/K,
correlation Δ↔K, marginal K-group/context TV и joint K-group×context TV против
observed targets; mean/p99 S, target requested-resource work и horizon. Нет
детерминированного сопоставления synthetic jobs с историческими job IDs.

Главный descriptive loss Q **отдельно по каждому origin** — equal-policy mean absolute log error MC mean
mean/p99 T против observed этого completed-only сценария. Парные seed-level
контрасты L(a)−L(b), где L — mean по 6×2 метрикам |log(metric)−log(observed)|:
joint_iid−gap_independent; block20−shuffle20; recent_iid−expanding_iid;
каждый новый generator−fixed_coarse. Отрицательное лучше левый. 95% t-CI по
8 seed differences, условные на fit/cohorts; не CI по зависимым origins или
causal production effect. Q(mean metrics) не подменяется mean L(seed).
Policy regret с полным списком observed ties; нулевой regret при ties не валидация.
Regret отдельно по mean T/p99 T: observed(chosen)/min_policy observed−1, chosen
по минимуму MC mean, точные ties в порядке перечисленных policies. Reference ties
isclose(rtol=1e-12,atol=1e-9). p99 — NumPy linear quantile, затем mean quantiles
по seeds, не pooled quantile. Положительные finite T обязательны для log loss.
Это численные соглашения [EPIC-061](../queue_aware_selection.md), но не его
lifecycle population. Всего 24 Q, 32 descriptive paired CI и 48 policy decisions;
нет multiplicity-adjusted подтверждающих тестов или объединения origins в CI.

- [x] Строгий marked-arrival API, prefix audit, matched controls.
- [x] Runner, fixed coarse, frozen forecasts и независимые diagnostics.
- [x] Синтетические/аналитические/regression/leakage тесты.
- [x] 1176 replay, полный побайтовый повтор, независимый аудит.
- [x] Полный/быстрый pytest, black/isort/pylint/pre-commit.
- [x] Методика/отчёт, README/models/roadmaps и reader review.

## Результаты

Реализованы `ArrivalMark`, `MarkedArrivalBootstrap`, строгая генерация без jitter,
согласованные перестановки и отдельный runner. Диспетчеры и прежние модели не менялись.
[Методика/API](../joint_marked_arrivals.md),
[отчёт](../research/joint-marked-arrivals-results-2026-10.md),
[артефакты](../../works/joint_marked_arrivals/README.md).

1176 расписаний и полный повтор дали шесть побайтово одинаковых JSON; проверены
20 implementation и пять output hashes. Независимый аудит: 196 workload tapes,
32 matched block pairs, все 24 observed schedules, 1176 rows, 1008 model intervals,
24 Q, 32 paired contrasts и 48 policy decisions. Диспетчер общий, не независимая
вторая реализация. Протокол и metadata записаны до replay.

Устойчивого выигрыша joint/block нет. В SDSC .90 все новые generators хуже
fixed coarse по seed-level loss, в Kalos .65 некоторые лучше. В Kalos .70
recent joint horizon=0.70 дня против observed 19.24, K-TV=0.6292; completed-prefix
отстаёт от cutoff на 4.63 дня. Q от MC means и средний seed-loss могут менять
знак сравнения. Это не причинное объяснение drift и не новый выбранный победитель.
Kalos observed имеет шесть ties; zero regret не валидирует scheduler.

Добавлено 59 offline-тестов. Полный pytest **1740 passed**, быстрый **1731 passed**,
оба с 16 warnings; CI-policy разрешала один повтор AssertionError, он не понадобился.
Black/isort/pre-commit прошли, новый модуль и runner pylint **10/10**, библиотека
**9.98/10**, без новых замечаний. Reader review до запуска уточнил prefix,
warmup/shuffle и оцениваемые величины; финальная сверка отчёта с JSON пройдена.
README EN/RU, models, simulation и оба roadmap синхронизированы.

Следующий этап: аудит нового источника с наблюдаемыми ресурсными ограничениями
и submission-time marks. Отдельная будущая ветка — arrival history без устаревающего
completed-prefix и с явным учётом цензурирования S. Fixed-arrival coarse сохранить;
не подбирать capacity/window/block по рассмотренным origins.
