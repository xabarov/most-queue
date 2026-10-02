# EPIC-060: Эффективная ёмкость и ресурсные ограничения GPU

- **Статус:** done
- **Создан:** 2026-10-02
- **Roadmap:** [Реальные трассы](../roadmaps/real-trace-calibration.md)

## Цель и наблюдаемость

Проверить чувствительность очереди и выбора дисциплины к размеру доступного
пула и эксклюзивному резервированию узлов. Это **сценарная модель**, не оценка
реальной effective capacity. В закреплённом Acme/Kalos есть requested GPU/node
counts, но нет per-job placement, событий доступности узлов, quota/partition
membership и истории квот. Поле `queue` — длительность ожидания, не ID очереди;
user не превращается в tenant/quota без доказательств. CPU/memory запросы сами
по себе не восстанавливают совместную доступность/размещение.

Источники: [схема AcmeTrace](https://github.com/InternLM/AcmeTrace/blob/f9fdf591b4876c2875a9e3d28adb1bda8120dfcb/README.md),
неизменённый hash-pinned Kalos из EPIC-058/059. Не загружаем многогигабайтную
телеметрию: utilization не тождественна доступному scheduler capacity.

## Предписанный эксперимент

Протокол фиксируется до нового replay. Историческое W не используется для выбора
capacity, моделей или сценариев. Пересматриваются известные периоды; нет заявления
нового blind holdout. Ни один сценарий не выбирается как «лучшая реконструкция».

| Origin, доля raw submit span | Warmup completed | Measured completed | Назначение |
| --- | ---: | ---: | --- |
| .35 | 200 | 1000 | ранний активный период |
| .50 | 200 | 1000 | другой активный период |
| .70 | 100 | 300 | поздняя изменившаяся смесь |

Fixed cohorts: первые warmup+targets completed после cutoff, source-order ties.
Отбор outcomes ретроспективен. Все пригодные cancelled/failed competitors в
интервале и partial running/waiting snapshot сохраняются, как carry_terminal
EPIC-058. Для любого counterfactual сохраняются те же IDs/arrival/elapsed age;
S одинаков между ресурсными сценариями внутри одного variant/seed. Initial
running/waiting S всегда записанное, их остаток не ресэмплируется;
историческое waiting не навязывается новым arrivals. CPU-only не добавляются.

Матрица ресурсов:

- `gpu_pool`: demand=requested GPU K, capacity=8×N.
- `exclusive_nodes`: demand=recorded node_num, capacity=N. Работа резервирует
  узлы целиком независимо от K; это допущение, не восстановленное размещение.
- N заранее [302,226,151,113], то есть GPU inventories [2416,1808,1208,904].
  Все значения показываются; нет выбора по fit к W. 302 соответствует прежнему
  nominal pool, остальные — floor(302×[.75,.5,.375]) как sensitivity.
- Если отдельный job больше capacity либо исходный running sum больше capacity,
  ячейка `infeasible`, без удаления/масштабирования jobs или освобождения carry-in.
  В ней нет расписаний и нет выдуманных latency/zero-success значений.

Observed S reference плюс **один** recent-coarse empirical baseline (history≤4000,
minimum=20), восемь seeds 60000–60007. Никакого повторного выбора типа модели.
Train только completed end<cutoff, coarse groups по исходному K: 1 / 2–8 /
9–32 / >=33, иначе pooled. Общий U для одного ID/seed/origin во всей ресурсной
матрице. Forecast неизменный expanding-coarse p90 исходного K, численно один
для всех resources/capacities/seeds; sampled S не раскрывается диспетчерам.

FCFS/FirstFit/MSF/Adaptive Quickswap/EASY/Conservative не меняются. Максимум
3×2×4×(1+8)×6=1296 расписаний. По coverage audit до измерения очередей ожидаются
4 infeasible cells (при .50/.70 N=113, обе модели), 20 feasible cells и **1080
расписаний**. Фактический envelope записывается до запусков; несовпадение с
ожиданием расследуется, а не исправляется подгонкой.

## Метрики и проверка

Для фиксированных targets: mean/p99 T, mean/p99 W, K-weighted T (вес всегда
исходный GPU K), классовые counts/T/W по исходному K. Дренирование полное.
Все targets completed, так как новых runtime limits нет; labels окружения
сохраняются отдельно. Сравнение с observed **того же resource/capacity сценария**.

Занятый ресурс выражается явно: reserved units (GPU либо node), reserved
GPU-equivalent=units×(1 либо 8), и requested GPU time отдельно. Из временных
массивов независимо интегрируются requested-GPU utilization и ledger; slack
между ними и эксклюзивным node reservation не называется физической загрузкой.
Observation window от первого target arrival до последнего, ledger включает drain
и окружение, carry-running учитывает только остаток после boundary.

Сохраняются MC-интервалы модели и парные изменения mean/p99 T относительно
nominal GPU pool (общие seeds), отдельно deterministic observed contrasts.
Policy regret по mean/p99 T и tie count reference; нулевой regret при равенстве
policies не свидетельствует о валидности выбора. Нет CI по зависимым origins,
нет статистического доказательства production scheduler superiority.

`Observed` — replay с записанными S, не исторические T/W. Choice по минимуму
MC-среднего выбранной метрики, ties в порядке POLICIES. Relative regret =
observed(chosen)/min_policy observed−1, в той же ячейке. Reference ties:
isclose(rtol=1e-12, atol=1e-9). Нормативные формулы и алгоритмы воспроизведения
приведены в [методике](../gpu_resource_envelope.md); independent reader запросил
эти уточнения до прогона.

Отдельный аудит наблюдаемости: full accepted trace single-job/peak concurrent
GPU и node requests, интеграл превышения каждого scenario-capacity. Это
ретроспективная **совместимость** recorded occupation с допущением, не estimator
capacity. Характеристики исторического W — описательные, не objective настройки.

## Работы

- [x] Аудит доступных полей и границ идентификации capacity/quotas.
- [x] Resource-demand projection и строгая проверка feasible envelope.
- [x] Зафиксированный paired runner, явные units и неподменённые cohorts.
- [x] Offline-тесты, аналитические малые примеры, regression nominal replay.
- [x] Полная матрица, побайтовый повтор и независимый аудит.
- [x] Полный/быстрый pytest, форматирование, pylint/pre-commit.
- [x] Методика/отчёт, README/models/roadmaps и независимая reader-проверка.

## Результаты

Выполнено **1080 расписаний** в 20 feasible cells; четыре N=113 клетки .50/.70
сохранены как infeasible без удаления oversized jobs. Повтор на другом числе
worker дал побайтовое совпадение всех пяти JSON. Независимый аудитор восстановил
180 service лент, пересчитал 120 observed schedules и проверил 1080 строк,
1200 модельных интервалов, 720 контрастов и 40 решений.

Observed ожидание появляется при меньшем ресурсе/эксклюзивных узлах, но mean T
разделяет policies только в двух ячейках, а p99 T — ни в одной. Нулевой regret
во всех 40 решениях не валидирует выбор дисциплины. Ошибки генератора recent-coarse
при снижении capacity могут резко увеличиваться. Историческое W не использовалось
для выбора capacity. Реальная реконструкция квот, переменная capacity и
node-placement остаются за границей без необходимых наблюдений.

55 новых offline-тестов; полный pytest **1632 passed**, быстрый **1623 passed**.
Pylint нового модуля/runner — 10/10, библиотеки — 9.98/10 с прежними замечаниями.
Форматирование и pre-commit проходят. Предварительный стохастический сбой старого
теста и успешные повторы описаны в отчёте; допуски не менялись.
Reader-проверка уточнила единицы, правило выбора, границы наблюдаемости и три
группы policies в .70/N=151; финальные таблицы сверены с JSON, блокирующих
неоднозначностей не осталось.

[Методика/API](../gpu_resource_envelope.md),
[результаты и ограничения](../research/gpu-resource-envelope-results-2026-10.md),
[артефакты и аудит](../../works/gpu_resource_envelope/README.md).
Следующий эпик — queue-aware выбор модели на раннем replay с отдельной временной
проверкой, coarse baseline и контролем class coverage; не перенастройка по уже
просмотренным результатам.
