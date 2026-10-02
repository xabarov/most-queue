# EPIC-061: Выбор модели обслуживания по ошибкам очереди

- **Статус:** done
- **Создан:** 2026-10-02
- **Roadmap:** [Реальные трассы](../roadmaps/real-trace-calibration.md)

## Цель

Проверить, переносится ли выбор генератора S по ранним ошибкам mean/p99 T
лучше выбора по CRPS обслуживания. Это отдельный предписанный эксперимент,
не замена победителя EPIC-059 после просмотра test. Coarse baseline обязателен;
новых дисциплин, настройки capacity, совместной генерации arrival/K/context
и восстановления latent successful S из cancelled/failed нет.

Используются прежние hash-pinned SDSC SP2 и Acme/Kalos. Эти периоды уже изучались:
временное разделение алгоритма не делает исследование новым слепым тестом.
Kalos nominal GPU pool назначен контролем ожидаемой слабой различимости policies;
фактические ties будут измерены, не предполагаются заранее.

## Протокол до новых replay

| Source | Validation origins | Test origins | Warmup | Validation targets | Test targets | Capacity |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| SDSC | .60, .70 | .85, .90 | 200 | 400 | 1000 | 128 processors |
| Kalos | .35, .50 | .65, .70 | 100 | 300 | 300 | 2416 requested GPU |

Origins — доли полного raw submit span. Cohort — первые warmup+targets eligible
completed с submit>=cutoff в source-order. Выбранные warmup+target блоки не
пересекаются по arrivals/IDs; окружение может повторяться, история переиспользуется.
Недостаточный размер или пересечение отклоняют весь протокол, не меняют cohort.
Fit — последние 4000 по submit order completed с end<cutoff; minimum cell=20,
pooled minimum=2. Прогноз диспетчера — общий expanding-coarse p90 на cutoff,
не зависит от кандидата или generated completion и не обновляется внутри replay.
S генерируется только для выбранных новых completed, включая warmup; carry-in
и cancelled/failed companions сохраняют observed S, elapsed age, outcome.

Partial carry-in и terminal environment как EPIC-059. Все записанные outcomes
validation cohort **и его окружения** должны иметь end<первого test cutoff своего источника;
иначе весь протокол отклоняется, без удаления долгих jobs. Это ретроспективное
окружение, не доступный online snapshot полных длительностей. В test refit использует
только завершённую к его cutoff историю, в том числе доступные ранние outcomes.

Три неизменённых кандидата на источник: SDSC coarse/request_bin/request_ratio;
Kalos coarse/type_coarse/type_exact. Requested time — известный request, type —
ретроспективный exported label, не гарантированно online. Модели и fallback
EPIC-059, coarse группы K=1/2–8/9–32/>=33. Восемь seeds: validation 61000–61007,
test 62000–62007. U общий между кандидатами/сценариями/policies для ID/seed/origin;
генератор и ECDF inverse как EPIC-059, cutoff/source входят в SeedSequence.

Шесть прежних policies: FCFS, FirstFit, MSF, Adaptive Quickswap, EASY, Conservative.
Основной сценарий без cap: SDSC carry_cancelled, Kalos carry_terminal.
На двух SDSC test origins отдельно requested_limit, **не участвующий в выборе**:
короткий terminal T из-за timeout не считается улучшением успешного обслуживания.
Итого 8 origins×(1 observed+3×8)×6 + 2 SDSC cap×(1+3×8)×6 = **1500 расписаний**.

## Правила выбора и проверки

Для каждого источника один queue-aware выбор:

`Q(m) = mean_{validation origin, policy, metric in [mean T,p99 T]}
          abs(log(mean_seed metric(m)) - log(metric(observed)))`.

Observed — replay с записанным S того же сценария, **не историческое W/T**.
Равные веса origins, policies и двух метрик. T строго положительно; нули,
nonfinite и несовпадающие shapes отклоняются, без epsilon или broadcasting.
Это проектный dimensionless loss, не proper scoring rule полного распределения.
Mean берётся по seeds **до** log/error. Точные ties — порядок кандидатов выше.
CRPS comparator минимизирует mean по двум validation origins от mean job CRPS
measured targets; warmup в loss/CRPS не входит. Фиксированный coarse — третий метод.

Validation JSON пишутся до selection.json; selection.json обоих источников —
до test scoring/replay. Тип выбранной модели заморожен; параметры каждого
кандидата refit на каждом test cutoff, все три показаны, нет выбора по test.
Leave-one-origin-out и leave-one-seed-out выборы — диагностика чувствительности
validation, не новая процедура выбора или CI вероятности правильного решения.

Test: same-cell ошибки mean/p99 T, W, weighted T, counts/T/p99 четырёх coarse
K-групп, CRPS, fallback levels/support sizes по этим группам. Support size —
число history samples в выбранной CDF, не p90 coverage. TV расстояние отдельных
marginal need-group/context proportions recent history (≤4000) против measured
targets без warmup — описательный drift, не joint TV или significance test.
Context: фиксированный request bucket SDSC либо exported type Kalos, None отдельная категория.
Отдельно success/timeout и успешный T в cap-сценарии с явным denominator.
В каждом origin/scenario Q test и парные seed-wise изменения log-error выбранной
модели против CRPS/coarse. Для каждого seed L=mean по 6 policies×2 T-метрикам
от |log(metric(seed))−log(observed)|. Contrast = L(queue)−L(comparator), отрицательное
значение лучше queue. t-CI по восьми таким парным разностям; mean individual
log-errors не подменяет Q(mean metrics).
95% t-CI условны на fit/выбор/arrivals/окружении, не CI по зависимым origins.
Policy regret и все reference ties сохраняются; zero regret при шести ties не
валидирует scheduler. Пустой класс даёт null, не нулевую задержку.

## Работы

- [x] Строгий API queue loss/selection, временной контракт и leakage-тесты.
- [x] Runner, фиксируемый validation-выбор, coverage/drift и тестовые контрасты.
- [x] Полные 1500 replay, побайтовый повтор, независимый аудит.
- [x] Offline аналитические/regression тесты, полный/быстрый pytest, lint.
- [x] Методика/отчёт, README/models/roadmaps, независимый reader review.

## Результаты

1500 расписаний выполнены; полный повтор на другом числе worker дал 10
побайтово одинаковых JSON. Независимый аудит: 200 service/250 workload лент,
24 scoring/coverage блока, 60 observed schedules, 1500 строк, 1980 интервалов,
360 per-policy contrasts, 60 decisions, два frozen selections и 12 test contrasts.

SDSC queue-selected request_bin улучшает точечный uncapped Q относительно
CRPS-selected request_ratio в обоих test-блоках, но не относительно coarse.
Ratio лучше воспроизводит success под cap. В .85 обе selected models ошибаются
по p99 policy (regret 47.188%); меньшая ошибка summaries не гарантирует выбор
дисциплины. Kalos type_exact/type_coarse после refit дают одинаковые test ленты;
в .70 сильная смена K/context mix и 106/300 pooled fallback, без причинного вывода.

49 новых offline-тестов, полный pytest **1681 passed**, быстрый **1672 passed**,
без фактических reruns. Pylint нового модуля/runner 10/10, библиотеки 9.98/10.
Black/isort, pre-commit, локальные ссылки и hashes проверены. Reader review
уточнил временные границы, paired loss и exact tie Kalos; итоговые таблицы
сверены с JSON, блокирующих замечаний нет.
[Методика/API](../queue_aware_selection.md),
[отчёт и ограничения](../research/queue-aware-selection-results-2026-10.md),
[артефакты](../../works/queue_aware_selection/README.md).
Следующий эпик — совместная генерация arrival/K/признаков с раздельными controls
зависимости/drift и сохранённым fixed-arrival coarse reference.
