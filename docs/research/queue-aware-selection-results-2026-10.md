# Перенос выбора модели по ошибкам очереди на поздние периоды

Дата: 2026-10-02. [Методика](../queue_aware_selection.md),
[эпик](../epics/EPIC-061-queue-aware-model-selection.md),
[артефакты](../../works/queue_aware_selection/README.md).

Сравниваются validation-выбор по mean/p99 T, выбор по CRPS обслуживания и
фиксированный coarse. Источники и периоды уже исследовались; новый предписанный
алгоритм выбора не превращает их в слепой holdout. Capacity и policies неизменны.

**Итог:** queue-aware выбор не дал устойчивого преимущества над простым coarse.
На SDSC он лучше CRPS-selected по точечному uncapped Q в обоих test-периодах,
но хуже coarse. На Kalos queue/CRPS выбирают разные имена моделей, однако после
refit их service ленты и результаты совпадают. Успех на validation, даже
устойчивый к исключению seed, не гарантирует перенос.

## Ранний выбор

Два validation origins на источник, восемь seeds, шесть policies, 24 равновесные
origin/policy/mean-or-p99 ячейки в Q. Mean по seeds берётся до log/error.
CRPS — среднее origin-level job scores в секундах, без MC. Все значения ниже
из validation; `selection.json` записан до scoring/replay обоих test-источников.

| Source | Кандидат | Validation Q | Validation CRPS, s | Выбран по |
| --- | --- | ---: | ---: | --- |
| SDSC | coarse | 0.491343 | 7104.94 | fixed baseline |
| SDSC | request_bin | 0.084696 | 4234.42 | queue |
| SDSC | request_ratio | 0.192213 | 3514.40 | CRPS |
| Kalos | coarse | 0.727368 | 347.72 | fixed baseline |
| Kalos | type_coarse | 0.635472 | 343.47 | CRPS |
| Kalos | type_exact | 0.607546 | 350.12 | queue |

SDSC request_bin сохраняется в обоих leave-one-origin-out и всех восьми
leave-one-seed-out выборах. Kalos type_exact сохраняется при исключении любого
seed, но при исключении origin .35 оставшийся .50 даёт точный tie type_coarse
и type_exact: по заранее заданному порядку выбирается type_coarse. При исключении
.50 выбирается type_exact. Смена имени при tie не означает строгого преимущества
другой модели. Эта диагностика не оценивает вероятность переноса.

## Поздние ошибки и выбор дисциплины

Выполнено **1500 расписаний**: 600 validation, 900 test, включая отдельные SDSC
runtime-cap сценарии. Ниже uncapped test Q от MC means; меньше лучше. Выбранные
типы заморожены, все три кандидата refit на completed end<своего test cutoff.
Observed reference — same-cell replay с записанным S, не историческое T/W.

| Source / origin | Fixed coarse | Queue-selected | CRPS-selected |
| --- | ---: | ---: | ---: |
| SDSC .85 | 0.176076 | 0.744417 | 0.793454 |
| SDSC .90 | 0.475758 | 0.493615 | 0.637140 |
| Kalos .65 | 1.197262 | 1.202568 | 1.202568 |
| Kalos .70 | 1.913480 | 2.041069 | 2.041069 |

Coarse имеет наименьший точечный Q во всех четырёх строках, но **не объявляется
новым выбранным победителем после test**. У SDSC .85 парный seed-wise
queue-minus-CRPS log-error −0.05027, 95% CI [−0.11672, 0.01617]; .90 — −0.13676
[−0.21737, −0.05614]. Против coarse .85: +0.49731 [0.36091, 0.63371]; .90:
+0.00837 [−0.13140, 0.14813]. Интервалы условны на фиксированном выборе и истории;
это **средние seed-wise errors**, не разности Q от MC means из таблицы.
Нельзя делать общий тест по двум зависимым периодам или считать данные CI
неопределённостью переобучения селектора.

На Kalos type_exact и type_coarse имеют одинаковые полные service hashes во всех
восьми seeds **обоих** test origins, включая warmup. Их paired difference=0
не является доказательством эквивалентности семейств на других данных.
В .65 mean T модели с type=151.48 s против observed 622.44 s, W=0 у всех.
В .70 среднее по policies/seeds model W=14581.07 s против observed W=0;
ошибка модели не объясняется сравнением с исторической очередью.

Для SDSC без cap все три модели выбирают FirstFit по mean T на обоих test-блоках,
совпадая с observed. По p99 в .85 observed best=MSF, coarse выбирает FirstFit
(regret 0.743%), queue/CRPS выбирают EASY (regret **47.188%**). В .90 все выбирают
верный FirstFit по p99. Таким образом, lower queue-summary loss относительно
CRPS не гарантирует верный выбор хвостовой policy. На Kalos оба test-блока
имеют шесть равных observed policies по mean/p99; zero regret не валидирует выбор.

### Runtime caps и успешное обслуживание

Cap не участвовал в выборе. Доля timed_out среди фиксированных targets, %:

| SDSC origin | Observed | Coarse | Queue request_bin | CRPS request_ratio |
| --- | ---: | ---: | ---: | ---: |
| .85 | 0.0000 | 14.8000 | 7.2250 | 0.3625 |
| .90 | 0.2000 | 21.8500 | 6.5250 | 0.1125 |

Модельные значения — среднее восьми seeds; для одного seed доля не зависит от
policy при полном drain и service-clock cap. Queue-selected не оптимизировался
на timeout: ratio заметно ближе к observed success в обоих периодах.
В .85 capped Q queue=0.62830 против CRPS=0.77415, но это не устраняет разницу
по успешности. В .90 capped Q queue=0.71665 против CRPS=0.68155: преимущество
uncapped Q не переносится даже на эту соседнюю цель. Successful T с изменяющимся
denominator и числом успехов сохранён в JSON; короткий terminal T сам по себе
не трактуется как улучшение обслуживания.

## Покрытие классов и ограничения

SDSC queue-selected request_bin использует context cell для 998/1000 targets
в каждом test-блоке; только две работы переходят на coarse fallback. Counts
K-групп .85: [122,447,331,100], .90: [162,399,335,104]. Context TV recent history
против targets=0.2290/0.0925, K-group TV=0.1200/0.04675. Хорошее формальное
покрытие ячеек не гарантировало меньший Q, чем у coarse.

Kalos .65: все 300 targets — Eval с K=1, exact-context support=3911 history
samples; остальные reporting classes пустые, latency=null. В .70 counts
[107,75,31,87], хотя recent history [3933,6,9,52]. Поэтому **106/300** targets
групп 2–8/9–32 получают pooled fallback; у 87 широких jobs context support
всего 24–28 samples. K-group TV=0.62658, context TV=0.41108. Новых названий
context нет: отсутствие unseen labels не означает достаточную историю комбинаций.
Это описательная смена состава, не доказательство причины ошибок.

Условные генераторы всё ещё фиксируют arrivals/K/context и не моделируют их
совместную динамику. Наблюдаемая cancelled/failed занятость не равна скрытой
успешной потребности, carry-in частичен, Kalos type ретроспективен. Нет оценки
production quotas/placement, независимости origins, blind generalization,
online-доступности всего информационного набора или универсальной лучшей policy.

## Проверки и продолжение

Полный запуск на восьми worker и повтор на шести дали **10 побайтово одинаковых
JSON**, включая оба выбора и manifest. Проверены 19 implementation hashes и
девять output hashes. Независимый аудитор восстановил 200 service лент и 250
scenario-specific workload лент, проверил 24 scoring/coverage блока и пересчитал
все **60 observed schedules**. Проверены 1500 строк, 1980 модельных интервалов,
360 per-policy парных интервалов, 60 policy decisions, два frozen selections
и 12 test-контрастов. Диспетчер общий, а не вторая независимая реализация.

Добавлено 49 offline-тестов: аналитические свойства log loss, exact ties,
строгие shapes/finite bounds, разные queue/CRPS choices, временные отказы для
targets и окружения, отсутствие влияния поздних S на validation, сохранение
выбора до test scoring, deterministic parallel replay и regression старых
формул summaries. Полный pytest — **1681 passed**, быстрый — **1672 passed**,
оба с 16 warnings; использована CI-политика одного повтора AssertionError,
фактических повторов не потребовалось. Новые модуль и runner: pylint **10/10**,
библиотека: **9.98/10**, без новых замечаний.
Black/isort и pre-commit проходят. Независимая reader-проверка уточнила
временной контракт и paired loss до запуска, а после сверки JSON — exact tie
Kalos при исключении origin. Финальная проверка интерпретаций пройдена.

Следующий эпик по roadmap — совместная генерация приходов/K/признаков с
раздельными контролями зависимости и временного drift. Сохранять fixed-arrival
coarse baseline; новый протокол должен отделять эффект arrival mix от изменения
S и от выбора scheduler. Queue-aware выбор не заменяет baseline автоматически.
Реальные quotas/placement/variable capacity требуют другого источника данных,
а сравнение adaptive policies — различающего их reference.
