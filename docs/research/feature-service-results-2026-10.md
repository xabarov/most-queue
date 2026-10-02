# EPIC-059: признаки улучшают S, но не гарантируют верную очередь

Дата: 2026-10-02. [Протокол/API](../feature_service.md),
[эпик](../epics/EPIC-059-feature-conditional-service.md),
[артефакты](../../works/feature_service/README.md).

На позднем SDSC requested-time ratio снижает CRPS на **46.69%** и исправляет
избыточные модельные timeout. Но без enforcement средние/хвостовые задержки
становятся менее точными, а выбранная по p99 дисциплина ошибочна. В Kalos модель
с type выигрывает ранний validation, но проигрывает coarse на изменившейся
поздней нагрузке. Условное распределение полезно, однако не заменяет проверку
очереди, временного переноса и доступности признаков.

## Что было зафиксировано до результатов

Три кандидата на источник, окно 4000 completed, минимум cell=20, фиксированные
K/request группы, 400 validation jobs, один поздний test, восемь seed и шесть
неизменённых диспетчеров. Выбор по **mean validation CRPS**, не по test latency.
`selection.json` записан до первого test replay. Ранние validation периоды
ранее встречались в треке; это не слепой внешний эксперимент.

SDSC validation cutoff=44 677 443.8, test=57 280 676.6 секунды исходной шкалы.
Test-когорта submit 57 302 857–60 704 152: 200 warmup и 1000 measured targets.
Kalos validation cutoff=1 689 047 952.4, test=1 689 835 734.8 UTC seconds;
test submit 1 689 844 701–1 691 650 103: 100 warmup и 300 measured targets.
Полные когорты не пересекают прежние test-блоки EPIC-057/058.

Всего **450 расписаний**: SDSC carry_cancelled/requested_limit — 300,
Kalos carry_terminal — 150. Во всех одинаковы приходы/K, численные forecasts
и наблюдаемое окружение; генерируется только S выбранных completed, включая warmup.
Полный повтор дал побайтовое совпадение selection, обоих результатов и manifest.

## Выбор и качество распределения

CRPS в секундах, меньше — лучше. Test-score исключает warmup.
Звёздочка означает **заранее выбранную** по validation модель; по test она не заменяется.

| Источник / кандидат | Validation CRPS | Test CRPS | Test MAE mean prediction | Test p90 coverage |
| --- | ---: | ---: | ---: | ---: |
| SDSC coarse | 4271.90 | 4695.90 | 7449.19 | 87.70% |
| SDSC request_bin | 2911.41 | 2756.22 | 3852.74 | 86.60% |
| SDSC request_ratio * | 2393.03 | 2503.50 | 3597.12 | 87.80% |
| Kalos coarse | 442.35 | 2346.82 | 4682.47 | 61.67% |
| Kalos type_coarse | 420.68 | 2514.27 | 4724.72 | 58.67% |
| Kalos type_exact * | 419.17 | 2514.27 | 4724.72 | 58.67% |

SDSC ratio использует joint cell для 998/1000 targets, coarse-ratio fallback
для двух. Ни один request здесь не отсутствует; API-пропуски проверены отдельно.
Predicted mean S=5573.45 s против observed 6810.30: лучший CRPS не означает
точного среднего, хвоста или сохранения временной зависимости обслуживания.

Kalos validation содержит Eval 391/400, Pretrain 8 и Debug 1. В test targets:
Eval 170/300, Pretrain 50, Debug 52, Other 28. Это одновременно изменение типов
и K: для K=2–8 и 9–32 recent history содержит лишь 6 и 9 наблюдений.
Поэтому **106/300** целей всех моделей получают pooled CDF.
У type_exact ещё 107 exact-context и 87 coarse-context predictions.

В позднем refit type_exact и type_coarse дают одинаковые CDF на выбранных целях
и одинаковые generated tapes, несмотря на разные названия fallback. Это не два
независимых подтверждения. Их test CRPS хуже coarse на **7.14%**. Внутри Pretrain
CRPS растёт с 8190.56 до 10108.22, тогда как Debug улучшается с 1006.48 до 130.37.
Агрегат без class coverage скрыл бы эту неоднородность. Type предоставлен
ретроспективно; даже такой доступ не обеспечивает перенос качества.

## SDSC: runtime limits и completion

Все 1000 target-ID неизменны. Доли не зависят от policy: при полном drain и
service-clock cap outcome задаётся S/request, а не ожиданием.

| Вариант | Predicted P(S>request), test CDF | Brier | Replay completed под cap | Replay timeout |
| --- | ---: | ---: | ---: | ---: |
| Observed | фактически 0.200% | — | 99.8000% | 0.2000% |
| Coarse | 21.509% | 0.07393 | 78.0750% | 21.9250% |
| Request bin | 6.557% | 0.01378 | 93.3375% | 6.6625% |
| Request ratio * | 0.166% | 0.00199 | 99.8625% | 0.1375% |

CDF exceedance вычислена точно, replay — среднее восьми генераций, поэтому они
не обязаны совпадать. Для ratio 95% conditional MC CI completed:
**[99.754%, 99.971%]**. Для coarse: [77.121%, 79.029%]. Никакого clipping R≤1
нет: редкие R>1 остаются в fit. Model S не нормируется по test observations.

Это исправляет конкретное несоответствие S/request среди **исторически completed**.
Не восстановлены latent S, вероятность отмены, retries и неизвестный ресурсный
след неуспешных jobs. Совпадение completion нельзя переносить на production failures.

## Очередь: выигрыш зависит от сценария

MAPE ниже — среднее абсолютных относительных ошибок MC-средних по шести policies
**одного** source/scenario. Это шесть зависимых ячеек, не CI и не независимые folds.

| Источник / сценарий / модель | MAPE mean terminal T | MAPE p99 terminal T |
| --- | ---: | ---: |
| SDSC carry_cancelled / coarse | 36.37% | 29.17% |
| SDSC carry_cancelled / request_bin | 32.73% | 30.31% |
| SDSC carry_cancelled / request_ratio * | 45.08% | 40.72% |
| SDSC requested_limit / coarse | 87.42% | 80.87% |
| SDSC requested_limit / request_bin | 49.62% | 43.97% |
| SDSC requested_limit / request_ratio * | 45.12% | 41.09% |
| Kalos carry_terminal / coarse | 303.58% | 413.43% |
| Kalos carry_terminal / type_coarse | 531.21% | 641.36% |
| Kalos carry_terminal / type_exact * | 531.21% | 641.36% |

На SDSC без cap выбранный ratio ухудшает оба агрегата. Например, paired change
абсолютной ошибки mean T для FirstFit: **+4998.65 s**, conditional 95% CI
[+1396.18,+8601.12]. Под cap тот же contrast **−13760.27 s**,
CI [−16685.40,−10835.14]. Точнее воспроизведённая доля успеха не делает
оставшиеся ошибки задержки малыми. Successful T учитывается отдельно со своим
знаменателем; его нельзя подменять terminal T.

Во всех SDSC вариантах mean-T choice сохраняется: FirstFit. По p99 coarse и
request_bin также выбирают FirstFit, но **selected ratio выбирает MSF**:

| Сценарий | Observed p99 FirstFit | Observed p99 MSF | Regret выбранного ratio |
| --- | ---: | ---: | ---: |
| carry_cancelled | 313917.18 s | 510934.13 s | 62.76% |
| requested_limit | 397223.58 s | 510704.10 s | 28.57% |

Более качественный CRPS не гарантирует нужного ранжирования tail latency.
Мы не заменяем победителя на request_bin после просмотра этой таблицы.
Следующая проверка queue-aware selection потребовала бы отдельного протокола
с собственной ранней очередью и поздним holdout.

Kalos при номинальных 2416 GPU вновь имеет **нулевое observed target waiting**
для всех шести policies, mean T=2342.70 s, p99=21016.21 s. Поэтому reference
tie count=6 и нулевой regret не валидирует policy choice. Модели, напротив,
создают очереди: среднее модельное W по policies/seeds равно 6649.61 s у coarse
и 11670.48 s у type. Это показывает чувствительность к сгенерированным ресурсным
интервалам; не доказывает правильность производственной модели или capacity.

## Проверки и ограничения

Новый API и runner имеют 59 offline unit-тестов: CRPS против независимой попарной
формулы, scale identity, inverse-CDF ties, fallback и missing values, сохранение
R>1, ошибочные входы, строгие временные split, запрет доступа selection к test,
воспроизводимость и независимый пересчёт итогов. Старые адаптеры/диспетчеры не менялись.

Независимый аудитор восстанавливает CDF/генерацию из raw-файлов без импортов новых
fit/runner, проверяет hashes и метрики; диспетчер переиспользован, независимость
его алгоритма не заявляется. Проверены 12 score-блоков, 50 service tapes,
75 workload tapes, 450 строк, 594 MC-интервала, 108 парных интервалов,
18 policy decisions и все 18 observed schedules с интегрированием ресурса
по событиям. Полный pytest: 1577 passed; быстрый: 1568 passed.
Первый быстрый прогон имел один сбой старого stochastic M/G/1 warmup-теста;
изолированный и полный повтор быстрого набора прошли без изменений теста.
Глобальный `np.random.seed` в нём не фиксирует generator симулятора, что оставлено
отдельным test-infrastructure долгом, а не скрыто изменением допуска.
Читательская проверка привела к явным определениям eligibility, прогрева,
пропусков, pooled minimum и неизменных численных forecasts.

Один validation/test переход на источник, selected-completed population,
ретроспективный carry-in/type и общий номинальный GPU-пул ограничивают вывод.
MC-интервалы не включают неопределённость истории, выбора модели и неизвестных
ограничений ресурса. Прогнозы резервирования остались прежними: возможны
нарушения обещаний EASY/Conservative из-за неточного p90; это не переполнение
capacity. Проверка бюджета/ресурсного ledger отделена от обещаний диспетчера.

## Решение и следующий шаг

1. Сохранить coarse как обязательный baseline. SDSC ratio — полезный кандидат
   для согласования S/request, не универсальная замена модели очереди.
2. Не объявлять Kalos type лучшей моделью поздней нагрузки: выбранный по validation кандидат хуже
   на позднем блоке; нужны достаточная class coverage и проверяемая доступность
   признаков. Совместная генерация приходов/K/type и queue-aware selection
   остаются отдельными, ещё не выполненными направлениями.
3. Следующий эпик трека — эффективная capacity, квоты и ограничения GPU-нагрузки
   либо явно модельная sensitivity-сетка. Не подбирать capacity по этому test W
   и не внедрять новую адаптивную дисциплину до различающего политики контроля.
