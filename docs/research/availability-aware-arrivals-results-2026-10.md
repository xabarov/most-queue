# Availability-aware arrival history: результаты

Дата: 2026-10-05. [Эпик](../epics/EPIC-065-availability-aware-arrival-history.md),
[методика](../availability_aware_arrivals.md),
[артефакты](../../works/availability_aware_arrivals/README.md).

**Итог:** устранение completed-prefix staleness (EPIC-062) снижает prefix_lag
на порядки (с дней до минут/часов) и примерно удваивает донор-пул на всех
четырёх origin, но **не даёт равномерного улучшения Q**: на SDSC оно заметно
ухудшает ошибку очереди, на Kalos .70 — заметно улучшает, на Kalos .65 —
меняется в пределах шума. Выбор дисциплины по mean T не меняется ни на одном
origin ни для одного варианта. Это diagnostic/generative исправление
известного артефакта эксперимента, не универсальное улучшение точности или
причинное объяснение ошибок очереди.

## Prefix lag: до и после

Те же четыре origin, split и held-out cohort, что в EPIC-062; сравниваются
только donor-пулы для arrival-fit.

| Origin | Stale lag (дни) | Availability-aware lag (дни) | Stale donors | Availability donors |
| --- | ---: | ---: | ---: | ---: |
| SDSC .85 | 3.030 | 0.0524 | 39629 | 49610 |
| SDSC .90 | 8.204 | 0.0088 | 40902 | 51346 |
| Kalos .65 | 0.070 | 0.0699 | 9667 | 17694 |
| Kalos .70 | 4.629 | 0.167 | 9667 | 18717 |

На Kalos .65 lag не меняется (там и не было стоящего job на cutoff — EPIC-062
уже отметил это как origin без staleness), но донор-пул всё равно почти
вдвое больше: availability-aware донор включает cancelled-записи целиком,
а не только completed, и не требует отдельного "resolved" условия вообще.
На SDSC .90 задержка падает с 8.2 дней до 12.7 минут — именно тот случай,
который EPIC-062 выделил как самый дорогой.

## Эффект на Q (ошибку очереди)

Q — среднее абсолютных log-errors MC means по шести policies и mean/p99 T
(`loss_of_mc_means`, меньше лучше), как в EPIC-062.

| Origin | fixed_coarse | recent_joint_iid (stale) | recent_availability_iid | expanding_availability_iid |
| --- | ---: | ---: | ---: | ---: |
| SDSC .85 | 0.2300 | 0.3960 | **0.9686** | 0.9564 |
| SDSC .90 | 0.0928 | 0.7657 | **1.3036** | 1.5912 |
| Kalos .65 | 1.2318 | 0.5262 | **1.2089** | 0.8874 |
| Kalos .70 | 1.6151 | 1.7286 | **0.4272** | 1.1328 |

Парный контраст (recent_availability_iid − recent_joint_iid_stale, по тем же
8 seeds) положителен и значим на SDSC (0.475 и 0.522, CI не пересекает 0),
отрицателен и значим на Kalos .70 (−1.130, CI [−1.598, −0.662]), и не значим
на Kalos .65 (0.204, CI [−0.323, 0.731]). Направление эффекта **меняется по
origin** — устранение staleness не является безусловным улучшением.

## Вероятный механизм (наблюдение, не доказанная причина)

Mean gap донор-пула смещается в РАЗНЫХ направлениях относительно observed
на разных источниках, и это совпадает с направлением эффекта на Q:

| Origin | observed mean gap, s | stale mean gap, s | availability mean gap, s |
| --- | ---: | ---: | ---: |
| SDSC .90 | 2881.15 | 2115.43 (ближе к observed) | 1932.79 (дальше от observed) |
| Kalos .70 | 5558.69 | 90.92 (сильно сжат) | 322.01 (менее сжат, ближе к observed) |

На SDSC availability-aware пул включает cancelled-заявки как дополнительные
арривалы, сжимая средний gap ещё сильнее относительно observed (хуже). На
Kalos .70 то же самое включение, наоборот, **раздвигает** сильно сжатый stale
gap в сторону observed (лучше), потому что там stale-префикс обрывался всего
через 0.07 дня после начала cutoff-периода (см. "Kalos .65" case — тот же
donor-pool до 624 уже известных исходов, упомянутый в отчёте EPIC-062) и
содержал непропорционально частые арривалы. need_group_tv при этом ВСЕГДА
ближе к observed у availability-aware, чем у stale, на обоих origin — то
есть по составу K донор-пул улучшился в обоих случаях; расхождение в Q
объясняется именно темпом арривалов (gap), не миксом K. Это согласуется с
наблюдением, не доказывает единственный causal механизм.

## Выбор дисциплины

Регret по mean T для выбора дисциплины **не меняется** ни на одном из четырёх
origin между stale и availability-aware вариантами (SDSC .85: 0%, SDSC .90:
0.623%, Kalos .65/.70: 0%, все ties с полным списком дисциплин на Kalos,
т.к. observed W=0 там). Исправление prefix staleness меняет Q, но не меняет
решение о дисциплине в этих четырёх origin.

## Ограничения

Failed-статус (SWF status=0) и genuinely unresolved-at-export Acme-записи
(`state` RUNNING/PENDING без start/end) остаются вне availability-aware
donor-пула — уже отфильтрованы существующими парсерами (резерв, не решено
здесь). Сравнение ограничено iid-сэмплингом (без block/shuffle осей EPIC-062,
по изолирующему скоуп этого эпика). Четыре origin — те же, что в EPIC-062, не
новый blind holdout; вывод не переносится автоматически на другие трассы,
сплиты или cutoff-фракции. Service-fit остаётся completed-only и неизменным;
это не прогноз latent successful S для отменённых/неуспешных заявок.

## Проверки

16 новых юнит-тестов (`tests/units/test_workload_trace.py`,
`tests/units/test_availability_aware_arrivals_experiment.py`). Полный
`pytest tests/ -m "not slow" -n auto` — **1804 passed**. pylint `most_queue`
без новых замечаний (9.98/10, без изменений), pylint нового кода — 10/10,
black/isort чисты. Побайтовый повтор (`--workers 8` против `--workers 6`,
другой `--output-dir`) подтверждён для всех шести выходных JSON.
