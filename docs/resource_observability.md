# Аудит ресурсных ограничений в реальных трассах

[EPIC-063](epics/EPIC-063-resource-observability-audit.md) проверяет, какие
наблюдения позволяют перейти от общего GPU-пула к виртуальным кластерам и
меняющейся ёмкости. Это аудит источников и данных, **не новая модель очереди**.
Сравнены шесть кандидатов; полный количественный аудит выполнен только для
Helios. Для остальных проверены опубликованные схемы и условия, а не сырые строки.

Решение: Helios подходит для следующего **явно модельного** сценария с дневной
ёмкостью и VC; он не разрешает заявлять восстановление точных production quotas,
размещения или rankings. Матрица источников, количественные результаты и
go/no-go находятся в [отчёте](research/resource-observability-results-2026-10.md).

## Что требуется наблюдать

Для job replay нужны arrival, доступные при submission требования и ограничения,
жизненный цикл попыток, фактические интервалы занятости и условия успешного
завершения. Для ресурса — ёмкость с интервалами действия, membership VC/tenant,
квоты/borrowing, topology и правила уменьшения ёмкости. Отдельно нужен provenance:
когда каждое значение стало доступно планировщику, а не только когда его экспортировали.

Число узлов `node_num` не даёт node IDs. GPU utilization не является доступной
capacity. VC label не является численной квотой. Дневное число GPU не определяет
момент переключения внутри дня, hard enforcement или передачу ресурса между VC.
Отсутствие строки не означает нулевой ресурс. Failed/cancelled execution не
раскрывает latent successful S; scheduled-to-deletion не обязательно service.
Эти различия — критерии отбора, а не поля, которые можно заполнить подгонкой к W.

## Закреплённый Helios

Источник: SenseTime / S-Lab-System-Group, Hu et al., SC 2021,
[Characterization and Prediction of Deep Learning Workloads in Large-Scale GPU Datacenters](https://doi.org/10.1145/3458817.3476223).
[Описание](https://github.com/S-Lab-System-Group/HeliosData/blob/159f0caeec16600b9b6017862952a36aae01c43f/README.md),
[CC-BY-4.0](https://github.com/S-Lab-System-Group/HeliosData/blob/159f0caeec16600b9b6017862952a36aae01c43f/LICENSE.txt).
Код Most-Queue MIT не меняет условия исходных данных. Источник предоставлен as-is,
без endorsement. Здесь данные преобразованы в агрегаты, hashes и проверки;
исходные пользовательские/job IDs и raw строки не публикуются.

Revision `159f0caeec16600b9b6017862952a36aae01c43f`, archive SHA-256
`3d22a5f6c0ae669e2fcbfe4200fa9c48664507bc397c677bad8f085222c032ac`.
Архив 36 437 672 bytes содержит четыре `cluster_log.csv` и четыре
`cluster_gpu_number.csv`: Earth, Saturn, Uranus, Venus. Читаются только восемь
allowlisted payloads, без extractall или исполнения чужого кода. Каждый payload
имеет отдельные bytes/SHA-256 в `helios-audit.json`. Directory entries не считаются
дополнительными таблицами; неизвестные/повторные file entries отклоняются.

## Алгоритм полного аудита

Сначала проверяются точные заголовки и дневные конфигурации. Даты и VC-columns
уникальны, resource counts — неотрицательные целые. Дубликат даты, неправильная
ширина строки или malformed count конфигурации отклоняют аудит: lookup иначе
неоднозначен. Пропущенные дни, изменения counts и несовпадение суммы VC с total
считаются, но не исправляются. История не заполняется назад или вперёд.

Далее читаются **все job rows**, без completed-only отбора для общей статистики.
Подсчитываются исходные states, GPU>0 и CPU-only, duplicate job IDs внутри
cluster, неизвестные VC, malformed widths, resources и timestamps. Отсутствующий
Counter key означает ноль наблюдений этой категории. Дубликаты не удаляются:
аудит должен показать raw population до выбора deduplication policy.

Clock — naive source calendar с точностью секунды; timezone не сообщена в
прочитанной схеме, поэтому UTC не назначается. Проверяются start>=submit,
end>start, duration=end−start и queue=start−submit. Нулевые/отрицательные/неизвестные
интервалы учитываются отдельно, не превращаются в S=0. `duration_equals_sojourn`
считает совпадения end−submit; при W=0 это одновременно execution и sojourn,
поэтому этот счётчик сам по себе не означает ошибку поля.

Для каждого GPU job, независимо от terminal state, отдельно сопоставляются
submit и start с VC-count на **ту же дату**. Счётчики: invalid time, missing date,
unknown VC, compared, zero count, request выше daily count. Это проверка
совместимости с гипотезой, не реальная проверка admission или online features.

Для occupancy берутся GPU>0 и валидные submit<=start<end со статусом
COMPLETED/CANCELLED/FAILED/TIMEOUT/NODE_FAIL. RUNNING/SUSPENDED и другие статусы
не становятся завершениями даже при заполненном end. Интервал `[start,end)`
полуоткрытый; одновременные события складываются до учёта следующего интервала.
Фактически вычисляется **requested occupancy**, не GPU SM utilization:
`O(t)=Σ gpu_num_i × 1(start_i<=t<end_i)`.

Full work=Σ gpu_num×(end−start) по всей выбранной занятости, без clipping.
Covered work — её интеграл только по опубликованным датам конфигурации.
Peak считается по всему execution span. Для диагностической гипотезы
`C(t)=daily_count(date(t))` вычисляются ∫max(O−C,0)dt и число секунд O>C:
отдельно для общего пула и каждого VC. Сумма VC-excess seconds — **VC-секунды**,
не длительность объединения событий во времени. Нулевые counts остаются нулевыми;
неизвестные даты/VC исключаются из сравнения, но не из full work.
`compared_seconds` — число опубликованных дней×86400 общего пула.

Эта гипотеза постоянного значения с начала календарного дня не утверждает ни
фактического момента смены конфигурации, ни online-доступности daily row.
Превышение может отражать различную семантику VC, borrowing, внутридневную смену
или данные; причины по текущей трассе не идентифицированы. Отсутствие общего
превышения также не доказывает полноту журнала, физическую capacity или квоты.

Mean/p99 S и записанного W считаются отдельно только для валидных completed GPU
rows; p99 — linear sample quantile. Это описательные свойства отобранной
популяции, не iid закон S, fitted service generator, MC или confidence interval.
В этом эпике **ноль scheduler runs**, нет подгонки ёмкости и выбора дисциплины.

## Повторение и независимая сверка

```bash
.venv/bin/python -m examples.resource_observability_audit --download --output-dir works/resource_observability
.venv/bin/python -m examples.resource_observability_audit --output-dir /tmp/most-queue-resource-observability-repeat
.venv/bin/python -m works.resource_observability.verify
```

Download opt-in, pinned hash обязателен до сохранения; существующий cache никогда
не перезаписывается. Raw остаётся в ignored `.cache/resource_observability/`.
Основной runner создаёт `helios-audit.json` и `manifest.json`, без времени запуска,
чтобы повтор был побайтовым. Manifest фиксирует source/license, hash реализации,
версию NumPy, output hash и scheduler_runs=0.

Независимая сверка не импортирует primary runner: pandas parsing, timestamp
differences, join по date/VC, vectorized unique/cumsum event integration.
Она проверяет raw counts/states/IDs, конфигурации, service/wait, full/covered work,
peak и оба excess integrals; отдельно классифицирует duplicate IDs/unknown VC.
`verification.json` содержит hash проверенного результата, verifier и pandas.
Это независимый пересчёт данных, а не независимая валидация истинности источника.

[Артефакты](../works/resource_observability/README.md),
[результаты и ограничения](research/resource-observability-results-2026-10.md).
