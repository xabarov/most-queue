# Реальные трассы и калибровка обслуживания

Первый эпик: [EPIC-055](../epics/EPIC-055-real-trace-calibration.md).
Цель трека — проверять модели по ошибке задержек и выбора дисциплины, а не только
по близости нескольких моментов распределения обслуживания.

## Первый этап

EPIC-055 выполнен 2026-10-02. [Отчёт](../research/real-trace-calibration-results-2026-10.md):
верный выбор по mean T не означал точных задержек; p99-решения не сохранились.

Один исторический источник SDSC SP2, проверенный формат SWF, фиксированная копия
с SHA-256, аудит исключений. Реализация — независимый адаптер входа и эксперимент
поверх `MsjGeneralSim`; существующий планировщик не меняется.

До измерений зафиксированы split, группы K, четыре семейства, seed, шесть
дисциплин и основная метрика. [Контракт и команды](../real_trace_calibration.md).
Эмпирический iid-resampling истории — отдельная модель, не исходная трасса.
Контроль сохраняет реальные длительности и их порядок в выбранной подвыборке.

Сравнение состоит из 18 контрольных replay и 576 модельных replay. Оценки
генерации условны на одной обучающей истории и трёх фиксированных блоках.
Исследование не воспроизводит отмены, исходный backlog, приоритеты очередей,
топологию или поведение пользователей исторического кластера.

## Второй этап

[EPIC-056](../epics/EPIC-056-real-trace-temporal-dependence.md) выполнен 2026-10-02.
Четыре заданных заранее origin, семь вариантов и 1368 расписаний; отдельно
обновление среднего/формы, точный K и блоки рангов с согласованным iid-контролем.
[Методика](../real_trace_temporal.md),
[отчёт](../research/real-trace-temporal-results-2026-10.md).

Recent-coarse снижает агрегированную ошибку mean T с 94.57% до 44.12%, но хуже
expanding в двух периодах. Точный K и блочная зависимость не дают устойчивого
выигрыша. Это контролируемые изменения генератора, не причинное разложение
производственных эффектов. Все семь моделей сохраняют выбор FirstFit по mean T;
выбор победителя по p99 сохраняется не во всех периодах, и regret зависит
от периода/модели.

Сохранить recent-coarse как простой baseline. Не подбирать окно, K-группы
или длину блока по этим же тестам без отдельной обучающей/проверочной схемы.

## Третий этап

[EPIC-057](../epics/EPIC-057-real-trace-initial-state-cancellation.md) выполнен
2026-10-02. Отдельный `MsjLifecycleSim` с начальными running/waiting,
отменённым занятым ресурсом и runtime limits; completed/cancelled/timeout
учитываются раздельно. Четыре сценария, 1632 расписания и полный побайтовый повтор.
[Методика](../real_trace_lifecycle.md),
[отчёт](../research/real-trace-lifecycle-results-2026-10.md).

200 работ прогрева не устранили эффект начальной очереди. Добавление наблюдаемой
отменённой нагрузки сменило mean-T победителя на одном периоде; обе модели
ошиблись по p99 во всех четырёх, со средним regret 60.51%. Под жёстким лимитом
completed составил 99.75% у observed и 75.34%/80.36% у expanding/recent.
Короткий terminal T нельзя считать улучшением успешного обслуживания.

Snapshot частичный и ретроспективный. Из 16 598 status=5 включены 10 917 с
положительным наблюдаемым занятием; 5681 без такого runtime не превращены
в нулевую нагрузку. Полный latent S, неизвестные моменты отмены в очереди,
retries, приоритеты и память исходного планировщика остаются ненаблюдаемыми.

## Четвёртый этап

[EPIC-058](../epics/EPIC-058-modern-gpu-trace-validation.md) выполнен 2026-10-02: независимый источник
Acme/Kalos 2023 после аудита лицензии, timestamp и terminal labels. 1224 расписания
и полный побайтовый повтор. Выявлено, что duration включает ожидание; используется
end-start. Отказы учитываются отдельно. Requested time отсутствует, поэтому
enforcement исключён. [Методика](../modern_gpu_trace.md),
[отчёт](../research/modern-gpu-trace-results-2026-10.md).

При номинальном общем пуле 2416 GPU контрольное ожидание всех целей нулевое,
дисциплины равны; нулевой regret не валидирует их выбор. Recent слегка снижает
агрегированную ошибку mean T, но ухудшает p99. Реальная эффективность планировщика
требует наблюдений о квотах, эффективной capacity и ресурсных ограничениях.
Отдельный будущий sensitivity-протокол допустим, но не подгонка capacity по test W.

## Пятый этап

[EPIC-059](../epics/EPIC-059-feature-conditional-service.md) выполнен 2026-10-02: условная генерация
S при фиксированных K/request/type, ранний validation-выбор по CRPS и поздние
test-блоки, 450 расписаний с побайтовым повтором. [Методика](../feature_service.md),
[отчёт](../research/feature-service-results-2026-10.md).

SDSC request_ratio снижает test CRPS на 46.69% и избыток timeout с 21.925% до
0.1375% при observed 0.2%, но без cap ухудшает mean/p99 ошибки очереди и
выбирает неверную p99-policy. У Kalos validation-selected type_exact хуже coarse
на поздней изменившейся смеси; 106/300 targets используют pooled fallback.
Это условная, **не полная совместная** модель нагрузки, type ретроспективен.

## Шестой этап

[EPIC-060](../epics/EPIC-060-gpu-resource-envelope.md) выполнен 2026-10-02:
заранее заданная чувствительность к GPU capacity и эксклюзивным узлам, не
восстановление квот. 24 ячейки, четыре явно infeasible, **1080 расписаний**,
побайтовый повтор и независимый аудит. [Методика](../gpu_resource_envelope.md),
[отчёт](../research/gpu-resource-envelope-results-2026-10.md).

Меньший пул создаёт observed ожидание, но mean T разделяет политики только
в двух из 20 feasible cells; p99 T не разделяет их нигде. Нулевой regret
не подтверждает выбор. Ошибки recent-coarse могут резко усиливаться в меньшем
пуле. Capacity не подгонялась к историческому W; node_num не доказывает
эксклюзивность или размещение. Для реальных quotas/placement/variable capacity
нужны новые наблюдения, а не оптимизация на этой трассе.

## Седьмой этап

[EPIC-061](../epics/EPIC-061-queue-aware-model-selection.md) выполнен 2026-10-02:
выбор по ошибкам mean/p99 T на двух ранних replay-блоках против CRPS и fixed coarse,
два поздних блока на источник. 1500 расписаний, полный побайтовый повтор,
независимый аудит. [Методика](../queue_aware_selection.md),
[отчёт](../research/queue-aware-selection-results-2026-10.md).

SDSC queue-selected request_bin лучше CRPS-selected request_ratio по точечному
uncapped test Q в обоих периодах, но хуже coarse; ранняя устойчивость выбора не
гарантирует перенос. По timeout ratio ближе к observed. На Kalos после refit
type_exact/type_coarse дают одинаковые поздние ленты; .70 имеет 106/300 pooled
fallback targets и сильную смену K/context mix. Это не причинное объяснение
ошибок, не online-валидация и не новый blind holdout. Capacity не подбиралась.

## Восьмой этап

[EPIC-062](../epics/EPIC-062-joint-marked-arrivals.md) выполнен 2026-10-02: совместная генерация
gap/K/context/request, общий coarse S|K и fixed-arrival baseline. 1176 empty-start
расписаний, побайтовый повтор и независимый аудит.
[Методика](../joint_marked_arrivals.md),
[отчёт](../research/joint-marked-arrivals-results-2026-10.md).

Устойчивого выигрыша joint/block нет. На SDSC .90 все новые generators хуже
fixed coarse по seed-level loss; на Kalos .65 некоторые лучше, но меняют
ресурсный состав и work. На Kalos .70 recent joint horizon 0.70 дня против
observed 19.24, K-TV=0.629; полностью разрешённый arrival prefix отстаёт от
cutoff на 4.63 дня. Контракт ретроспективен, лаг зафиксирован до replay, не
исправлялся после test. Смена loss от MC means на mean seed-loss может менять
вывод; обе оценки сохранены. К lifecycle EPIC-061 эти empty-start числа не
приравниваются. Новый победитель после просмотра test не выбирается.

## Девятый этап

[EPIC-063](../epics/EPIC-063-resource-observability-audit.md): сопоставлены шесть
источников, полностью проверены 3 362 981 записи Helios и 724 daily configuration
rows, закреплены source/license/hashes. [Методика](../resource_observability.md),
[отчёт](../research/resource-observability-results-2026-10.md).

Helios выбран для bounded daily-capacity/VC сценариев, не production reconstruction.
17 459 GPU-записей стартуют при нулевом same-day VC-count. Общий пул не превышен
в пределах известных дат, но отдельные VC превышены. Реальные hard quotas,
intraday changes и borrowing не определены; данные не подгонялись к W.
Philly/PAI/Alibaba/Google остаются отдельными ветками для attempts, placement,
sharing и resource events, а не взаимозаменяемыми scalar-S трассами.

## Десятый этап

[EPIC-064](../epics/EPIC-064-msj-capacity-calendar.md) выполнен 2026-10-05:
opt-in `CapacityCalendar`/`MsjLifecycleSim.run_capacity_calendar`, shrink/growth
без обязательного arrival, автоматический grandfathering, явные
infeasible/unresolved результаты на границе домена календаря и обнаружение
(не молчаливое поглощение) Conservative reservation/calendar mismatch. 23
юнит-теста фиксируют механику на синтетике; три предписанных сценария
(fixed/daily/isolated-VC pool) на одном предписанном 7-дневном окне Helios
Venus — 18 расписаний, побайтовый повтор и независимая pandas-сверка.
[Методика](../msj_capacity_calendar.md),
[отчёт](../research/msj-capacity-calendar-results-2026-10.md).

В выбранном окне общий пул Venus не менялся ни разу (первое изменение total —
только на 98-й день всей 181-дневной трассы), поэтому fixed/daily-сценарии
совпали; shrink/growth-механику на реальных данных это окно не демонстрирует
намеренно — окно выбрано предписанным правилом («первые семь дней»), а не по
результату, чтобы не настраивать его под желаемую находку. Ни одного infeasible
job и ни одного reservation mismatch не возникло ни у одной из шести дисциплин;
EASY/Conservative реально backfill'ят (до 155/990 заданий у isolated VC), не
меняя итоговые completed/running counts относительно FCFS/First-Fit/MSF/Adaptive
QuickSwap в этом окне. Это проверка корректности и безопасности календаря на
реальных данных, не ranking дисциплин и не подтверждение/опровержение EPIC-063
daily-VC-гипотезы.

## Одиннадцатый этап

[EPIC-065](../epics/EPIC-065-availability-aware-arrival-history.md) выполнен
2026-10-05: `availability_prefix` строит arrival-mark donor-пул из
submit<cutoff поверх уже распарсенной популяции completed+cancelled (SDSC) /
completed+cancelled+failed+timeout+node_failed (Kalos), не останавливаясь на
первом незавершённом к cutoff задании. Те же четыре origin, split, cohort,
policies и replications, что в EPIC-062. 792 расписания, побайтовый повтор,
независимый аудит prefix_lag/donor-pool-size из сырых источников.
[Методика](../availability_aware_arrivals.md),
[отчёт](../research/availability-aware-arrivals-results-2026-10.md).

Prefix lag падает на порядки (SDSC .90: 8.2 дня → 12.7 минут; Kalos .70: 4.6
дня → 4 часа), donor-пул примерно удваивается на каждом origin. Это **не**
равномерное улучшение: Q заметно хуже на SDSC, заметно лучше на Kalos .70, в
пределах шума на Kalos .65. Выбор дисциплины по mean T не меняется ни на одном
origin. Правдоподобный (не доказанный) механизм — сдвиг среднего gap донор-пула
относительно observed в разные стороны на разных источниках. Failed-статус SWF
и genuinely unresolved-at-export Acme-записи остаются вне scope (резерв).

## Следующие этапы

1. Проверять новые адаптивные дисциплины на различающем reference. Прерывания
   потребуют измеренных либо отдельно обозначенных модельных расходов
   checkpoint/resume.
2. Отдельный будущий прогон EPIC-064 на окне с реальным intra-window изменением
   total/VC (например, вокруг 98-го дня трассы Venus) — предписанным заранее,
   не выбранным по итогу первого прогона.
3. Расширение availability-aware ingestion на failed-статус SWF и genuinely
   unresolved-at-export Acme-записи (EPIC-065, резерв) — отдельный эпик, не
   попытка задним числом улучшить уже полученный результат.

Каждый пункт — отдельный будущий эпик. Выполненные этапы не доказывают универсальное
преимущество PH, lognormal, recent-history, availability-aware истории или
какого-либо планировщика.
