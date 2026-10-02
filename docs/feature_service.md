# Условное обслуживание по requested time и типу работы

[EPIC-059](epics/EPIC-059-feature-conditional-service.md) проверяет, улучшает ли
условная эмпирическая CDF воспроизведение длительностей и очередей. Модель
выбирается по раннему validation, до измерения позднего test. Это генерация
S при **фиксированных** приходах, K и признаках, не совместная модель всей нагрузки.
Новые диспетчеры и изменение capacity в этот эпик не входят.

## Данные и доступность признаков

Используются неизменённые источники и адаптеры
[SDSC SP2](real_trace_lifecycle.md) и [Acme/Kalos](modern_gpu_trace.md).
Нормативные правила eligibility, exclusions, units и частичного carry-in описаны
в этих методиках. SHA-256 и лицензии закреплены также в
[manifest](../works/feature_service/manifest.json). Raw-файлы остаются в `.cache`,
не распространяются под MIT вместе с библиотекой.

Для SDSC пригодны status=1 с положительным runtime, известным wait, целым
allocated K в [1,128], requested K=allocated K, без явного predecessor.
S=runtime, end=submit+wait+runtime. Положительные status=5 добавляют только
занятое отменённое окружение. Нулевые/неизвестные интервалы не заменяются нулём.
Для Kalos пригодны завершённые интервалы с положительным целым GPU request,
согласованными timestamp и allocation; обучение только COMPLETED, S=end-start.
CPU-only, отсутствующее окончание и нулевые интервалы исключаются.

| Признак | Использование | Ограничение |
| --- | --- | --- |
| K | Сохранённый запрос ресурса, группы 1 / 2–8 / 9–32 / >=33 | Отбор согласованных request/allocation ретроспективен |
| SDSC requested time | Предоставленный пользователем лимит, секунды | По [SWF specification](https://www.cs.huji.ac.il/labs/parallel/workload/swf.html); история изменения значения не наблюдается, предполагается исходный request |
| Kalos type | Метка Eval/Pretrain/Debug/Other из выгрузки | Доступность при submit не подтверждена; сценарий с предоставленной ретроспективной меткой, не online-предиктор |
| Recorded S / completed | Обучение завершённых историй и оценка | Будущий успех неизвестен online; не учим latent success demand по неуспешным интервалам |

Адаптер SDSC превращает неположительный request в None; это не нулевой бюджет.
Библиотечный API требует None либо положительное конечное значение, ошибочные
ноль/отрицательное число отвергает. None — отдельная context-категория.
В закреплённом Kalos пустых type нет; API допускает None как отдельную категорию,
пустую строку отвергает, неизвестная непустая категория использует fallback.

## Временной протокол и выбор

Cutoff=first_raw_submit+fraction×(last_raw_submit−first_raw_submit), без округления,
в секундах исходного источника. Все равные submit сохраняют порядок файла.

| Источник | Validation cutoff | Validation jobs | Test cutoff | Warmup / measured targets | Capacity |
| --- | ---: | ---: | ---: | ---: | ---: |
| SDSC | .70 | 400 | .90 | 200 / 1000 | 128 processors |
| Kalos | .60 | 400 | .70 | 100 / 300 | 2416 requested GPUs |

Fit использует только completed с end **строго меньше** cutoff, последние 4000
в submit-порядке (всю историю, если меньше). Validation — первые 400 eligible
completed с submit>=validation cutoff. Fit заморожен на весь блок. Если хотя бы
один из этих 400 end>=test cutoff, runner останавливается, не отбирая быстрые
исходы. Недостаточный размер блока также ошибка, не автоматическое уменьшение.

Выбирается один кандидат на источник по минимуму mean validation CRPS в секундах.
При точном равенстве float используется порядок кандидатов в таблице ниже;
численный tolerance при выборе модели не вводится. `selection.json` записывается
**до** test-score и первого scheduler replay. Затем все три кандидата refit на
completed до test cutoff; уже известный validation может войти в эту историю.
`selected` остаётся ссылкой на выбранного кандидата, не четвёртой моделью.

Test — первые warmup+targets eligible completed с submit>=test cutoff.
Первые warmup работ участвуют в генерации и очереди, но не в test CRPS и target
latency. Все test-score и target метрики относятся только к следующим targets.
Выбранные поздние блоки не перекрывают прежние test-когорты EPIC-057/058.
SDSC .85 пересекал прежний .80 блок, поэтому .90 закреплён после проверки
покрытия, до scoring. Ранние validation периоды уже встречались в прежних
исследованиях: это не слепая внешняя валидация и не оценка на независимых folds.

## Три кандидата на источник

| Кандидат | Распределение и fallback |
| --- | --- |
| `coarse` (оба) | ECDF S по K-группе → pooled S |
| SDSC `request_bin` | ECDF S по K-группе × request bucket → K-группа → pooled |
| SDSC `request_ratio` | ECDF R=S/request по K-группе × request bucket → K-группа ratio → pooled ratio; S=request_target×R |
| Kalos `type_coarse` | ECDF S по K-группе × type → K-группа → pooled |
| Kalos `type_exact` | ECDF S по exact K × type → K-группа × type → K-группа → pooled |

Request buckets: (0,300], (300,1800], (1800,7200], (7200,28800],
(28800,86400], (86400,∞), отдельно None. Границы и минимум 20 для fitted cells
не подбираются по validation/test. Pooled ECDF требует лишь 2 наблюдения,
не 20. Ratio обучается только на известных положительных requests. Если таких
меньше двух или target request неизвестен, используется **абсолютный** coarse
того же окна. R>1 сохраняется, S не обрезается по request при генерации.

CRPS оценивает полное распределение, не восемь случайных предсказаний:

`CRPS(F,y) = E|X-y| − 0.5 E|X-X'|`.

Для sorted empirical x_i вторая часть равна Σ(2i−n−1)x_i/n², i=1,…,n.
В ratio-модели сначала масштабируется support, поэтому score остаётся в секундах.
[Gneiting & Raftery, 2007](https://doi.org/10.1198/016214506000001437) — источник
proper scoring rule; формула отдельно проверяется попарным расстоянием в тестах
и интегралом `(F(x)−1[x>=y])²` в независимом аудите.

## Генерация и replay

Seeds 59000–59007. NumPy default_rng получает SeedSequence из seed, source index
(SDSC=0, Kalos=1) и двух little-endian uint32 слов float64 test cutoff.
В submit-порядке выбранной когорты рисуются целые U_int∈[0,2^52), затем
U=(U_int+0.5)/2^52. Для каждого target-ID U одинаков у всех кандидатов/сценариев.
Inverse ECDF: x_[ceil(U×n)−1], без интерполяции. U независимы и равномерны;
условные CDF различаются при разных признаках, поэтому S независимы при
фиксированных признаках, но не обязаны быть одинаково распределены.
Временная зависимость рангов из EPIC-056 не восстанавливается.

Forecast — линейный выборочный p90 **всей** completed train-истории по coarse K,
fallback pooled. Он фиксируется до replay; численно одинаков для каждого K во
всех кандидатах, seeds и сценариях источника. По сгенерированным завершениям
не обновляется. EASY/Conservative получают forecast, не фактическое S.

SDSC использует `carry_cancelled` и `requested_limit`, Kalos — `carry_terminal`.
Начальная boundary — первый выбранный submit. Carry: submit<boundary<end;
start<=boundary означает running, иначе waiting. Начальные running сохраняют
исторический остаток, waiting — исходное S после нового старта. Начальные
прогнозы учитывают elapsed age; истёкшее обещание не раскрывает остаток.
Snapshot частичный, остатки/labels ретроспективны; память reservations сбрасывается.
Все пригодные неуспешные arrivals в [boundary,last_selected_submit] фиксированы
между кандидатами. Их наблюдаемая занятость не превращается в успешное S.

В `requested_limit` cap=request применяется только к **новым** arrivals, в том
числе cancelled окружению; initial running/waiting сохраняются без нового cap.
None означает отсутствие cap. Если S>cap, ресурс освобождается через cap от
симулированного старта, outcome=timed_out; равенство не timeout. Это service-clock
сценарий, не реконструкция абсолютного времени отмены пользователем.

Шесть неизменённых политик: FCFS, FirstFit, MSF, Adaptive Quickswap, EASY,
Conservative. SDSC: 2×(1 observed+3×8)×6=300; Kalos: 1×(1+3×8)×6=150.
`observed` использует записанные S при том же сценарии, не исторические W/T.

## Метрики и интерпретация

На validation/test: mean CRPS, MAE predictive mean, p90 coverage, mean predicted S,
context coverage и fallback counts. SDSC также mean P(S>request), фактическая
частота и Brier score `(p−1[S>request])²` среди targets с известным request.
Это calibration среди completed, не вероятность production failure.

Для фиксированных targets W=start−submit, terminal T=release−submit. Дренирование
полное. Weighted T=ΣK_i T_i/ΣK_i; историческое имя JSON
`node_weighted_mean_release_t` сохранено, unit явно записан отдельно.
Successful T имеет собственный изменяющийся знаменатель; при нуле успехов null.
p99 — NumPy linear quantile; итог модели — среднее восьми выборочных p99,
не quantile объединённых реализаций.

Utilization/idle-with-queue интегрируются от первого measured target arrival до
последнего, без drain. Resource ledger включает окружение, warmup и drain;
для initial running только остаток после boundary. Это зарезервированный ресурс,
не аппаратная utilization GPU. Неуспешные labels учитываются раздельно.

Относительная ошибка: mean(model metric)/observed−1; при observed=0 — null.
Парный contrast: |candidate−observed|−|coarse−observed| для того же seed,
в секундах для T и долях для success. 95% t-интервалы характеризуют только
MC-генерацию при фиксированных fit/истории/окружении, не неопределённость выбора
модели или репрезентативность источника. CRPS fit детерминирован и не имеет MC CI.

Policy выбирается по минимуму MC mean T либо MC mean p99 T в фиксированном
порядке политик; это ретроспективная replay-оценка, **не повторный выбор модели**.
Regret=observed(chosen)/min_policy observed−1. Reference ties определяются
`isclose(rtol=1e-12, atol=1e-9)`; список и число ties явно сохранены.
Нулевой regret при шести tied policies ничего не доказывает о выборе дисциплины.
Ошибки не объединяются между разными источниками/сценариями в универсальный рейтинг.

## API и воспроизведение

```python
from most_queue.random.feature_service import FeatureConditionalEmpirical

model = FeatureConditionalEmpirical.fit(
    needs=[1, 1, 2, 2], services=[2, 12, 8, 24], contexts=["a"] * 4,
    scales=[2, 4, 4, 8], minimum=2,
)
prediction = model.predict(1, "a", scale=10)
assert prediction.mean == 20
assert prediction.probability_exceeding(10) == 0.5
print(prediction.level, prediction.quantile(0.9), prediction.crps(12))
```

```bash
.venv/bin/python -m examples.feature_service_experiment --download \
  --output-dir works/feature_service
.venv/bin/python -m examples.feature_service_experiment \
  --output-dir /tmp/most-queue-feature-repeat
.venv/bin/python -m works.feature_service.audit
```

Download opt-in, hash/size проверяются до принятия cache. CLI использует только
закреплённые параметры; синтетические unit-тесты задают маленький FeatureConfig.
Manifest сохраняет версии окружения, 17 implementation hashes и hashes outputs.
Аудитор не импортирует runner, adapters или fit-классы: заново читает источники,
восстанавливает CDF, выбор, generated tapes и интервалы, пересчитывает observed
метрики из событий. Диспетчеры переиспользует, их независимой реализации не заявляет.

[Результаты](research/feature-service-results-2026-10.md) отделены от протокола.
