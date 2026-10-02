# EPIC-059: Условное обслуживание по запросу времени и типу работы

- **Статус:** done
- **Создан:** 2026-10-02
- **Roadmap:** [Реальные трассы](../roadmaps/real-trace-calibration.md)

## Цель

Проверить, помогает ли распределение S при фиксированных K и доступных признаках
воспроизводить сервис, задержки и completion под runtime limit. Выбор модели
проводится на отдельном раннем validation, не по последующим test-очередям.
Это условная генерация S, не полная совместная генерация приходов/K/request/type.

## Предписанный протокол

Используются прежние hash-pinned SDSC SP2 и Acme/Kalos без изменения адаптеров,
capacity, времени приходов и семантики диспетчеров. Только completed служат
источником обучения S. K-группы 1 / 2–8 / 9–32 / >=33, minimum=20, history=4000
последних завершённых работ в submit-порядке. Неуспешное окружение остаётся
наблюдаемым; latent success demand не выводится из времени до отмены/отказа.

| Источник | Validation cutoff, доля raw span | Validation | Test cutoff | Прогрев + targets | Сценарии |
| --- | ---: | ---: | ---: | ---: | --- |
| SDSC | .70 | 400 completed arrivals | .90 | 200 + 1000 | carry_cancelled, requested_limit |
| Kalos | .60 | 400 completed arrivals | .70 | 100 + 300 | carry_terminal |

История fit строго end < соответствующий cutoff. Validation — первые 400
eligible completed с submit >= validation cutoff; **все** их end должны быть
строго до test cutoff, иначе протокол останавливается. Test — первые заданные
completed с submit >= test cutoff. Ties сохраняются из файла. Выбор ID и labels
ретроспективен; поздние test-блоки не служат для выбора кандидата. Ранние validation
периоды уже затрагивались прежними исследованиями; этот протокол не называется
слепой внешней валидацией. SDSC .90 выбран после проверки перекрытия: .85 ещё
пересекает прежний блок EPIC-057 .80. Августовский хвост Kalos имеет другую смесь типов.

На каждом источнике заранее заданы три кандидата:

- `coarse`: прежняя empirical CDF по K-группе, иначе pooled.
- SDSC `request_bin`: CDF S по K-группе × bucket(request), fallback K-группа →
  pooled. Границы в секундах [300, 1800, 7200, 28800, 86400], верхняя включается;
  неизвестный request получает отдельный None, а не ноль.
- SDSC `request_ratio`: CDF R=S/request по тому же joint bucket, fallback
  K-группа ratio → pooled ratio, обучение ratio только на положительных requests.
  Генерация S=request_target × R; неизвестный request/отсутствие ratio-истории
  возвращает абсолютный coarse baseline. R>1 сохраняется; clipping/capping нет.
- Kalos `type_coarse`: CDF S по K-группе × type → K-группа → pooled.
- Kalos `type_exact`: CDF S по exact K × type → K-группа × type → K-группа → pooled.

Fallback зависит только от train-count. Тип Kalos — метка из выгрузки; доступность
в момент submit не подтверждена. Поэтому type-варианты — ретроспективный сценарий
с предоставленным признаком, не заявление готовности online-предиктора.

Выбор один на источник: минимум среднего CRPS на validation в **секундах**,
ties в порядке coarse, второй, третий кандидат. CRPS(F,y) = E|X-y| − .5 E|X-X'|
для полного empirical распределения, не по восьми сгенерированным длительностям.
Источник правила: [Gneiting & Raftery, 2007](https://doi.org/10.1198/016214506000001437).
Никакого переобучения по test-критерию: после выбора все три модели refit на
completed history до test cutoff, чтобы показать также заранее заданные ablations.
Результат `selected` — ссылка на уже выбранного кандидата, не четвёртый fit.

450 расписаний: SDSC 2 × (1 observed + 3 × 8 seeds) × 6 policies = 300;
Kalos 1 × (1 + 3 × 8) × 6 = 150. Seeds 59000–59007; U одинаковый для target-ID
при source/cutoff/seed. S прогрева тоже генерируется. Forecast остаётся одинаковым
expanding-coarse p90: новая условная модель не улучшает одновременно резервирование.
Начальное состояние и terminal-конкуренты фиксированы, сценарии парные по seed.

## Метрики и границы вывода

Нормативные детали eligibility, прогрева, пропусков, замороженных forecasts,
PRNG и формул метрик: [методика/API](../feature_service.md). Уточнены после
независимой читательской проверки протокола; кандидаты и критерий не изменены.

Primary selection: mean validation CRPS. На test — CRPS, MAE predictive mean,
p90 coverage, mean predicted S, fallback-counts и coverage по типам/запросам.
Для SDSC также P(S>request), Brier этой вероятности относительно recorded S>request;
это распределение среди completed, не production failure model.

Replay: fixed-target mean/p99 release T, W, weighted T, completed/timeout доля,
mean/p99 successful T с собственным знаменателем, utilization и resource ledger.
Под cap terminal T не выдаётся за successful service; условный success subset
сам меняется с моделью. Ошибки относительно observed того же сценария, выбор
policy и regret по mean/p99 terminal T, с числом tied reference policies.
95% t-интервалы только условной генерации; pairwise ошибки кандидатов относительно
coarse используют одинаковый seed. CRPS самого fit детерминирован, MC CI ему
не приписывается. Источники/сценарии не объединяются в один «универсальный» рейтинг.

## Работы

- [x] Библиотечный feature-conditional empirical fit, fallback/ratio и точный CRPS.
- [x] Зафиксированный train/validation/test runner, выбор до test, аудит признаков.
- [x] Offline-тесты формул, ties, пропусков, утечки, воспроизводимости и regression.
- [x] 450 расписаний и полный повтор; независимая проверка метрик и выбора.
- [x] Полный/быстрый pytest, black/isort/pylint/pre-commit.
- [x] Методика, отчёт, артефакты, README/models/roadmap и reader-проверка.

## Результаты

Добавлены `FeatureConditionalEmpirical`, `EmpiricalPrediction`, exact CRPS и
runner с отдельно записанным validation selection. Старые адаптеры и диспетчеры
не изменялись. [Методика/API](../feature_service.md),
[отчёт](../research/feature-service-results-2026-10.md),
[артефакты и независимый аудит](../../works/feature_service/README.md).

На SDSC выбран request_ratio: test CRPS −46.69%, timeout 0.1375% вместо
21.925% coarse при observed 0.2%. Однако без cap ошибки mean/p99 очереди выросли,
а p99-choice стал неверным: MSF вместо FirstFit, regret 62.76% без cap и 28.57%
под cap. На Kalos выбран type_exact; поздний CRPS хуже coarse на 7.14%,
106/300 targets используют pooled fallback. Нулевая observed очередь и шесть
tied policies снова не валидируют выбор планировщика.

Проверки:

- 450 расписаний, полный повтор: четыре JSON побайтово идентичны; 17 code hashes проверены.
- Независимый аудит: 12 score-блоков, 50 service tapes, 75 workload tapes, 450 rows,
  594 MC-интервала, 108 парных интервалов, 18 decisions, 18 observed schedules.
- 59 новых offline-тестов; полный `pytest tests/ -n auto`: **1577 passed**;
  быстрый `pytest tests/ -m 'not slow' -n auto`: **1568 passed** (8 workers).
  В первом быстром прогоне старый `test_mg1_warm` дал stochastic-сбой высокого
  момента; отдельно и при полном повторе быстрого набора прошёл. Его seed не
  фиксирует новый RNG симулятора; тест и tolerance не менялись, долг зафиксирован.
- Black/isort, pre-commit, diff whitespace checks прошли. Новый library/runner
  pylint 10/10; общий `pylint most_queue` — 9.98/10 со старыми замечаниями
  в неизменённых файлах, без новых предупреждений.
- Независимая читательская проверка уточнила eligibility, прогрев, None/pooled
  fallback, численно фиксированные forecasts и разницу iid U / conditional S.
  Выводы и числа повторно сверены; чрезмерных online/causal claims не найдено.

Полная совместная генерация приходов/K/признаков не выполнена и перенесена в
отдельное продолжение roadmap; здесь реализовано только условное S.
Следующий эпик трека — эффективная capacity/квоты/ограничения GPU либо явно
объявленная sensitivity-модель. Queue-aware selection также требует нового
протокола, а не замены победителя по уже просмотренному test.
