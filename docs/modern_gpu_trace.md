# Калибровка обслуживания по современной GPU-трассе

[EPIC-058](epics/EPIC-058-modern-gpu-trace-validation.md) переносит простые
эмпирические baseline на Acme/Kalos 2023. Проверяется ошибка модельного replay,
не точность реконструкции исторического планировщика. Исходный журнал, обучающая
история и модельный эксперимент — разные объекты.

## Источник и пределы наблюдаемости

Используется [официальный Kalos CSV](https://github.com/InternLM/AcmeTrace/blob/f9fdf591b4876c2875a9e3d28adb1bda8120dfcb/data/job_trace/trace_kalos.csv)
из AcmeTrace Shanghai AI Laboratory, Hu et al., NSDI 2024.
[Схема](https://github.com/InternLM/AcmeTrace/blob/f9fdf591b4876c2875a9e3d28adb1bda8120dfcb/README.md)
определяет submit/start/end и отдельные terminal states.
[Лицензия](https://github.com/InternLM/AcmeTrace/blob/f9fdf591b4876c2875a9e3d28adb1bda8120dfcb/LICENSE.txt)
— CC-BY-4.0, не MIT пакета. Raw CSV хранится только в `.cache`; в репозитории
сохраняются собственный код, агрегаты и указание преобразований.

Commit `f9fdf591b4876c2875a9e3d28adb1bda8120dfcb`, SHA-256
`7c5a7845da1d66448fa668342e4ecb622fd9a005118080660ec3ad2788d4c9bf`.
Даты в `submit_time`, `start_time`, `end_time` имеют явный timezone и переводятся
в секунды UTC; порядок одинаковых submit сохраняется из файла. Покрытие именно
этой выгрузки — submit с 2023-05-17 11:00:58 UTC по 2023-08-16 15:18:02 UTC,
не весь шестимесячный Acme из публикации.

В отличие от описания derived-колонки, в закрытых строках этого файла
`duration = end_time − submit_time`. Адаптер вычисляет `S = end_time − start_time`,
`W_history = start_time − submit_time`; готовые duration/gpu_time только проверяет.
Иначе историческое ожидание было бы повторно внесено в обслуживание. Аудит
каждого расхождения и исключения сохраняется в manifest; исходник не исправляется.

| Поле или механизм | Что используем | Что не утверждаем |
| --- | --- | --- |
| GPU request K | Точное положительное целое число | Не измеренная SM utilization |
| S | Положительный end-start | Не duration из CSV, не latent demand при отказе |
| COMPLETED | Обучение и целевые jobs | Будущий успех неизвестен online |
| CANCELLED / FAILED | Отдельные интервалы занятости и метки | Не успех и не причина/модель повторной попытки |
| TIMEOUT / NODE_FAIL | Поддерживаемые отдельные метки | В закреплённом файле таких строк нет |
| RUNNING без end | Аудит исключения | Ни нулевое S, ни прогноз окончания |
| Requested runtime | Отсутствует | Нет сценария enforcement |
| Capacity | Постоянные номинальные 2416 GPU | Не восстановленная доступная ёмкость во времени |

[Авторское описание кластера](https://www.usenix.org/publications/loginonline/understanding-workload-characteristics-large-language-model-development)
указывает A100 и 2416 GPU для Kalos. Это основание для однородного номинального
пула, но не доказательство доступности всего пула каждому job. В симуляции
работа резервирует K единиц на весь S. Ресурсные метрики означают **модельное
резервирование**, не аппаратную загрузку. CPU/memory, размещение по узлам,
сеть, квоты, приоритеты, outages и зависимости приложений не реконструируются.
Fractional GPU не округляются; CPU-only и неподдерживаемые запросы исключаются
с отдельными счётчиками. Аудит проверяет K <= 8 × node_num для Kalos.

## Почему выбран этот источник

Выбор сделан по наблюдаемости, а не по величине ошибки моделей.

| Кандидат | Наблюдаемость и решение |
| --- | --- |
| [Alibaba GPU 2023](https://github.com/alibaba/clusterdata/blob/cb65b488983ce23efb09eb891f796c214e0a3fd3/cluster-trace-gpu-v2023/README.md) | Creation/scheduled/deletion и фазы pod; deletion не гарантирует completion, есть sharing и разные GPU. Нельзя автоматически получить rigid successful-service replay. |
| [Alibaba GPU 2025](https://github.com/alibaba/clusterdata/blob/cb65b488983ce23efb09eb891f796c214e0a3fd3/cluster-trace-gpu-v2025/README.md) | Долгоживущие inference instances с пропусками границ жизни. Не лента успешно обработанных запросов. |
| [Alibaba GPU 2026](https://github.com/alibaba/clusterdata/blob/cb65b488983ce23efb09eb891f796c214e0a3fd3/cluster-trace-gpu-v2026/docs/schema.md) | Часовые observations, execution spans и дробные requests. Нет точной arrival/start/end ленты для данного протокола. |
| Acme/Kalos | Раздельные timestamp и outcomes, homogeneous GPU source; выбран ограниченный scalar-pool replay. |

Это проверка схем, не сравнение всех возможных датасетов и не отказ от
multi-resource/sharing моделей. Новизна года выпуска сама по себе не компенсирует
недостающие события. У Acme также не предполагается точность всех derived fields.

## Обучение и фиксированные цели

Cutoff = first_raw_submit + f × (last_raw_submit − first_raw_submit),
f = .35, .45, .50, .55. Доли закреплены после аудита наличия completed работ,
до измерения расписаний. Прежние .65/.80 не дают полного блока из 1200;
выбранные блоки не перекрываются, но истории пересекаются. Это активная часть
июня–июля, не проверка малонасыщенного августовского хвоста.

Train: только eligible COMPLETED с end **строго меньше** cutoff. Heldout: первые
1200 eligible COMPLETED с submit **не меньше** cutoff в исходном submit-порядке.
Первые 200 — прогрев, следующие 1000 — фиксированные target-ID. Отбор по конечным
меткам ретроспективный; он не имитирует online-знание того, какие работы завершатся.
Точные cutoff/границы измерения записаны в каждом origin JSON.

Expanding-coarse учит эмпирическую CDF по всей train-истории; recent-coarse —
по последним 4000 train jobs в submit-порядке (или всей истории, если её меньше).
Четыре K-группы: 1, 2–8, 9–32, >=33. При n < 20 группа получает pooled CDF
**того же окна**. Сэмплирование iid по обратной эмпирической CDF; K и arrival
сохраняются. Точного K-fit, ранговых блоков или нового семейства здесь нет.

Восемь seed 58000–58007; один U на выбранный job при фиксированном seed/cutoff
общий для обеих моделей и всех сценариев/политик. Генерируется S только 1200
выбранных successful jobs. Никакой нормировки по test S, изменения arrival,
подбора capacity или постфактум выбора лучшего окна.

Forecast — expanding-coarse p90 train S для группы K, одинаковый для всех
вариантов. EASY/Conservative используют его в резервировании; остальные четыре
дисциплины не получают фактическое S. У carry-running прогноз остатка равен
max(p90 − age, 0); истёкший прогноз означает overdue, а не знание истинного остатка.

## Жизненный цикл и сценарии

`observed` — **контрольный replay с записанными S**, не исторические T/W.
Оба генератора сравниваются с ним при одинаковом сценарии:

1. `empty_completed`: 1200 выбранных completed с пустого старта.
2. `carry_completed`: также completed с submit < boundary < end, где boundary
   — первый выбранный submit. Если start <= boundary, работа running; иначе
   waiting. End = boundary уже освобождён, submit = boundary — новый приход.
3. `carry_terminal`: дополнительно все пригодные неуспешные running/waiting и
   новые неуспешные arrivals в [boundary, последний выбранный submit], включая
   прогрев и обе границы. Их S/метки не генерируются.

Наблюдаемый состав окружения и остатки ретроспективны и частичны. Running
оставляет K и исторический остаток до release. Waiting/new неуспешные jobs
занимают ресурс на исходное end-start **после нового симулированного старта**.
Такой service-clock сценарий проверяет чувствительность; он не переносит
абсолютный момент пользовательской отмены или отказа оборудования. Информация
об исходе не передаётся диспетчеру. Reservations/Adaptive phase начинаются заново.

Не восстанавливаются неуспешные без положительного наблюдаемого интервала,
queue abandonment, retries, latent S до успеха и отсутствие записей перед началом
файла. Начальный запрос сверх capacity вызывает ошибку, не clipping. Requested
runtime отсутствует: budgets остаются None, ограничения по факту будущего TIMEOUT
не выводятся. Этот сценарий EPIC-057 здесь явно исключён.

FCFS, FirstFit, MSF, Adaptive Quickswap, EASY, Conservative; 4 origins × 3 scenarios
× (1 observed + 2 модели × 8 seed) × 6 policies = **1224 расписания**.

## Метрики и неопределённость

Для 1000 фиксированных targets: W = simulated_start − submit,
T = simulated_release − submit. Дренирование полное, T не обрезается по последнему
arrival. Все targets completed, поскольку budgets отсутствуют; успех окружения
не смешивается с успехом целей. p99 — выборочный quantile NumPy с линейной
интерполяцией; модельный итог — среднее восьми выборочных p99, не pooled quantile.

Utilization и idle-with-queue интегрируются от arrival первого target до arrival
последнего target, без drain, с нормировкой на capacity × длину интервала.
Ledger K × занятое время включает всё окружение, прогрев и drain. Для начальных
running учитывается только остаток после boundary. Это GPU-request-seconds.

Для каждой policy/origin/scenario signed relative error = mean(model metric) /
observed metric − 1; при observed=0 относительная ошибка None, абсолютная разность
остаётся. MAPE отдельно для каждой пары model/scenario — среднее модуля ошибки
MC-средних по 24 origin/policy ячейкам. Mean T и p99 положительны.

Выбирается policy с минимальным MC-средним mean T либо p99 T; regret =
observed(chosen) / min_policy observed − 1. Равенства разрешает фиксированный
порядок POLICIES; разные названия могут иметь нулевой regret. 95% t-интервалы
условны на истории, составе ленты, fit, окружении и capacity. Разности сценариев
парные по одному seed; у детерминированного observed границы интервала None.
Нет CI по четырём «независимым» периодам, причинного вывода или гарантии SLO.

## API и воспроизведение

```python
from pathlib import Path
from most_queue.sim.utils.acme_trace import parse_acme_kalos, acme_snapshot

with Path(".cache/real_trace/acme-kalos.csv").open(encoding="utf-8") as stream:
    source = parse_acme_kalos(stream)
print(source.audit)  # completed cohort: source.jobs; all accepted: terminal_jobs
running, waiting = acme_snapshot(source, source.jobs[100].submit,
                                include_unsuccessful=True)
```

```bash
.venv/bin/python -m examples.modern_gpu_trace_experiment --download \
  --output-dir works/modern_gpu_trace
.venv/bin/python -m examples.modern_gpu_trace_experiment \
  --output-dir /tmp/most-queue-modern-repeat
```

Download строго opt-in, проверяется SHA-256, размер ограничен, существующий cache
не перезаписывается. `--jobs`, `--warmup`, `--replications`, `--fractions` служат
для smoke/custom runs; основной протокол использует defaults. Manifest закрепляет
версию окружения, исходник, implementation и каждый output hash. Исходные
данные/большая телеметрия не входят в git и не нужны offline-тестам.

`MsjLifecycleJob` теперь сохраняет completed/cancelled/timed_out/failed/node_failed.
Записанный timed_out сам по себе не задаёт runtime_limit. Старые три ledger-ключа
сохранены; failed/node_failed добавляются только при наличии таких исходов.
Обычный `MsjGeneralSim.run_trace` не менялся. Старый EPIC-057 воспроизводится
из его закреплённого commit `99b8500`; текущий lifecycle файл имеет новый hash.
