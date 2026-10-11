# ГЛ2: controlled joint-mark order pilot, 2026-10-11

**Результат: rank reversal не подтверждён на frozen pair.** Это отрицательный
pilot одной выбранной области, не доказательство отсутствия эффекта порядка.
[Протокол до replay](../epics/EPIC-092-llm-mark-order.md),
[полные агрегаты](llm-mark-order-results-2026.json).

## Источник и аудит

Wang et al., [BurstGPT2401.17644v5](https://arxiv.org/abs/2401.17644v5);
[официальный releasev2.0](https://github.com/HPMLL/BurstGPT/releases/tag/v2.0).
Файл SHA совпадает с digest asset, 52,283,111bytes, 1,429,737 строк данных.
Схема конкретного asset имеет только 6 полей: timestamp, model, request/
response/total tokens, log type. Session ID и Elapsed time отсутствуют; расширенная
схема README не переносится автоматически. Conversation исключены, никакой
восстановленной причинной структуры или prefix cache нет. Строки не сортируются
за пользователя: исходная хронология и total=S+O проверяются строго.
В JSON — взаимно исключающий аудит всего файла. Zero output не объявляется
цензурированным успешным запросом с известной длиной; потенциальное смещение
из-за исключённых отказов остаётся ограничением. Original GPU latency неизвестна.

## Проверка

Восемь prespecified day-prefix cohorts, 1024 измеряемых запроса каждый и128
прогрева. Одна и та же последовательность timestamps для всех конфигураций
и абляций. Три permutation seeds не увеличивают число наблюдаемых дней.
Прогрев закреплён: переставляются только1024 измеряемых pairs, и multiset
в измеряемой части проверяется точно. Первый software run пересекал warmup
границу; он сохранён приватно как недействующий, здесь только исправленный
run. Конфигурации, дни, seeds, порог и endpoint при исправлении не менялись.
Тесты дополнительно сверяют ручную boundary trajectory, heterogeneous B=1
через scalar FCFS/Lindley, circular adjacency и joint multiset каждого local окна.

Independent oracle проверяет replay-механику; это не аппаратная калибровка.
Итерации детерминированны по состоянию, future output length не участвует
в admission/scheduling, только в событии завершения. No KV/offload/eviction.
Дренирование нужно для полных TTFT cohort, occupancy/backlog измеряются до
drain. Синтетические costs не лицензируют утверждение об устойчивости GPU.

## Endpoint и вывод

Разность SLO fractions large-minus-small, equal day weights:

| Вариант | Средняя разность, п.п. | Conditional paired day CI95, п.п. |
|---|---:|---:|
| Observed | 5.5664 | [0,16.6748] |
| Global pairs permutation | 6.2744 | [0,18.7419] |
| Local300s (cohort-relative clock) | 4.8828 | [0,14.6484] |
| Block32 | 6.4535 | [0,18.7093] |
| Circular shift | 6.3436 | [0,18.1681] |

Primary global-minus-observed shift **0.7080 п.п., CI[−0.0244,2.0793]**.
Оба ранга не имеют противоположных ненулевых CI; practical threshold5п.п.
не пройден. На этих cohorts нет основания продвигать claim «порядок меняет
оптимальную конфигурацию». Средняя serialized isolated-work exposure для
малой конфигурации от0.00442 до0.04570, для большой от0.00109 до0.01116;
это диагностические отношения, не rho shared batching. Большие gaps и
локальные bursts не заменяются средним load. Много дней имеют малую нагрузку; результат
ограничен исходной временной шкалой, frozen pair и successful API population.
Bootstrap на8 днях условен; календарная разнесённость не доказывает независимости.
Secondary controls не превращаются в подтверждение через выбор удобного endpoint.

Для следующего исследования можно отдельно preregister нагрузочную область
и пару конфигураций по training days, затем проверить на другом asset/датасете
и аппаратно. Такой этап не выполнен и не выдаётся за результат этого пилота.
Синтетический proof-of-mechanism пока ограничен ручными trajectory/invariance
тестами, положительный synthetic rank reversal не предъявлен.

## Современные соседи

[Li et al.2606.18431v1](https://arxiv.org/abs/2606.18431v1), прочитаны выбранные
§2–3 и §4: general fragility/rank sensitivity to bursts и совместное управление
KV уже исследуются; в hardware protocol отключён prefix cache, есть mixture
workloads и scaled Azure timestamps. Это не наш exact-pair-permutation опыт,
но запрещает общую заявку «мы впервые заметили важность burstiness».
Nie/Shi/Zhao [DOI10.51903/jtie.v5i2.565](https://doi.org/10.51903/jtie.v5i2.565)
остаётся потенциальным соседом chronological BurstGPT capacity planning:
full text в этом этапе не получен; эквивалентность не заявлена.

## Воспроизведение

```bash
python -m examples.llm_mark_order_experiment --trace /path/BurstGPT_1.csv
pytest tests/units/test_llm_mark_order.py
```

Raw trace сохранён только в приватной диссертационной папке. Public JSON
не содержит пользовательских текстов или исходных строк, только агрегаты.

Итоговая совместная проверка ГЛ2/ГЛ3: fast3236passed, full3256passed,
16 прежних warnings;37 targeted checks; black/isort/pylint новых модулей чистые.
