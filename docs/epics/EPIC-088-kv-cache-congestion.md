# EPIC-088: Service-induced congestion при вытеснении KV-cache

- **Статус:** done (библиотечный контракт ниже)
- **Создан:** 2026-10-11
- **Roadmap:** [Л5](../roadmaps/llm_serving_catchup_roadmap.md)

## Контракт

Воспроизведение Ao, Dong, Luo, Simchi-Levi,
[arXiv:2606.15555v1](https://arxiv.org/html/2606.15555v1): deterministic
class-stage memory dynamics, homogeneous cycles, finite-input spectrum,
proportional saturated admissions и admission cap. Отдельно — независимые
целые запросы, LPF eviction/restart, Poisson/replay и конечный горизонт без drain.
Новизна не заявляется. Динамическая стабильность равновесия не равна
positive recurrence открытой очереди; integer FCFS не равен fluid proportional.

## Задачи

- [x] Основной текст и разделы приложений для реализуемых формул; локальная
      классика, processing-network источник и четыре направления поиска.
      Полный HTML сохранён приватно; глубина чтения фиксируется отдельно.
- [x] BaseQueue-калькулятор fixed point / worst homogeneous cycle / finite
      and limiting spectrum с numerical boundary diagnostics.
- [x] Saturated continuous trajectory, proportional LPF и rate cap.
- [x] Независимый integer DES с request/work conservation и явным restart.
- [x] Рациональная карта, ручные трассы, прямой stage Jacobian, limiting
      factors, exceptional balanced orbit, finite-input GCD counterexample.
- [x] Воспроизводимые trajectories / spectrum / stochastic runs и JSON.
- [x] EN/RU модель/схема, README/models/simulation, методический отчёт.
- [x] Целевые, slow, fast/full тесты, black/isort/pylint и ссылки.
- [x] Диссертационные источники, гипотезы/план/журнал, синхронизация архива.

## Результат

[Контракт/API](../models/kv-cache-congestion.md),
[методический отчёт](../research/kv-cache-congestion-validation-2026.md),
[JSON](../research/kv-cache-congestion-results-2026.json),
[скрипт](../../examples/kv_cache_congestion_experiment.py).

75 targeted (включая 30000-итерационный slow), full **3071 passed**,
fast **3061 passed**, 16 прежних warnings, без reruns. Black/isort,
новые production/experiment модули pylint 10/10; package 9.99/10 с прежними
диагностиками. JSON двух запусков совпадает побайтно: 10 trajectories,
48 spectra, 24 DES-настройки × 4 paired seeds.

Независимый рациональный Schur-тест и прямой stage Jacobian:
Figure3 для (2,7,p=.5) даёт threshold18 вместо печатного15;
Example9 (2,3) имеет unstable S=0 / boundary S=1. Глобальные теоремы
о достаточно большом S этим не опровергнуты. Код авторов не проверен.

Private dissertation сохраняет источник, карточки, результаты, гипотезы,
append-only журнал и 90 research/roadmap snapshots с SHA-256. Папка gitignored;
обычный push её не резервирует. Следующий пункт — Л6, SMDP запуска пакета.

## Границы

Без GPU/Vidur replication, prefill/recovery time model, unknown random output
lengths, оптимального маршрутизатора и распределений latency. Утраченную
decode-работу учитываем; платный wall-clock swap/recompute не добавляем к
доказательствам статьи. K-class/heterogeneous-input spectrum — численная
диагностика заданной recurrence, не новая глобальная теорема.
