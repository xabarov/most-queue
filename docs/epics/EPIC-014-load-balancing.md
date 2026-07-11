# EPIC-014: Балансировка нагрузки / диспетчеризация (mean-field)

- **Статус:** done (2026-07-11)
- **Создан:** 2026-07-11
- **Roadmap:** обзор трендов — [../research/queueing-trends-2026.md](../research/queueing-trends-2026.md) (вторая волна, п. A)

## Цель

Аналитические модели **политик диспетчеризации** для больших пулов серверов: JSQ, **JSQ(d) /
power-of-d**, **JIQ (join-the-idle-queue)**, random. Даёт распределение длины очереди и среднее время
отклика в mean-field пределе. Завершает уже начатую в туториале `power_of_two_choices` тему реальным
калькулятором.

## Контекст

Самая активная датацентр-тема балансировки. Mean-field (fluid) предел при N→∞:
- **power-of-d**: доля серверов с ≥k заявками `s_k = ρ^{(d^k − 1)/(d − 1)}` (двойная экспонента при
  d≥2 — «power of two choices»); d=1 даёт геометрию `s_k = ρ^k`.
- **JIQ**: асимптотически нулевое ожидание при ρ<1 (insensitivity, zero-waiting режимы).
- Среднее число в системе и время отклика — из `{s_k}`.

В most-queue уже есть **симулятор** power-of-d (в туториале). Первоисточники: Vvedenskaya–Dobrushin–
Karpelevich / Mitzenmacher (power-of-two), Lu et al. (JIQ), mean-field обзоры (см. research).

## Задачи

- [x] **`LoadBalancingMeanField`** (`theory/load_balancing.py`): power-of-d, JSQ, JIQ, random —
      хвост `s_k`, среднее число на сервер `L`, время отклика `W`.
- [x] **`LoadBalancingSim`** (`sim/load_balancing.py`): конечное N, политики JSQ/JSQ(d)/JIQ/random;
      сидируемый.
- [x] Валидация: d=1 = M/M/1 точно; двойная экспонента хвоста для d=2 (`s_k=ρ^(2^k−1)`); JIQ/JSQ
      zero-wait (`W=1/μ`); mean-field vs конечное-N (N=200) — <0.5% для power-of-d, ~2% JIQ/JSQ.
      Тесты `test_load_balancing.py` (9 шт.).
- [x] Каталог (EN+RU); связан с туториалом `power_of_two_choices`.

## Критерии готовности (DoD эпика)

Общий DoD ([../DOD.md](../DOD.md)). Специфично: d=1 = геометрия (M/M/1-пул), формула power-of-d
совпадает с симуляцией при большом N; хвост очереди для d=2 спадает двойной экспонентой.

## Результаты

- **`LoadBalancingMeanField`** — mean-field отклик для power-of-d / JSQ / JIQ / random;
  **`LoadBalancingSim`** — парный симулятор конечного пула. Валидировано (d=1=M/M/1, двойная
  экспонента, zero-wait, сходимость к mean-field). Каталог EN+RU, тесты (9 шт.).
