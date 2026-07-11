# EPIC-013: Планирование с предсказаниями (learning-augmented scheduling)

- **Статус:** done (2026-07-11)
- **Создан:** 2026-07-10
- **Roadmap:** обзор трендов — [../research/queueing-trends-2026.md](../research/queueing-trends-2026.md)

## Цель

Расширить уже существующие size-based дисциплины с предсказаниями (SPJF) до полноценного набора
**learning-augmented** инструментов: разные модели предсказаний, метрики устойчивости к ошибке
(graceful degradation), воспроизведение открытых задач из survey SIGMETRICS 2025 «Queueing,
Predictions, and LLMs». Это низкозатратное направление с прямой опорой на имеющийся код.

## Контекст

most-queue уже содержит `most_queue.theory.srpt` (SRPT/SJF/PSJF/SPJF) и `SizeBasedQsSim` с
предикторами (`PerfectPredictor`, `ExpNoisePredictor`, …), воспроизводит таблицу Mitzenmacher–Shahout.
Survey (Sohn & Maguluri, Stochastic Systems 2025) формулирует открытые задачи:
- **Graceful degradation / smoothness**: как деградирует время отклика при росте ошибки предсказания;
  показано, что наивный SPRPT не даёт константного фактора от SRPT даже при ограниченной ошибке.
- Чувствительность к типу ошибки (мультипликативная/аддитивная), к смещению предиктора.
- Сравнение дисциплин при одинаковом качестве предсказаний.

Первоисточники: см. [research](../research/queueing-trends-2026.md).

## Задачи

- [x] **Метрики деградации** (`theory/srpt/degradation.py`): `prediction_degradation_curve` —
      кривая «E[T]_SPJF vs шум предсказания σ» (лог-нормальный предиктор) в рамке SRPT / SJF /
      слепой FB. **Воспроизведена центральная открытая задача survey**: при достаточном шуме
      size-based SPJF становится **хуже слепой** политики → возвращается `breakeven_sigma`.
- [x] Валидация предельных случаев: SPJF(perfect) = SJF точно; σ→0 → SJF; монотонный рост E[T] по
      σ; SRPT — нижняя граница; пересечение слепой FB при большом σ. Тесты
      `test_prediction_degradation.py`.
- [x] Каталог (EN+RU): раздел «Graceful degradation предсказаний».
- [ ] *(будущее)* дополнительные предикторы (bin-based/quantized как в multi-bin batching),
      сравнительный бенчмарк дисциплин в `disciplines_comparison.ipynb`.

Использованы уже имеющиеся `MG1SpjfCalc` + `LognormalNoisePredictor` (σ — регулятор качества) —
низкозатратное расширение, как и планировалось.

## Критерии готовности (DoD эпика)

Общий DoD ([../DOD.md](../DOD.md)). Специфично: perfect-predictor сводит SPJF→SJF, SPRPT→SRPT точно;
кривые деградации согласуются с симуляцией; воспроизведён хотя бы один результат/наблюдение из survey.

## Результаты

- **`prediction_degradation_curve` / `DegradationCurve`** — анализ graceful degradation для M/G/1
  SPJF: E[T] по уровням шума σ + референсы SRPT/SJF/слепой FB + точка перелома, где размерная
  политика проигрывает слепой. Воспроизводит центральную тему survey SIGMETRICS 2025.
- Валидация: `test_prediction_degradation.py` (3 теста, зелёные) — редукции и монотонность.
- Каталог EN+RU обновлён.
