# EPIC-049 — MSJ прогнозы по признакам и калибровка длительности

- **Статус:** done
- **Создан:** 2026-10-02
- **Завершён:** 2026-10-02
- **Предшественник:** [EPIC-048](EPIC-048-msj-conservative-controlled-load.md)
- **Roadmap:** [MSJ](../roadmaps/msj-ph-backfilling.md)

## Цель и выбор

Выполнить следующий пункт MSJ-roadmap — условные прогнозы по наблюдаемым признакам.
EPIC-048 показал нарушения резервирований с class-mean оценками. Теперь заменить
знание истинной длительности или параметров закона обучением на завершённых
исторических работах и проверить компромисс между покрытием и задержками.

## Задачи

- [x] Log-linear регрессия длительности по признакам, доступным до запуска;
      позитивные оценки и валидация данных/жизненного цикла без новых зависимостей.
- [x] Односторонняя split-conformal калибровка по отдельной выборке;
      точная порядковая статистика и явный отказ при недостатке данных.
- [x] Детерминированные тесты формул и утечки данных, валидация marginal coverage,
      связка с replay и проверка M/E2/1 по общим допускам проекта.
- [x] Эксперимент: отдельные train/calibration/test, общие трассы дисциплин,
      обученное среднее класса, feature point, feature upper и явный oracle;
      штатный режим и сдвиг длительности, контроль ресурсной нагрузки.
- [x] Отчёт с покрытием по классам, средними/p99, нарушениями обещаний,
      независимыми повторами и парными интервалами; EN/RU документация и roadmap.
- [x] Быстрый и полный pytest, форматирование, pylint без новых замечаний.

## Результат и проверки

Реализованы [predictor API](../msj_runtime_prediction.md) и
[воспроизводимый эксперимент](../../examples/msj_runtime_prediction_experiment.py).
Выполнены 432 scheduler-прогона с Erlang-2/lognormal шумом, двумя нагрузками,
штатным режимом и двукратным slowdown без перекалибровки. Данные и команды:
[works/msj_runtime_prediction](../../works/msj_runtime_prediction/README.md).

[Отчёт](../research/msj-runtime-prediction-results-2026-10.md) разделяет точность
регрессии, покрытие и качество очереди. При iid/lognormal суммарное покрытие upper95
равно 94.75%, но для K=4 — 87.81%; при slowdown суммарное падает до 83.47%.
Улучшение средних и уменьшение нарушений не означают гарантию p99 или всех обещаний.
Следующее обоснованное направление — class-conditional калибровка.

- 48 новых тестовых случаев, включая две сверки с M/E2/1 по общим допускам.
- `.venv/bin/pytest tests/ -m 'not slow' -n auto -q --tb=short --show-capture=no`:
  **917 passed**, 16 прежних warnings.
- `.venv/bin/pytest tests/ -n auto -q --tb=short --show-capture=no`:
  **926 passed**, 16 прежних warnings.
- Black/isort: пять новых Python-файлов прошли проверки.
- Pylint: новых замечаний нет; по `most_queue` 43 до и после относительно
  `11f9064`, новый модуль — 10.00/10.
- Проверены все JSON-агрегаты, Student-интервалы и парные разности;
  пересозданы train/calibration/test и проверены отпечатки всех 12 seed-наборов;
  воспроизведены все prediction-метрики и четыре полных conservative replay.
- Пример документации исполнен; локальные Markdown-ссылки и whitespace проверены.

## Математический и информационный контракт

OLS для log S с intercept и стандартизацией X по train. Point-estimate использует
множитель среднего экспоненцированных train-остатков; это baseline, не обещание
точного условного среднего при неверной модели. На calibration считать signed score
`r_i = log(S_i) − log(point(X_i))`; выбрать k-ю порядковую статистику,
`k=ceil((n_cal+1)*coverage)`. Верхний прогноз `point(X)*exp(r_(k))`.
При k>n_cal конечной границы нет: API отвергает запрос, не обрезает ранг.

Marginal coverage относится к обменным calibration/test наблюдениям при модели,
обученной независимо. Это не покрытие каждого класса, не гарантия всей трассы,
не гарантия резервирований и не p99 SLO очереди. При сдвиге закона гарантия теряется.
Predict принимает только X, без S. Разделение и доступность признаков обеспечиваются
вызывающим кодом; API не может доказать происхождение переданных массивов.

## Границы

Эпик ограничен offline-прогнозами при поступлении и воспроизводимыми синтетическими
данными. Динамическая коррекция по возрасту, реальные кластерные трассы, адаптивная
калибровка при drift, class-conditional калибровка и MSFQ/ServerFilling — следующие
самостоятельные этапы roadmap. Существующие FCFS/EASY/conservative не меняются.

## Источники

- Tsafrir, Etsion, Feitelson. *Backfilling Using System-Generated Predictions Rather
  Than User Runtime Estimates*, IEEE TPDS, 2007,
  [DOI 10.1109/TPDS.2007.70606](https://doi.org/10.1109/TPDS.2007.70606).
  Мотивация прогнозов; мы не воспроизводим алгоритм динамической коррекции статьи.
- Lei, G'Sell, Rinaldo, Tibshirani, Wasserman. *Distribution-Free Predictive
  Inference for Regression*, JASA, 2018,
  [DOI 10.1080/01621459.2017.1307116](https://doi.org/10.1080/01621459.2017.1307116).
- Angelopoulos, Bates. *A Gentle Introduction to Conformal Prediction and
  Distribution-Free Uncertainty Quantification*,
  [arXiv:2107.07511](https://arxiv.org/abs/2107.07511): калибровка произвольного score
  и ограничения marginal coverage. Наша специализация — signed log-runtime score;
  научная новизна conformal prediction или backfilling не заявляется.
