Версия 9 статьи о СМО с отрицательными заявками, 08.11.2026. План и диагностика: PLAN.txt.
Документы: ../word/negative_queues_editable_v9.* и ../word/negative_queues_supplement_v9.*

Новые расчёты (все команды из корня репозитория, PYTHONPATH=. OPENBLAS_NUM_THREADS=1):
  diagnostics/            — проверка ошибки квадратуры при delta=0, J-последовательность case08,
                            проверка имитатора при delta=0, ТТ против прямого решения,
                            погрешность двухфазного закона (tt_law_substitution.txt).
  tt_tagged.py            — вероятности Такахаси–Таками (most_queue) + уравнения меченой заявки.
  tt_results.py           — удаление/очищение: основные и расширенные условия, таблица закона.
  correction_grids.py     — сетки при delta=0 для поправки квадратуры (группы refs, small,
                            jcheck, main-cv0.8, main-cv1.2, case09, case11; 2 процесса).
  apply_correction.py     — поправка m(δ)+[m_ref(0)−m_JK(0)], сравнение с ИМ.
  long_simulations.py     — 32 прогона × 2·10^6 уходов (имитатор v7 без изменений).
  figure1_rcs.py          — рисунок 1а (с символьной проверкой интенсивностей).
Сборка:
  .venv/bin/python works/negative_queues/revision_v9_20261108/make_tables_v9.py
  .venv/bin/python works/negative_queues/revision_v9_20261108/assemble_v9.py
  .venv/bin/python works/negative_queues/revision_v9_20261108/make_supplement_v9.py
  uv run --no-project --with pypandoc_binary --with python-docx --with lxml python works/negative_queues/revision_v9_20261108/build_word_v9.py [--supplement]
  cd works/negative_queues/word && soffice --headless --convert-to pdf negative_queues_editable_v9.docx negative_queues_supplement_v9.docx
  uv run --no-project --with lxml python works/negative_queues/revision_v9_20261108/check_artifacts_v9.py
Тесты: .venv/bin/python -m pytest -q tests/units/test_negative_*.py — 75 passed (tests.log).
Исправлен допуск rtol 1e-10 -> 1e-9 в test_exponential_restart_and_moment_existence
(платформенный шум 4,6e-10). Добавлен tests/units/test_negative_tt_v9.py.
Источники новых ссылок проверены по Crossref/OpenAlex (08.11.2026); для Бочарова и др. (2007)
цитируется англоязычная версия, русское название не подтверждено.
  cv_effect.py, make_figure2.py — рисунок 2 (влияние CV): cv-effect.json, figures/figure2*.
  diagnostics/effective_occupation.py — проверка формул времени занятия канала.
