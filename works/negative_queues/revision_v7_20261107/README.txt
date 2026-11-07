Исследовательские материалы к версии7 статьи о СМО с отрицательными заявками.
Предыдущая версия6 сохранена; контрольные суммы приведены в baseline.json.

Все команды расчёта запускаются из корня репозитория с PYTHONPATH=.
Рекомендуется OPENBLAS_NUM_THREADS=1; исследование использует несколько
отдельных процессов. Наличие результата JSON позволяет пропустить готовую
сетку. Не удалять исходные прогоны при воспроизведении.

1. Полные уравнения помеченной заявки:
 .venv/bin/python works/negative_queues/revision_v7_20261107/step1_calculate.py
 .venv/bin/python works/negative_queues/revision_v7_20261107/step1_simulate.py
 .venv/bin/python works/negative_queues/revision_v7_20261107/summarize_step1.py
2. Уточнения сеток:
 .venv/bin/python works/negative_queues/revision_v7_20261107/step2_calculate.py
Дополнительные сетки J5 и независимые решения больших систем описаны в
STEP2_PROTOCOL.txt. Они не заменяют физические переходы и не подбирают
параметры по имитации.
3. Расширенные условия:
 .venv/bin/python works/negative_queues/revision_v7_20261107/step3_calculate.py
 .venv/bin/python works/negative_queues/revision_v7_20261107/step3_simulate.py
 .venv/bin/python works/negative_queues/revision_v7_20261107/run_case11.py
 step3_fast.py содержит дополнительные строгие проверки общих прерываний;
 основной K12-результат берётся из run_case11.py.
 .venv/bin/python works/negative_queues/revision_v7_20261107/summarize_step3.py
4. Положительные PH:
 .venv/bin/python works/negative_queues/revision_v7_20261107/step4_calculate.py
 .venv/bin/python works/negative_queues/revision_v7_20261107/step4_simulate.py
 .venv/bin/python works/negative_queues/revision_v7_20261107/summarize_step4.py

Итоговые таблицы:
 .venv/bin/python works/negative_queues/revision_v7_20261107/summarize_step2.py
 .venv/bin/python works/negative_queues/revision_v7_20261107/plot_convergence.py
 .venv/bin/python works/negative_queues/revision_v7_20261107/assemble_article.py

Word:
 uv run --no-project --with pypandoc_binary --with python-docx --with lxml python works/negative_queues/revision_v7_20261107/build_word_v7.py
 Та же команда с --supplement создаёт отдельное приложение.
Для контроля PDF используется render_word.py из refinement_20261107.
Он временно подставляет совместимый Liberation Serif при экспорте LibreOffice,
поскольку установленный локальный Times New Roman некорректно отображает
кириллический курсив. Сам DOCX сохраняет Times New Roman и Cambria Math;
формулы остаются нативными редактируемыми OMML.

Научная интерпретация:
- Исходный закон при CV=.8 — Gamma, при CV=1.2 — положительный H2.
- Сохранённая длительность задаётся классом работы и эрлангизацией прогресса;
  это численное приближение исходного непрерывного процесса.
- Разности сеток не являются доказанными границами ошибки.
- Доверительные интервалы относятся к отдельным оценкам; все выходы за них
  сохранены. Никакие seeds не исключались по результату.
- Время указано для выполненных расчётов на совместно используемой машине,
  а не как строго контролируемый аппаратный benchmark.
- LITERATURE_CHECK.txt сообщает, какие источники и разделы были прочитаны.

Материалы v6 необходимы для исходной имитации, квадратуры работы и сравнения
решателей. Базовые утилиты распределений находятся в most_queue/random.
Исследовательские программы не объявляются уже опубликованными в публичном
репозитории. Отправка в Telegram и публикация этой версии не выполнялись.

Завершённый комплект и контроль:
 Все пять пунктов выполнены; см. FINAL_REPORT.txt и STEP1–STEP4_REPORT.txt.
 Основная статья — ../word/negative_queues_editable_v7.docx и .pdf.
 Приложение — ../word/negative_queues_supplement_v7.docx и .pdf.

Независимая сборка крупнейших сеток пункта 2:
 .venv/bin/python works/negative_queues/revision_v7_20261107/run_ilu.py --cv 0.8 --levels 100
 .venv/bin/python works/negative_queues/revision_v7_20261107/run_ilu.py --cv 1.2 --levels 100
 .venv/bin/python works/negative_queues/revision_v7_20261107/run_ilu.py --cv 1.2 --levels 160
 Эти команды пересчитывают и перезаписывают соответствующий численный JSON.
 Для точного сохранения комплекта запускать повторные эксперименты в копии.
 summarize_step2.py предпочитает эти результаты исходному медленному решателю.

Дополнительная чувствительность условия 9:
 .venv/bin/python works/negative_queues/revision_v7_20261107/case08_refine.py
 .venv/bin/python works/negative_queues/revision_v7_20261107/summarize_case08.py

Итоговая сверка Word (после экспорта PDF и копирования PDF в ../word):
 uv run --no-project --with lxml python works/negative_queues/revision_v7_20261107/check_artifacts_v7.py
 Проверка схемы выполняется validate.py из skill docx с зависимостями
 lxml и defusedxml; журналы word-validation.log и supplement-validation.log.

LaTeX: в этом каталоге и в подкаталоге supplement выполнить
 latexmk -pdf -interaction=nonstopmode -halt-on-error negative_queues_takahasi_takami.tex
 Эти PDF имеют собственную пагинацию; файлы ../word/*.pdf соответствуют Word.

Проверки расчётных методов:
 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/units/test_negative_*.py
 Проверено: 49 passed, журнал tests-expanded.log.
