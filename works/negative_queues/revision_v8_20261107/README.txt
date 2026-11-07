Редакционная версия 8 статьи о СМО с отрицательными заявками, 07.11.2026.

Готовые документы находятся в ../word/:
negative_queues_editable_v8.docx и .pdf — статья, 9 страниц PDF;
negative_queues_supplement_v8.docx и .pdf — дополнительные материалы, 6 страниц PDF.
Формулы в Word редактируемые: 216 объектов в статье и 92 в дополнительных материалах.
Предыдущая версия 7 сохранена. Численные расчёты при этой редактуре не менялись.

Что изменено

Термин «меченая заявка» (одна н) принят для этой статьи по последнему
уточнению автора. Это локальное предпочтение; прежнее «помеченная заявка»
в общем skill автоматически не заменялось. Названия распределений переданы
по-русски: гамма-распределение, бета-распределение, распределение Эрланга.
Раздел 5.1 назван «Условия численных экспериментов».

Из основного текста убрана история проверок и исправлений. Она сохранена
в working-diagnostics.txt как рабочий материал. Раздел 5.2 теперь содержит
результаты и их краткое обсуждение. Допущения модели, ограничения численного
метода и сведения о расхождениях сохранены; общая гарантия точности не заявлена.
Двухфазная схема очищения перенесена из внутреннего приложения в подраздел 3.4.
Отдельные подробные численные материалы по-прежнему сопровождают статью.

Таблицы 2 и 3 помещаются на странице 8; таблица 2 находится вверху.
Страница 7 заполнена без прежней большой пустоты. Подзаголовки групп имеют
вид «Коэффициент вариации: 0,8» / «Коэффициент вариации: 1,2», размер 10 пунктов.
Табличные формулы также имеют размер 10 пунктов. ИМ и ДИ — прописными,
без малых прописных. Заголовки повторяются при переносах таблиц.
В подписях и осях дополнительного графика используются десятичные запятые.

Редактура выполнена с учётом:
../../../.claude/skills/queueing-articles/SKILL.md
../../../.claude/skills/queueing-articles/glossary-ru.md
../../../.claude/skills/queueing-articles/references/scientific-prose.md
../../../.claude/skills/queueing-articles/references/model-description.md
../../../.claude/skills/queueing-articles/references/numerical-evidence.md
../../../.claude/skills/queueing-articles/references/presentation.md
/home/roman/.codex/skills/docx/SKILL.md
/home/roman/.codex/skills/pdf/SKILL.md

Проверки

Все 534 числовые ячейки совпали с манифестами версии 7 (72 + 462).
Число нативных формул соответствует исходному дереву документа; сохранены
три матрицы. Ссылки Word не повреждены; оба DOCX прошли XML-валидацию.
Проверены размер табличных формул и подзаголовков, регистр ИМ/ДИ и терминология.
Просмотрены все 9 страниц статьи и все 6 страниц дополнительных материалов.
В итоговых журналах LaTeX нет переполнений строк и неопределённых ссылок.
Хеши замороженных источников, данных и документов версий 6/7 совпадают.
Отчёты: ../word/verification_v8/ и ../word/verification_supplement_v8/.
Новые численные эксперименты и научные тесты в редакционной версии не запускались;
состав и результаты прежних проверок описаны в ../revision_v7_20261107/README.txt.

PDF получен через LibreOffice с временной подстановкой Liberation Serif
из-за ошибки местного Times New Roman в кириллическом курсиве. DOCX использует
Times New Roman и Cambria Math. Пагинация в Microsoft Word может слегка отличаться
в зависимости от шрифтов и версии редактора.

Повторная сборка из корня репозитория

python3 works/negative_queues/revision_v8_20261107/prepare_v8.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 .venv/bin/python works/negative_queues/revision_v8_20261107/plot_convergence.py
uv run --no-project --with pypandoc_binary --with python-docx --with lxml python works/negative_queues/revision_v8_20261107/build_word_v8.py
uv run --no-project --with pypandoc_binary --with python-docx --with lxml python works/negative_queues/revision_v8_20261107/build_word_v8.py --supplement
uv run --no-project --with lxml python works/negative_queues/refinement_20261107/render_word.py works/negative_queues/word/negative_queues_editable_v8.docx /tmp/negative-word-v8-20261107
uv run --no-project --with lxml python works/negative_queues/refinement_20261107/render_word.py works/negative_queues/word/negative_queues_supplement_v8.docx /tmp/negative-supplement-v8-20261107
cp /tmp/negative-word-v8-20261107/render/negative_queues_editable_v8.pdf works/negative_queues/word/
cp /tmp/negative-supplement-v8-20261107/render/negative_queues_supplement_v8.pdf works/negative_queues/word/
uv run --no-project --with lxml python works/negative_queues/revision_v8_20261107/check_artifacts_v8.py

Для LaTeX выполнить в этом каталоге и в supplement/:
latexmk -pdf -interaction=nonstopmode -halt-on-error negative_queues_takahasi_takami.tex

prepare_v8.py копирует рисунки версии 7, поэтому plot_convergence.py запускается
после него. После повторного экспорта требуется заново просмотреть страницы;
сохранённый visual-review.json относится только к указанному в нём хешу PDF.
