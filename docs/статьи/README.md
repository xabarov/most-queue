# Материалы статей и продолжение работы

По указанию автора от 07.11.2026 собственные рукописи, результаты расчётов,
рисунки, Word/PDF и скрипты включены в Git. Репозиторий публичный.
Ранее записанный в исторических отчётах приватный статус относится к моменту
подготовки этих отчётов. Загруженные сторонние PDF, книги и диссертация в этот
комплект не входят.

## Актуальные документы

| Статья | Исходник для правки | Документы и инструкция |
|---|---|---|
| СМО с отрицательными заявками, версия 8 | [LaTeX](../../works/negative_queues/revision_v8_20261107/negative_queues_takahasi_takami.tex) | [Word](../../works/negative_queues/word/negative_queues_editable_v8.docx), [PDF](../../works/negative_queues/word/negative_queues_editable_v8.pdf), [инструкция](../../works/negative_queues/revision_v8_20261107/README.txt) |
| Дополнительные материалы к версии 8 | [LaTeX](../../works/negative_queues/revision_v8_20261107/supplement/negative_queues_takahasi_takami.tex) | [Word](../../works/negative_queues/word/negative_queues_supplement_v8.docx), [PDF](../../works/negative_queues/word/negative_queues_supplement_v8.pdf) |
| Пакетное обслуживание, сокращённая редакция | [article.tex](batch-service-sla-single-short/latex/article.tex) | [Word](batch-service-sla-single-short/word/main.docx), [PDF](batch-service-sla-single-short/latex/main.pdf), [инструкция и приложение](batch-service-sla-single-short/README.md) |
| Пакетное обслуживание, полная редакция | [article.tex](batch-service-sla-single/latex/article.tex) | [Word](batch-service-sla-single/word/main_v3_editable.docx), [PDF](batch-service-sla-single/latex/main.pdf), [воспроизведение](batch-service-sla-single/supplement_README.md) |

Сохранены также предыдущие материалы `batch-service-sla`,
`batch-service-sla-multi`, `continuous-batching-occupancy-sla`, архив `batch-v1.zip`
и история полной редакции. Они не заменяют актуальные документы из таблицы.
Смысловые ограничения и нерешённые вопросы расчётов находятся в соответствующих
рукописях и отчётах; наличие PDF не означает принятие статьи журналом.

## Подготовка другого компьютера

После клонирования репозитория или `git pull --ff-only` команды выполняются
из его корня. Для расчётов подходят Python 3.10–3.12; сохранённые исследования
выполнялись на Python 3.12. Создание окружения:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pip install lxml python-docx pypandoc_binary
```

Для экспорта и проверки PDF нужны LibreOffice и Poppler (`pdftotext`,
`pdftoppm`). Для LaTeX нужны latexmk, Biber, русский языковой пакет, pdfLaTeX
для отрицательных заявок и XeLaTeX для пакетного обслуживания. В последнем
используются PT Serif и DejaVu Sans Mono. Word хранит Times New Roman и Cambria
Math; контрольный Linux-экспорт временно заменяет Times New Roman на Liberation
Serif. На другой системе пагинация может немного отличаться.

Скрипты текущей версии не требуют локального каталога Codex skills. В инструкциях
встречается `uv run --no-project --with ... python`; при установленных выше
зависимостях его можно заменить на `.venv/bin/python`.

## Проверка сохранённого комплекта

```bash
.venv/bin/python works/negative_queues/revision_v8_20261107/check_artifacts_v8.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/units/test_negative_*.py tests/units/test_bulk_service_revision.py docs/статьи/batch-service-sla-single/test_revision_export.py
```

Первая команда сверяет формулы, числовые ячейки, регистр ИМ/ДИ и хеши прежних
версий. Она читает PDF непосредственно из репозитория и не зависит от прежнего
каталога `/tmp`. JSON исходных прогонов и промежуточные численные результаты
сохранены рядом с программами в `works/negative_queues/` и `docs/research/`.

## Продолжение редактирования

Для отрицательных заявок править исходники версии 8, затем запускать
`build_word_v8.py` (также с `--supplement`) и экспорт PDF по её инструкции.
`prepare_v8.py` заново создаёт редакцию из замороженной версии 7: **не запускать
его поверх новых ручных правок версии 8**. Для следующей редакции лучше создать
соседний каталог и новые имена документов, сохранив отправленную версию.

Сокращённая пакетная статья собирается командами из её README; сначала LaTeX,
затем Word, поскольку преобразование использует сгенерированные перекрёстные
ссылки `.aux`. В full-редакции файлы расчётов и сборки перечислены в
`supplement_README.md`. Старые архивы воспроизведения сохраняют собственные
версии кода; их не следует распаковывать поверх текущего репозитория.

Правила научного изложения и словарь находятся в
[queueing-articles](../../.claude/skills/queueing-articles/SKILL.md).
Терминологию согласовывать внутри выбранной рукописи: в отрицательных заявках
версии 8 используется «меченая заявка», в сохранённой сокращённой пакетной
статье — «помеченная заявка».

## Проверка переноса 07.11.2026

Перед коммитом прошли 173 выбранных теста отрицательных заявок, одноканального
пакетного обслуживания и экспорта численных результатов статьи. Из отдельной
копии файлов, подготовленных к коммиту, без локальных игнорируемых материалов:

- проверен сохранённый комплект версии 8 и заново собраны оба Word-документа;
- сверены все 534 числовые ячейки и нативные формулы;
- выполнен экспорт статьи версии 8 в PDF через системный LibreOffice — 9 страниц;
- собраны LaTeX и Word сокращённой пакетной статьи и отдельного приложения;
- прошли 38 проверок пакетной модели и экспорта статьи в этой отдельной копии.

Отправленные автору Word/PDF и замороженные численные данные при проверке
переноса не перезаписывались. Прежние SVG, исходники и журналы сохранены
побайтно, включая их исходные пробелы и форматирование.
