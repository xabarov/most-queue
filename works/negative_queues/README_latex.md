## Сборка LaTeX-версии статьи

Файлы:

- `negative_queues_takahasi_takami.tex` — исходник статьи
- `citations.bib` — библиография (biblatex/biber)

Сборка (в папке `works/negative_queues/`):

```bash
pdflatex -interaction=nonstopmode -halt-on-error negative_queues_takahasi_takami.tex
biber negative_queues_takahasi_takami
pdflatex -interaction=nonstopmode -halt-on-error negative_queues_takahasi_takami.tex
pdflatex -interaction=nonstopmode -halt-on-error negative_queues_takahasi_takami.tex
```

Выходной файл: `negative_queues_takahasi_takami.pdf`.

## Воспроизводимость экспериментов

Расчёты и графики для статьи генерируются скриптом `work_mgn_negatives.py`. Параметры сетки и точности задаются в `base_parameters.yaml`. Запуск выполняется **из корня репозитория** (most-queue), с виртуальным окружением `./.venv`.

**Пересчёт результатов для дисциплины RCS** (рисунки и `results.json` в `negative_queues_figures/rcs/{utilization,coefs,channels}/`):

```bash
cd /path/to/most-queue
NEGATIVE_DISCIPLINE=RCS OUTPUT_DIR=works/negative_queues/negative_queues_figures/rcs ./.venv/bin/python works/negative_queues/work_mgn_negatives.py
```

**Пересчёт результатов для дисциплины DISASTER (катастрофы)**:

```bash
NEGATIVE_DISCIPLINE=DISASTER OUTPUT_DIR=works/negative_queues/negative_queues_figures/disaster ./.venv/bin/python works/negative_queues/work_mgn_negatives.py
```

После прогона в каждой из подпапок `rcs` и `disaster` должны появиться файлы `v_ave.png`, `w_ave.png`, `v_served_ave.png`, `v_broken_ave.png`, `*_err.png` и `results.json`. Затем можно заново собрать PDF (команды выше в папке `works/negative_queues/`).

**Обновление таблиц из результатов:** опционально можно сгенерировать фрагменты LaTeX-таблиц (booktabs) из сохранённых `results.json` скриптом:

```bash
./.venv/bin/python works/negative_queues/export_tables_from_results.py
```

Скрипт выводит готовые строки таблиц в stdout; при необходимости их можно вставить в статью вручную.

