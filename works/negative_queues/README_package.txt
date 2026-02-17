Исходники статьи: расширение метода Такахаси–Таками для многоканальных СМО с отрицательными заявками (RCS и катастрофы).

СБОРКА PDF
----------
  pdflatex negative_queues_takahasi_takami.tex
  biber negative_queues_takahasi_takami
  pdflatex negative_queues_takahasi_takami.tex
  pdflatex negative_queues_takahasi_takami.tex

Требуются: TeX Live (pdflatex, biber), пакеты biblatex-gost, csquotes, geometry, hyperref и др.

ДИАГРАММЫ ПЕРЕХОДОВ (рис. 1 и 2 в статье)
-----------------------------------------
  python generate_transition_diagrams.py

Создаёт PNG/SVG в negative_queues_figures/rcs/diagrams/ и disaster/diagrams/
(нужен Python 3, matplotlib).

ОСТАЛЬНЫЕ РИСУНКИ (графики v_ave, q и т.д.)
-------------------------------------------
Генерируются скриптами work_mgn_negatives.py, plot_*.py с использованием
библиотеки most_queue (корень репозитория most-queue). Текстовые данные
для графиков в статье уже лежат в negative_queues_figures/ (подкаталоги
rcs/remove, rcs/requeue, disaster/clear, disaster/requeue).

Структура архива
----------------
  negative_queues_takahasi_takami.tex  — основной файл статьи
  citations.bib                      — библиография
  negative_queues_figures/            — все рисунки для статьи
  generate_transition_diagrams.py     — генератор диаграмм переходов
  work_mgn_negatives.py, plot_*.py    — скрипты расчётов и графиков
