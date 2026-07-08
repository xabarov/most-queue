# Инфраструктура проекта

> Как собирать, тестировать и выпускать Most-Queue. См. также [PROJECT.md](PROJECT.md) и [DOD.md](DOD.md).

## Окружение

- Python ≥ 3.9 (target-версии в конфиге black: 3.9–3.12).
- Зависимости пакета (runtime): `numpy`, `scipy>=1.13,<2.0`, `pandas`, `matplotlib`,
  `networkx`, `graphviz`, `pyyaml`, `tqdm`, `colorama` — см. `[project.dependencies]` в `pyproject.toml`.
- `requirements.txt` — полный freeze dev-окружения (jupyter, sphinx, pytest, twine, hatch и пр.),
  не runtime-зависимости пакета.
- Установка для разработки: `pip install -e .`

## Сборка и публикация

- Сборщик: **hatchling** (`[build-system]` в `pyproject.toml`).
- Версия задаётся вручную в `pyproject.toml` (`[project] version`).
- Публикация на PyPI: `hatch build` / `twine upload` (см. `deploy.txt` в корне).
- Собранные артефакты — в `dist/` (в git не коммитятся новые сборки без релиза).

## Тесты

- Фреймворк: **pytest**. Тесты — в `tests/` (интеграционные, «теория vs симуляция»)
  и `tests/units/` (модульные).
- Локальный запуск: `pytest tests/` из корня репозитория.
- Быстрый прогон без медленных тестов: `pytest -m "not slow"` (маркер `slow`
  объявлен в `pyproject.toml`).
- Общие параметры тестов — `tests/default_params.yaml` (интенсивности, число заявок,
  допуски сравнения теории и симуляции).
- Docker-прогон против опубликованной версии: `tests/run.sh` (собирает образ по
  `tests/Dockerfile`, ставит `most-queue` из PyPI и гоняет тесты). Для проверки
  локального кода использовать локальный pytest, а не Docker-вариант.

## CI

- CI-пайплайна в репозитории нет (`.github/workflows` отсутствует) — проверки запускаются
  локально. Прогон тестов перед коммитом — обязанность разработчика (см. [DOD.md](DOD.md)).

## Качество кода

- Форматирование: **black**, line-length 120; импорты — **isort** (profile black).
- Линтер: **pylint** (набор отключённых проверок — в `pyproject.toml`).
- Команды: `black most_queue tests`, `isort most_queue tests`, `pylint most_queue`.

## Документация

- `docs/` — markdown на русском; входная точка — `docs/README.md`.
- README.md в корне — английский, для PyPI/GitHub.
- В dev-окружении установлен Sphinx (в `requirements.txt`), но собранной sphinx-конфигурации
  в репозитории нет — актуальная документация ведётся в markdown.
- Jupyter-туториалы — `tutorials/` (перед коммитом очищать выводы, если они не нужны намеренно).

## Git

- Основная ветка: `main`, remote — `https://github.com/xabarov/most-queue`.
- Issue-трекер: GitHub Issues.
