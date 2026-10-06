---
name: telegram-notify
description: Notify the owner in Telegram as most-queue agent. Use when a work block or epic slice completes, a long-running task (test suite, article build) finishes, a blocker needs the owner's decision, or the owner explicitly asks to receive a file (e.g. a PDF) in Telegram. Sends via the shared Telegram bot using credentials from the repo .env.
---

# Telegram-уведомления владельцу (most-queue agent)

## Когда отправлять

- Завершён рабочий блок / эпик — даже частично (что готово, что осталось).
- Закончилась долгая работа: прогон тестов, сборка статьи/документа.
- Блокер или вопрос, требующий решения владельца.
- Владелец явно попросил прислать текст или файл (например, PDF статьи) в Telegram.

Когда НЕ отправлять: промежуточные шаги, рутинные tool calls.

## Как отправлять текст

```bash
bash .claude/skills/telegram-notify/scripts/send.sh "текст"
```

Многострочный текст — через stdin:

```bash
bash .claude/skills/telegram-notify/scripts/send.sh <<'EOF'
✅ most-queue agent — Блок: <название>

Сделано:
- ...

Дальше: <следующий шаг>
EOF
```

## Как отправлять файл (PDF и т.п.)

```bash
bash .claude/skills/telegram-notify/scripts/send_document.sh /path/to/file.pdf "подпись (опционально)"
```

## Подпись агента

Бот общий для нескольких проектов владельца. В этом проекте текстовые
уведомления начинаются с **`most-queue agent —`**. Другие проекты (SARA,
PentestLens) используют тот же бот со своими подписями — не путать.

## Требования и границы

- `.env` в корне репозитория содержит `TELEGRAM_BOT_TOKEN` и
  `TELEGRAM_CHAT_ID` (общий бот). Скрипты читают их сами; токен никогда не
  выводить в чат, логи или транскрипт.
- Доставка не блокирует работу: при ошибке — одна повторная попытка, затем
  сообщить о неудаче доставки в финальном ответе.
- Содержимое .env (токены) в текст уведомления или подпись файла не включать.
