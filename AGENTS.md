# Public showcase

Python 3.12+, uv, Jinja2, vanilla browser JavaScript. Preserve offline demo and public/private state boundaries.

Use `uv sync --locked`, `uv run pytest -q`, `uv run ruff check .`. Tests use synthetic inputs and mock providers; never run paid calls, send email, deploy, or access real credentials for validation.

Run demo in a fresh directory. Original deployment workflows are inert examples. Do not enable schedules or copy private state/fixtures/history into this snapshot.
