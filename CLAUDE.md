# CLAUDE.md

Guidance for Claude Code (and any agent) working in the Ajax repository.

## Commit requirement

**Every commit must pass tests, pre-commit, and the CI checks — verify
this BEFORE creating the commit.** All of the following must pass on the
state being committed:

1. **pre-commit** — `poetry run pre-commit run --all-files`
   (ruff lint, ruff-format, mypy).
2. **Test suite + coverage** — `poetry run coverage run -m pytest
   --deselect tests/agents/test_probing.py`, then
   `poetry run coverage report --fail-under=70`.
3. **Probing tests** — `poetry run pytest tests/agents/test_probing.py`.

These are exactly the checks in `.github/workflows/ci.yml`. Do not create
a commit while any of them is red. If a failure is pre-existing and
unrelated to the change, call it out explicitly instead of committing
over it.

See `CONTRIBUTING.md` for repository layout, the agent-implementation
template, and the composable-hook conventions.
