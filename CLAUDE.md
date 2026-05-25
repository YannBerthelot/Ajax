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
template, and the Extension framework conventions.

## Refactoring & code-quality standards

Heavy or risky changes go on a dedicated branch — never committed
directly to a shared branch. Tests ship with new code: no new module
or function lands without a covering test. Full CI must be green on
every commit; never stack a red commit on a red parent (if a failure
is pre-existing and unrelated, call it out explicitly instead of
committing over it).

Improve quality as you go — readability, refactoring, performance —
but never at the expense of clarity or reliability. No tricks, no
hacks: find the root cause and fix it. If a change is a deliberate
workaround, say so explicitly in the commit message and in a code
comment so future readers know.

Treat experiments as opportunities to improve the tools, not to
overfit them. An improvement prompted by one experiment must be
general; do not bake experiment-specific assumptions into the
framework. If a feature only makes sense for one paper, it belongs
as an Extension or a downstream script — not in the core agent.

We are not in a rush; do things as cleanly as possible.

### Extensions are self-contained

An :class:`Extension` is a composable mutation of an agent's training
loop. The base agent must NOT hardcode any extension's hyperparameters
or know about any specific extension's existence. Extensions own their
own frozen-dataclass fields and (when they need agent context) carry a
:meth:`bind_to_agent` method to attach it without touching the agent.

If you find yourself adding a per-extension kwarg to an agent's
``__init__`` to make a new feature work, **stop**: make the feature
an Extension instead. The agent surface is closed — it accepts only
the algorithm's own hyperparameters plus ``extensions=()``. Past
violations (the SAC legacy back-compat shim that grew ``__init__`` to
~100 kwargs) were the worst offender; Phase 5 of the architecture
rework deleted them. Don't reintroduce that pattern.

### Performance no-regression guardrail

Cross-agent perf baseline lives in ``benchmarks/agent_baseline.jsonl``;
per-phase deltas in ``benchmarks/agent_phaseN.jsonl``. Any agent that
regresses beyond ~10% on the standardized ``benchmarks/agent_bench.py``
run (fixed env, fixed n_envs / n_timesteps / seed) is investigated to
root cause **before** the change lands. Small explained deltas are
recorded in ``PERFORMANCE_REPORT.md``. Benchmarks are CPU-only by
default — never compete with running GPU experiments.
