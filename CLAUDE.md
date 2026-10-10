# CLAUDE.md

Guidance for Claude Code (and any agent) working in the Ajax repository.

## Commit requirement

**Nothing merges into main unless every check below is green.** The full
test suite is expensive (about an hour of CPU), so it runs in GitHub CI on
the pull request, not locally before each commit. Before committing, run
the cheap checks locally: pre-commit (step 1), `poetry run pytest
--collect-only -q`, and the test files that exercise what you changed.
Then push and let CI run the rest; fix anything it reports on the same
branch. The checks:

1. **pre-commit** — `poetry run pre-commit run --all-files`
   (ruff lint, ruff-format, mypy).
2. **Tests not marked slow + coverage** — `poetry run coverage erase`
   (parallel mode never removes old data files), then `poetry run
   coverage run -m pytest -m "not slow" --deselect
   tests/agents/test_probing.py --ignore=tests/probing`, then `poetry run
   coverage combine`
   (coverage measures subprocesses, one data file each), then `poetry
   run coverage report --fail-under=70`.
3. **Slow tests** — `poetry run pytest -m slow --deselect
   tests/agents/test_probing.py --ignore=tests/probing`.
4. **Probing tests** — `poetry run pytest tests/agents/test_probing.py
   tests/probing`.

`make ci` runs all four locally when you do want them. These are the
checks in `.github/workflows/ci.yml`, which runs steps 2-4 in parallel
jobs. Never merge while any of them is red. If a failure is
pre-existing and unrelated to the change, call it out explicitly instead
of merging over it.

See `CONTRIBUTING.md` for repository layout, the agent-implementation
template, and the Extension framework conventions.

## Refactoring & code-quality standards

Heavy or risky changes go on a dedicated branch — never committed
directly to a shared branch. Tests ship with new code: no new module
or function lands without a covering test. Full CI must be green
before a branch merges; never build new work on a red branch (if a
failure is pre-existing and unrelated, call it out explicitly instead
of merging over it).

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

### Refactoring rules

1. **Pin behaviour first.** Before moving code, a test must record what
   it does today (goldens, `tests/probing`, the downstream API pin).
2. **Restructure or change behaviour, never both** in one commit. A fix
   that moves numbers re-records its goldens in its own commit, saying why.
   Bitwise reproduction of earlier outputs is not a goal: the target is
   each paper's algorithm and performance. Exact-output pins are change
   detectors, not truth; a behaviour-neutral change that shifts them only
   by floating-point drift re-records them in the same PR.
3. **Small steps**, each PR green in CI before merging, and bisectable.
4. **One source of truth per piece of knowledge.** Descendants import
   the parent's maths. Merge only what is truly the same; abstract on the
   third occurrence, not the second.
5. **Delete before abstracting.** Dead code and unused options go first.
6. **Same problem, same solution:** one way to build a training loop, log,
   resume and fold extensions across all agents.
7. **No hidden global state:** RNG keys, config and caches are passed in.
8. **Code says what, comments say why.** Docstrings and comments that
   restate the code go; keep reasons, citations and deviations from the
   paper.
9. **Budget size up front.** State a line budget before a step and measure
   it after; code generated without a budget is presumed too long.
10. **Expand, then contract** for public names, checkpoints and config
    keys: add the new form, migrate the downstream projects, then remove
    the old.
11. **One concern per PR**, small enough to read.

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
