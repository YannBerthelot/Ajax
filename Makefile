SHELL=/bin/bash

# Strict CPU isolation -- required when running tests while a live experiment
# uses the GPU on this machine. CUDA_VISIBLE_DEVICES="" hides the GPU from
# the CUDA driver entirely so even CUDA init can't probe it. JAX_PLATFORMS
# (newer) + JAX_PLATFORM_NAME (older) are belt-and-suspenders.
CPU_ENV := CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu

.PHONY: ci ci-precommit ci-test ci-slow ci-probe test-cpu probe-cpu help

help:  ## Show this help message
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ---------------------------------------------------------------------------
# Canonical local-CI -- mirrors .github/workflows/ci.yml: the same commands,
# CPU environment and markers. CI splits ci-test across three runners and
# only adds reporting flags (--durations, -v). Use these instead of ad-hoc
# commands so the local checks match what merges.
# ---------------------------------------------------------------------------

ci: ci-precommit ci-test ci-slow ci-probe  ## Full local CI: pre-commit, tests not marked slow + coverage, slow tests, probing

ci-precommit:  ## pre-commit on all files (ruff lint + ruff-format + mypy)
	uv run pre-commit run --all-files

ci-test:  ## Tests not marked slow (deselects probing) + coverage >= 70
	uv run coverage erase
	$(CPU_ENV) uv run coverage run -m pytest -m "not slow" --ignore=tests/probing
	uv run coverage combine
	uv run coverage report --fail-under=70

ci-slow:  ## Tests marked slow (deselects probing), without coverage
	$(CPU_ENV) uv run pytest -m slow --ignore=tests/probing

ci-probe:  ## Cross-agent probing tests (the known-answer probes)
	$(CPU_ENV) uv run pytest tests/probing

# ---------------------------------------------------------------------------
# Convenience: fast CPU-only test runs that skip the coverage gate.
# ---------------------------------------------------------------------------

test-cpu:  ## Run tests on CPU only (no coverage gate)
	$(CPU_ENV) uv run pytest --tb=short --disable-warnings

probe-cpu:  ## Run probing on CPU only
	$(CPU_ENV) uv run pytest tests/probing -v
