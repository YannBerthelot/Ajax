#!/usr/bin/env bash
# Bisect a red slow-tier run (.github/workflows/slow.yml), when cheap:
#   slow_bisect.sh LAST_GREEN NODE [NODE ...]
# The failing tests are re-run at the head first: passing there, they are
# flaky and the bisect stops. Otherwise git bisect walks the first-parent
# commits (the merges) since the last green run; a commit where a test does
# not exist yet is skipped. Writes result=flaky or result=<first bad sha>.
set -euo pipefail
good=$1
shift
check() {
  uv sync --locked -q
  uv run pytest -q -p no:cacheprovider "$@" && return 0
  local code=$?
  # 4: a named test does not exist at this commit; 5: none collected.
  if [ "$code" -eq 4 ] || [ "$code" -eq 5 ]; then return 125; fi
  return 1
}
export -f check
if check "$@"; then
  echo "result=flaky" >> "$GITHUB_OUTPUT"
  exit 0
fi
git bisect start --first-parent HEAD "$good"
git bisect run bash -c 'check "$@"' _ "$@" || true
echo "result=$(git rev-parse refs/bisect/bad)" >> "$GITHUB_OUTPUT"
git bisect reset
