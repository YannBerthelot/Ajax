#!/usr/bin/env bash
# Recompute every committed table from the committed results and check that
# it equals the committed one (README.md, "Reproducing the tables").
#
#   bash reproduce_tables.sh [PYTHON]
#
# PYTHON: a python with numpy, scipy, matplotlib and PyYAML (the Ajax venv;
# default: $AJAX_PYTHON, else python3). Nothing is trained and nothing under
# results/ is written: the outputs go to a temporary directory, which is
# printed and kept.
#
# Runs compare.py on round 1 and on round 2's pairs E and R, and analyze3.py
# on round 3, then diffs compare.md / analysis3.md (every line but the "Root"
# line, which names the directory the tables were computed from: the scratch
# output directory for the committed tables, results/ here) and summary.json /
# analysis3.json (whole file) against results/. compare.py's provenance checks
# verify the commits the records name: the reference's 29eb964 (pinned in
# compare.py) and Ajax's AJAX_COMMIT below, the commit README.md says the
# committed results ran (at run time those checks confirm which code was run;
# here they confirm the records name the documented commits).
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1  # nothing written next to the scripts either
PY=${1:-${AJAX_PYTHON:-python3}}
R=$HERE/results
TMP=$(mktemp -d "${TMPDIR:-/tmp}/reference_comparison_tables.XXXXXX")

AJAX_COMMIT=5cb0738f3a2e127a3f8d64994eac3be1bc8c926e  # README.md, "Setup"

status=0
check() {  # check NAME COMMITTED_DIR NEW_DIR MD JSON
  local name=$1 old=$2 new=$3 md=$4 js=$5
  if diff <(grep -v '^Root' "$old/$md") <(grep -v '^Root' "$new/$md") > "$TMP/$name.md.diff" \
     && diff "$old/$js" "$new/$js" > "$TMP/$name.json.diff"; then
    echo "$name: identical ($md except its Root line, $js)"
  else
    echo "$name: DIFFERS (see $TMP/$name.*.diff)"
    status=1
  fi
}

run_compare() {  # run_compare NAME REF_ROOT AJAX_ROOT COMMITTED_DIR
  local name=$1 ref_root=$2 ajax_root=$3 old=$4
  "$PY" "$HERE/compare.py" --ref-root "$ref_root" --ajax-root "$ajax_root" \
    --root "$ref_root" --out "$TMP/$name" --seeds 0 1 2 \
    --ajax-commit "$AJAX_COMMIT" > "$TMP/$name.stdout"
  check "$name" "$old" "$TMP/$name" compare.md summary.json
}

run_compare round1 "$R/round1" "$R/round1" "$R/round1"
run_compare round2_pairE "$R/round2" "$R/round1" "$R/round2/pairE"
run_compare round2_pairR "$R/round1" "$R/round2" "$R/round2/pairR"

"$PY" "$HERE/analyze3.py" --root "$R/round3" --out "$TMP/round3" > "$TMP/round3.stdout"
check round3 "$R/round3" "$TMP/round3" analysis3.md analysis3.json

echo "outputs in $TMP"
exit $status
