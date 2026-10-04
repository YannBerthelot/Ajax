#!/usr/bin/env bash
# Launch one round of the Ajax-vs-reference DreamerV3 CartPole comparison
# (README.md; PROTOCOL.md 3.6, PREREG2.md, PREREG3.md): one reference process
# per seed and one vmapped Ajax run over all seeds (as
# benchmarks/learning_checks.py runs a check), all in parallel, CPU only.
#
#   bash launch.sh --round {1,2,3} [--out DIR] [--smoke]
#                  [--reference-checkout DIR] [--reference-python PY]
#                  [--ajax-root DIR] [--ajax-python PY]
#
# Rounds (PREREG2.md, PREREG3.md):
#   1  run_reference.py         + run_ajax.py            seeds 0-2, 20000 rows, eval every 2000
#   2  run_reference_exact.py   + run_ajax_refdecode.py  seeds 0-2, 20000 rows, eval every 2000
#   3  run_reference_round3.py  + run_ajax.py            seeds 0-4, 24000 rows, eval every 400
# Every round evaluates on 10 episodes. --smoke (plumbing only): seeds 0 1,
# 1600 rows, eval every 400 on 2 episodes, tiny widths (the runners' --tiny).
#
# Defaults (HERE = this directory, REPO = the repository holding it):
#   --out                 HERE/.work/round<N>   (HERE/.work/round<N>_smoke with --smoke)
#   --reference-checkout  HERE/.work/dreamerv3_29eb964  (danijar/dreamerv3 at 29eb964)
#   --reference-python    HERE/.work/venv/bin/python    (the reference venv, README.md)
#   --ajax-root           REPO  (the Ajax checkout whose src/ is run; the committed
#                         results ran 5cb0738, see README.md)
#   --ajax-python         REPO/.venv/bin/python
# HERE/.work/ is git-ignored.
#
# Everything goes to OUT, which must not exist yet:
#   ref_s<S>/       reference outputs (run_reference*.py)  ref_s<S>.log  stdout+stderr
#   ajax/           Ajax outputs (run_ajax*.py)             ajax.log      stdout+stderr
#   <name>.DONE     written when the process exits: its exit code (0 = success)
#   <name>.nohup    output of the wrapper itself (normally empty)
#   pids.txt        wrapper pids (the python process is the wrapper's child)
#   launch_info.txt provenance (time, OS and architecture, load, revisions and
#                   dirty flags, flags); no host name and no absolute path:
#                   locations relative to named roots (README.md "Recorded
#                   locations")
# The runs are finished when every ref_s<S>.DONE and ajax.DONE exists. The
# script prints the analysis command for the round (compare.py, analyze3.py).
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)

ROUND=""
FULL=""
SMOKE=0
REF="$HERE/.work/dreamerv3_29eb964"
RPY="$HERE/.work/venv/bin/python"
AJ="$REPO"
APY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --round) ROUND=$2; shift 2 ;;
    --out) FULL=$2; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --reference-checkout) REF=$2; shift 2 ;;
    --reference-python) RPY=$2; shift 2 ;;
    --ajax-root) AJ=$2; shift 2 ;;
    --ajax-python) APY=$2; shift 2 ;;
    -h|--help) sed -n '2,38p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done
APY=${APY:-$AJ/.venv/bin/python}

case "$ROUND" in
  1) REF_RUNNER=run_reference.py;        AJAX_RUNNER=run_ajax.py
     SEEDS=(0 1 2);       ROWS=20000; EVERY=2000 ;;
  2) REF_RUNNER=run_reference_exact.py;  AJAX_RUNNER=run_ajax_refdecode.py
     SEEDS=(0 1 2);       ROWS=20000; EVERY=2000 ;;
  3) REF_RUNNER=run_reference_round3.py; AJAX_RUNNER=run_ajax.py
     SEEDS=(0 1 2 3 4);   ROWS=24000; EVERY=400 ;;
  *) echo "--round must be 1, 2 or 3 (see --help)" >&2; exit 2 ;;
esac
EPISODES=10
TINY=()
SUFFIX=""
if [ "$SMOKE" = 1 ]; then
  SEEDS=(0 1); ROWS=1600; EVERY=400; EPISODES=2; TINY=(--tiny); SUFFIX=_smoke
fi
FULL=${FULL:-$HERE/.work/round$ROUND$SUFFIX}

# Thread limits. Measured on the machine of the committed runs (14 cores,
# shared; load 20-39 from other jobs):
# * reference (jax 0.4.26): "--xla_cpu_multi_thread_eigen=false" is the only
#   flag that limits XLA's CPU threads (single-threaded Eigen ops);
#   "intra_op_parallelism_threads=N" has no effect. A/B, two full-size
#   reference processes side by side at load ~35: single-thread 2.02 s per
#   update, default threading 1.60 s per update. Default threading is faster
#   even when shared, so it is the default here; export
#   REF_XLA_FLAGS="--xla_cpu_multi_thread_eigen=false" to pin each reference
#   process to one core instead (more predictable, ~25 % slower).
# * Ajax (jax 0.7.2, thunk runtime): neither flag limits threads; left default.
# Threading changes speed only, not the algorithm.
REF_XLA_FLAGS="${REF_XLA_FLAGS:-}"
AJAX_XLA_FLAGS="${AJAX_XLA_FLAGS:-}"

for f in "$RPY" "$APY" "$HERE/$REF_RUNNER" "$HERE/$AJAX_RUNNER" "$HERE/run_reference.py" \
         "$HERE/run_ajax.py" "$HERE/train_instrumented.py" "$HERE/cartpole_env.py" \
         "$HERE/namemap.py" "$HERE/provenance.py" "$REF/dreamerv3/main.py" "$AJ/src/ajax/__init__.py"; do
  [ -e "$f" ] || { echo "missing: $f (see --help and README.md)" >&2; exit 1; }
done
REF=$(cd "$REF" && pwd)
AJ=$(cd "$AJ" && pwd)
if [ -e "$FULL" ]; then
  echo "$FULL exists (the stock checkpoint logic would resume from ref_s*/; run_ajax.py refuses an existing ajax/): move it away first" >&2
  exit 1
fi
mkdir -p "$FULL"
FULL=$(cd "$FULL" && pwd)

# HERE relative to its repository (named <repo>), else its directory name.
case "$HERE/" in
  "$REPO/"*) HERE_REC="<repo>/${HERE#"$REPO/"}" ;;
  *) HERE_REC=$(basename "$HERE") ;;
esac
dirty() {  # dirty DIR: "yes" if git status lists anything under DIR, else "no"
  if [ -n "$(cd "$1" && git --no-optional-locks status --porcelain -- .)" ]; then
    echo yes
  else
    echo no
  fi
}
{
  echo "launched: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "host: $(uname -sm)"
  echo "load: $(uptime | sed 's/.*load average/load average/')"
  echo "round: $ROUND$( [ "$SMOKE" = 1 ] && echo ' (smoke)'); runners $REF_RUNNER, $AJAX_RUNNER (in $HERE_REC)"
  echo "reference: <reference-checkout> @ $(git -C "$REF" rev-parse HEAD); dirty: $(dirty "$REF")"
  echo "ajax: <ajax-root> @ $(git -C "$AJ" rev-parse HEAD); uncommitted diff vs HEAD sha1:" \
       "$(git --no-optional-locks -C "$AJ" diff HEAD -- src | shasum | cut -d' ' -f1);" \
       "src dirty: $(dirty "$AJ/src")"
  # The harness (this directory: the runners, train_instrumented.py,
  # cartpole_env.py, ...) is code that ran too; it need not be in <ajax-root>.
  if HARNESS_COMMIT=$(git -C "$HERE" rev-parse HEAD 2>/dev/null); then
    echo "harness: $HERE_REC @ $HARNESS_COMMIT; dirty: $(dirty "$HERE")"
  else
    echo "harness: $HERE_REC (not in a git repository)"
  fi
  echo "REF_XLA_FLAGS='$REF_XLA_FLAGS' AJAX_XLA_FLAGS='$AJAX_XLA_FLAGS'"
  echo "rows $ROWS, every $EVERY, eval episodes $EPISODES, seeds ${SEEDS[*]}"
} > "$FULL/launch_info.txt"

# launch NAME CWD CMD...: runs CMD in CWD under nohup, stdout+stderr to
# $FULL/NAME.log; when CMD exits, its exit code is written to $FULL/NAME.DONE
# (via a .tmp file and mv, so a DONE file is never seen half-written).
launch() {
  local name=$1 cwd=$2
  shift 2
  nohup /bin/bash -c '
    name=$1; cwd=$2; full=$3; shift 3
    if cd "$cwd"; then
      "$@" > "$full/$name.log" 2>&1
      code=$?
    else
      code=97
    fi
    echo "$code" > "$full/$name.DONE.tmp" && mv "$full/$name.DONE.tmp" "$full/$name.DONE"
    exit "$code"
  ' "launch_$name" "$name" "$cwd" "$FULL" "$@" > "$FULL/$name.nohup" 2>&1 < /dev/null &
  echo "$name wrapper pid $!" | tee -a "$FULL/pids.txt"
}

# PYTHONDONTWRITEBYTECODE=1: nothing (not even __pycache__) is written under the
# reference checkout, the Ajax checkout or this directory.

# Reference: one process per seed (cwd = reference checkout, as main.py).
for s in "${SEEDS[@]}"; do
  launch "ref_s$s" "$REF" \
    env JAX_PLATFORMS=cpu XLA_FLAGS="$REF_XLA_FLAGS" PYTHONDONTWRITEBYTECODE=1 \
        DREAMERV3_REF="$REF" PYTHONPATH="$REF:$HERE" \
    "$RPY" "$HERE/$REF_RUNNER" --seed "$s" --rows "$ROWS" --eval-every "$EVERY" \
      --eval-episodes "$EPISODES" --out "$FULL/ref_s$s" ${TINY[@]+"${TINY[@]}"}
done

# Ajax: one vmapped run over the seeds (cwd = FULL so nothing is written under
# the Ajax checkout).
launch ajax "$FULL" \
  env JAX_PLATFORMS=cpu XLA_FLAGS="$AJAX_XLA_FLAGS" PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH="$AJ/src" AJAX_SRC="$AJ/src" \
  "$APY" "$HERE/$AJAX_RUNNER" --seeds "${SEEDS[@]}" --rows "$ROWS" --log-every "$EVERY" \
    --eval-episodes "$EPISODES" --out "$FULL/ajax" ${TINY[@]+"${TINY[@]}"}

names=""
for s in "${SEEDS[@]}"; do names="$names ref_s$s.DONE"; done
echo "outputs and logs: $FULL (ref_s*.log, ajax.log; progress: ref_s*/progress.jsonl)"
echo "finished when these exist in $FULL:$names ajax.DONE"
echo "then (Ajax venv):"
if [ "$ROUND" = 3 ] && [ "$SMOKE" = 0 ]; then
  echo "  $APY $HERE/analyze3.py --root $FULL"
elif [ "$ROUND" = 2 ] && [ "$SMOKE" = 0 ]; then
  # PREREG2.md: each round-2 side is compared with the other side of round 1.
  echo "  # pair E (exact expectation on both sides): round-2 reference vs round-1 Ajax"
  echo "  $APY $HERE/compare.py --ref-root $FULL --ajax-root ROUND1_OUT --out $FULL/pairE \\"
  echo "    --ajax-commit ROUND1_AJAX_COMMIT"
  echo "  # pair R (reference expectation on both sides): round-1 reference vs round-2 Ajax"
  echo "  $APY $HERE/compare.py --ref-root ROUND1_OUT --ajax-root $FULL --out $FULL/pairR \\"
  echo "    --ajax-src $AJ/src"
else
  smoke_flag=""
  [ "$SMOKE" = 1 ] && smoke_flag=" --smoke"
  echo "  $APY $HERE/compare.py --root $FULL --seeds ${SEEDS[*]}$smoke_flag \\"
  echo "    --ajax-src $AJ/src"
fi
