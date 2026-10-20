#!/usr/bin/env bash
# The runs of the Compile preview workflow (.github/workflows/compile-preview.yml),
# each on a compile-heavy subset of the fast tests:
#   produce ISA          compile the subset into an empty cache, $RUNNER_TEMP/cache
#   consume ISA REPLICA  run it again on that cache, moved under this runner's
#                        fingerprint, as a key without the CPU's flags would find it
#   levers REPLICA       without a cache, on one runner: as today, at XLA's
#                        optimisation level 0, and with 2 and 4 xdist workers
# ISA "avx2" caps XLA's code generation at AVX2 (every runner has it); "native"
# leaves it at the runner's own CPU. Each run leaves stats/NAME.json (the
# compile statistics, tests/compile_stats.py) and stats/NAME.meta.json.
set -euo pipefail
SUBSET=${SUBSET:-"tests/agents/test_agent_contract.py tests/agents/test_weight_decay.py
  tests/environments/test_wrapper_contract.py tests/agents/test_agent_pins.py
  tests/networks/test_memory.py"}
CPU=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | xargs || uname -m)
ROOT="$RUNNER_TEMP/cache"
mkdir -p stats

measure() {  # NAME XLA_FLAGS PYTEST_ARGS...: the subset, without coverage
  local name=$1 flags=$2 start=$SECONDS status=0
  shift 2
  XLA_FLAGS="$flags" COMPILE_STATS="stats/$name.json" uv run ${WITH[@]+"${WITH[@]}"} python -m pytest -q \
    -p no:cacheprovider -p tests.compile_stats -m "not slow" $SUBSET "$@" \
    > "stats/$name.log" 2>&1 || status=$?
  local warned
  warned=$(grep -c "could lead to SIGILL" "stats/$name.log" || true)
  printf '{"run": "%s", "cpu": "%s", "wall_s": %d, "status": %d, "isa_warnings": %d}\n' \
    "$name" "$CPU" $((SECONDS - start)) "$status" "$warned" > "stats/$name.meta.json"
  echo "$name on $CPU: $((SECONDS - start)) s, exit $status, $warned ISA warnings"
}

isa_flags() { [ "$1" = avx2 ] && echo "--xla_cpu_max_isa=AVX2" || echo ""; }
WITH=()
case $1 in
  produce)
    AJAX_JAX_COMPILE_CACHE_DIR=$ROOT measure "produce-$2" "$(isa_flags "$2")" ;;
  consume)
    own=$(AJAX_JAX_COMPILE_CACHE_DIR=$ROOT uv run python -c \
      "import os; from ajax._compile_cache import cache_dir_for, host_fingerprint as h
print(cache_dir_for(os.environ['AJAX_JAX_COMPILE_CACHE_DIR'], h()))")
    made=$(find "$ROOT" -mindepth 1 -maxdepth 1 -type d)
    [ "$made" = "$own" ] || mv "$made" "$own"
    AJAX_JAX_COMPILE_CACHE_DIR=$ROOT measure "consume-$2-$3" "$(isa_flags "$2")" ;;
  levers)
    export AJAX_NO_COMPILE_CACHE=1
    measure "levers-$2-today" ""
    measure "levers-$2-O0" "--xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true"
    WITH=(--with pytest-xdist==3.6.1)
    measure "levers-$2-xdist2" "" -n 2
    measure "levers-$2-xdist4" "" -n 4 ;;
esac
