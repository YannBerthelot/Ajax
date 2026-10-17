#!/bin/zsh
# Record a reference comparison. Usage, from the repository root:
#   benchmarks/reference/record.sh reference AGENT [SEEDS] [SB3_PYTHON]
#   benchmarks/reference/record.sh ajax AGENT [SEEDS]     (both arms)
# References are recorded once (they do not depend on Ajax); the Ajax arms
# are re-run on demand. SB3 seeds run as 8 single-thread processes; JAX
# runs train every seed in one program.
set -e
cd ${0:A:h}
mkdir -p results
what=$1 agent=$2 seeds=${3:-16} sb3_python=${4:-python}
ref=$(uv run --no-sync python -c "from refcmp_specs import COMPARISONS as C; print(C['$agent'].reference)")
if [[ $what == reference && $ref == sb3 ]]; then
  per=$(( (seeds + 7) / 8 ))
  for (( first = 0; first < seeds; first += per )); do
    n=$(( seeds - first < per ? seeds - first : per ))
    $sb3_python refcmp_sb3.py $agent $first $n >> results/sb3.jsonl &
  done
  wait
elif [[ $what == reference ]]; then
  uv run --no-sync python refcmp_jax.py $agent 0 $seeds >> results/$ref.jsonl
else
  for arm in matched defaults; do
    uv run --no-sync python refcmp_ajax.py $agent $arm 0 $seeds >> results/ajax.jsonl
  done
fi
