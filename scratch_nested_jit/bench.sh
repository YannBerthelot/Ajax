#!/bin/bash
# bench.sh OUTDIR ROUNDS TRIALS AGENT... : agent_bench per NESTED_JIT mode
# (off, family; DQN also step), modes alternating, order flipped each round.
# CPU only. Run from anywhere inside the repository; nothing else should run
# on the machine meanwhile.
set -u
REPO=$(git rev-parse --show-toplevel)
mkdir -p "$1"; OUT=$(cd "$1" && pwd); ROUNDS=$2; TRIALS=$3; shift 3
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
cd "$REPO"
for round in $(seq 1 "$ROUNDS"); do
  for agent in "$@"; do
    modes="off family"; [ "$agent" = DQN ] && modes="off step family"
    if [ $((round % 2)) -eq 0 ]; then modes=$(echo $modes | tr ' ' '\n' | tac | tr '\n' ' '); fi
    for mode in $modes; do
      NESTED_JIT=$mode uv run --frozen python -W ignore benchmarks/agent_bench.py --tag "$mode" \
        --out "$OUT/$agent-$mode-$round.jsonl" --only "$agent" --trials "$TRIALS" 2>&1 | grep -E "^  $agent"
      uptime >> "$OUT/load.log"
    done
  done
done
echo DONE
