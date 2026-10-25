#!/bin/bash
# hlo.sh OUTDIR : the optimised HLO stats of each agent's train program per mode.
set -u
REPO=$(git rev-parse --show-toplevel)
mkdir -p "$1"; OUT=$(cd "$1" && pwd)
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
cd "$REPO"
for agent in DQN SAC TD3 PPO PQN; do
  modes="off family"; [ "$agent" = DQN ] && modes="off step family"
  for mode in $modes; do
    NESTED_JIT=$mode uv run --frozen python -W ignore scratch_nested_jit/hlo_dump.py "$agent" "$OUT/$agent-$mode" 2>&1 | tail -1 | cut -c1-400
  done
done
