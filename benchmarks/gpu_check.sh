#!/usr/bin/env bash
# Does an NVIDIA GPU train Ajax's agents faster than the same machine's CPU,
# and from how many seeds? One command on Linux with an NVIDIA GPU, from a
# checkout:
#
#     bash benchmarks/gpu_check.sh
#
# Pre-registered question: on one NVIDIA GPU, at which seed counts does each
# agent run more steps per second than on the same machine's CPU?
# agent_bench times PPO, PQN, SAC and DQN at 1, 16 and 128 seeds on the GPU
# and at 1 and 16 on the CPU (128 seeds on a CPU would take hours), 3 trials
# each, into benchmarks/gpu/; device_report.py then answers the question.
# About 1.5-2.5 hours.
set -euo pipefail
cd "$(dirname "$0")/.."

out=benchmarks/gpu
agents=(PPO PQN SAC DQN)
mkdir -p "$out"

uv sync --extra cuda
# Stop before hours of timing if JAX does not see the GPU.
uv run python -c 'import jax; assert jax.default_backend() == "gpu", jax.devices()'

for seeds in 1 16 128; do
    uv run python benchmarks/agent_bench.py --only "${agents[@]}" \
        --n-seeds "$seeds" --trials 3 --tag gpu --out "$out/gpu-$seeds-seeds.jsonl"
done
for seeds in 1 16; do
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run python benchmarks/agent_bench.py \
        --only "${agents[@]}" --n-seeds "$seeds" --trials 3 --tag cpu \
        --out "$out/cpu-$seeds-seeds.jsonl"
done

uv run python benchmarks/device_report.py "$out"/*-seeds.jsonl | tee "$out/report.md"
