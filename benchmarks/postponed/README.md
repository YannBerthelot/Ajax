# Postponed speed-up measurements (not for main)

The before/after driver for the speed-up branches postponed on 2026-10-14:
perf/td3-actor-q1-only, perf/eval-while-loop, perf/eval-shape-skeleton,
perf/model-reference-sampling, perf/row-collector-single-trace,
perf/transformer-kv-cache, perf/redq-subset-targets,
perf/ppo-gae-associative, perf/udrl-stream-associative (and
docs/memory-costs). All were branched from main c97630d. Each lands only
if `agent_bench --compare` (or its micro-timing) shows a gain on a quiet
machine. `bench.sh` hard-codes the scratch paths of the session that made
it; point it at fresh checkouts of the base and of each branch.
