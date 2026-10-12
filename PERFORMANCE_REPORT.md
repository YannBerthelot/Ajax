# Ajax Performance Report

Consolidated summary of memory and speed improvements landed across
four audit rounds. Source data and per-round narrative live in
[PERFORMANCE_LOG.md](PERFORMANCE_LOG.md). This document is the
executive summary.

## Headline numbers

The biggest single win is the persistent JAX compile cache (patch #10),
which dwarfs every per-step micro-optimisation combined.

It only applies to programs without host callbacks: JAX never writes a
program that calls `jax.debug.callback` to the persistent cache, so a
run with logging enabled recompiles on every fresh process. The numbers
below are from benchmark runs with logging off. The cache keeps one
subdirectory per host fingerprint (CPU, Python, JAX and jaxlib
versions; see [`ajax/_compile_cache.py`](src/ajax/_compile_cache.py)).

|                              | before | after  | delta            |
| ---------------------------- | -----: | -----: | :--------------- |
| **Wall-clock (steady-state, stress_sac)** | 11.98s | 11.33s | **-5.4%**        |
| **Wall-clock (warm-process, p3dcircle)**  | 16.26s | 3.35s  | **-79%**         |
| **Compile + warmup (warm-process, p3dcircle)** | 21.5s  | 4.9s   | **-77%**         |
| **Peak GPU memory (warm-process, p3dcircle)** | 302 MB | 15 MB  | **-95%**         |
| **Seed throughput (Pendulum, N=32)**       | -      | 28x    | (scaling)        |
| **Seed throughput (Plane3DCircle, N=100)** | -      | 86x    | (scaling)        |

"steady-state" = single fresh-process run, JIT cache hot from the
warmup in the same process. "warm-process" = a second invocation of
the same script, so `import ajax` lands the disk-cache hit before any
training starts. The two columns answer different questions: the
first is "how fast is each training step now"; the second is "how
fast does a relaunch of the same program reach steady state". A
distinct HPO trial is another program: its hyperparameters are
closed over as Python constants and compiled in, so it misses the
cache (see [HPO trials](#hpo-trials-scaled-worse-than-the-benches-2026-05-08)).

## What changed and why it helped

### Memory wins

| Patch | Where it lives | Effect | Workload-conditional? |
| ----- | -------------- | ------ | --------------------- |
| #1 obs-norm shape `(n_envs,*) -> (1,*)` | `init_agent_obs_norm`, `update_obs_norm` ([obs_norm.py](src/ajax/environments/obs_norm.py)) | n_envs× drop on running stats; redundant per-call `nanmean` reductions removed | Material when n_envs and obs_dim both grow (image obs, large vmap). Invisible at Pendulum scale. |
| #6 network init shape `(n_envs,*) -> (1,*)` | `get_initialized_actor_critic`, `get_initialized_critic` ([networks.py](src/ajax/networks/networks.py)) | One-shot allocation drop at agent build time | Material when obs is high-dim (images) or n_envs is large. |
| #4 carry-only inner update scan | `final_aux_scan` ([perf_utils.py](src/ajax/perf_utils.py)), used by SAC's, REDQ's and DQN's inner updates | Drops `[num_critic_updates, *aux]` of per-step diagnostics that get discarded | Material under vmap-over-seeds with large `num_critic_updates` (REDQ-style UTD>1). |
| #5 `donate_argnames` on top-level train jit | `build_resumable_train` ([perf_utils.py](src/ajax/perf_utils.py)) | XLA reuses caller's `initial_state` buffer instead of cloning | Material on resume-from-checkpoint paths. No-op (and free) on init paths. |
| **#10 persistent JAX compile cache** | [`ajax/_compile_cache.py`](src/ajax/_compile_cache.py) | **Eliminates ~290 MB of XLA transient compile-time scratch** | Material on every fresh process that runs a program already cached: relaunches of the same configuration, repeated agent rebuilds. Not distinct HPO trials. |

The peak-memory column on benchmark rows tracks the per-fresh-process
peak, which is dominated by JIT compilation transients on first
invocation. With the disk cache hit, those transients never allocate,
so the steady-state memory floor (around 15 MB on Plane3DCircle)
becomes the actual peak.

### Speed wins

| Patch | Where it lives | Effect | Workload-conditional? |
| ----- | -------------- | ------ | --------------------- |
| #2 gate redundant `q_preds_for_var` | `update_value_functions` ([SAC/train_SAC.py](src/ajax/agents/SAC/train_SAC.py)) | Pure SAC saves a full `[N_critics, B]` critic forward per critic step | Skipped unless an extension folds `on_target` (IBRL, LCBGatedBootstrap, CriticBlend, MCVarianceCorrection). Bigger relative win at higher `num_critics`. |
| #7 `nan_safe=False` opt-in for agent obs | `online_normalize` ([utils.py](src/ajax/utils.py)), `update_obs_norm` ([obs_norm.py](src/ajax/environments/obs_norm.py)) | `mean` instead of `nanmean` per call (skips the per-element NaN-mask compare) | Always on. AVG keeps `nan_safe=True` because of its NaN-sentinel `G_return`. |
| #9 reuse buffer-mix sample indices | deleted with the buffer-mix path (88124c5) | One `randint` call replaced N `random.choice` calls inside `tree.map` | Only triggered on the IBRL/expert buffer-mix path. |
| **#10 persistent JAX compile cache** | [`ajax/_compile_cache.py`](src/ajax/_compile_cache.py) | **~17s of HLO compile saved per fresh process when shapes match** | Hits on second and subsequent invocations of the same code. |

The per-step patches have preconditions. A configuration whose
extensions fold `on_target` (IBRL, for one) with one critic update per
step fires neither #2 nor #4, so its projected steady-state gain is about
1-2% instead of SAC's 5%. edge-qa was projected there too in 2026-05 (its
expert-Q path kept #2's forward); its `EDGEExploration` folds no
`on_target`, so #2 now fires for it. The compile cache and the seed
scaling below apply to every variant.

### Scaling wins (already present, now verified)

The vmap-over-seeds path in `ActorCritic.train`
([base.py](src/ajax/agents/base.py)) was structurally correct from the
start. Empirically the scaling is near-ideal up to compute saturation:

|             | Pendulum-v1            | Plane3DCircle |
| ----------- | ---------------------- | ------------- |
| N=1   wall  |   9.81s                | 15.77s        |
| N=2   wall  |  10.37s   (1.89x tput) | -             |
| N=4   wall  |  10.51s   (3.73x tput) | 17.46s (3.61x tput)  |
| N=8   wall  |  10.23s   (7.67x tput) | -             |
| N=16  wall  |  10.43s  (15.05x tput) | 18.75s (13.46x tput) |
| N=32  wall  |  11.08s  (28.34x tput) | 19.12s (26.40x tput) |
| N=50  wall  |  -                     | 18.99s (41.51x tput) |
| N=100 wall  |  -                     | 18.37s (85.86x tput) |

SAC with the edge-qa options (LCB gating, expert-state augmentation, obs
norm, a stub expert) scaled the same way on Plane3DCircle: 83% of ideal
at N=100 (16.42 s at N=1, 19.77 s at N=100), about 12 MB per seed.

Operational guidance from this:

- **More seeds is cheaper per seed.** Per-seed time at N=100 on
  Plane3DCircle is half that of N=50 (0.184s vs 0.380s). The fixed
  ~15s of compile + dispatch + kernel-launch overhead amortises
  across the parallel seeds. Do not run small-N sweeps to "save GPU";
  you are paying full overhead for fewer results.
- **Memory caps the seed count, not compute.** On Plane3DCircle the
  marginal cost is roughly 10-11 MB per seed past N=16. On a 35.7 GB
  GPU that leaves headroom for many hundreds of seeds before OOM.
  Wall clock stays nearly flat well past the point where naive
  intuition expects saturation.

### HPO trials scaled worse than the benches (2026-05-08)

A scan of past HPO trials (AjaxExperiments' `hp_results/`; the script,
`benchmarks/hpo_seed_scan.py`, was removed in 2026-10 and is in git
history) compared the cost per seed-step at N=20 (phase 1) and N=50
(phases 2-3), at width 512 with 4 critics. Of 13 (env, method) pairs,
as the N=50/N=20 ratio, 4 got cheaper (plane3dcircle sac 0.84, ibrl
0.63, jsrl 0.53; cheetahrun jsrl 0.59), 2 stayed flat (plane3dcircle
edge-qa 1.01, fourtank sac 0.94) and 7 got dearer (plane3dcircle
jsrl_curriculum 1.81, residual 1.12; cheetahrun sac 1.72, edge-qa 1.27;
glassfurnace edge-qa 1.53, residual 1.27; fourtank edge-qa 1.15). The
likely cause is a memory-bandwidth-bound regime once the per-seed working
set (about 8 MB of parameters per seed) outgrows the GPU cache; whether a
configuration saturates depends on how XLA fuses its shapes. This
supports HPO's 20 seeds (50 in the final phase): measure the
per-seed-step cost before a large-N study. None of the patches here
changes this curve, and the compile cache does not shorten a distinct
trial: its hyperparameters are compiled in, and a trial that logs is
never cached. Levers not landed: a smaller width or fewer critics (the
HPO explores both), bf16 (rejected, below), and sharding the seeds
across devices.

## What did not pan out

| Attempted | Why it was reverted |
| --------- | ------------------- |
| Cascade obs-norm shape into env-side `init_norm_info` ([wrappers.py](src/ajax/wrappers.py)) | Brax `VmapWrapper.step` calls `jax.vmap(env.step)(state, action)` over axis 0 of the entire env state. Env-side stats nested in that pytree must keep leading axis `n_envs`, otherwise vmap raises "inconsistent sizes". The narrow agent-side fix is fine because agent state is not vmapped by Brax. |
| Patch #4 (no-ys scan) on PPO / APO / AVG / ASAC | Their inner update scans intentionally aggregate the full ys axis (`flatten` then `mean`) for logging. Converting would silently change reported metrics. Left as-is. |
| bf16 mixed precision, patches #11-#12 (2026-05-08): bf16 compute for the critic, then the actor and critic (`AJAX_BF16_NETS`), and bf16 parameter storage behind an fp32 Adam state (`AJAX_BF16_PARAMS`) | Removed with its env vars and `benchmarks/convergence_check.py` (in git history). Plane3DCircle at the HPO config: bf16 compute saved 4.4% wall at N=20 and 7.7% at N=50, and 1.8-2.9% memory. bf16 parameters saved 26-27% peak memory but did not learn: updates below bf16's epsilon round to zero (Pendulum 30k, 8 seeds: mean -1307 against fp32's -568; Plane3DCircle 300k, 20 seeds: 0 seeds above -9000 against 14). bf16 compute matched fp32 on Pendulum (-600) and beat it on Plane3DCircle (17 of 20 seeds above -9000 against 14, median -3990 against -5928), within the seed spread (std about 3150) and itself unexplained. Decision: fp32 everywhere; single-digit wall-clock gains do not justify a numerical variable in precision-sensitive comparisons. |
| Fusing the two `predict_value` calls of the edge-qa LCB scores into one doubled batch (round 5, 2026-05-08) | XLA already fused them: -1.8% to +3.8% at N=1 to 100, memory unchanged; the explicit concatenate adds a temporary ([log](PERFORMANCE_LOG.md#round-5-edge-qa-lcb-predict-pair-fuse-reverted-2026-05-08)). |

## How wins were verified

All numbers came from
[`benchmarks/perf_bench.py`](benchmarks/perf_bench.py), one fresh
Python process per measurement so the JIT cache and the XLA memory
allocator both start cold inside the process. Scenarios:

- `pure_sac`: Pendulum-v1, n_envs=4, num_critics=2, 64-wide nets.
  Cheapest; smoke-test scenario.
- `obs_norm_sac`: same but `normalize_obs_running=True`. Exercises
  the agent obs-norm path.
- `stress_sac`: Pendulum-v1, n_envs=64, num_critics=10, 256-wide nets,
  obs norm on. Headline-number scenario.
- `stress_sac_utd`: same, with `num_critic_updates=10` (REDQ-style).
- `p3dcircle_sac`: target_gym `Plane3DCircle` (17-dim obs, 3-dim
  action), 256-wide nets. Used for cache + seed-scaling tests.

Reproduce a single before/after comparison:

```bash
git stash push -- <patched files>
python benchmarks/perf_bench.py --tag baseline   --scenario stress_sac --trials 3 --n-envs 64
git stash pop
python benchmarks/perf_bench.py --tag patched    --scenario stress_sac --trials 3 --n-envs 64
python benchmarks/seed_scaling_report.py benchmarks/results.jsonl   # for seed sweeps
```

Reproduce the cold-vs-warm cache result:

```bash
rm -rf ~/.cache/ajax/jax_compile_cache
python benchmarks/perf_bench.py --tag cache_cold --scenario p3dcircle_sac
python benchmarks/perf_bench.py --tag cache_warm --scenario p3dcircle_sac
```

## Playground fresh auto-reset (2026-10-02)

`build_env_from_id(env_id, fresh_reset=True)` swaps playground's
`BraxAutoResetWrapper` (every episode of env `i` restarts from one cached
state) for `ajax.wrappers.FreshAutoResetWrapper` (a new initial state per
episode). Fresh resets are the default since 2026-10-03, so the costs
below now apply to every playground run; `fresh_reset=False` restores the
cached behaviour and its timings. CPU, `JAX_PLATFORMS=cpu`, MJX
`impl="jax"`; measured while an unrelated CPU job used ~5.5 cores, so
single-digit-% deltas are noise.

Env step alone (zero actions, episode length 1000, vmapped over 2 seeds as
every Ajax agent runs), ms per env step:

| env | n_envs | cached (default) | fresh | playground `full_reset=True` |
|---|---|---|---|---|
| CheetahRun | 1 | 0.073 | 0.076 | 13.26 |
| CheetahRun | 16 | 0.33-0.38 | 0.43-0.46 | 60.7 |
| CartpoleBalance | 1 | 0.005 | 0.006 | 0.007 |

CheetahRun's reset runs a 200-step stabilisation, so a fresh episode start
costs ~200 steps of physics. Fresh mode pays it once per episode (the
16-env delta is mostly that: ~0.06 of the ~0.09 ms/step); `full_reset=True`
pays it every step, and in Ajax's stack it is also incorrect (on done steps
it zeroes `info["truncation"]` and overwrites `info["final_obs"]` with the
reset observation). Two gating designs were measured and rejected:
`lax.cond` on `done.any()` lowers to `select` under the seed vmap and costs
as much as `full_reset` (9.25 ms/step, CheetahRun, 1 env); a `while_loop`
whose body closes over a constant key gets the reset hoisted out of the loop
by XLA (13.24 ms/step). `_call_if_any` threads the key through the loop
carry to avoid the latter.

SAC end to end, execution only (train compiled once, min of 3 runs), 1 env,
3000 steps, `learning_starts=500`, ms per iteration: CartpoleBalance 2.80
cached / 2.70 fresh (noise), CheetahRun 3.51 / 3.58 (+1.9%). Compile plus
first run grows, because a reset is now compiled into the training and eval
loops: CartpoleBalance 18.0 -> 22.4 s, CheetahRun 36.1 -> 58.7 s per cold
process (the persistent compile cache amortises it across processes).
`benchmarks/agent_bench.py` does not cover this: it only runs gymnax envs,
whose code path the change does not touch.

## Brax auto-reset: reset only when an env is done (2026-10-03)

`ajax.wrappers.AutoResetWrapper` (Ajax's brax stack, `_build_brax_env`)
evaluated a full batched reset on every env step and discarded it unless
some env was done. Its comment claimed JAX reuses a cached result for an
unchanged seed; under jit nothing is cached, and the reset is part of every
compiled step. It is now gated with `_call_if_any` (the `while_loop` of the
playground fresh reset above, with an `advance` hook so the seed stream is
unchanged). `build_env_from_id(..., differentiable_reset=True)` keeps the
old reset-every-step path for reverse-mode gradients through the reset
itself (see `build_env_from_id`).

CPU, `JAX_PLATFORMS=cpu`, brax `generalized` backend, 16 envs vmapped over 2
seeds, zero actions. Measured while unrelated jobs kept the load average at
15-35 on 14 cores: whole-rollout timings swung by +-15% between runs (a
never-reset variant once came out slower than the reset-every-step one), so
the robust numbers below are the minimum over 500-1500 interleaved calls of
one jitted `step`, with XLA single-threaded, plus XLA's static cost
analysis, which load cannot affect. ms per call:

| env | old (reset every step) | gated, no env done | gated, envs done | never reset | reset alone |
|---|---|---|---|---|---|
| inverted_pendulum | 0.737 | 0.717 | 0.758 | 0.684 | 0.133 |
| ant | 3.61 | 3.76 | 3.88 | 3.75 | 0.467 |

The reset is 31% (ant) and 50% (inverted_pendulum) of a step's flops in the
old step (XLA cost analysis: 4.72e6 vs 3.25e6 flops with and without it on
ant, 2.68e5 vs 1.34e5 on inverted_pendulum), with similar shares of memory
traffic. On CPU at this batch size, though, the brax step is bound by
per-op overhead rather than arithmetic, and the reset is ~7-13% of the wall
time (the "reset alone" call includes dispatch). Gating removes that on
steps where no env is done and adds ~3% on steps where it runs. On ant the
~10% effect sits within the remaining noise (multi-threaded runs gave old
5.27 / gated-no-done 4.42 / never 4.92 ms). The net gain depends on how
many steps contain a done somewhere among `n_envs` x seeds: 73%
(inverted_pendulum) and 42% (ant) of steps with zero actions, close to 0
once a policy keeps episodes to the time limit. GPU was not measured; the
flops share suggests a larger gain there, where arithmetic counts more.
XLA compile time is unchanged within noise (2-4 s for a 1000-step rollout).

Exactness: the seed stream, done flags, step counters and reset
observations are bit-identical to the old wrapper. Brax states match to
float32 rounding only: inside the loop body XLA compiles the fields that
`pipeline.init` derives from positions (centre of mass, inertia, ...)
differently, moving them by ~1 ulp, which later steps carry on. On chaotic
contact dynamics that compounds: over 1000 zero-action ant steps the two
seeds had 265 and 239 steps containing a done instead of 268 and 246. Runs are therefore not
bitwise reproducible against the old wrapper.
`differentiable_reset=True` computes what the old wrapper did and was
bit-identical to it on inverted_pendulum on macOS ARM. The test holds it
only to float32 rounding, because whether two equivalent programs compile
to the same bits is up to XLA and differs across backends (CI is Linux
x86).

## Recurrent off-policy replay: head queries and exact time limits (2026-10-11)

Two changes to the recurrent SAC/ASAC/REDQ/TD3 replay
([recurrent.py](src/ajax/agents/recurrent.py)) move its cost. First, a
recurrent Q-critic embeds the observation and the previous action before
its memory and reads the queried action at a head after it, an MLP of the
critic's architecture (`Critic.query_dim`): every loss is one parallel
pass, and each critic call runs the encoder, the memory and the head.
Second, a window's bootstrap runs over a stream holding each time limit's
final observation after its row (`_bootstrap_stream`), exact for any
number of time limits in a window, so twice the rows long: the target
critic's and the bootstrap actor's forward passes grow from the 16 next
rows to 32 steps. The stream is built and read with one-hot selects;
building it with scatters, which XLA runs one index at a time on CPU,
cost about 15% more on the transformer.

SAC on Pendulum, 4 envs, 2,048 steps, actor and critic `("64", "relu")`,
memory width 32 (window 16), CPU, the second of two identical calls,
base and branch interleaved while unrelated jobs kept the load at 6-12 on
14 cores (single runs swing by about 15%); medians of five rounds (mamba
three), seconds:

| memory | base | both changes | delta |
|---|---|---|---|
| gru | 13.61 | 15.92 | +17% |
| transformer | 12.97 | 16.09 | +24% |
| mamba | 80.4 | 95.2 | +18% |

Where it goes, from the variants timed on the way (each against its own
interleaved base): the head without an embedder before the memory, which
then read the raw 4-wide input, measured -13% on the GRU and par on the
transformer; adding the exact stream to it, -12%, +17% and +25% (mamba).
That layout learned the probes' step counting more slowly (p0b's
cue-free value read 0.1 instead of 0 at 2,500 steps), so the embedder
came back by the owner's decision, for the last third of the deltas
above: the memory's input projection is 64 wide again. Attention
cost grows with the square of the stream's length and the state-space
scan with its length, on two forward passes per update. A stream holding
only a window's first time limit (two rows longer than the window)
measured +1% on the transformer and par on mamba, but cut the later time
limits' targets in windows holding several (episodes shorter than the
window); exactness was kept by the owner's decision. Holding the final
observations elsewhere (per-step carries, or extra query tokens) needs
per-memory-kind changes and costs as many tokens. A narrower embedder
(Ni et al.'s are small linear layers) would cut the memory's input
projection; not tried.

## The refactor, before and after (a1658d9 -> 91ee10f, 2026-10-11)

`benchmarks/agent_bench.py` at the audit commit a1658d9, before the
refactoring roadmap's merges, and at main 91ee10f, after them: Pendulum
(CartPole for the discrete agents, as the bench's presets), 4 envs, 8,000
steps, 3 trials, compile cache off, on a quiet Mac (CPU). Raw rows:
[`agent_refactor_before.jsonl`](benchmarks/agent_refactor_before.jsonl)
and [`agent_refactor_after.jsonl`](benchmarks/agent_refactor_after.jsonl);
`python benchmarks/agent_bench.py --compare` on them reports no regression
beyond 10%. Median steps/s and compile (plus warm-up) seconds:

| agent | steps/s before | after | delta | compile s before | after | delta |
|---|---|---|---|---|---|---|
| APG | 5594 | 5494 | -1.8% | 1.52 | 1.48 | -2% |
| APO | 4217 | 4486 | +6.4% | 1.92 | 2.03 | +6% |
| ASAC | 4164 | 5006 | +20.2% | 2.24 | 1.55 | -31% |
| AVG | 1787 | 4056 | +126.9% | 1.95 | 1.58 | -19% |
| DQN | 5547 | 6375 | +14.9% | 1.17 | 0.94 | -20% |
| DreamerV3 | 287 | 273 | -5.1% | 11.39 | 12.31 | +8% |
| PPO | 3904 | 4281 | +9.6% | 2.21 | 1.98 | -10% |
| PQN | 6364 | 7332 | +15.2% | 1.11 | 1.01 | -9% |
| REDQ | 3859 | 4868 | +26.2% | 2.14 | 1.58 | -26% |
| SAC | 4403 | 5051 | +14.7% | 4.46 | 4.71 | +6% |
| TD3 | 5317 | 6163 | +15.9% | 1.52 | 1.30 | -14% |
| TDMPC2 | 247 | 243 | -1.3% | 6.65 | 7.25 | +9% |

The SAC, TD3, REDQ and ASAC rows time collection only (see
[the next section](#the-off-policy-bench-rows-timed-collection-only-2026-10-11)).
SafeSAC, benchmarked before (4091 steps/s), was deleted in between
(#77). The causes, where known:

- AVG: the shared loop (a5ad4e8): its metrics are no longer stacked per
  step, and its length-1 inner scans became direct calls (88124c5).
- ASAC, REDQ: the shared loop (no per-step metric stacking) and the
  plain-MLP default (#87: no encoder LayerNorm), plus the length-1 scans
  gone.
- TD3: the shared loop and its plain critic (#82).
- SAC, DQN, PPO: the plain-MLP default (#87); DQN and PPO also moved
  onto the shared loop (#78, #79).
- PQN, APO: the shared loop (#79) and the dead work removed (#71); not
  attributed further.
- APG, TDMPC2, DreamerV3: within run-to-run spread; not attributed.

## The off-policy bench rows timed collection only (2026-10-11)

`benchmarks/agent_bench.py` ran SAC, TD3, REDQ and ASAC at their default
`learning_starts=10_000`, past the end of its 8,000-step run. Their
steps/s figures, in the table above, in its two raw files and in the
baselines before 2026-10-12, measured collection alone, with no gradient update. Their presets now set
`learning_starts=1000`, DQN's default, so the updates cover seven eighths
of the run, and `tests/benchmarks/test_agent_bench.py` checks from the
configs that every preset updates within the run. These four agents'
figures will drop: do not compare them with the earlier rows. The new
baseline is [below](#cpu-baseline-after-the-close-out-batch-1-2026-10-12).

The other presets already updated in the run: DQN from step 1,000, AVG
from the first step, PQN 16 rollouts, APG 125, DreamerV3 496 updates and
TD-MPC2 7,996. PPO and APO make one update: their default 2,048-step
rollout over 4 envs is 8,192 steps, more than the run, and the loop
always runs one rollout, so their steps/s count 8,000 of the 8,192 steps
they take (2.4% low, the same in every row).

## Where to look for more

The [audit-pending section](PERFORMANCE_LOG.md#audit-items-still-pending)
of the running log lists items not landed (cosmetic-only) or
out-of-scope (env-side stat refactor blocked by Brax shape coupling).
The set of patterns now codified in
[`ajax/perf_utils.py`](src/ajax/perf_utils.py) covers the universally
applicable wins; future agents only need to build their train with
`build_resumable_train` and use `final_aux_scan` for diagnostic-only
inner scans to inherit the benefit.

## Where the wins come from (mental model)

1. **Most "memory cost" was JAX compile-time transient scratch, not
   training memory.** Once the disk cache short-circuits compile, peak
   memory drops by 95% on a fresh process (a cached program: no host
   callbacks, so no logging). This is the largest single effect we
   measured.
2. **Most "speed cost" was waiting for `jit(train)` to compile.**
   Same fix; same source. Steady-state per-step compute is already
   tight (1-5% wins from the audit, which is the right order of
   magnitude for a mature JIT codebase).
3. **Seed parallelism was already correct in the structure.** What
   was missing was confidence that more seeds are free (they are on
   the benches; not on every HPO configuration, see above).
4. **The remaining audit-flagged items are either
   correctness-defensive (no observable wins at current scale) or
   blocked by external library shape coupling (Brax VmapWrapper). The
   diminishing-returns line is reached.**

## CPU baseline after the close-out batch 1 (2026-10-12)

`benchmarks/agent_baseline.jsonl` now holds this run: main `a8a13c2`, macOS
on Apple silicon (CPU only, `JAX_PLATFORMS=cpu`), the persistent compile
cache off (`AJAX_NO_COMPILE_CACHE=1`), the defaults of
`benchmarks/agent_bench.py` (4 envs, 8,000 steps, 1 seed, 3 trials), and no
other Ajax work running. The load average was about 5, from macOS background
services (Spotlight indexing).

Two methodology breaks with every earlier `benchmarks/agent_*.jsonl` file:
every preset now updates within the run (SAC, TD3, REDQ and ASAC start at
step 1,000), and `exec_s` times the run's execution alone, without tracing,
lowering or compiling (`measured_s` still includes them). Compare with
`--compare`, which reads `exec_steps_per_s` when both files have it.

| Agent | Env | exec steps/s | measured steps/s | compile s |
|---|---|---|---|---|
| SAC | Pendulum-v1 | 2,218 | 1,635 | 3.8 |
| SAC | inverted_pendulum (terminates) | 1,090 | 723 | 5.3 |
| TD3 | Pendulum-v1 | 2,659 | 1,961 | 1.4 |
| REDQ | Pendulum-v1 | 47 | 47 | 1.7 |
| ASAC | Pendulum-v1 | 1,254 | 1,032 | 1.6 |
| AVG | Pendulum-v1 | 9,460 | 4,027 | 1.5 |
| PPO | CartPole-v1 | 17,817 | 4,298 | 2.0 |
| APO | Pendulum-v1 | 16,390 | 4,024 | 2.0 |
| DQN | CartPole-v1 | 16,662 | 6,395 | 0.9 |
| PQN | CartPole-v1 | 36,961 | 7,358 | 1.0 |
| UDRL | Pendulum-v1 | 12,575 | 4,441 | 1.4 |
| APG | Pendulum-v1 (tracking) | 34,958 | 5,339 | 1.6 |
| DreamerV3 | CartPole-v1 | 367 | 275 | 12.9 |
| TDMPC2 | Pendulum-v1 | 292 | 245 | 7.3 |

Medians of the three trials. REDQ is slow by design: 20 critic updates of
10 critics per environment step after step 1,000. UDRL's trials spread the
most (9,604 to 15,069 exec steps/s); judge it on more trials before calling
a regression.

## Continuous actions mapped onto the env's box (2026-10-12)

Every model-free env step now maps the agent's `[-1, 1]` action onto the
env's box (`agent_action_to_env`, one `c + h * clip(a, -1, 1)`), and
experts are read in the same units. On `[-1, 1]` boxes every run is
bitwise unchanged. On Pendulum's `[-2, 2]` the squashed agents reached only half the
torque before.

`benchmarks/agent_bench.py` defaults, CPU, before and after the change,
medians of 3 trials, exec steps/s: SAC (Pendulum) 2,138 -> 2,141 (+0.1%),
TD3 (Pendulum) 2,699 -> 2,776 (+2.8%), PPO (CartPole, discrete: the map
passes it through) 16,784 -> 16,957 (+1.0%). All within noise.

Learning check: gymnax Pendulum-v1, 20k steps, the final evaluation return
(10 episodes) of seeds 0 and 1.

| Agent | before | after |
|---|---|---|
| SAC | -262, -291 | -166, -167 |
| TD3 | -242, -147 | -180, -87 |
| PPO | -1537, -1313 | -1400, -1334 |

20k steps are too few for PPO on Pendulum; its Gaussian is now read in
normalised units (its torque std doubles), so this row only shows it runs.

## TD-MPC2 agent (M4b): CPU learning checks and bench entry (2026-10-03)

`benchmarks/world_models/learning_checks.py` (`docs/world_models/DESIGN.md` §0, §10): TD-MPC2
at `model_size=1` with every other hyperparameter at the paper default (512
planning samples including 24 policy-prior trajectories, 64 elites, 6 MPPI
iterations, batch 256, UTD 1, one env, `seed_steps = max(1000, 5 T)`, `gamma`
from `T`), evaluated by the planner in `eval_mode`. Seeds 0 and 1, run on the
uncommitted M4b worktree on top of `af0295e`, on a shared M-series CPU at load
average 25-40 (other jobs running).

| Check | Budget (agent steps) | Eval returns per seed (at steps) | Final | Bar | Verdict | Wall (2 seeds) |
|---|---|---|---|---|---|---|
| gymnax Pendulum-v1 (T = 200, bounds [-2, 2] mapped) | 6,000 | seed 0: -177, -134, -143; seed 1: -203, -287, -124 (2k, 4k, 6k) | -143, -124 | > -400 | PASS | 45 min |
| playground CartpoleBalance, action repeat 2 (T = 500) | 10,000 (20,000 frames) | seed 0: 656, 810; seed 1: 533, 854 (5k, 10k) | 810, 854 | > 800 | PASS | 88 min |

Pendulum is solved after the first 1,000 updates (a random policy scores about
-1200). CartpoleBalance is still improving at the end of its budget (533-656 at
5k steps, 810-854 at 10k; the paper's curve sits near 1000 from 100k frames).
Seed 0 clears its bar by only 10, so the bar is not met with much margin at
this budget.

Bench entry (`benchmarks/agent_bench.py`, preset documented there): one run
at load average ~35 then gave 64-66 steps/s (8,000 env steps, 4 envs, 7,996
updates; compile 46 s). Its baseline row is now in
[the 2026-10-12 baseline](#cpu-baseline-after-the-close-out-batch-1-2026-10-12).

The checks are the entries `tdmpc2-pendulum` and `tdmpc2-cartpole-balance` of
the shared `benchmarks/world_models/learning_checks.py` registry (merged with DreamerV3's in
M7; same agent arguments, budgets, cadences and bars as the runs above).

## DreamerV3 agent (M7): CPU learning checks, reference comparison and bench entry (2026-10-03)

`benchmarks/world_models/learning_checks.py`: DreamerV3 at `model_size="1m"` (`d = 64`: deter
512, 4 classes) with every other hyperparameter at the paper-era default (16
envs, train ratio 512, batches of 16 x 64, imagination horizon 15), 20,000 rows
(the agent's unit: one row per env per vector step, reset rows included), so
about 9,500 updates; evaluated every 2,000 rows on 10 episodes of sampled
actions from fresh resets (the reference has no deterministic mode). Seed 0, run
on the uncommitted M7 worktree on top of `db65f10`, both checks side by side on a
shared M-series CPU at load average 25-30.

| Check | Eval returns at 2k, 4k, ..., 20k rows | Final | Bar | Verdict | Wall |
|---|---|---|---|---|---|
| gymnax Pendulum-v1 (bounds [-2, 2] mapped) | -1182, -1198, -1118, -1252, -1177, -620, -264, -250, -198, -222 | -222 | > -400 | PASS | 68 min |
| gymnax CartPole-v1 (returns at most 500) | 22, 100, 378, 123, 182, 219, 186, 126, 121, 155 | 155 | > 400 | **FAIL** | 68 min |

Pendulum swings up from 12k rows on and stays up. CartPole learns but does not
reach the bar at this budget, and its evaluation curve is not monotonic.
Diagnostic runs (same configuration, outside the registry):

- Seeds 1 and 2, 20k rows, evaluated every 992 rows: both rise from about 19
  to 286-342 by 6k rows, then oscillate (peaks 462 and 342; finals 487 and
  121).
- Seed 0, 40k rows: the first 20k rows follow the check's curve (21, 105,
  390, 126, ..., 157 at 20k), then the policy reaches the 500 cap at 22k, 30k, 32k and 40k rows, but scores 150-400 on
  the evaluations in between. The training-episode return (the reference's
  score, a rolling mean of each env's last 10 episodes) rises monotonically,
  from 20 to 112 at 20k rows and 211 at 40k.

The agent learns CartPole, but this first check failed. The bar (400 at the
last log) is ours, not the paper's: DreamerV3 reports no gymnax CartPole result.

### Reference comparison (2026-10-03 to 10-04)

The first CartPole check failed a bar of our own (400 at the last log), and the
question was whether that is normal DreamerV3 behaviour or an end-to-end bug.
The real reference code (`danijar/dreamerv3` at `29eb964`, the version Ajax
follows) was run on the identical protocol in a throwaway venv (jax 0.4.26):
- a numpy port of gymnax CartPole-v1, checked step by step against gymnax;
- the 1m widths set key by key (637,511 vs 637,381 parameters; the 130 extra
  are the reference's padded two-hot output column);
- float32, 16 envs, train ratio 512, and 10-episode evaluations of the sampling
  policy from fresh resets at fixed seeds.

Both sides make their first update at row 1,040 and 9,481 updates by 20,000
rows. The statistics and decision rules were written down before each round's
results. The harness, protocol, pre-registrations, root-cause investigation and
compact results of all three rounds are in `docs/world_models/reference_comparison/`
(`README.md` there says how to rerun each round and recompute every table).

**Round 1** (3 seeds per side, 20k rows). What matched:
- the training-episode score, within 1% at 20k rows (106.6 vs 107.6); along
  the way the means were up to 22% apart (42.1 vs 33.0 at 8k rows);
- the dynamics, representation and reward losses, within 1.1% from 4k rows.

What did not:
- the pre-registered evaluation statistic, the mean evaluation at 16k-20k rows,
  failed: Ajax 204 vs reference seeds 241-386;
- the reconstruction loss was 16-25% lower in Ajax at 10k, 12k and 16k-20k
  rows (seed ranges disjoint only at 10k);
- two training statistics differed: the actor's entropy was lower in Ajax in
  every window from 4k to 18k rows (window means 6-23% lower, seed ranges
  disjoint in 5 of 10 windows), and the mean |encoder output| was 5-10% lower,
  the gap shrinking over training.

**Root causes**, from an independent investigation with every claim
adversarially verified:
- **Encoder output:** the seeds' initial draws, not code. With the same
  parameters both encoders agree to 2e-6, and 300 initialisations per side give
  the same mean.
- **Entropy:** deviation D22, the two-hot expectation. Under jit, XLA contracts
  the reference's mirror-pair sum into fused multiply-adds. Early in training
  this adds noise of std about 0.085 to every reward and value prediction.
  Ajax's mirror-difference form is exact. Patching the reference formula into
  Ajax reproduces the reference's entropy, advantage spread and losses at
  4k rows. `deviations.md` D22 now says so; it previously called the effect
  negligible.

**Round 2** (3 seeds per side, both directions): the reference with Ajax's
exact expectation, and Ajax with the reference's.
- With the same expectation on both sides, the entropy seed ranges overlap in
  all ten windows in both pairs. In round 1, five windows did not overlap.
- The evaluation statistic still failed, driven by one checkpoint. All six
  reference runs scored at least 399 at 20k rows (1 of 6 at 18k, 0 of 6 at
  16k); one of the six Ajax runs did (round 1 seed 2, 500), the other five
  scored 109-245.

**Round 3** (5 seeds per side, 24k rows, evaluation every 400 rows). The
reference used the exact expectation and a replay sampler seeded per run, as
Ajax does (D2). The stock `make_replay` seeds every run's sampler with 0, so the
reference's runs had drawn identical replay indices and were not independent.

The 20k concordance disappears: 2 of 5 runs reach 400 at 20k on each side. It
came from those correlated reference runs. Pre-registered results:

| Statistic (per seed) | Reference mean [range] | Ajax mean [range] | Rule | Welch p |
|---|---|---|---|---|
| mean eval over [12k, 24k] (primary) | 312.7 [270.3, 346.7] | 265.4 [198.3, 321.1] | fails: Ajax mean 4.9 below the lowest reference seed | 0.12 |
| mean eval at 16k, 18k, 20k | 282.1 [201.5, 358.1] | 305.8 [250.7, 373.8] | passes (within range) | 0.52 |

**Conclusion: no Ajax bug was found.**
- With D22 matched and the replay sampler seeded per run (round 3, 5 seeds per
  side), the seed ranges of the 90 training statistics both sides log overlap
  in at least 52 of the 58 400-row windows that contain updates: the
  replay-value loss in 52, the advantage spread in 53, every other statistic in
  54 or more, the non-overlapping windows lying between 7.6k and 20k rows.
  Three exceptions carry no information: the update counter `opt_grad_steps`
  (its reference window mean is 0.5 lower because the reference returns the
  previous call's metrics) and `ent/action/max` / `rand/action/max`, which sit
  at their maximum (ln 2, 1) on both sides and differ only in the last float32
  digit (0.693147 vs 0.693146). In round 2's pair E (3 seeds, D22
  matched, the stock reference sampler), Ajax's imagined return and value were
  17-31% higher at 10k-18k rows, with disjoint seed ranges in 5 of 10 windows.
- On evaluation, Ajax's late-phase mean is 15% below the reference's. This is
  not significant at 5 seeds (p = 0.12). The strict pre-registered rule (Ajax
  mean at least the lowest reference seed) fails by 4.9 points.
- The remaining structural difference is the reference's asynchronous acting
  (D1: parameters about 16 updates stale). It cannot be removed without
  changing the reference.

### The CartPole check, redefined

`dreamerv3-cartpole` now runs round 3's protocol: 24k rows, evaluated every 400
rows. Its statistic is the mean evaluation return over [12k, 24k] rows (31
logs), and the bar is > 156: half the reference's mean of that statistic (312.7).
A random policy scores about 22.

The check asks whether the agent learns. Whether it matches the reference is
answered by the comparison above. All five of round 3's Ajax seeds pass (310.6,
321.1, 198.3, 232.6, 264.6), and so do all five of the reference's.

Seed 0 run through the script (`--only dreamerv3-cartpole --seeds 0`, on
`aef355f` plus this change): statistic 295.1, PASS, 11,481 updates, 70 min at
load average about 15. Its curve up to 20k rows is the same as the first
check's, value for value (22, 100, 378, 123, ..., 155), because it is the same
program; it then reaches 500 at 22k rows.


### Bench entry

Bench entry (`benchmarks/agent_bench.py`, preset documented there: the 1m model
on 8 windows of 16 + 1 rows, train ratio 8, about 500 updates in 8,000 rows, 4
envs): one run at load average about 25 then gave a median of 80.9 steps/s
(3 trials at 66.5, 82.1 and 80.9 steps/s; compile 41 s). Its baseline row is
now in [the 2026-10-12 baseline](#cpu-baseline-after-the-close-out-batch-1-2026-10-12).
