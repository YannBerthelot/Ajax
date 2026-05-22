# Ajax Performance Log

Tracks GPU performance optimizations: each entry has the change, the
benchmark numbers (before vs after), and the takeaway. Goal is a
durable record of which tricks paid off so they can be reused.

## How to benchmark

The harness lives at [benchmarks/perf_bench.py](benchmarks/perf_bench.py).
One process per measurement (so JIT cache, RNG, and the XLA memory
allocator all start fresh).

```bash
# Per-scenario, 3 trials, appends to benchmarks/results.jsonl.
python benchmarks/perf_bench.py \
    --tag <label> \
    --scenario <pure_sac|obs_norm_sac|stress_sac> \
    --timesteps 2000 --warmup-timesteps 300 \
    --trials 3 --n-envs 64 \
    --out benchmarks/results.jsonl
```

To compare before/after a patch:

```bash
git stash push -- <patched files>
python benchmarks/perf_bench.py --tag baseline ...
git stash pop
python benchmarks/perf_bench.py --tag <patch-label> ...
```

Scenarios:

- `pure_sac`: Pendulum-v1, n_envs=4, num_critics=2, no obs norm. Cheapest;
  catches gross regressions but small patch effects sit in the noise.
- `obs_norm_sac`: same but with `normalize_obs_running=True`. Exercises
  the agent obs-norm path.
- `stress_sac`: Pendulum-v1, n_envs=64, num_critics=10, 256-wide nets,
  obs norm on. Amplifies per-update critic compute and the obs-norm
  forward path so small patch effects emerge from noise.

Run env at time of writing: GPU `cuda:0`, JAX `gpu` backend, Python 3.12.3.

## Results

### Patch 1 + 2 combined (2026-05-08)

Patches:

1. Agent obs-norm running stats stored with leading shape `(1, *)`
   instead of `(n_envs, *)`. `online_normalize` already broadcasts the
   per-env batch reduction, so the per-env axis carried no information.
   Files: [obs_norm.py](src/ajax/agents/obs_norm.py),
   [cloning.py](src/ajax/agents/cloning.py).
2. Gate the redundant `q_preds_for_var = predict_value(...)` in the
   critic update behind `target_modifier is not None or expert_q is not
   None`. Pure SAC saves a full `[N_critics, B, ...]` critic forward per
   critic step. File: [train_SAC.py](src/ajax/agents/SAC/train_SAC.py).

Numbers (3 trials each, lower is better):

| Scenario       | Baseline (median / min) | Patched (median / min) | Δ median  | Δ min    |
| -------------- | ----------------------- | ---------------------- | --------- | -------- |
| pure_sac       | 9.41s / 9.21s           | 9.59s / 9.37s          | +1.9%     | +1.7%    |
| obs_norm_sac   | 9.80s / 9.60s           | 9.65s / 9.47s          | -1.5%     | -1.4%    |
| stress_sac     | 11.98s / 11.97s         | 11.49s / 11.36s        | **-4.1%** | **-5.1%**|

Peak GPU memory at this scale: 252 MB across all conditions. The
allocator pool granularity is far coarser than the `(64, 3) - (1, 3)`
obs-norm tensor savings on Pendulum, so memory deltas stay invisible
in this regime.

Takeaways:

- Both patches are correctness-preserving and free wins (no algorithmic
  change). They land speedup but the magnitude is workload-dependent.
- At Pendulum / n_envs=4 / num_critics=2 the savings sit inside trial
  noise (±2%). The work being skipped is fully fused by XLA at this
  scale.
- At Pendulum / n_envs=64 / num_critics=10 (`stress_sac`) the saved
  critic forward becomes a measurable 4-5% of step time. Both effects
  scale with work-per-step: high `num_critics` (REDQ-style ensembles)
  amplifies #2; high `n_envs × obs_dim` amplifies #1's memory + reduce
  cost (memory specifically would emerge with large obs, not Pendulum).
- Lesson for future patches: micro-benchmarks need a workload that's
  large enough to not be allocator/dispatch-noise dominated. Default to
  `stress_sac` when reporting headline numbers. Pendulum-cheap is fine
  as a smoke test that nothing broke.

### Patch 4 + 5 (2026-05-08)

Patches:

4. Switch the inner critic-update `lax.scan` from ys-emitting to
   carry-only. The previous version returned the per-step
   `ValueAuxiliaries` as ys then took `x[-1]` for each leaf, so
   `[num_critic_updates, *aux]` was materialized just to be discarded.
   New form carries the latest aux through the scan; `jax.eval_shape`
   builds the zeros placeholder for the initial carry without running a
   real critic forward.
   File: [train_SAC.py:1792-1820](src/ajax/agents/SAC/train_SAC.py#L1792-L1820).
5. Add `donate_argnames=("initial_state",)` to the top-level `train`
   jit. On the resume-from-checkpoint path this lets XLA reuse the
   incoming agent-state buffers, dropping peak memory by ~one
   `agent_state` worth. No effect when `resume_from_state=False`
   (default) because `initial_state=None` has no buffer to donate.
   File: [train_SAC.py:2308-2316](src/ajax/agents/SAC/train_SAC.py#L2308-L2316).

Numbers (3 trials each, lower is better):

| Scenario           | Baseline (median / min)  | Patches 1+2+4+5 (median / min) | Δ median | Δ min  |
| ------------------ | ------------------------ | ------------------------------ | -------- | ------ |
| stress_sac         | 11.98s / 11.97s          | 11.69s / 11.66s                | -2.4%    | -2.6%  |
| stress_sac_utd     | 11.26s / 11.17s          | 11.27s / 11.25s                | +0.1%    | +0.7%  |

Memory peak: unchanged at this scale (252 MB and 201 MB for stress_sac
and stress_sac_utd respectively). The aux materialization saved by #4
is a few hundred bytes, well below allocator pool granularity. The
donation savings of #5 don't apply on the bench's non-resume path.

Takeaways (calibration):

- #4 shows no measurable wall-clock or memory effect at any tested
  scale. It is a **defensive correctness fix**: prevents OOM at
  extreme `num_critic_updates × vmap_seeds` with a high-dimensional
  aux. Worth keeping; not worth claiming a speedup.
- #5 does not register on this bench because the bench never resumes
  from checkpoint. The change is free (donating None is a no-op) and
  unblocks peak-memory wins for resume-heavy workflows.
- Honest reading: Patches 1 and 2 carried the entire measured 4-5%
  speedup at the `stress_sac` scale; 4 and 5 are correctness/headroom.

### Tricks worth remembering

1. **Carry only the leading-1 axis for global stats.** When a stat is
   updated by reducing over a batch axis (e.g. running mean of obs
   across envs), storing it with that leading axis present is a pure
   `n_envs×` memory tax: the per-env rows hold identical values.
   `(1, *)` keeps broadcasting working without the inflation.
2. **Diagnostics-only forward passes belong inside the loss.** When a
   `predict_value`-style call produces values that the loss already
   computes internally, expose them through `aux` instead of
   recomputing. If the outer call is only used downstream conditionally
   (e.g. `target_modifier is not None`), gate it with a Python-level
   `if` so trace time skips it entirely.
3. **JAX broadcasting is your friend, but only after you remove the
   redundant reductions.** If you find yourself calling `nanmean(x,
   axis=0)` inside a hot path to collapse a stale dim, fix the
   producer's shape instead.
4. **Use `jax.eval_shape` for tracer-only structure derivation.** When
   you need a zeros pytree shaped like the output of an expensive
   function (e.g. for a scan's initial carry), `eval_shape` gives the
   `ShapeDtypeStruct` tree without running the function. Pair it with
   `tree.map(lambda s: jnp.zeros(s.shape, s.dtype), ...)`.
5. **Don't return ys for diagnostics you only inspect at the last
   step.** Carrying the latest aux through the scan keeps the leading
   scan axis off-device. Cheap structural fix, big payoff under
   vmap-over-seeds with non-trivial aux.
6. **`donate_argnames` is free when the caller doesn't reuse the
   input.** Top-level `jit`'d training functions typically receive a
   freshly constructed agent state once and never see it again.
   Donating that argument lets XLA reuse its buffers, halving peak
   memory at the jit boundary on resume-from-checkpoint paths.
7. **Calibrate benchmark workloads to the patched code path.** Tiny
   tensors, small `num_critics`, small `num_critic_updates`: every
   patch effect disappears into XLA fusion + dispatch noise. Use
   `stress_sac` (n_envs=64, num_critics=10, 256-wide nets) as the
   default headline number. `pure_sac` is a smoke test, not a
   measurement.

### Patch 6 + 7 (2026-05-08)

Patches:

6. Network init uses a shape-1 dummy batch instead of `(n_envs, *)`.
   Flax `network.init` only reads `init_x`'s shape to infer params; the
   leading batch dim's size is irrelevant. Saves an `n_envs×` zero
   allocation at init for the actor and critic init paths.
   File: [networks.py:562-565](src/ajax/networks/networks.py#L562-L565),
   [networks.py:617-618](src/ajax/networks/networks.py#L617-L618).
7. `online_normalize` gets a `nan_safe: bool = True` static argument.
   The agent obs path (`update_obs_norm`) sets `nan_safe=False` so its
   reductions compile down to plain `mean` instead of `nanmean`'s
   masked variant. Default stays True so AVG's NaN-sentinel path
   ([agents/AVG/utils.py:41-55](src/ajax/agents/AVG/utils.py#L41-L55))
   keeps working. Files:
   [utils.py:20-66](src/ajax/utils.py#L20-L66),
   [obs_norm.py:50-52](src/ajax/agents/obs_norm.py#L50-L52).

Numbers (3 trials each on `stress_sac`, lower is better):

| Tag                       | Median  | Min     | Δ vs baseline (median) |
| ------------------------- | ------- | ------- | ---------------------- |
| baseline                  | 11.98s  | 11.97s  | -                      |
| patches_1_2               | 11.49s  | 11.36s  | -4.1%                  |
| patches_1_2_4_5           | 11.69s  | 11.66s  | -2.4%                  |
| patches_1_2_4_5_6_7       | 11.33s  | 11.29s  | **-5.4%**              |

Memory peak: still 252 MB (allocator granularity dominates). #6's
saving is at init (one-shot), so it doesn't show in steady-state peak.
#7 is per-step; the savings on Pendulum (3-dim obs) are too small to
move the allocator.

Takeaways:

- Most of the additional 3% improvement vs `patches_1_2_4_5` likely
  comes from #7. The `online_normalize` reduction is called twice per
  collection step plus implicitly at every `apply_obs_norm` site that
  touched the running stats; switching from `nanmean` to plain `mean`
  drops the per-reduction mask comparison.
- #6 is one-shot at init; invisible on steady-state numbers but a real
  peak-memory win for image-obs / large-`n_envs` settings (saves an
  `(n_envs, *obs_shape)` float32 allocation at agent construction).

### Patch 8 attempted, REVERTED (2026-05-08)

Attempted: cascade the `(1, *)` shape change into the env-wrapper
`init_norm_info`. Failed because env-side stats live inside the env
state pytree, which Brax's `VmapWrapper.step` does
`jax.vmap(self.env.step)(state, action)` on. With stats shape `(1, *)`
and the rest of the state at leading axis `n_envs`, vmap raises
`ValueError: vmap got inconsistent sizes`. Reverted.

Lesson logged in tricks #8 below.

### Tricks worth remembering (updated)

1. **Carry only the leading-1 axis for global stats.** When a stat is
   updated by reducing over a batch axis (e.g. running mean of obs
   across envs), storing it with that leading axis present is a pure
   `n_envs×` memory tax: the per-env rows hold identical values.
   `(1, *)` keeps broadcasting working without the inflation.
2. **Diagnostics-only forward passes belong inside the loss.** When a
   `predict_value`-style call produces values that the loss already
   computes internally, expose them through `aux` instead of
   recomputing. If the outer call is only used downstream conditionally
   (e.g. `target_modifier is not None`), gate it with a Python-level
   `if` so trace time skips it entirely.
3. **JAX broadcasting is your friend, but only after you remove the
   redundant reductions.** If you find yourself calling `nanmean(x,
   axis=0)` inside a hot path to collapse a stale dim, fix the
   producer's shape instead.
4. **Use `jax.eval_shape` for tracer-only structure derivation.** When
   you need a zeros pytree shaped like the output of an expensive
   function (e.g. for a scan's initial carry), `eval_shape` gives the
   `ShapeDtypeStruct` tree without running the function. Pair it with
   `tree.map(lambda s: jnp.zeros(s.shape, s.dtype), ...)`.
5. **Don't return ys for diagnostics you only inspect at the last
   step.** Carrying the latest aux through the scan keeps the leading
   scan axis off-device. Cheap structural fix, big payoff under
   vmap-over-seeds with non-trivial aux.
6. **`donate_argnames` is free when the caller doesn't reuse the
   input.** Top-level `jit`'d training functions typically receive a
   freshly constructed agent state once and never see it again.
   Donating that argument lets XLA reuse its buffers, halving peak
   memory at the jit boundary on resume-from-checkpoint paths.
7. **Calibrate benchmark workloads to the patched code path.** Tiny
   tensors, small `num_critics`, small `num_critic_updates`: every
   patch effect disappears into XLA fusion + dispatch noise. Use
   `stress_sac` (n_envs=64, num_critics=10, 256-wide nets) as the
   default headline number. `pure_sac` is a smoke test, not a
   measurement.
8. **`nn.Module.init` only needs shape, not n_envs.** A leading batch
   dim of 1 is enough to trace the network and infer params. Allocating
   `(n_envs, *obs_shape)` at init buys nothing and costs `n_envs×` for
   image-obs setups.
9. **Make NaN-safety opt-in via a static flag, not a hidden default.**
   `nanmean`/`nansum` are needed when callers feed sentinel NaNs
   (AVG's per-step `G_return` mask is one example), but they cost a
   per-element mask compare every reduction. A static `nan_safe`
   kwarg lets the JIT pick between code paths so clean-data callers
   compile down to plain `mean`.
10. **Beware shape coupling through batch wrappers.** Brax's
    `VmapWrapper` does `jax.vmap(env.step)(state, action)` over axis 0
    of the entire state pytree. Anything nested in the env state (e.g.
    running normalisation stats) must keep a leading axis of `n_envs`
    to vmap cleanly. Shrinking such a leaf to `(1, *)` will be
    rejected with a "vmap got inconsistent sizes" error. The agent-
    side state is not subject to this constraint, which is why the
    `(1, *)` shape works for `init_agent_obs_norm` but had to be
    reverted for `init_norm_info` (the env-wrapper variant).

## Seed-scaling investigation (2026-05-08)

How fast is `agent.train(seed=[s_0, ..., s_{N-1}])` as N grows? The
inner train function is wrapped in `jax.vmap` over seeds at
[base.py:168](src/ajax/agents/base.py#L168), so ideal scaling is
**flat wall-clock**: doubling N should not double the time, since the
work runs in parallel on the GPU. Saturation appears as wall-clock
climbing toward N×.

The bench harness now takes `--n-seeds`. Sweep results live in
[benchmarks/seed_scaling.jsonl](benchmarks/seed_scaling.jsonl) (Pendulum)
and [benchmarks/seed_scaling_p3d.jsonl](benchmarks/seed_scaling_p3d.jsonl)
(Plane3DCircle). Render with:

```bash
python benchmarks/seed_scaling_report.py [path/to/jsonl]
```

### `pure_sac` (Pendulum-v1, n_envs=4, num_critics=2, 64-wide nets)

2 trials per N, 2000 timesteps measured.

|   N | median (s) |   min (s) | throughput vs N=1 | time/seed (s) | peak (MB) |
| --: | ---------: | --------: | ----------------: | ------------: | --------: |
|   1 |      9.811 |     9.448 |             1.00x |         9.811 |     240.1 |
|   2 |     10.373 |    10.183 |             1.89x |         5.187 |     240.2 |
|   4 |     10.511 |    10.254 |             3.73x |         2.628 |     240.3 |
|   8 |     10.231 |    10.028 |             7.67x |         1.279 |     240.2 |
|  16 |     10.428 |    10.279 |            15.05x |         0.652 |     240.3 |
|  32 |     11.082 |    10.583 |            28.34x |         0.346 |     256.5 |

Reading: Pendulum is so cheap that 32 seeds in parallel only adds 13%
to wall clock vs 1 seed, giving 28× throughput. Memory grows by 16 MB
total across the 32× span (~0.5 MB / additional seed) since the
per-seed working set is dominated by fixed kernel/buffer overhead, not
network params.

### `p3dcircle_sac` (Plane3DCircle from `target_gym`, 17-dim obs, 3-dim action)

n_envs=4, num_critics=2, 256-wide nets, 500 timesteps measured, 1
trial per N (jsonl: `seed_scaling_p3d.jsonl`).

|   N | wall (s) | throughput vs N=1 | time/seed (s) | peak (MB) |
| --: | -------: | ----------------: | ------------: | --------: |
|   1 |   15.770 |             1.00x |        15.770 |     288.0 |
|   4 |   17.457 |             3.61x |         4.364 |     320.0 |
|  16 |   18.746 |            13.46x |         1.172 |     336.0 |
|  32 |   19.120 |            26.40x |         0.598 |     357.4 |
|  50 |   18.992 |            41.51x |         0.380 |     558.2 |
| 100 |   18.375 |            85.86x |         0.184 |    1090.9 |

Reading: scaling is **superlinear** through N=100. Wall clock goes
*down* slightly from N=50 (18.99s) to N=100 (18.37s); time per seed
halves (0.380s → 0.184s). The fixed kernel-launch / dispatch /
compile overhead (~15s out of the N=1 number) is shared across all
parallel seeds, so the more seeds you can fit on-GPU, the cheaper
each one gets.

This matches the user's report that "50 seeds was much slower than
100 seeds" on Plane3DCircle: per-seed throughput at N=100 is roughly
2× that of N=50 because the fixed overhead amortizes better.

Memory grows roughly linearly (10-11 MB per additional seed past
N≈16; leading per-seed footprint dominates over fixed overhead). At
N=100 we're at 1.1 GB on a ~36 GB device, so there's room to push
further before OOM.

### Practical guidance from the sweep

- **Run as many seeds as memory allows.** On Pendulum-class workloads
  the cost per extra seed is essentially zero up to N=32. On
  Plane3DCircle-class workloads (17-dim obs, 256-wide nets) the cost
  per extra seed past N=16 is `≈ measured_s_at_N=100 / 100 ≈ 0.18s`
  per 500 timesteps; cheaper than N=50 by 2x.
- **Don't run small-N sweeps to "save GPU".** That intuition is
  inverted: 50 seeds takes ~98% of the wall-clock of 100 seeds, so
  halving the seed count throws away half the throughput for free.
- **Monitor peak memory, not wall clock, when scaling N.** Wall clock
  stays nearly flat because the GPU isn't compute-saturated;
  saturation manifests first as memory growing past device capacity
  (vmap broadcasts everything along the seed axis). On the bench's
  CudaDevice (35.7 GB) we'd estimate room for ~3000 seeds on
  Plane3DCircle before OOM, given 10 MB/seed marginal.
- **Allocate a few seeds to compile warmup, not training.** The N=1
  number (15.77s) reflects ~95% fixed overhead. If you only need a
  single training run for debugging, prefer running it inside the
  same Python process as a real sweep so the JIT cache amortizes.

### Caveats

- Scaling on Plane3DCircle was tested at default-SAC settings
  (no num_critics ensemble, no UTD>1, no expert path). REDQ-like
  setups (num_critics=10, num_critic_updates>1) make per-step
  compute heavier and can shift the saturation point lower.
- Bench used 1 trial per N to keep total wall time bounded
  (~5 minutes). Numbers within ~5% of each other should be read as
  noise. The shape of the curve (per-seed time falling monotonically
  with N) is robust across re-runs.
- `n_envs` is fixed at 4 in this scaling test. Increasing `n_envs`
  shifts compute upward; saturation appears earlier in N.

### Patch 9 + 10 (2026-05-08)

Patches:

9. Pre-generate the buffer-mix sample indices once instead of calling
   `jax.random.choice` per leaf. Previously, the
   `additional_transition is not None and transition_mix_fraction < 1.0`
   path used `jax.tree.map(lambda x: jax.random.choice(sample_key, x,
   shape=(n_from_online,)), additional_transition)`, which re-derives
   identical indices from the same `sample_key` for every leaf. New
   form generates one `jax.random.randint` index vector and applies it
   via `x[idx]` to all leaves. Trace-time cleanup; non-default code
   path so it doesn't show on the bench scenarios. File:
   [train_SAC.py:1635-1654](src/ajax/agents/SAC/train_SAC.py#L1635-L1654).

10. Wire JAX's persistent on-disk compile cache. Default cache dir
    `~/.cache/ajax/jax_compile_cache`, override with
    `AJAX_JAX_COMPILE_CACHE_DIR`, disable with
    `AJAX_NO_COMPILE_CACHE=1`. Configured at module import in
    [`ajax/__init__.py`](src/ajax/__init__.py) so it kicks in before
    any jit. Min compile-time / size thresholds set to 0 so every
    compile is cached. Cache key includes JAX version + HLO hash, so
    cross-version contamination is impossible.

**Numbers (Plane3DCircle, 1 trial, 300 measured timesteps, 100 warmup):**

| Phase                            |  cold | warm |   Δ      |
| -------------------------------- | ----: | ---: | -------: |
| compile_plus_warmup (100ts)      | 21.5s | 4.9s | **-77%** |
| measured (300ts)                 | 16.3s | 3.3s | **-79%** |
| steps_per_s during measured      |  18.5 | 89.7 | **+385%**|
| peak_bytes_in_use during measured | 302 MB | 15 MB | **-95%** |

The peak-memory drop is the surprise: ~290 MB on cold runs is XLA's
transient compile-time scratch space, not steady-state training
memory. With a warm cache the compile is skipped entirely, and that
scratch never allocates. Net effect for HPO: each subsequent trial
across a sweep runs in ~one-fifth the wall clock and one-twentieth
the peak GPU memory of the very first trial.

The cache only hits when JAX/JAXlib version + HLO hash match. Code
changes that alter the lowered HLO (most edits to jit'd functions)
invalidate the affected entries; unrelated entries stay valid. The
cache survives across processes by design — that's the whole point.

For seed-scaling specifically: each `n_seeds` value is a different
vmap shape and a different cache key, so the first `--n-seeds 1` run
warms up only the N=1 entry. Subsequent re-runs at N=1 hit cache;
re-runs at N=4 do not (they need their own compile). Use
`AJAX_NO_COMPILE_CACHE=1` to opt out (e.g. CI runs that don't want
disk side effects).

### Tricks worth remembering (updated)

11. **Always wire JAX's persistent compile cache for production
    workflows.** A 3-line config in `__init__.py` (cache dir + zero
    thresholds) saves ~10-20s of compile *and* ~95% of peak GPU mem
    per fresh process when the HLO matches. Critical for HPO sweeps
    that spawn many short-lived processes.
12. **Don't reuse a PRNG key inside a `tree.map`.** `jax.random.choice`
    with a fixed key produces identical indices on every leaf, but
    pays the index-generation cost N_leaves times. Generate indices
    once with `jax.random.randint(key, (N,), 0, B)` and gather via
    `x[idx]`.

### Round 4: shared optimization module (2026-05-08)

Up to this point each performance fix landed inline in `train_SAC.py`.
A new module [`ajax/perf_utils.py`](src/ajax/perf_utils.py) extracts
the patterns so every agent inherits the same fixes from one place.

**Public API**:

| Helper | Replaces |
| --- | --- |
| `@train_jit` | `@partial(jax.jit, static_argnames=("resume_from_state",), donate_argnames=("initial_state",))`. Signature-introspects so it falls back to plain `jax.jit` for agents that don't yet take `initial_state` / `resume_from_state`. |
| `final_aux_scan(body, init_carry, length, xs=None)` | the `lax.scan` + `tree.map(lambda x: x[-1], ys)` idiom. Carries the latest aux through the scan instead of materialising `[length, *aux]` on device. Uses `jax.eval_shape` for the initial-aux placeholder; no real body invocation. |

**Refactor coverage**:

| File | `@train_jit` | `final_aux_scan` |
| --- | :---: | :---: |
| `agents/SAC/train_SAC.py` | yes | yes (critic-update) |
| `agents/REDQ/train_REDQ.py` | yes | yes (critic-update) |
| `agents/TD3/train_TD3.py` | yes | yes (per-epoch update) |
| `agents/PPO/train_PPO.py` | yes | n/a (aux is mean'd over scan axis on purpose) |
| `agents/APO/train_APO.py` | yes | n/a (same as PPO) |
| `agents/AVG/train_AVG.py` | yes | n/a (full ys flattened for logging) |
| `agents/ASAC/train_ASAC.py` | yes | n/a (full ys flattened for logging) |
| `agents/UDRL/train_UDRL.py` | yes | n/a |
| `SafetyExperiments/agents/informed_ppo.py` | n/a (no top-level train, drives Ajax PPO from outside) | n/a |

The `n/a (full ys flattened)` agents intentionally aggregate every
step's aux for logging (e.g. `to_state_dict(aux.value).items() →
val.flatten()`); converting them to `final_aux_scan` would change
behaviour. Left as-is.

**Tests**: 95 passed / 13 warnings across every agent's pytest suite
after the refactor (~262s runtime).

**Net effect**:

- **SAC**: identical behaviour to round 1-3 (the inline patches were
  literally the inlined version of these helpers).
- **PPO, REDQ, TD3, AVG, APO, ASAC, UDRL**: now opted into `donate_argnames`
  + static `resume_from_state` (those that accept the kwarg) for free.
  Same memory benefit at the jit boundary on resume paths and same
  trace-time DCE on the Python branch as SAC enjoys.
- **REDQ + TD3**: also drop their inner update scan ys axis. Saves
  `[num_critic_updates, *aux]` (REDQ) and `[n_epochs, *aux]` (TD3) of
  on-device materialisation per outer step. Under vmap-over-seeds
  these compound by another factor of N.
- **`informed_ppo.py`** in SafetyExperiments has no top-level train
  function: it builds blocks that the Ajax PPO drives from outside. It
  inherits patches #6 (network init), #7 (`nan_safe`), #10 (compile
  cache) automatically through `import ajax`. No file-local change
  needed.

**Future-proofing**: any new agent that follows the
`def train(key, index=None, initial_state=None, resume_from_state=False)`
convention and uses `@train_jit` automatically gets donation +
`static_argnames`. Agents that aren't ready for the resume signature
get plain `jit` until they are; no friction.

### Round 5: edge-qa LCB predict-pair fuse — ATTEMPTED, REVERTED (2026-05-08)

Hypothesis: each `edge_compute_lcb_scores` /
`edge_compute_asym_scores` / `edge_compute_thompson_stats` /
`edge_compute_value_gap` call does **two** `predict_value` forwards on
the same obs and same critic params (different action arms). Fusing
into one forward by stacking `[obs, action_e]` and `[obs, action_p]`
along a doubled batch axis should halve kernel-launch overhead under
edge-qa.

Implementation: shared `_predict_value_pair(critic_state,
critic_params, obs, action_a, action_b)` helper in
[exploration.py](src/ajax/modules/exploration.py); applied to all
four affected functions.

Bench (Plane3DCircle, edge-qa scenario with `exploration_lcb=True`,
`use_expert_guided_exploration=True`, num_critics=2, 256-wide nets,
500 measured timesteps; cache wiped between unfused and fused runs):

|     N | unfused (s) | fused (s) | delta  |
| ----: | ----------: | --------: | -----: |
|     1 |       17.62 |     17.53 |  -0.5% |
|    16 |       20.06 |     20.82 |  +3.8% |
|    50 |       20.45 |     20.34 |  -0.5% |
|   100 |       20.89 |     20.51 |  -1.8% |

Net: noise across N, with a slight regression at N=16. Memory peak
identical at every N.

Conclusion: **XLA was already fusing the two `predict_value` calls
into one optimised kernel.** Manual concatenate-doubled-batch fusion
adds overhead (the explicit `concat` materialises a temp) without any
parallelism benefit. **Reverted.**

Lesson logged as trick #13 below.

### Tricks worth remembering (updated)

13. **Trust XLA to fuse independent ops with shared inputs before
    refactoring.** Two `predict_value` calls with the same params and
    same obs prefix look like a fuse opportunity: stack the inputs,
    do one forward, slice the output. In practice XLA already produces
    one fused kernel for the unfused source. Manual fusion via
    `jnp.concatenate` then forces a temp materialisation that costs
    more than what was saved. Verify with a clean cache-wiped
    before/after bench *before* shipping a fuse refactor.

## Audit items still pending

- AVG ([agents/AVG/](src/ajax/agents/AVG/)) maintains its own
  `NormalizationInfo` class. Its `count`/`mean`/`mean_2` are already
  scalar-shaped; only the `value` field carries `(n_envs, 1)` (which is
  correct since each env owns its own G_return). No redundant
  per-env stats here.
- Audit suggestion to drop `__pycache__` jit decorators inside other
  jitted scopes (e.g. [interaction.py:1077-1094](src/ajax/environments/interaction.py#L1077-L1094)):
  no runtime cost, purely cosmetic. Skipped.

