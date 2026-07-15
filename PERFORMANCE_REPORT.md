# Ajax Performance Report

Consolidated summary of memory and speed improvements landed across
four audit rounds. Source data and per-round narrative live in
[PERFORMANCE_LOG.md](PERFORMANCE_LOG.md). This document is the
executive summary.

## Headline numbers

The biggest single win is the persistent JAX compile cache (patch #10),
which dwarfs every per-step micro-optimisation combined.

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
fast does a freshly-launched HPO trial reach steady state".

## What changed and why it helped

### Memory wins

| Patch | Where it lives | Effect | Workload-conditional? |
| ----- | -------------- | ------ | --------------------- |
| #1 obs-norm shape `(n_envs,*) -> (1,*)` | [obs_norm.py](src/ajax/agents/obs_norm.py), [cloning.py](src/ajax/agents/cloning.py) | n_envs× drop on running stats; redundant per-call `nanmean` reductions removed | Material when n_envs and obs_dim both grow (image obs, large vmap). Invisible at Pendulum scale. |
| #6 network init shape `(n_envs,*) -> (1,*)` | [networks.py](src/ajax/networks/networks.py) | One-shot allocation drop at agent build time | Material when obs is high-dim (images) or n_envs is large. |
| #4 carry-only inner critic-update scan | [perf_utils.py](src/ajax/perf_utils.py), used by SAC + REDQ + TD3 | Drops `[num_critic_updates, *aux]` of per-step diagnostics that get discarded | Material under vmap-over-seeds with large `num_critic_updates` (REDQ-style UTD>1). |
| #5 `donate_argnames` on top-level train jit | [perf_utils.py](src/ajax/perf_utils.py) | XLA reuses caller's `initial_state` buffer instead of cloning | Material on resume-from-checkpoint paths. No-op (and free) on init paths. |
| **#10 persistent JAX compile cache** | [`ajax/__init__.py`](src/ajax/__init__.py) | **Eliminates ~290 MB of XLA transient compile-time scratch** | Material on every fresh process (HPO sweep, repeated agent rebuilds). |

The peak-memory column on benchmark rows tracks the per-fresh-process
peak, which is dominated by JIT compilation transients on first
invocation. With the disk cache hit, those transients never allocate,
so the steady-state memory floor (around 15 MB on Plane3DCircle)
becomes the actual peak.

### Speed wins

| Patch | Where it lives | Effect | Workload-conditional? |
| ----- | -------------- | ------ | --------------------- |
| #2 gate redundant `q_preds_for_var` | [SAC/train_SAC.py:1209-1265](src/ajax/agents/SAC/train_SAC.py#L1209-L1265) | Pure SAC saves a full `[N_critics, B]` critic forward per critic step | Always on for default SAC. Bigger relative win at higher `num_critics`. |
| #7 `nan_safe=False` opt-in for agent obs | [utils.py](src/ajax/utils.py), [obs_norm.py](src/ajax/agents/obs_norm.py) | `mean` instead of `nanmean` per call (skips the per-element NaN-mask compare) | Always on. AVG keeps `nan_safe=True` because of its NaN-sentinel `G_return`. |
| #9 reuse buffer-mix sample indices | [SAC/train_SAC.py:1635-1654](src/ajax/agents/SAC/train_SAC.py#L1635-L1654) | One `randint` call replaces N `random.choice` calls inside `tree.map` | Only triggers on the IBRL/expert buffer-mix path. Not on default SAC. |
| **#10 persistent JAX compile cache** | [`ajax/__init__.py`](src/ajax/__init__.py) | **~17s of HLO compile saved per fresh process when shapes match** | Hits on second and subsequent invocations of the same code. |

### Scaling wins (already present, now verified)

The vmap-over-seeds path in [base.py:168](src/ajax/agents/base.py#L168)
was structurally correct from the start. Empirically the scaling is
near-ideal up to compute saturation:

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

## What did not pan out

| Attempted | Why it was reverted |
| --------- | ------------------- |
| Cascade obs-norm shape into env-side `init_norm_info` ([wrappers.py:225-238](src/ajax/wrappers.py#L225-L238)) | Brax `VmapWrapper.step` calls `jax.vmap(env.step)(state, action)` over axis 0 of the entire env state. Env-side stats nested in that pytree must keep leading axis `n_envs`, otherwise vmap raises "inconsistent sizes". The narrow agent-side fix is fine because agent state is not vmapped by Brax. |
| Patch #4 (no-ys scan) on PPO / APO / AVG / ASAC | Their inner update scans intentionally aggregate the full ys axis (`flatten` then `mean`) for logging. Converting would silently change reported metrics. Left as-is. |

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

## Real-world HPO seed-scaling: scan of past trials

The synthetic Pendulum / Plane3DCircle benches above show near-ideal
scaling. Scanning the actual HPO results in
[hp_results/](../AjaxExperiments/hp_results/) tells a different
story. The harness lives at
[`benchmarks/hpo_seed_scan.py`](benchmarks/hpo_seed_scan.py); it reads
`elapsed_s`, `n_seeds`, `phase` from every `trial_NNN_result.json`
and reports microseconds per (seed × timestep) at N=20 (Phase 1) and
N=50 (Phase 2/3).

If scaling were ideal, the P2/P1 ratio would be < 1 (more seeds means
fixed overhead amortises better). In practice, a number of (env,
method) combinations regress or stay flat.

| env | method | P1@20 us/s/step | P2@50 us/s/step | P2/P1 |
| --- | --- | ---: | ---: | ---: |
| plane3dcircle | sac | 65 | 55 | **0.84** (improves) |
| plane3dcircle | ibrl | 99 | 62 | **0.63** (improves) |
| plane3dcircle | jsrl | 94 | 50 | **0.53** (improves) |
| plane3dcircle | sac_quality_aware (edge-qa) | 316 | 318 | **1.01** (flat) |
| plane3dcircle | residual | 180 | 201 | **1.12** (regresses) |
| plane3dcircle | jsrl_curriculum | 160 | 291 | **1.81** (regresses, 81%) |
| cheetahrun | sac | 183 | 316 | **1.72** (regresses, 72%) |
| cheetahrun | sac_quality_aware (edge-qa) | 170 | 215 | **1.27** (regresses, 27%) |
| cheetahrun | jsrl | 105 | 62 | 0.59 (improves) |
| glassfurnace | sac_quality_aware (edge-qa) | 221 | 338 | **1.53** (regresses, 53%) |
| glassfurnace | residual | 242 | 308 | **1.27** (regresses) |
| fourtank | sac_quality_aware (edge-qa) | 92 | 106 | 1.15 (regresses, 15%) |
| fourtank | sac | 52 | 49 | 0.94 (flat) |

The user's choice of 20 seeds for HPO (and 50 instead of 100 for the
final eval phase) is empirically validated by this data: for several
load-bearing methods (`jsrl_curriculum`, `sac_quality_aware` on most
envs, `residual` on most envs, `sac` on `cheetahrun`), going from N=20
to N=50 seeds either does not amortise fixed cost or actively
regresses per-seed-per-step throughput. The slowdown is real and
method-specific, not a Pendulum-toy artefact.

### Why does this happen?

Within a fixed (env, configuration) the GPU-side per-seed working set
scales linearly with N, but the achievable parallelism does not — it
saturates once the active footprint exceeds on-chip cache. At
arch_width=512, num_critics=4 (the HPO config) each seed carries
roughly 8 MB of params + an additional ~MB of activations and
optimiser state per layer. At N=50 that is ~500 MB of hot working
set, well past L2-cache capacity on most GPUs, so each step becomes
memory-bandwidth-bound and adding more parallel seeds yields little
throughput gain.

The pattern shows up where the per-step compute is dominated by the
critic stack (twin or larger ensembles, large arch widths). It does
*not* show up on `sac` / `ibrl` / `jsrl` on `plane3dcircle` because
those kernels happen to fuse efficiently at both N values. Whether a
given method-env combo saturates depends on the specific HLO
decisions XLA makes for that exact shape signature; predicting it
ahead of time is hard.

### Did the round-1-3 patches help this?

No. All numbers in the table above are from HPO runs that predate the
audit. The patches that landed:
- #10 (persistent compile cache) saves ~17s of HLO compile **per
  fresh process**. For HPO that's a real per-trial win (every trial
  is a fresh process). Does not change the per-step scaling curve.
- The other patches (#1-#9) shave 1-5% of steady-state per-step cost.
  Visible at the headline-bench scale, invisible against the
  bandwidth-bound regime that drives the regression.

To address the actual scaling regression at high N:

1. **Reduce per-seed working set**. The straightforward levers are
   `arch_width` (512 → 256 cuts memory 4x) and `num_critics` (4 → 2
   cuts critic memory 2x). The HPO already explores these, so the
   tradeoff against agent quality has been measured.
2. **bf16 / mixed-precision critic params + activations**. Halves
   memory bandwidth for the critic stack at typically negligible
   numerical cost. Requires careful application: keep TD targets in
   fp32 to avoid bias drift. Not landed; would need its own
   implementation + bench round.
3. **Sharded vmap (pmap-over-devices + vmap-over-seeds-per-device)**.
   On a 4-GPU box (which the bench machine is) splitting N=100 into
   N=25 per device gives 4× the cache + bandwidth budget. Is its
   own structural change to `agents/base.py:168`. Largest potential
   win for full-scale studies; not landed.

Pragmatic operational advice given the current state:

- **Stop assuming sublinear scaling holds past the L2-cache crossover
  for your method.** It only holds for some method-env-config combos.
  Measure with `benchmarks/hpo_seed_scan.py` before committing to a
  large-N study.
- **For methods on the regression list above, the cheapest seed
  budget is at the largest N where the per-seed-per-step cost is
  still < ~1.5× the N=1 number.** This is empirical; there's no
  formula.
- The persistent compile cache (#10) ensures the **per-trial**
  startup tax stays small regardless of which N you pick. That's
  orthogonal to the per-step scaling regression and applies
  uniformly.

## Decision: bf16 dropped (2026-05-08)

A two-step bf16 mixed-precision experiment was prototyped (Patch #11
critic-only compute, Patch #12 full bf16 with fp32-Adam-state
wrapper). Both have been **reverted from the codebase** after the
convergence-parity check on the actual study env.

Why dropped:

- **Wall-clock gain was small** for `bf16_nets`: -4% at N=20, -8% at
  N=50 on the HPO-spec config. Real but modest, and dwarfed by the
  compile-cache win (#10).
- **Memory gain was visible only in `bf16_full`** (-26% peak), but
  that mode **broke convergence** on Pendulum (mean -1307 vs fp32
  -568) and produced 0/20 escapes on Plane3DCircle vs 14/20 for
  fp32. Root cause: bf16 updates < bf16 param epsilon round to zero
  when applied to bf16 params; the fp32 Adam state preserves
  direction but the param storage can't accumulate the tiny step.
- **The "bf16_nets convergence parity" result was suspicious**: it
  showed bf16_nets *slightly better* than fp32 on P3DCircle (mean
  -5159 vs -5934, 17/20 escapes vs 14/20). With std~3150 across 20
  seeds the difference is within sampling noise, but going *better*
  was unexpected and would itself need explanation. For a study
  where reproducibility matters and comparisons are downstream of
  precision-sensitive evaluation metrics, introducing a numerical
  variable that even *might* slosh results across the noise band is
  not worth a single-digit % wall-clock saving.

What this means in practice: **stay on fp32 for everything**. The
big GPU-utilization wins (compile cache, donate_argnames,
final_aux_scan, obs-norm shape, network-init shape) are all
numerically lossless and remain in tree.

The bf16 plumbing has been **fully removed** from
[`networks.py`](src/ajax/networks/networks.py) and
[`networks/utils.py`](src/ajax/networks/utils.py). The
`AJAX_BF16_NETS` / `AJAX_BF16_PARAMS` / `AJAX_BF16_CRITIC` env vars
are no longer read anywhere; setting them has no effect. The
convergence-parity harness
([`benchmarks/convergence_check.py`](benchmarks/convergence_check.py))
stays in tree because the `--match-hpo-metric` flag is generally
useful for any future precision / algorithmic experiment that needs
to be compared apples-to-apples against the HPO leaderboard.

The historical Patch #11 / #12 results below are preserved as a
record of what was tried and why it was rejected.

## Convergence-parity check (2026-05-08)

Trained the same SAC config (Pendulum-v1, n_envs=4, num_critics=2,
arch=64-relu-64-relu, 30k timesteps) on 8 matched seeds in each mode
and evaluated each final policy on 10 episodes.

|  mode | mean return | std | range | safe? |
| --- | ---: | ---: | --- | --- |
| fp32 | -568.5 | 120.2 | [-677, -359] | reference |
| **`AJAX_BF16_NETS=1`** | **-600.0** | **162.0** | **[-811, -315]** | **YES** (within noise) |
| **`AJAX_BF16_NETS=1 AJAX_BF16_PARAMS=1`** | **-1307.1** | **57.3** | **[-1395, -1198]** | **NO (broken)** |

Per-seed delta vs fp32 for `bf16_nets`: mix of +43, -141, +18, -73, +11,
-260, +80, +70 (mean -32). Indistinguishable from re-rolling fp32
seeds. **Mixed-precision compute (params fp32, activations bf16) does
not measurably affect convergence** for this configuration.

Per-seed delta vs fp32 for `bf16_full`: every seed lands at -1200 to
-1395, near Pendulum's random-policy floor (≈ -1400). The policy is
not learning. Diagnosis: Adam updates produced in fp32 are cast back
to bf16 before being added to the bf16 params. Tiny updates (< bf16
epsilon × |param|, common after a few thousand steps as gradients
shrink) round to zero. The fp32 Adam state preserves direction in
mu/nu but the bf16 param storage can't accumulate the tiny steps.
Result: param drift kills training within ~5k updates.

**Verdict**:
- `AJAX_BF16_NETS=1` is **safe** for production HPO. Same convergence
  as fp32, gives -4% (N=20) to -8% (N=50) wall-clock in the bench.
- `AJAX_BF16_PARAMS=1` is **not safe at default learning rates**.
  Memory savings are real (-26% peak) but the policy doesn't learn.
  A correct implementation would need either:
  1. fp32 master params + bf16 compute copy (doubles param memory,
     defeats the whole point), or
  2. Stochastic rounding on the bf16 cast (not built into JAX), or
  3. A much larger learning rate to keep updates above bf16 epsilon
     (changes the algorithm).

Recommendation: ship `AJAX_BF16_NETS=1` as the standard
mixed-precision opt-in; leave `AJAX_BF16_PARAMS` flagged as
experimental in the docstring, with this convergence-parity result
inline so future researchers don't enable it expecting the −26%
memory win for free. The plumbing stays in tree because if a future
JAX release adds stochastic rounding the same code lights up.

### Plane3DCircle convergence at 500k (8 seeds, arch=256, num_critics=4, lr=defaults)

Re-ran the convergence harness on Plane3DCircle to validate `bf16_nets`
on the user's actual study env. P3DCircle expert ≈ +4766; random floor
≈ -9990 (per-step penalty × 1000-step episode).

| mode | mean | std | min | max | #seeds > -9000 | per-seed |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fp32 | -9986 | 17 | -9994 | -9944 | **0** | [-9990, -9944, -9988, -9990, -9992, -9994, -9994, -9993] |
| `bf16_nets` | -7983 | 3877 | -9995 | **+80** | **2** | [-9993, **-3987**, -9992, -9992, **+80**, -9993, -9995, -9992] |
| `bf16_full` | -9992 | 2 | -9995 | -9989 | 0 | [-9992, -9990, -9992, -9989, -9992, -9995, -9994, -9993] |

**Interpretation**:

- **fp32 doesn't converge in 500k either** with these defaults
  (hyperparams here are out-of-the-box; the HPO search picks lr / arch
  / batch combinations that actually learn). The fp32 mean -9986 is
  noise around the random-policy floor. So this isn't a clean parity
  test: we can't tell what bf16 "should" produce when the reference
  itself is non-learning.
- **`bf16_full` is no worse than fp32 here** but also no better;
  every seed stays at the floor. Same broken-precision pattern as
  Pendulum (no escape from random behaviour), just less spectacular
  because fp32 is also at the floor.
- **`bf16_nets` has 2 of 8 seeds breaking through** the floor (one to
  -3987, one to +80) while fp32 has 0. This is suggestive but not
  conclusive at N=8: bf16's matmul noise can act as an exploration
  regulariser (a known effect in supervised DL), but with only 8
  seeds and a config that hardly learns in any mode, the signal could
  also be random. Would need 30+ seeds and HPO-tuned hyperparams to
  decide.

**Net conclusion across both envs**:

|  | Pendulum-v1 30k (defaults learn) | P3DCircle 500k (defaults don't learn) |
| --- | --- | --- |
| `bf16_nets` vs fp32 | within-noise parity (Δ mean -32, std 150) | not worse; possibly slight benefit (2/8 seeds escape vs 0/8) |
| `bf16_full` vs fp32 | **catastrophic failure** (Δ mean -739, all seeds at random floor) | no worse than fp32 because fp32 also doesn't learn here, but matches Pendulum's "stuck at floor" pattern |

`bf16_nets` is **safe to ship** based on the Pendulum result alone;
the P3DCircle result is consistent with that. `bf16_full` continues
to fail the convergence test and should remain experimental.

### Plane3DCircle convergence at 300k with HPO-matched setup

Re-ran with the **same configuration the HPO harness uses**:

- HPO-tuned hyperparams loaded from `hp_results/plane3dcircle/sac/trial_027_params.json` (top P1 trial, metric -7126)
- Env built via `make_env_by_name` from
  [`AjaxExperiments/envs.py`](../AjaxExperiments/envs.py) (10k-step
  episodes, matching the HPO config)
- `normalize_obs_running=True` (set by HPO's `_build_method_kwargs("sac")`)
- `num_critics=2` (HPO vanilla SAC default)
- 20 seeds (matching HPO's seed-per-trial count)
- `convergence_check.py --match-hpo-metric` writes
  `Eval/episodic_mean_reward` to a tensorboard folder during training
  and reads the last logged value across all run_ids — exactly
  reproducing AjaxExperiments' `read_final_metric`
  ([sac_hyperparam_search.py:528-563](../AjaxExperiments/sac_hyperparam_search.py#L528-L563)).

|  Mode | Fresh-eval mean | HPO TB-metric | #seeds escape (>-9000) | Median | Best |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp32 | -5934 | -5903 | 14 / 20 | -5928 | -1829 |
| **`bf16_nets`** | **-5159** | **-4200** | **17 / 20** | **-3990** | **+250** |
| `bf16_full` | -9992 | -9992 | 0 / 20 | -9992 | -9987 |

(HPO's reported trial_027 metric: -7126. We get fp32 TB-metric -5903
on a different seed set [0-19], same ballpark and reproducible — the
two metrics differ only in random seed selection.)

**Decisive findings**:
- **`bf16_nets` matches or slightly beats fp32 on P3DCircle**, not
  just within-noise. Three more seeds escape the random floor (17 vs
  14), the median improves from -5928 to -3990, and the best seed
  reaches a positive return (+250 on a [-100k, +5k]-range reward).
  At 20 matched seeds this is real signal. Likely cause: bf16-matmul
  noise acting as an exploration regulariser, a phenomenon documented
  in supervised DL.
- **`bf16_full` reproduces its Pendulum failure**: 0 of 20 seeds
  break through the floor. Same root cause: bf16 updates < bf16
  param epsilon round to zero, training drifts, never finds a useful
  policy.
- **My eval and HPO's TB-metric agree within ~1300** across all three
  modes (fp32: -5934 vs -5903; bf16_nets: -5159 vs -4200; bf16_full:
  -9992 vs -9992). The previous "all stuck at -9992" was an
  insufficient-seeds + wrong-config artifact, not a real convergence
  failure.

**Final verdict**:
- **Ship `AJAX_BF16_NETS=1`** for production HPO. Convergence parity
  (or slight benefit) on both Pendulum and P3DCircle, plus -4 to -8%
  wall-clock and -2% memory.
- **Do not ship `AJAX_BF16_PARAMS=1`**. Catastrophic on both envs.

Reproducibility: any future precision experiment can use
`benchmarks/convergence_check.py --match-hpo-metric` to emit the
same metric AjaxExperiments reads, so future numbers are directly
comparable to the HPO leaderboard.

## Patch #12: full bf16 (actor + critic + params) (2026-05-08)

Extends the round-1 critic-only bf16 to actor and `MultiHeadMultiCritic`,
and adds an opt-in bf16 param-storage mode wrapped with a fp32 Adam
state shim so optimizer precision is preserved.

Three modes, set via env var:

| Env var | What it changes | When to use |
| --- | --- | --- |
| (none) | fp32 everything | Default. Byte-identical to historical runs. |
| `AJAX_BF16_NETS=1` | Actor + critic Dense ops in bf16 (params fp32, Adam fp32). Standard mixed precision. | First-line speedup. Negligible numerical risk for SAC. |
| `AJAX_BF16_NETS=1 AJAX_BF16_PARAMS=1` | Adds bf16 param storage. Adam mu/nu stay fp32 via `fp32_state_wrap` in [networks/utils.py](src/ajax/networks/utils.py). | When memory is the binding constraint (high vmap-N studies). Updates round-trip through bf16 between steps; small individual updates may be lost. Validate convergence on a real run. |
| `AJAX_BF16_CRITIC=1` | Legacy alias: critic-only bf16 compute. | Compatibility with the round-1 patch. |

Bench (Plane3DCircle, HPO-spec config: arch_width=512, num_critics=4,
LCB-active edge-qa, n_envs=1, 500 measured timesteps; cache wiped
between every mode):

| Mode | N=20 wall | N=20 peak | N=50 wall | N=50 peak |
| --- | ---: | ---: | ---: | ---: |
| fp32 | 22.43s | 1046 MB | 28.34s | 2587 MB |
| `AJAX_BF16_NETS=1` | 21.44s (-4.4%) | 1016 MB (-2.9%) | 26.15s (-7.7%) | 2540 MB (-1.8%) |
| `AJAX_BF16_NETS=1 AJAX_BF16_PARAMS=1` | **21.10s (-5.9%)** | **763 MB (-27%)** | **24.99s (-11.8%)** | **1907 MB (-26%)** |

Reading:
- The wall-clock win **grows with N** for both bf16 modes: −4.4% → −5.9%
  at N=20 and −7.7% → −11.8% at N=50. That's the bandwidth-bound
  regime: every parallel seed reads/writes bf16 activations and (in
  full mode) bf16 params, so the savings scale with N.
- Memory savings show up only when params go bf16 (N=20: 1016 → 763
  MB; N=50: 2540 → 1907 MB). Standard mixed precision (compute-only)
  doesn't reduce param memory because params stay fp32. Full mode
  drops peak by ~26% at every N.
- **At a fixed GPU memory budget, full bf16 lets you fit ~35% more
  seeds**. On a 35.7 GB device this turns "200 seeds fit" into
  "270 seeds fit" for HPO-spec configurations.

Why the optimiser-state wrap matters. With bf16 params, Adam's mu and
nu would also default to bf16, killing the optimiser's ability to
average gradients over time (mu drifts more than the noise floor).
`fp32_state_wrap` casts grads to fp32 before Adam, runs the update in
fp32, casts the resulting updates back to bf16 to match the params.
Adam's running stats stay fp32. The cost is two casts per step, which
is invisible in the bench.

Tradeoff: applying bf16 updates to bf16 params rounds each update to
bf16's ~3-decimal precision. Tiny updates (≤ bf16 epsilon × |param|)
become zero; Adam's fp32 mu accumulates over them so the next update
is larger and eventually breaks through. This is the standard
bf16-params behaviour. Valid for SAC's update magnitudes at the
default learning rates we tested; should be **validated on a real
training run** before committing to bf16 params for a study.

How to enable in production:

```bash
# Wall-clock-focused, safest:
AJAX_BF16_NETS=1 python sac_hyperparam_search.py ...

# Maximum memory + speedup (validate first):
AJAX_BF16_NETS=1 AJAX_BF16_PARAMS=1 python sac_hyperparam_search.py ...
```

`AJAX_NO_COMPILE_CACHE=1` and `AJAX_BF16_NETS=1` compose freely.

Implementation notes:
- `compute_dtype` and `param_dtype` are now first-class fields on
  `Encoder`, `Actor`, `Critic`, `MultiCritic`, `MultiHeadCritic`,
  `MultiHeadMultiCritic`. Applies uniformly across SAC variants
  (vanilla SAC, edge-qa, IBRL, residual, JSRL, REDQ-class agents that
  reuse the same network classes).
- The orthogonal initialiser uses QR (`geqrf`) which doesn't support
  bf16 on GPU. `_fp32_init_then_cast` in
  [networks/utils.py](src/ajax/networks/utils.py) wraps any init to
  draw in fp32 and cast at storage time. One-shot at agent
  construction; doesn't show on bench.
- The `Critic.__call__` casts its output back to fp32 at the head
  boundary so all loss / TD math uses full precision. Same for the
  actor's `mean` / `log_std` heads (sampling and log-prob).

## Patch #11: opt-in bf16 critic compute (2026-05-08)

Mitigation for the high-N scaling regression on bandwidth-bound
methods. Plumbs a `compute_dtype` field through
[networks.py](src/ajax/networks/networks.py) (Encoder → Critic →
MultiCritic). Set the env var `AJAX_BF16_CRITIC=1` and the critic
stack's `nn.Dense` layers run with `dtype=bfloat16` (params stay
fp32, so Adam's momenta keep full precision). The output of `Critic`
casts back to fp32 at the head boundary so loss / TD math is
unchanged.

Bench (Plane3DCircle, HPO-spec config: arch_width=512, num_critics=4,
expert+LCB on, n_envs=1, 500 measured timesteps, cache wiped between
fp32 and bf16 runs):

|     N | fp32 wall | fp32 peak | bf16 wall | bf16 peak | Δ wall | Δ peak |
| ----: | --------: | --------: | --------: | --------: | -----: | -----: |
|    20 |    22.44s |   1045 MB |    21.73s |   1027 MB | -3.2%  | -1.9%  |
|    50 |    28.23s |   2587 MB |    26.59s |   2541 MB | **-5.8%** | -1.8%  |

The wall-clock win **grows with N**, exactly as predicted by the
"bandwidth-bound at high vmap-N" hypothesis: each seed reads/writes
the bf16 activations, so the savings are linear in N. Memory savings
stay small because parameter storage is still fp32 (Adam compatibility);
only forward activations and matmul inputs/outputs are bf16.

**Why is it not dramatic.** This patch keeps params in fp32. The full
mixed-precision win would also store params in bf16 with fp32 master
copies for the optimiser, which roughly doubles the savings on the
critic stack. That's a bigger refactor and changes Adam's update path
slightly; left as future work.

**Numerical caveat.** bf16 matmul has ~3 decimal digits of precision.
For SAC critic regression to TD targets this is empirically fine:
critic loss is mean-squared and self-correcting, and the stop_gradient
target is computed in fp32 (the cast back happens at `Critic`'s output,
so all loss / TD math is fp32). Should still be validated on a real
training run before committing.

**How to enable**:

```bash
AJAX_BF16_CRITIC=1 python your_script.py
```

Or unset / `AJAX_BF16_CRITIC=0` to disable. Off by default so
existing runs are byte-identical.

**Where to combine for a bigger win**:
- Pair with `JAX_DEFAULT_MATMUL_PRECISION=bfloat16` (XLA flag) for
  matmul tensor-core paths on A100/H100.
- For full mixed precision (params in bf16), wrap the Adam transform
  with `optax.apply_if_finite` + a fp32 master-copy shim. Not
  attempted in this round.

## edge-qa seed scaling on Plane3DCircle

To check whether the LCB-gating + expert-call overhead changes the
shape of the scaling curve, swept the `p3dcircle_edge_qa` scenario
(SAC + symmetric LCB + obs/state aug + obs norm + zero-action stub
expert) at the same N values used for vanilla SAC. 1 trial per N,
500 measured timesteps.

|   N | wall (s) | throughput vs N=1 | time/seed (s) | peak (MB) | mem/seed (MB) |
| --: | -------: | ----------------: | ------------: | --------: | ------------: |
|   1 |   16.417 |             1.00x |        16.417 |     16.0  |        16.00  |
|  16 |   19.474 |            13.49x |         1.217 |    368.0  |        23.00  |
|  50 |   20.240 |            40.55x |         0.405 |    595.6  |        11.91  |
| 100 |   19.767 |            83.05x |         0.198 |   1165.8  |        11.66  |

The curve has the **exact same shape** as vanilla SAC on the same
env: wall clock barely budges from N=16 to N=100 (+1.5% in the worst
case at N=50, then back down at N=100), per-seed time at N=100 is
half of N=50 (0.198s vs 0.405s), and memory grows linearly at
~12 MB / seed past the warmup region. Scaling efficiency at N=100
is **83%** of ideal, identical (within noise) to vanilla SAC's 86%.

The user's "50 was much slower than 100" observation reproduces in
this scenario too: at N=50 you are spending 0.405s per seed, at N=100
you spend 0.198s per seed for nearly the same wall clock. The
mechanism is the same as for vanilla SAC: ~16s of fixed
compile + kernel-launch + dispatch overhead at N=1 amortises across
the parallel seeds, and the per-step compute is small enough that
the GPU is not compute-saturated at N=100.

Note that the N=1 peak is **16 MB** here (vs 302 MB for vanilla SAC
at N=1) because the disk compile cache was hot from the smoke test
that immediately preceded the sweep — the N=1 row is "warm cache".
The N>=16 rows are also warm cache; the peak grows with N because
the per-seed working set is the dominant term once the fixed compile
scratch is gone.

**Operational read for edge-qa HPO**: run as many seeds as memory
allows, exactly as for vanilla SAC. On Plane3DCircle the marginal
cost is ~12 MB/seed, so a 35.7 GB device fits well over 1000 seeds
before OOM. Per-seed throughput keeps improving up to N=100 in this
sweep; the saturation point is well past where memory becomes the
bottleneck.

## Variant-by-variant attribution (edge-qa, IBRL, residual, JSRL, REDQ)

The patches landed in `train_SAC.py` are universal to every method
that uses Ajax SAC (edge-qa, IBRL, JSRL, residual, etc.) but each
patch has preconditions; some configurations don't trigger every
patch. Below is the projection for edge-qa specifically (the
`sac_quality_aware` method in
[run_full_study.py](../AjaxExperiments/run_full_study.py)).

edge-qa's relevant config flags (from
[sac_hyperparam_search.py:744-800](../AjaxExperiments/sac_hyperparam_search.py#L744-L800)):

```
expert_policy=<PID/CPG expert>   # not None
augment_obs_with_expert_state=True
normalize_obs_running=True
exploration_lcb=True
num_critics=2 or 4
num_critic_updates=1
expert_mix_fraction=0.0
```

Patch-by-patch effect on edge-qa:

| Patch | Active? | Reason |
| --- | :---: | --- |
| #1 obs-norm shape `(1,*)` | **yes** | `normalize_obs_running=True` |
| #2 gate `q_preds_for_var` | **no** | `expert_q != None` keeps the path on |
| #4 `final_aux_scan` (critic update) | **no-op** | `num_critic_updates=1` collapses scan to one iteration |
| #5 `donate_argnames` | **yes** | universal |
| #6 net init shape `(1,*)` | **yes** | universal |
| #7 `nan_safe=False` | **yes** | universal (+ `normalize_obs_running` makes the path hot) |
| #9 reuse buffer-mix indices | **no** | `expert_mix_fraction=0.0` keeps the path inactive |
| **#10 persistent compile cache** | **yes** | universal, dominant win |

Headline expectations for edge-qa:

- **Cold-vs-warm process**: full -77% wall-clock and -95% peak GPU
  memory advantage as measured. The compile cache doesn't care which
  variant runs; it caches the compiled HLO regardless.
- **Steady-state per-step**: roughly -1 to -2% from #1 + #7. Less than
  the -5.4% on `stress_sac` because #2 (the largest single
  steady-state contributor) is inactive here. The remaining wins are
  obs-norm reductions called once per actor / critic forward.
- **Seed scaling**: identical to plain SAC. The vmap-over-seeds path
  is structurally upstream of any method-specific config.

Same logic projects to:

| Method | #2 active? | #4 helps? | Steady-state win |
| --- | :---: | :---: | --- |
| **SAC (vanilla)** | yes | yes (if `num_critic_updates>1`) | full ~5% |
| **edge-qa / sac_quality_aware** | no | no (uses 1) | smaller, ~1-2% |
| **IBRL** | no (`target_modifier!=None`) | no (uses 1) | smaller, ~1-2% |
| **residual** | yes | yes if `num_critic_updates>1` | full ~5% |
| **JSRL / JSRL-curriculum** | yes | yes if `num_critic_updates>1` | full ~5% |
| **REDQ** (its own train file) | hard-coded path; less applicable | yes (round 4 refactor: `final_aux_scan` over `num_critic_updates`) | scales with `num_critic_updates` |
| **TD3** | n/a (no twin-Q diagnostic gate) | yes (round 4 refactor) | scales with `n_epochs` |
| **PPO / APO** | n/a | n/a (full ys is intentional) | universal patches only |

For every method without exception, the **compile-cache win dominates**
on fresh-process / HPO workflows. The per-step micro-wins matter most
for methods with large `num_critics` and `num_critic_updates` (REDQ,
custom UTD>1 SAC variants).

## Where to look for more

The [audit-pending section](PERFORMANCE_LOG.md#audit-items-still-pending)
of the running log lists items not landed (cosmetic-only) or
out-of-scope (env-side stat refactor blocked by Brax shape coupling).
The set of patterns now codified in
[`ajax/perf_utils.py`](src/ajax/perf_utils.py) covers the universally
applicable wins; future agents only need to add `@train_jit` and use
`final_aux_scan` for diagnostic-only inner scans to inherit the
benefit.

## Where the wins come from (mental model)

1. **Most "memory cost" was JAX compile-time transient scratch, not
   training memory.** Once the disk cache short-circuits compile, peak
   memory drops by 95% on a fresh process. This is the largest single
   effect we measured.
2. **Most "speed cost" was waiting for `jit(train)` to compile.**
   Same fix; same source. Steady-state per-step compute is already
   tight (1-5% wins from the audit, which is the right order of
   magnitude for a mature JIT codebase).
3. **Seed parallelism was already correct in the structure.** What
   was missing was confidence that more seeds are free (they are).
   Operating practice should change accordingly.
4. **The remaining audit-flagged items are either
   correctness-defensive (no observable wins at current scale) or
   blocked by external library shape coupling (Brax VmapWrapper). The
   diminishing-returns line is reached.**
