# DreamerV3 and TD-MPC2: version choices and deviations

The fidelity rule (see `DESIGN.md` §0): reproduce the code that produced each paper's
published numbers, adopt only the bug fixes the authors made later, make asynchronous or
unseeded infrastructure synchronous and seeded, and treat paper typos as typos. This file
records (1) every place where the paper-era code and the latest official code differ and
which side Ajax follows, and (2) every place where Ajax departs from the paper-era code.
Each entry is also cited in the docstring of the code it concerns.

Pinned sources:
- DreamerV3: paper arXiv:2301.04104v2 (= Nature 2025); paper-era code
  `danijar/dreamerv3@2411f7d`; bug fix `29eb964`; latest checked `e3f0224`.
- TD-MPC2: paper arXiv:2310.16828v2 (ICLR 2024); paper-era code
  `nicklashansen/tdmpc2@b67b21c` (curves) / `5f6fade` (identical agent algorithm);
  latest checked `e9f5932`.

## 1. DreamerV3: paper-era (2411f7d) vs latest (e3f0224)

| Aspect | 2411f7d (paper era) | e3f0224 (latest) | Ajax |
|---|---|---|---|
| DMC-proprio preset | 12M, train ratio 512, action repeat 2, 16 envs (Table 2) | 1M, ratio 1024, repeat 1 | 2411f7d, as a documented protocol |
| Training-start gate | `len(replay) >= batch_size` (16 items); `when.Ratio` returns 1 on its first call | `len(replay) >= B*T` (1024 items) | 2411f7d |
| Discrete actor | one-hot with 1 % unimix; log-prob and entropy of the mixed distribution | categorical, no unimix | 2411f7d |
| Start-state reward / continuation in imagination | from replay data: `reward_t`, hard `1 - is_terminal_t` | model predictions (`ĉ_0 ≈ 0.997` enters the weight) | 2411f7d |
| Vector-decoder output scale | 0.1 (Dist default) | 1.0 | 2411f7d |
| Symlog-MSE | squared errors < 1e-8 zeroed; no ½ factor | no tolerance; no ½ factor | 2411f7d |
| Slow critic | separate f32 module, hard copy after the first optimizer step, then EMA 0.02 | copy at creation, EMA 0.02 from step 1 | 2411f7d rule (`mix = 1` at update 0, then 0.02); created as a copy of the critic, which predicts what 2411f7d's separate module predicts until the first update replaces it (both output layers are zero) |
| First `prevact` in the replay context | bug: previous unrelated batch's carry | `data[action][:, K-1]` | **fixed** (29eb964, an upstream bug fix) |
| RMSNorm statistics | input dtype (bf16) | f32 | n/a (Ajax computes in f32, see D4) |
| Policy tanh / sigmoid | after the f32 cast | on bf16 outputs | n/a (f32) |
| Integer vector observations | cast to f32 and symlogged | one-hot, not symlogged | 2411f7d (Ajax observations are float) |
| Stored stochastic latent | int32 class indices | f32 one-hot | indices (uint8 when classes ≤ 256) |
| Key order / decoder input order | obs-space order / concat(deter, stoch) | sorted / concat(stoch, deter) | layout only |
| Two-hot output width (reward head; critic) | `Linear(bins + 1 = 256)`, last logit dropped (`nets.py:432-438`) | `Linear(bins = 255)` (`embodied/jax/heads.py:132-135`) | 255, layout only: the dropped column starts at 0 (outscale 0) and gets exactly zero gradient, so it stays 0 under AGC + LaProp and training is identical; `units + 1` fewer parameters per head |
| LaProp β2 | 0.999 (paper text says 0.99) | 0.999 | 0.999 (the code behind the curves) |
| Paper Table 3, 12M recurrent units | — | — | 2048 (= 8d; the printed 1024 is a typo) |
| Paper Table 11 task mean / median rows | — | — | swapped in the paper; mean ≈ 754, median ≈ 871 at 500K |

## 2. TD-MPC2: paper-era (5f6fade = b67b21c agent) vs latest (e9f5932)

| Aspect | 5f6fade / b67b21c (paper era) | e9f5932 (latest) | Ajax |
|---|---|---|---|
| Policy entropy bonus | `+β·log_pi`, `log_pi = n·(Σ(−½ε²−logσ) − ½ln2π) − Σ log(relu(1−tanh(u)²)+1e−6)`; tanh term unscaled, with gradient | tanh term without gradient; ½ln2π per dim | paper era |
| Task-embedding gradient in the policy loss | none (`track_q_grad(False)`) | leaks into the next world-model step | paper era (stop-gradient) |
| World-model clip norm | includes the previous update's post-clip policy gradients | world-model gradients only | paper era (reproduced exactly; one carried scalar) |
| Termination / episodic mode | absent | optional head + masking | absent; terminations refused (T10) |
| Q-ensemble init | each member N(0, 0.02), last weight 0 | same (restored in e9f5932 after a 2024-09 regression) | same |
| Q-ensemble dropout (p = 0.01) | active in **every** Q pass, eval mode included: TD target (target Q), value and policy losses, planning. The `combine_state_for_ensemble` functional module lives only in the `torch.vmap` closure, so `train()` / `eval()` never reach it (`common/layers.py:12-21`; verified by running 5f6fade, recorded by `parity/tdmpc2_update_fixtures.py`; `plan()` runs in eval mode, recorded by `parity/tdmpc2_plan_fixtures.py`) | registered submodule: dropout only in train mode (value and policy losses) | paper era: the TD target takes a dropout key (M2); planning takes one per MPPI iteration for its terminal Q pass (M3) |
| Replay buffer | b67b21c: whole episodes, capacity in episodes, uniform episode × random crop | rows + SliceSampler | b67b21c episode buffer |
| TD bootstrap state | code: encoded next observation `h(s')` (paper Eq. 3 writes the dynamics prediction) | same | code |
| Loss normalisations | /H, /(H·Nq), latent-mean MSE, H terms (paper writes sums over H+1 terms) | same | code |
| Encoder learning rate | 3e-4 × 0.3 = 9e-5 (paper Table 8: 1e-4) | same | code |
| Planner σ warm start | μ only; σ reset to max_std (paper: warm-start μ and σ) | same | code |
| Executed action | score-sampled elite + σ₀ noise (paper: sample N(μ*, σ*)); the elite by `np.random.choice(p=score)` (inverse CDF of one uniform) | same (Gumbel-max instead of `np.random.choice`) | code; the paper-era inverse CDF, in float32 |
| Terminal action in the value estimate | `π(z_H)` (paper Eq. 6: planner Gaussian) | same | code |
| Paper Eq. 4 sign | literally minimises entropy (typo) | — | code sign (maximise Q/S + β·entropy) |
| SimNorm temperature | none (paper Eq. 5 vs App. H contradict each other) | none | none |
| task_dim | 96, but 64 for mt30 at 5/19/48M ("account for slight inconsistency") | same | 96 (T14) |

## 3. Deviations register

| # | Agent | Reference behaviour | Ajax behaviour | Why / impact |
|---|---|---|---|---|
| D1 | DreamerV3 | Asynchronous: acting parameters ≈ 2 driver iterations stale; one-batch prefetch; write-back and metrics one train call late; first batch prefetched before the gate | Synchronous | Infrastructure, not algorithmic |
| D2 | DreamerV3 | Replay sampler RNG fixed to seed 0; the report stream pops the online queue every 180 s of wall-clock | Seeded from the run seed; no report stream | Infrastructure |
| D3 | DreamerV3 | Updates interleaved between the per-env adds of one vector step | Updates after the tick's adds | Same rate; intra-tick order only |
| D4 | DreamerV3 | bf16 compute (2411f7d: bf16 RMS statistics, f32 slow critic) | f32 | Numerics only; bf16 revisited at the GPU milestone |
| D5 | DreamerV3 | One decoder head per observation key (dict observations) | One head for the flat observation vector | Loss identical (sum of means); the per-tensor AGC partition differs (≤ 8 % clipped-gradient difference, probed) |
| D6 | DreamerV3 | Optimizer and normaliser state checkpointed; Ratio state, online queue and replay RNG not | Everything checkpointed | Resume fidelity |
| T7 | TD-MPC2 | Single environment | `n_envs > 1`: per-env t0 and warm-start mean, `n_envs` updates per stepping tick, seed phase / burst rule of `DESIGN.md` §4.4 | Generalisation; exact for `n_envs = 1` |
| T8 | TD-MPC2 | The first planned step after seeding warm-starts from the step-0 evaluation's leftover mean | Zeros | Negligible |
| T9 | TD-MPC2 | Evaluation reuses the training env instance and planner mean; the first evaluation is at step 0 (untrained agent) | Rebuilt evaluation env, separate planner carry; each evaluation from fresh initial states (its key folded with the number of evaluations so far: deterministic per seed, fresh across evaluations as the reference's resets of its running env are); evaluations every `log_frequency` env steps from the start of training, the first after `log_frequency` steps (on the held tick ending an episode when `log_frequency` is a multiple of `T·n_envs`, as the reference's `eval_freq` is) | Evaluation protocol only |
| T10 | TD-MPC2 | Paper era: no termination guard (any done treated as a time limit); latest: ValueError | Terminations counted (episode ends off the `T`-step schedule, and true terminations on its last step); ValueError raised after the run | Loud failure instead of silently wrong targets |
| T11 | TD-MPC2 | Seed-phase actions unseeded (gym space RNG) | Seeded | Reproducibility |
| T12 | TD-MPC2 | Adam state and RunningScale not checkpointed | Checkpointed | Resume fidelity |
| T13 | TD-MPC2 | Matmul precision: TF32 (latest) / full fp32 (paper era) | `highest` in parity tests; set explicitly in the GPU report | Numerics |
| T14 | TD-MPC2 MT | task_dim 64 for mt30 at 5/19/48M (an acknowledged accident) | 96 (paper) | Ajax's task set is not mt30 |
| T15 | TD-MPC2 MT | Evaluation-time embedding renorm persisted in place | Renorm applied but not persisted at evaluation | Evaluation cannot write parameters |
| T16 | TD-MPC2 MT | `evaluate.py` normalised score with exploration noise on; the offline trainer evaluates after its first update, then every `eval_freq` | Offline-trainer protocol (`eval_mode`, 10 episodes per task, `t0` at each episode's first step), every `log_frequency` updates of the run, the first after `log_frequency` updates; each evaluation from fresh initial states; normalised score `mean(return / 10)` (the DMC normalisation) | The trainer protocol is the one behind the paper's training curves |
| T17 | TD-MPC2 MT | Datasets mt30 / mt80 (replay of 240 single-task agents, random to expert) | Self-generated: full training history of single-task Ajax runs on the playground versions of the 19 original mt30 DMC tasks | Validates the mechanisms, not the paper's numbers |
| E18 | both | dm_control (MuJoCo C) | mujoco_playground MJX ports: reduced solver iterations, some `sim_dt` changes (e.g. reacher 0.005 vs 0.02), `impl='jax'` on CPU. QuadrupedRun / DogRun and the 11 custom TD-MPC2 tasks are unavailable | Environment, not algorithm |
| E19 | both | dm_control action repeat stops at the episode end | brax `EpisodeWrapper` repeat continues after a termination | None on DMC (never terminates) |
| E20 | both | Fresh random initial state every episode | Same: static reset mode resets freshly itself; dynamic mode uses the auto-reset, fresh on every backend since playground envs default to `fresh_reset=True` (#54). Only a playground env built with `fresh_reset=False` restarts from a cached first state in dynamic mode (warned) | None with the default env stack |
| E21 | both | — | Golden checksums are CPU-only regression guards | GPU nondeterminism (TF32, scatter-add order) |
| D22 | DreamerV3 | Two-hot bins `symexp(jnp.linspace(-20, 0, 128))` evaluated in float32 inside the jitted step; expectation `p_m·b_m + Σ (p_i·b_i + p_j·b_j)` over mirror pairs, which XLA fuses into multiply-adds under `jit`, so a zero-initialised head predicts 0.07 to 0.16 (CPU, depending on the batch shape), not the 0 of paper p.18 | Bins: float32 rounding of the float64 values, computed once with numpy (the same constants eagerly, under `jit` and on every backend); expectation `Σ (p_j − p_i)·b_j` over mirror pairs, exactly 0 for uniform p also under `jit` | Same expectation in exact arithmetic: bins within 1.3e-6 relative of the reference's; encode and loss bit-identical given the same bins; decode within 3·ε·E_p\|b\| (ε = float32 epsilon). **Not negligible in training.** Once the logits move, the reference's contracted pair sum adds noise of std ≈ 0.085 (max ≈ 0.3) to every reward, value and slow-critic prediction, plus a constant at zero logits that depends on the compiled program (−0.17 to +0.16); the noise fades as the outer bins lose mass (CartPole, 1m: value std 0.085, 0.052, ≈ 0.01 over the 2000-, 4000- and 6000-row windows). It slows the early decline of the actor's entropy (CartPole, 1m, 4000-row window: 0.51 with the reference's sum, 0.46 with Ajax's, measured in both codebases); with the same expectation on both sides (and the replay sampler seeded per run), the training statistics' seed ranges overlap the reference's in at least 52 of the 58 windows with updates (`PERFORMANCE_REPORT.md`, DreamerV3 reference comparison, round 3, for the exceptions). Ajax's form is exact only while the outer logits stay bitwise mirror-symmetric, which depends on the compiled program (a one-seed run and a three-seed vmapped run differ): a float32 fragility of the bins (±4.85e8, holding ≈ 1/255 of the mass early on) that both codebases share |
| D23 | DreamerV3 | Kernel init `1.1368 · sqrt(1/fan_in) · TruncatedNormal(−2, 2)` | `variance_scaling(outscale², 'fan_in', 'truncated_normal')`, i.e. the exact factor 1/0.87962566 | Same draws up to a constant factor: 4.16e-5 relative |
| T24 | TD-MPC2 | Two-hot over `torch.linspace(vmin, vmax, 101)` (middle bin −1.5e-7, not 0), any `vmin < vmax`; target by the floor formula `k = floor((x − vmin)/Δ)`; naive expectation `Σ p_k·b_k`; `F.log_softmax` | Bins: float32 rounding of the float64 grid, mirrored (within 4.8e-7 of torch's); target by linear interpolation between the neighbouring bins; symmetric expectation as in D22; `logits − logsumexp(logits)`; symmetric ranges only (`vmax = −vmin`, as in every TD-MPC2 config) | Float32 rounding: weights within 1.1e-5 of the floor formula and closer to the exact ones (2e-6 vs 8.6e-6); decode within ε·(1 + 6·E_p\|b\|) in symlog space (≤ 4.8e-6); log-probabilities within 3·ε·(max\|logits\| + loss) |
| T25 | TD-MPC2 | RunningScale: sort-and-interpolate `_percentile`; `S.lerp_(v, τ)`, i.e. `S + τ·(v − S)` | `jnp.percentile` (linear: the same formula); `(1 − τ)·S + τ·v` (`optax.incremental_update`) | Float32 rounding (≤ 1e-6 relative over 30 updates, tested) |
| T26 | TD-MPC2 | b67b21c replay capacity `floor(min(buffer_size, steps) / T)` episodes (`buffer.py:40`) | `ceil(min(buffer_size, total) / (T·n_envs))` rounds of `n_envs` episodes, plus one staging round that is written in place and never sampled; `total` = the run's env steps when the current `train()` call ends, so a resumed run (chunked training, or a checkpoint restored by a new agent) sizes its ring for the whole run and moves the carried episodes into it (`EpisodeBuffer.adopt`) | Identical when `buffer_size` is a multiple of `T·n_envs` (DMC: 1e6 / 500 = 2000 episodes) and whenever the run never fills the buffer; otherwise at most one more round. A run trained in chunks replays exactly what the uninterrupted run does |
| D26 | DreamerV3 | `when.Ratio` keeps `prev` in float64: `prev += repeats / r` | The exact closed form `1 + ⌊(t − t0)·r⌋` on the absolute tick (`r` the exact fraction of `train_ratio / (B·T)`) | Identical whenever `1/r` is exact in float64 (every reference configuration: `r = 2^k`); otherwise the reference's rounding sometimes runs an update one transition late, and Ajax is at most one update ahead (tested) |
| D27 | DreamerV3 | Replay capacity in items (5e6), FIFO over the items of all workers | Per-env rings of `C = min(ceil(replay_capacity / n_envs), rows per env of the run)` rows, overwritten in lockstep | Never binds in the paper protocols (DMC: 250K rows ≪ 5e6), in one run or split into resumed calls (a resumed run grows a run-clamped ring to its total length); when it binds, every env drops its oldest row together and `L − 1` fewer items per env are kept than rows |
