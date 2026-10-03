# DreamerV3 and TD-MPC2 in Ajax — architecture design (v2)

Status: v2, revised after a six-lens adversarial review of v1 (Ajax rules, DreamerV3
fidelity, TD-MPC2 fidelity, JAX/GPU, environment feasibility with CPU probes,
simplicity/testability). Every blocker and major finding is resolved below; a summary
of the review is in the description of the PR that added this file.

Companion documents (same directory): `dreamerv3_spec.md`, `tdmpc2_spec.md`,
`shared_blocks.md` (verified against the papers and the official code),
`deviations.md` (version choices and the deviations register).

## 0. Decisions taken with the user

- **Fidelity rule.** Reproduce the code that produced each paper's numbers:
  DreamerV3 `danijar/dreamerv3@2411f7d` (arXiv v2 / Nature recipe) and TD-MPC2
  `nicklashansen/tdmpc2@b67b21c` (agent algorithm identical to `5f6fade`). Adopt only
  bug fixes the authors made later (DreamerV3 `29eb964` prevact fix). Make asynchronous
  or unseeded infrastructure synchronous and seeded. Treat paper typos as typos. Where
  the paper text and the paper-era code disagree, follow the code. Every deviation is
  registered in `deviations.md` and cited in the relevant docstring.
- **Scope.** DreamerV3 revised recipe, vector observations only (no CNN). TD-MPC2
  single-task online and multi-task (mechanisms + offline trainer). Multi-task is
  validated on data we generate, never on mt30/mt80.
- **Surface.** Flat `__init__` kwargs (algorithm hyperparameters only) + a `model_size`
  preset; `extensions=()`.
- **Validation.** CPU merge gates in CI (unit, parity, smoke, resume, extension tests).
  CPU learning checks run as scripts before merging each agent PR (results recorded in
  the PR and `PERFORMANCE_REPORT.md`), not in CI. GPU paper-curve report later.

## 1. Principles

1. One jitted program per seed through `build_resumable_train`; seeds vmapped outside.
   Collection, planning, imagination, replay and update scheduling are on-device,
   fixed-shape and host-free.
2. **Every schedule is static arithmetic on the absolute, unbatched tick index.**
   Never gate on per-seed state (it turns `lax.cond` into a select that runs both
   branches; a probe showed a 90x per-tick slowdown with a 400 MB replay).
3. Faithful-and-simple: single-consumer code lives in the agent package and is promoted
   to a shared module on its second consumer. Shared modules hold only what both agents
   use.
4. Agents own collection and evaluation (APG precedent) through small shared helpers,
   instead of bending the SAC-shaped `collect_experience` / `evaluate_and_log`.
5. Closed surface; unsupported Extension phases rejected at construction.
6. Fidelity is **tested**, not asserted: literal transcriptions of the reference
   functions serve as oracles (§10).

## 2. Units and counters

- **Tick** = one scan iteration = one row per env (see §5.2). The absolute tick index
  `i` is the scan input (`offset + arange`, §7.4).
- **DreamerV3:** `n_timesteps` counts **rows** (the reference's `step` clock, reset rows
  included). `collector_state.timestep = n_envs * ticks`.
- **TD-MPC2:** `n_timesteps` counts **env steps** (agent steps). With fixed-length
  episodes the tick ↔ step map is static: per episode T+1 ticks, T stepping ticks.
  `collector_state.timestep` = env steps.
- Both log `env_frames = env_steps * action_repeat` (the papers' x-axes) next to the
  house keys (`timestep`, `Eval/episodic mean reward`, `Train/episodic mean reward`).
- **Episode length in agent steps** `T = agent_episode_length(env, env_params,
  action_repeat)`: brax/playground `env.episode_length // action_repeat` (EpisodeWrapper
  counts simulator steps), gymnax `env_params.max_steps_in_episode // action_repeat`;
  divisibility asserted. Never read from the `episode_length` kwarg directly.
  Tests: Pendulum-v1 → T=200 (γ 0.975, S 1000); playground DMC at repeat 2 → T=500
  (γ 0.99, S 2500).

## 3. Shared numerics and blocks (M1)

Only blocks both agents use. Each ships with oracle tests (§10).

- `src/ajax/distributional.py`
  - `symlog`, `symexp` (log1p / expm1).
  - `TwoHot` (frozen dataclass): `bins` (in interpolation space), `transform`
    (`identity` | `symlog`). Encode: two-hot weights by linear interpolation between the
    neighbouring bins of `transform(y)`, edge-clipped. Decode: **symmetric** summation
    of `p * bins`, written `Σ (p_j − p_i)·b_j` over mirror pairs so that it is exactly 0
    for uniform p also under `jit` (XLA's multiply-add fusion breaks DreamerV3's
    `p_i·b_i + p_j·b_j` pairs, D22); within `ε·(1 + 6·E_p|b|)` (≤ 4.8e-6) of TD-MPC2's
    naive sum in symlog space; then the inverse transform. Bins are float32 roundings
    of float64 constants (D22, T24). `loss(logits, y) = -Σ w·(logits − logsumexp(logits))`.
    - DreamerV3: `bins = symexp(linspace(-20, 20, 255))` built from a mirrored half,
      `transform = identity` (raw-space interpolation, raw-space expectation).
    - TD-MPC2: `bins = linspace(-10, 10, 101)`, `transform = symlog`
      (clip in symlog space, decode `symexp(E_p[bins])`); symmetric ranges only (T24).
- `src/ajax/normalizers.py`: one percentile helper (`jnp.percentile`, linear) and two
  thin normalisers sharing it, each EMA an `optax.incremental_update`:
  - `ReturnNormalizer` (DreamerV3 retnorm): lo/hi EMAs at rate 0.01, init 0, no
    debias, **update then read**, `scale = max(1, hi - lo)`.
  - `RunningScale` (TD-MPC2): `S ← lerp(S, max(1, p95 - p5), 0.01)`, init 1, update
    before divide (EMA form: T25).
- `src/ajax/networks/blocks.py`
  - `linear(features, kernel_init, bias_init='zeros', outscale)` → `nn.Dense`;
    initializers registered in the existing initializer registry (`networks/utils.py`):
    DreamerV3 fan-in truncated normal (`variance_scaling(outscale², 'fan_in',
    'truncated_normal')`, D23; BlockLinear fan_in = full input width), TD-MPC2
    `normal(0.02)` (torch's absolute ±2 trunc bounds are effectively untruncated;
    `jax truncated_normal(0.02)` is wrong), `zeros`.
  - `NormedMLP(layers, units, act, norm ∈ {layer, rms}, norm_eps, kernel_init, dropout)`:
    hidden layer = Dense → [Dropout (first layer only)] → Norm → Act.
    DreamerV3: RMSNorm eps 1e-4 (f32 statistics, scale only), SiLU, bias kept.
    TD-MPC2: LayerNorm eps 1e-5 (`use_fast_variance=False`), Mish, dropout 0.01 in the
    first layer of each Q member only.
  - `parse_activation` gains `silu` and `mish`.
- Everything else is agent-local (BlockLinear, RSSM, OneHot-ST, BoundedNormal,
  LaProp/AGC, λ-return in DreamerV3; SimNorm, Ensemble, TanhGaussianPrior, param-group
  Adam, MPPI in TD-MPC2). A block is promoted when a second consumer appears.
  `random_pair_reduce` is shared with REDQ's subset-min only if it reproduces REDQ's
  RNG stream exactly; otherwise it stays in TD-MPC2.

## 4. TD-MPC2 single-task (M2 world model + update, M3 planner, M4 agent)

### 4.1 Surface
```
TDMPC2(env_id, n_envs=1, model_size=5, enc_dim=None, mlp_dim=None, latent_dim=None,
       num_enc_layers=None, num_q=None, simnorm_dim=8, num_bins=101, vmax=10.0,
       dropout=0.01, horizon=3, rho=0.5, consistency_coef=20.0,
       reward_coef=0.1, value_coef=0.1, learning_rate=3e-4, enc_lr_scale=0.3,
       grad_clip_norm=20.0, pi_eps=1e-5, tau=0.01, entropy_coef=1e-4,
       log_std_min=-10.0, log_std_max=2.0, batch_size=256, buffer_size=1_000_000,
       seed_steps=None, gamma=None, discount_denom=5, discount_min=0.95,
       discount_max=0.995, iterations=6, num_samples=512, num_elites=64,
       num_pi_trajs=24, min_std=0.05, max_std=2.0, temperature=0.5,
       episode_length=1000, action_repeat=1, env_params=None, extensions=())
```
- `model_size ∈ {1, 5, 19, 48, 317}` fills the `None` widths from the reference table;
  an explicit width overrides it. `iterations += 2` when the action dim ≥ 20.
- `gamma=None` → `discount(T)`; `seed_steps=None` → `max(1000, 5T)` (T from §2).
- `vmax` bounds the two-hot bins at `±vmax` in symlog space (`TwoHot.tdmpc2(limit=vmax)`).
  The reference's separate `vmin` is not exposed: the mirrored bins and the symmetric
  sum need a symmetric range, which every TD-MPC2 config has (`vmin: -10, vmax: +10`;
  T24).
- Continuous actions only (raise on discrete, like SAC). Paper DMC protocol:
  `action_repeat=2, episode_length=1000` (documented in the docstring).
- Learning rates accept `float | Callable` (house schedulable rule); static
  hyperparameters (sizes, bins, horizons) are listed as such in the docstring.

### 4.2 Algorithm (b67b21c/5f6fade semantics)
Spec `tdmpc2_spec.md` §1–§3 with the paper-era column everywhere, in particular:
- update order and parameter freshness (spec 2.19): TD target from pre-step params with
  the online encoder `h(s')`; world-model step; policy loss on **pre-step** detached
  latents and **post-step** Q (param stop-grad, dropout on); target-Q EMA (τ 0.01, Q only)
  after both steps;
- PE entropy: `+β·log_pi`, `log_pi = n_valid·(Σ_d(−½ε²−logσ) − ½ln2π) −
  Σ_d log(relu(1−tanh(u)²)+1e−6)` (tanh term unscaled, with gradient);
- loss normalisations of the code (/H, /(H·Nq), latent-mean MSE; H terms);
- world-model optimizer: `clip_by_global_norm(20)` then Adam(eps 1e-8) with encoder
  lr × 0.3 (param groups via `optax.multi_transform`); **the clip norm reproduces PE**:
  `‖(g_wm, g_π,prev)‖` where `g_π,prev` is the previous update's post-clip policy
  gradient (PE never clears it before the world-model clip; carried as one scalar);
- policy optimizer: Adam(3e-4, eps 1e-5), clip 20 (built with the existing `get_adam_tx`);
- RunningScale on `Qp[0]`, update before divide.
- Q-ensemble dropout is active in **every** Q pass at PE, the TD target's target-Q pass
  included (found by running 5f6fade in M2; deviations §2, spec §0.5 item 10).
- **Randomness seams:** `update(state, batch, noise)` and `plan(..., noise)` take their
  random draws (ε, Q-pair indices, MPPI noise, elite draw) as explicit arrays produced by
  `draw_*_noise(key)`, so oracle tests can inject identical draws. Dropout enters as one
  key per Q pass (M2): the per-member masks are drawn inside `nn.vmap`, and torch's masks
  inside `torch.vmap` cannot be recorded, so the parity fixture runs with dropout 0 and
  dropout is tested on the Ajax side.

### 4.3 Planner (agent-local `agents/TDMPC2/planner.py`)
Spec §3.C exactly, paper-era: 512 candidates including 24 π trajectories (never
resampled, re-scored each iteration), μ warm start shifted by one with last row 0, σ
reset to `max_std` each decision, clamp before evaluation, value = Σγ^t r̂ + γ^H
avg-of-2 online Q at `π(z_H).sample` with Q dropout on (PE; only the drawn pair of
members is evaluated, where the reference runs all 5 and keeps 2: the same values, the
same dropout distribution, 20-30% less planning time at the paper size on CPU), top-64
elites, `score = exp(0.5(V − max V))`, biased weighted std around the new mean clamped
to [0.05, 2], 6 iterations (+2 when A ≥ 20; `lax.scan` over the per-iteration draws, H-step
rollouts unrolled in Python), executed action = the first action of the elite drawn
from the last score (inverse CDF of one uniform, PE's `np.random.choice`) + σ₀ noise
unless `eval_mode`. `plan(wm_params, pi_params, obs, prev_mean, t0, noise, *, config,
gamma, eval_mode)` for one env, vmapped over envs; per-env `prev_mean [n_envs, H, A]`;
`t0 = is_first`. The planner is not called during the seed phase (static cond) —
prev_mean stays at zeros, so the reference's stale warm start from the step-0 eval is
not reproduced (registered).

### 4.4 Collection, replay and schedule
- Collector in **static reset mode** (§5.2): fixed-length lockstep episodes; held tick
  `i mod (T+1) == T`; fresh `env.reset` on the held tick.
- **Terminations are refused.** The collector counts off-schedule `done`s and the loop
  counts true terminations on the schedule's last step (`is_terminal` rows); the agent
  raises `ValueError` on the host after `train()` returns (and the counts are logged).
  The guard's precedent is HEAD's ValueError; PE silently treated any done as a time
  limit (registered).
- **Episode buffer** (`agents/TDMPC2/buffer.py`, the b67b21c design): ring of
  `R` rounds × `n_envs` episode slots `[R·n_envs, T+1, ·]`, where
  `R = ceil(min(buffer_size, total_env_steps) / (T·n_envs)) + 1` (one staging round).
  Rows of the in-progress round are written in place; a round becomes sampleable when its
  held tick passes. Sampling: uniform committed episode × uniform offset `s ∈ [0, T−H]`.
  Slice conversion (rows are obs-aligned, §5.2): obs rows `s..s+H`, actions rows
  `s..s+H−1`, rewards rows `s+1..s+H`. The sampler cannot draw from zero episodes
  (build-time assertion that the burst happens after ≥ 1 committed round).
- **Schedule** `n_updates(i)` (static, 5 lines in `train_TDMPC2.py`), one
  `fori_loop(0, n_updates(i), update)` per tick:
  - seed phase: random actions on every stepping tick up to and including the stepping
    tick on which total env steps first exceed S, and at least one round has been
    committed;
  - burst: `S` updates right after that tick;
  - afterwards `n_envs` updates per stepping tick, 0 on held ticks (UTD 1 per env step).
  For `n_envs = 1` this reproduces b67b21c exactly (S+1 random actions at steps 0..S,
  burst after step S with ⌊S/T⌋ completed episodes, then 1 update per step) — pinned by
  a test against a Python port of the reference loop. `n_envs > 1` is a generalisation
  (registered).

### 4.5 State
`TDMPC2State(BaseAgentState, kw_only)`: `world_model_state` (encoder, dynamics, reward,
Q ensemble; `target_params` = target Q via `soft_update` on the Q subtree),
`actor_state` (policy prior), `critic_state=None`, `q_scale`, `pi_gradnorm_sq`,
`collector_state` (row collector state + `prev_mean`), `buffer_state`.

### 4.6 Evaluation
`evaluate_policy` (§5.4) with the planner in `eval_mode` (no final noise; everything
else stochastic, as in the reference), `num_episode_test` episodes on a rebuilt env with
the training `action_repeat` and episode length, each evaluation from fresh initial
states (its key folded with the evaluation count); separate planner carry (registered:
the reference reuses the training env and prev_mean).

## 5. Shared backbone (lands with its first consumer, M4)

### 5.1 Environment plumbing
- `action_repeat` is an explicit argument of `build_env_from_id` / `prepare_env` /
  `ActorCritic.__init__` and is stored on `EnvironmentConfig`. brax/playground: passed to
  `EpisodeWrapper(episode_length, action_repeat)` (repeat does not stop on termination —
  identical on DMC, which never terminates; registered). gymnax: `action_repeat > 1`
  raises (no paper use). Registered builders: called with `action_repeat=` only when
  it is > 1 (legacy two-argument builders raise in that case). Prebuilt env with
  `action_repeat > 1` raises.
- `setup_environment` (eval rebuild) takes `action_repeat` and the training
  `episode_length` explicitly; `_infer_max_eval_steps` is replaced at the new call sites
  by `agent_episode_length`.
- **Action bounds.** The collector and `evaluate_policy` map the agent's action to the
  env: `a_env = low + (clip(a, −1, 1) + 1)/2 · (high − low)` for finite Box bounds
  (DreamerV3 reference `NormalizeAction` + `ClipAction`; TD-MPC2 `action_scale`).
  Replay stores the agent's raw action. Test with Pendulum's [−2, 2].

### 5.2 Row collector (`src/ajax/environments/row_collector.py`)
One row per env per tick, obs-aligned (DreamerV3's replay convention; also exactly
TD-MPC2's T+1 rows per episode):

| field | meaning |
|---|---|
| `obs` | observation acted on at this tick |
| `reward` | reward received on entering `obs` (0 when `is_first`) |
| `is_first` / `is_last` / `is_terminal` | reset obs / final obs of an episode / true termination |
| `action` | raw agent action chosen at this tick, zeroed when `is_last` |
| extras | agent-defined per-row outputs (e.g. DreamerV3 posterior `deter`, `stoch`) |

Per tick: emit the row; call `policy_fn(carry, obs, is_first, key) -> (action, carry,
extras)` (or uniform random actions in a seed phase selected by the caller with a static
predicate); if `is_last`, do not step (hold) and emit the reset obs next tick
(`is_first=1, reward=0`); otherwise step, and if the step ends the episode emit its
terminal obs (`get_final_obs`) next tick as `is_last` with `is_terminal = terminated`.

Two reset modes, chosen by the agent:
- **static** (fixed-length lockstep episodes; all DMC tasks, all TD-MPC2 runs): the hold
  tick is `i mod (T+1) == T` for every env; on it the collector calls `env.reset` with a
  fresh key inside `lax.cond` on the unbatched tick. This gives a freshly randomised
  initial state every episode on every backend, whatever the env's own auto-reset does
  (a playground env built with `fresh_reset=False` returns a cached first state), at
  ≈0.1 ms/tick amortised. Off-schedule `done`s are counted as
  errors.
- **dynamic** (data-dependent episode ends: gymnax / brax tasks with terminations):
  per-env hold via `jnp.where` on a snapshot of the env state; reset obs = the
  auto-reset obs, fresh on gymnax, brax and playground (Ajax's playground stack uses
  `FreshAutoResetWrapper` by default since #54). Only a playground env built with
  `fresh_reset=False` restarts from a cached first state; dynamic mode warns about it
  (deviation E20).

State: `RowCollectorState(CollectorState)` adds `reward`, `is_first`, `is_last`,
`is_terminal`, `reset_obs`, `env_steps`, `rows`, `n_offschedule_dones`, `policy_carry`.
`timestep` follows §2. Episode return/length tracking reuses the existing rolling-mean
machinery.

### 5.3 Update loop helper (`perf_utils.final_aux_fori`)
`final_aux_fori(body, carry, n)`: `fori_loop(0, n, ...)` that carries only the last
aux (zeros via `eval_shape` when n = 0), one trace of `body`. `n` is computed from the
unbatched tick, so it lowers to an unbatched while loop under the seed vmap (probed).
Extension `post_update` is folded after **each** update inside the loop (CONTRIBUTING's
contract), with `step = collector_state.timestep`.

### 5.4 Evaluation and logging
- `evaluate_policy(env_args, policy_fn, init_carry, num_episodes, key, T, ...)` in
  `evaluate.py`: rebuilds the env (n_envs = num_episodes, training action_repeat and
  episode length), resets with the given key (fresh initial states), runs every env for
  one episode with the agent's stateful policy (`is_first` on the first step), returns
  mean return and length. Tested against `evaluate()` for a feedforward actor.
- `maybe_eval_and_log` lifted from APG's `_maybe_log` into `log.py` in a
  behaviour-preserving first commit (APG tests unchanged and green), parameterised by
  `metrics_fn(agent_state, aux)` and `evaluate_fn`.

### 5.5 Resume
`build_resumable_train` gains an unbatched `iteration_offset` (default 0): the scan input
becomes `offset + arange(num_updates)`. `ActorCritic.train` passes
`self.resume_iteration_offset(initial_state)` (default 0, so existing agents — including
APG's per-stage cadence — are unchanged; the new agents return their tick counter,
asserted equal across seeds). All schedules (seed phase, burst, gate, ratio, online
queue, static reset) therefore continue correctly after resume. Test: resumed run applies
the same number of updates as an uninterrupted one and never repeats the seed phase.

### 5.6 Replay capacity
Static per `train()` call: `replay_capacity` / `buffer_size` are clamped to the run
length (`min(capacity, rows or env steps the run has taken when this call ends)`, the
references' own `min(·, steps)`, which saves memory without changing what is replayed:
a buffer that holds the whole run never evicts). Updated after the M4b review: the
first version resolved the capacity once, at the first `train()`, and kept it; a run
trained in chunks then kept the first chunk's (smaller) buffer for good, and a
checkpoint restored by a new agent met a buffer of another size. Instead, a resumed
run sizes its buffer for the whole run so far and moves the carried data into it
(TD-MPC2: `EpisodeBuffer.adopt`), so the resume skeleton's (`n_timesteps=0`) buffer
shapes do not matter and a chunked run replays exactly what an uninterrupted one does.
Capacity and bytes per seed are recorded on the agent and in its run config; the
docstring lists the memory per seed (seed vmap multiplies it).

### 5.7 Extension support
`ActorCritic.supported_extension_phases: frozenset = frozenset(PHASES)` checked in
`ActorCritic.__init__` against `extension_stack.implemented_phases()` (existing agents:
unchanged). Both new agents declare `{pretrain, post_update, eval_metrics}` (+
`init_state`). `on_target`, `critic_loss`, `actor_loss`, `on_obs`, `on_batch`,
`action`, `eval_action` are rejected with a clear message until a concrete use defines
their semantics on latent agents.

## 6. DreamerV3 (M5 world model, M6 actor-critic + optimizer, M7 agent)

### 6.1 Surface
```
DreamerV3(env_id, n_envs=16, model_size='12m', units=None, deter=None, hidden=None,
          classes=None, stoch=32, blocks=8, enc_layers=3, dec_layers=3,
          rew_layers=1, con_layers=1, actor_layers=3, critic_layers=3, bins=255,
          free_nats=1.0, unimix=0.01, actor_unimix=0.01, dyn_scale=1.0, rep_scale=0.1,
          rec_scale=1.0, rew_scale=1.0, con_scale=1.0, actor_scale=1.0,
          critic_scale=1.0, repval_scale=0.3, train_ratio=512, batch_size=16,
          batch_length=64, imag_horizon=15, return_horizon=333, lam=0.95,
          repval_lam=0.95, actent=3e-4, slowreg=1.0, slow_rate=0.02,
          retnorm_rate=0.01, retnorm_limit=1.0, minstd=0.1, maxstd=1.0,
          learning_rate=4e-5, agc=0.3, agc_pmin=1e-3, beta1=0.9, beta2=0.999,
          eps=1e-20, warmup=1000, replay_capacity=5_000_000, episode_length=1000,
          action_repeat=1, env_params=None, extensions=())
```
- Presets (d = units = hidden, deter = 8d, classes = d/16): `1m` (code only), `12m`,
  `25m`, `50m`, `100m`, `200m`, `400m`. Explicit widths override.
- `gamma = 1 − 1/return_horizon` (renamed from the reference's `horizon` to avoid the
  clash with APG/TD-MPC2; the continue target folds γ in, `contdisc`).
- No boolean variant flags: replay context, online queue and contdisc are always on
  (paper-era behaviour). f32 only (bf16 deferred to the GPU milestone, registered).
- Discrete and continuous actions; `n_timesteps` counts rows.
- Paper DMC-proprio protocol (documented): `model_size='12m', n_envs=16,
  train_ratio=512, action_repeat=2, episode_length=1000, n_timesteps=250_000`.

### 6.2 Algorithm (2411f7d + 29eb964)
Spec `dreamerv3_spec.md` Algorithms A–J with every 2411f7d choice where HEAD differs
(full adoption table in `deviations.md`): discrete actor one-hot with 1% unimix (log-prob
and entropy of the mixed distribution); vector-decoder output outscale 0.1;
symlog-MSE tolerance 1e-8 (no ½ factor); start-state reward and continuation **from
replay data** (`reward_t`, hard `1 − is_terminal_t`) in imagination; slow critic set to a
hard copy of the critic after the first optimizer step (`mix = 1 if updates == 0 else
0.02`, applied after the step); λ-return with next-state r̂/ĉ/v bootstrapped from the
**online** critic; retnorm update-then-read; REINFORCE for both action types;
repval (not stop-gradiented, trains the world model, fixed γ, ~is_last mask);
`Σ scale · mean(term)` reductions.
- **One joint gradient.** One forward pass and one `jax.value_and_grad(total_loss)`
  over (world model, actor, critic) params at their pre-update values; imagination,
  the retnorm update, repval and the write-back latents all come from that forward. Then
  three optimizer instances (LaProp: per-tensor AGC → RMS(β2 0.999, eps 1e-20 after
  sqrt, bias-corrected) → momentum(β1 0.9, bias-corrected) → −lr with 1000-step linear
  warmup from 0) — bit-identical to one optimizer over all modules (probed), asserted
  by a test that also shows a sequential variant differs. Step counters asserted equal.
- **Randomness seams** as in §4.2 (posterior samples, imagined actions, replay draws).

### 6.3 Replay (`agents/DreamerV3/replay.py`)
- Per-env stream ring `[n_envs, C, ·]` (lockstep write pointer), rows per §5.2 plus the
  posterior latents `deter` (f32) and `stoch` (class indices, uint8 when C ≤ 256) of
  each row, `C = min(ceil(replay_capacity / n_envs), rows per env of the run)`.
- Items = all fully written windows of `L = batch_length + 1 = 65` rows (stride 1,
  crossing episodes). Physical indices are `(start + k) mod C`, read with `take` and
  written with modular scatters (never `dynamic_slice` on the ring).
- **Online queue:** item `q` = env `q mod n_envs`, start `1 + L·(q div n_envs)` (lockstep
  FIFO by push tick then env id); pending items are popped first (at most B per batch),
  the rest of the batch is drawn uniformly over items with replacement.
- Batch annotation: `is_first[:, 0] = True`; `is_last |= next is_first` (last column
  excluded).
- Replay context: carry from the stored latents at index 0; obs/flags from 1..64;
  `prevact = action[0:64]` (29eb964).
- **Write-back:** after the optimizer step, `fori_loop` over the B rows **in batch
  order**, each writing its 64 posteriors (indices 1..64, never the context) — later rows
  win, deterministically, as in the reference's sequential loop. Test with forced
  overlaps against a NumPy sequential loop.

### 6.4 Schedule (2411f7d training start + `when.Ratio`)
Static, per tick, from the transition-level reference (`train.py` + `when.Ratio`):
transitions are numbered within ticks; the gate opens at the first transition with
`len(replay) ≥ batch_size` items (items per env = rows − 64), where Ratio returns 1;
afterwards cumulative updates after transition t are `1 + ⌊(t − t0)·r⌋` with
`r = train_ratio / (B·T)`. `n_updates(i)` = difference of that closed form between the
ends of ticks i and i−1. With 16 envs the first update is at tick 64. Pinned by a test
against a Python port of the 2411f7d driver for `n_envs ∈ {1, 4, 16, 32}`. Updates are
run after the tick's rows are added (registered: the reference interleaves them between
the per-env adds of one vector step).

### 6.5 Collection and evaluation
Collector in static mode on fixed-length tasks (DMC), dynamic mode otherwise (e.g.
CartPole). Acting = posterior filter with the carried (deter, stoch, prevact), sampled
posterior, **sampled** action (no deterministic mode in the reference); prevact = raw
sample. Primary metric = training-episode returns of the stochastic policy (reference);
`evaluate_policy` eval episodes also sample, starting from a zero carry with
`is_first = True`.

### 6.6 State
`DreamerV3State(BaseAgentState, kw_only)`: `world_model_state` (enc, rssm, dec, rew,
con; the reference's `dyn` is `rssm`), `actor_state`, `critic_state` (`target_params` = slow critic), `retnorm`,
`collector_state` (row collector + policy carry), `replay_state` (+ online-queue
counter).

## 7. TD-MPC2 multi-task (M8)

- Mechanisms in `agents/TDMPC2/multitask.py` (the single-task code is imported, the
  lineage rule): task embedding `U(−0.02, 0.02)`, `task_dim=96` (paper; the code's 64 for
  mt30 at 5/19/48M is an acknowledged accident tied to that dataset — registered), max-norm
  1 by **look-up-time renorm with write-back** at the two points the reference renorms
  (before the TD target; after the world-model step, before the policy loss); renorm
  without persisting at eval; obs zero-padded at the end; prefix action masks applied to
  π mean/logσ/ε and to planner candidates and to mean/std after the std clamp; per-task
  discount table; `n_valid` entropy scaling; padded-A iteration rule.
- `TDMPC2MultiTask`: offline trainer, env-free update scan (batch 1024 slices uniform over
  pooled episodes; no planning in training). The dataset enters as an **unbatched
  argument shared across seeds** (new optional `shared` slot of `build_resumable_train` /
  `train`, `in_axes=None`) — never closed over (HLO constant bloat) nor copied per seed.
  `n_timesteps` counts updates. Evaluation is a host loop over tasks between scan chunks,
  each a per-task jitted `evaluate_policy` with the padded planner (`eval_mode`, 10
  episodes per task, the offline-trainer protocol; evaluate.py's noisy protocol
  registered as not reproduced).
- Dataset schema: `obs [N, T+1, O_max]` (zero-padded), `action [N, T+1, A_max]` (zeros on
  invalid dims and at row 0 convention §5.2), `reward [N, T+1]`, `task [N]`, plus `T` per
  task; written by a general export utility from the single-task episode buffer.
- **CI gate:** mechanisms + trainer smoke on an in-memory synthetic dataset of 2–3 toy
  tasks with different obs/action dims. **Validation (M9):** dataset = full training
  history (every completed episode from step 0, no FIFO eviction) of k seeds per task on
  the playground versions of the 19 original mt30 DMC tasks (mt30 order as task ids,
  action repeat 2); acceptance = the offline model reaches a stated fraction of each
  source agent's final return; reported as validating the mechanisms, not the paper's
  numbers.

## 8. Probing

- TD-MPC2: the stock probes are skipped (they terminate after 1–2 steps; the paper's
  TD-MPC2 is for non-terminating tasks and its slices need T+1 ≥ H+1 rows). Replaced by
  Ajax-local fixed-length non-terminating probe envs (constant reward → known discounted
  value via time-limit bootstrap; reward = action → planner action → +1).
- DreamerV3: local adaptor (tiny config, `return_horizon = 1/(1−γ)`, value read at the
  posterior after filtering the canonical trajectory from `is_first`); added to the
  skip set with the existing "too slow" reason if a check exceeds ~60 s in CI.

## 9. Performance guardrail
PR #48 (restore `benchmarks/`) merges first; a fresh CPU baseline is captured on this
machine before M4. After M4 and M7, `agent_bench.py --compare` runs for all registered
agents. APG and both new agents get bench entries with a small documented preset.

## 10. Testing

- **Oracles.** Two kinds; Ajax's modular implementation is compared with the oracle
  at tiny sizes with injected noise (atol ~1e-5, matmul precision `highest`): every
  loss term, per-module gradients, post-update parameters, normaliser state, target
  EMA, plan output.
  - *Reference fixtures*, where the pinned reference code can be run: a generator per
    agent in `docs/world_models/parity/` (committed, not collected by pytest, run by
    hand in a throwaway venv with the reference's pinned dependencies, as its docstring
    says) runs the real reference at tiny float32 sizes, records its random draws
    without changing what it computes, and writes small `.npz` files (parameters under
    their reference names, inputs, recorded draws, outputs, gradients) committed under
    `tests/agents/<A>/fixtures/`. The tests map the reference's parameter names onto
    Ajax's tree, force the recorded draws and compare. The fixtures' sizes are pairwise
    distinct where Ajax could confuse two of them, and the hyperparameters other than
    widths are the reference's defaults, against which Ajax's defaults are pinned.
    Pinned this way so far: the TD-MPC2 update (M2,
    `parity/tdmpc2_update_fixtures.py`: consecutive real 5f6fade `update()` calls,
    replayed by `test_tdmpc2_parity.py` through Ajax's jitted update), the TD-MPC2
    planner (M3, `parity/tdmpc2_plan_fixtures.py`: real `act()` decisions with their
    draws, per-iteration values, elites, scores, mean and std, replayed chained by
    `test_tdmpc2_planner_parity.py` through Ajax's jitted `plan`) and the DreamerV3
    world model (M5, `parity/dreamerv3_world_model_fixtures.py`: `29eb964`'s own
    `Agent.train`).
  - *Transcriptions*: `tests/world_models/reference_impls.py` (shared blocks) and
    `tests/agents/<A>/reference_<a>.py`, literal jnp transcriptions of the pinned
    reference functions (MIT-licensed, attributed line by line), where running the
    reference is impractical.
- **Control-flow parity.** Python ports of the reference loops (b67b21c online trainer;
  2411f7d driver + replay add/sample/online queue + train gate + Ratio) on a counter env:
  identical row streams, seed/burst/gate ticks, queue pops, update counts.
- **Structure.** Parameter-count tests (TD-MPC2 walker 4,960,618; MT 5,389,930; DreamerV3
  12M core), chex shape tests, gradient-routing tests (which params get non-zero grads
  from which loss).
- **Backbone.** Episode length per backend; action_repeat parity train/eval; static and
  dynamic reset modes (T+1 rows, held env bit-identical, consecutive episodes start from
  different states on playground); replay wrap-around; write-back with overlaps;
  `final_aux_fori` single trace and exact counts; resume offset; extension rejection;
  `evaluate_policy` vs `evaluate`; APG logging unchanged.
- **Agent.** Module-scoped tiny-config fixtures (train once, assert smoke / metrics /
  resume / extensions on the same run); ≤ 3 min of CI per agent test module; regression
  goldens labelled as such.
- **Learning checks** (`benchmarks/learning_checks.py`, outside CI, run before merging
  each agent PR): TD-MPC2 Pendulum-v1 (Pendulum's [−2, 2] bounds mapped), playground
  CartpoleBalance (repeat 2); DreamerV3 CartPole-v1 and Pendulum-v1 at small presets.
  Results recorded in the PR and `PERFORMANCE_REPORT.md`.

## 11. Milestones (vertical slices; one branch + PR each, CI green per commit)

- **M0** docs: specs, this design, deviations register (+ PR #48 merged, CPU baseline).
- **M1** shared numerics and blocks (§3) + block oracles.
- **M2** TD-MPC2 world model + update as pure functions (+ oracle parity, param counts).
- **M3** TD-MPC2 planner (+ oracle parity).
- **M4** TD-MPC2 agent: env plumbing (§5.1), row collector (§5.2), episode buffer,
  schedule, `final_aux_fori`, `evaluate_policy`, logging lift (first commit), resume
  offset, extension support, local probes, bench entry, learning checks.
- **M5** DreamerV3 world model (RSSM, encoder/decoder, heads, KL) + parity on fixtures
  from the real reference.
- **M6** DreamerV3 actor-critic (imagination, λ-returns, retnorm, repval, slow critic) +
  LaProp/AGC + oracle parity (joint-gradient equivalence).
- **M7** DreamerV3 agent: stream replay with context, online queue, write-back, schedule,
  dynamic-mode collector, probes, bench entry, learning checks.
- **M8** TD-MPC2 multi-task mechanisms + offline trainer + synthetic-data gate.
- **M9** validation scripts: GPU report (DreamerV3 DMC-proprio Table 11 tasks available in
  playground; TD-MPC2 DMC subset), multi-task dataset generation and validation.

## 12. Deviations register

The register lives in `deviations.md` (version-choice tables for both papers plus every
Ajax departure, numbered D*, T*, E*). Code cites the entry it implements.
