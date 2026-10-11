# Multi-task map: code and task facts

Source: `clone_BATCH` at b84469b (main + expert batch), read only. Paths are relative to `src/ajax/`
unless they start with `docs/`, `benchmarks/` or `site-packages/`. CPU (macOS ARM), playground 0.2.0,
mujoco/mjx 3.7.0, brax 0.14.2, jax 0.7.2. Scripts: `multitask_protocol/{dmc_facts,two_task_jit}.py`.

## 1. TD-MPC2 multi-task contracts

### 1.1 Data side (`agents/TDMPC2/dataset.py`)
| Object | Contract |
|---|---|
| `TaskEpisodes` :100-135 | one task, unpadded host numpy: `obs [N,L,O]`, `action [N,L,A]` (row L-1 is 0), `reward [N,L]` (row 0 is 0), `episode_length` T (agent steps), `name`. Needs T >= L-1. Rows are obs-aligned: row k = (o_k, a_k, r_{k-1}), the layout of the single-task ring. |
| `export_episodes(agent, state, seed_index, name, allow_evicted)` :142-216 | reads the single-task run's **fixed-length** episode ring (`state.buffer_state`, `collector_state.rows`), rebuilds an `EpisodeBuffer`, returns every committed episode oldest first; raises on eviction unless `allow_evicted` (a full-history dataset needs `buffer_size >=` run length). Only `EpisodeBuffer` layouts: an episodic run's `EpisodicBufferState [n_envs, C, ...]` gives `n_rounds = 1` and fails `EpisodeBuffer.__post_init__` (`buffer.py:135-139`), with no dedicated message. |
| `concatenate_episodes` :219-246 | joins runs (seeds) of one task; rows, dims, T and name must agree. |
| `MultiTaskDataset` :249-379 | pytree `obs [N,L,O_max]`, `action [N,L,A_max]`, `reward [N,L]`, `task [N]` int32 + static per-task `obs_dims`, `action_dims`, `episode_lengths`, `names`. `check()` :291-342 enforces shapes, unique names, every task present, exact zero padding (obs beyond `obs_dims[i]`, actions beyond `action_dims[i]`), `min(T_i) >= L-1`. |
| `MultiTaskDataset.sample(key, B, H)` :344-367 | episode uniform over the **pooled** episodes, crop uniform in [0, L-1-H]; returns `(TDMPC2Batch, task [B])`. No per-task balancing: a task's batch share = its episode share (`episode_counts()` :287-289). |
| `pool_tasks(tasks)` :387-424 | zero-pads obs/actions at the end to the max dims, task i = `tasks[i]`, **one L for all tasks** (raises otherwise; cutting longer episodes into segments, as mt80 probably did, is not implemented), places arrays on device, runs `check()`. |
| `save_dataset` / `load_dataset` :433-496 | `.npz` + JSON meta, no pickle, format version 1, atomic rename. |

- **Per-task episode length**: metadata only (sets the discount and eval length); rows per episode are
  shared (`L`). A dataset of DMC tasks at AR 2 has L = 501.
- **Terminations**: none. The schema has no `terminated`; `sample` builds an `EpisodeBufferState`
  without `terminal`, so `slice_batch` (`buffer.py:386-416`) returns `terminated=None`. Episodic
  (terminating) tasks cannot enter the offline multi-task path today.
- **Memory**: `(O_max + A_max + 1) * 4` bytes per row plus 4 bytes per episode. mt30-like padding
  (24, 6): 124 B/row. Ajax validation set (19 tasks x 3 seeds x 1000 eps x 501 rows = 28.6M rows):
  3.5 GB. Paper mt30 (345.7M rows) 42.9 GB; mt80 (39+6+1 floats, 550.5M rows) 101 GB. A Humanoid task
  (67, 21) would pad every task's row to 356 B. The whole dataset is `device_put` once
  (`TDMPC2MultiTask.py:202`): it must fit device memory, no host streaming.

### 1.2 Mechanisms (`agents/TDMPC2/multitask.py`, `core.py`)
- `TaskSet` (multitask.py:57-171), static and hashable: per-task obs dims, action dims, episode lengths,
  discounts, names. `create` :99-133 derives each discount by `discount_from_episode_length(T_i)`
  (denom 5, min 0.95, max 0.995; DMC T=500 gives 0.99). `obs_dim`/`action_dim` = max; `action_masks`
  :150-155 prefix masks `[K, A_max]`; `discount_table` :157-161 float32; `context(task)` :163-167 builds a
  `core.TaskContext(ids, mask)`; `discount(task)` :169-171 gathers per-sample gamma.
- `pad_observation` :174-182 zero-pads at the end. The env receives `action[:, :action_dims[task]]`
  (`task_planner_policy` :243-278).
- Task embedding: `PAPER_TASK_DIM = 96` (:53-54; deviation T14 vs the code's 64 for mt30).
  Table `wm_params["task_emb"]` (`core.TASK_EMB` :127), `[num_tasks, task_dim]`, U(-0.02, 0.02), created by
  `core.create_update_state(..., task_dim, num_tasks)` :714-761; gather `task_embedding` :232-243;
  `nn.Embedding(max_norm=1)` look-up-time renorm with write-back `renorm_task_embedding` :246-264, applied
  in `core.update` (:795-) at the two reference points (:837, :873). Policy loss reads it stop-gradiented.
- `core.TaskContext` :212-229 is the only task hook: every single-task function takes
  `task: Optional[TaskContext] = None` (None = single-task computation unchanged). Masks multiply the
  policy's mean, log-std and noise and count valid dims in the entropy (`squashed_gaussian` :283-315).
- One `RunningScale` (Q scale for the policy loss) serves every task (core.py:605-644, :626; spec 2.A).
- `multitask.update(state, batch, task, noise, config, tasks)` :214-240 = `core.update` with
  `gamma=tasks.discount(task)` and `task=tasks.context(task)`.

### 1.3 Trainer (`TDMPC2MultiTask.py`, `train_TDMPC2MultiTask.py`)
- `TDMPC2MultiTask(dataset, eval_envs=None, <TD-MPC2 hyperparameters>, batch_size=1024,
  episode_length=1000, action_repeat=1, extensions=())` :124-239. Not an `ActorCritic`: it owns no env.
  The dataset fixes the tasks (:203-211). `horizon <= L-1` checked (:195-199).
- `eval_envs` :242-292: one env id or prebuilt env per task, built with `prepare_env(..., n_envs=1,
  action_repeat)`; each must match the task's `(obs_dim,)`, action dim and agent-step episode length;
  continuous only. Evaluation (:379-406) is a host loop over tasks, one jitted, seed-vmapped
  `evaluate_task` each (`train_TDMPC2MultiTask.py:226-263`): planner in eval_mode, 10 episodes, logs
  `Eval/<name>/...`, mean over tasks, and `Eval/normalized score = mean(return/10)` (:57, DMC only).
- Train: `n_timesteps` counts updates. One update (`update_step` :97-123) = `dataset.sample` + `multitask.update`
  + `fold_post_update`. Chunks are `lax.scan`s through `build_resumable_train` with the dataset as an
  **unbatched shared argument** (`jit(vmap(resume, in_axes=(0,0,0,None)), donate_argnums=(2,))`,
  :304-338); chunk boundaries = absolute multiples of `log_frequency` (:492-517).
- Resume: `initial_state` continues; `resume_update_offset` :294-301 requires equal update counts across
  seeds. Separate jitted init program (:340-359). Extensions: `LOOP_PHASES` only (pretrain,
  post_update, eval_metrics; :122).

### 1.4 Single-task TD-MPC2 episodic mode (deviation T29)
- `TDMPC2(..., episodic=False, termination_coef=1.0)` (TDMPC2.py:168-169, doc :111-117). Off: fixed-length,
  non-terminating episodes; terminations or off-schedule dones raise after `train`
  (`refuse_terminations` :392-407; T10). On: termination head (BCE), `1 - terminated` in TD targets and
  planner, `EpisodicSchedule` (:259-262), `EpisodicBuffer` per-env rings of `ceil(buffer_size/n_envs)`
  rows (buffer.py:57-65, :419-, :439-), `n_timesteps` counts rows. core.py:42-50: **single task
  only**; nothing in `multitask.py` or the dataset carries `terminated`.

### 1.5 What `docs/world_models` records
- Reference: multi-task is **offline only** (spec 4.1, `train.py:52`): no online MT path, no online
  finetuning except the paper-only few-shot experiment (spec 4.25: 19M pretrained on 70 tasks,
  full model finetuned online on a held-out task, new embedding initialised from a source task).
- Suites (spec 4.15): mt30 = 19 original DMC + 11 custom DMC (padded obs 24, act 6); mt80 = mt30 +
  50 Meta-World (obs 39, act 6). Order = task id = embedding row. MultitaskWrapper (4.16): pad obs at
  end, env gets action prefix, `reset(task_idx)` picks the sub-env.
- Datasets (4.18): replay buffers of 240 single-task agents, random to expert; mt30 690k episodes x
  501 rows (345M transitions), mt80 5.45M episodes x 101 rows (545M); mt80's DMC part inferred to be
  mt30 cut into 101-row segments (unproven; 4.B). No `terminated` key.
- Training (4.19): 10M iterations default, batch 1024, slices uniform over pooled episodes, no
  per-task balancing, no planning in training. Eval and score (4.20; T16): trainer protocol kept.
- Sanity targets (4.28): mt30 normalised score 18.9/28.3/54.2/59.4/71.4 (1M..317M); mt80
  16.0/49.5/57.1/68.0/70.6; embedding norm ablation 46.6 vs 54.2.
- Deviations: T14 (task_dim 96), T15 (eval renorm not persisted), T16 (eval protocol), T17 (self-made
  datasets on playground versions of the 19 original tasks). DESIGN.md §0: "Multi-task is validated
  on data we generate, never on mt30/mt80"; §7: mechanisms as data (TaskSet, TaskContext), the
  dataset an unbatched shared argument.
- Validation pipeline (VALIDATION.md §4; `benchmarks/world_models/multitask_validation.py`, 583 lines):
  stages sources -> dataset -> train -> report, driven by a frozen `MultiTaskProtocol` dataclass
  (:102-127; `PROTOCOL` :145-162: 19 tasks, 3 source seeds x 500k agent steps, 19M offline model,
  1M updates). A multi-stage "protocol" object therefore already exists, as a downstream script.
- The 11 custom tasks are not in playground (VALIDATION.md §1 table): mt30 maps to 19 tasks only.

## 2. Padding, task identity and multi-env stepping elsewhere in Ajax

- `grep -i 'task_id|task_emb|num_tasks|multitask'` over `src/ajax` outside `agents/TDMPC2/`: only the
  `TDMPC2MultiTask` export in `__init__.py`. Padding: only batch padding (`agents/cloning.py:66-71`,
  `DreamerV3/replay.py:383-388`). `mask` hits are done/truncation/reset masks (PPO GAE, evaluate,
  memory carries, row collector), never action or observation masks. **No task identity, observation
  padding or action masking exists outside TD-MPC2.**
- Closest existing "family of tasks": `environments/system_class.py:1-15`, a distribution over
  `EnvParams` sampled as batched params (one system per env), vmapped in `interaction.reset/step`
  (:212-216, :272-279). **gymnax only** ("brax envs do not take per-step params"); same dims across
  members; used by APG / in-context control.
- One agent = one env: `EnvironmentConfig(env, env_params, n_envs, continuous, action_repeat)`
  (`state.py:77-86`); `ActorCritic.__init__(env_id: str | EnvType, ...)` (`agents/base.py:94`) calls
  `prepare_env` (:168) and builds it (:180). A prebuilt env is accepted, but with `action_repeat > 1`
  `prepare_env` raises unless the env comes from an id (`environments/create.py:423-442`).
- Env creation: `prepare_env` (create.py:393-454) -> `build_env_from_id` (:238-336: registered builders
  first, then playground registry, brax, gymnax) -> `add_ajax_wrappers` (:339-370: optional
  observation/reward normaliser, then action clip). Registered builders (`register_playground_builder`
  :112, `register_brax_builder` :97, called through `_call_registered_builder` :52-95, which forwards
  `action_repeat`) own their whole stack; evaluation rebuilds a brax/playground env **by its
  `_ajax_env_id`** with `n_envs=num_episodes` (`evaluate.py:80-120`; registered builders win). This is
  the existing seam through which a task-set env could reach every agent and its evaluation.
- Playground stack (`_build_playground_env` create.py:124-193): `registry.load(id, {"impl": "jax"} on
  CPU)` -> brax `EpisodeWrapper(episode_length, action_repeat)` (episode_length counts env control
  steps; repeat sums rewards, does not stop on termination, E19) -> `VmapWrapper` -> `FinalObsWrapper`
  (wrappers.py:995) -> `FreshAutoResetWrapper` (wrappers.py:1082; default `fresh_reset=True` since PR
  #54: a new initial state per episode) or playground's cached `BraxAutoResetWrapper` -> `BatchRngWrapper`
  (wrappers.py:964). Stepping (`interaction.step`, brax mode :319-329) calls `env.step` with no extra
  vmap; truncation from `info["truncation"]`, terminated = done x (1 - truncation).
- Normalisation: env-side `NormalizeVecObservation*` statistics are **pooled over all envs**
  (`online_normalize` collapses the env axis, wrappers.py:316-327; gymnax pooling interaction.py:283-300);
  agent-side `obs_norm.py` keeps one statistic (leading shape 1). With padded multi-task observations a
  pooled normaliser would mix tasks and turn padded zeros non-zero. TD-MPC2 uses raw observations.
- Evaluation returns the mean over episodes (`evaluate.py:700`): no per-task breakdown for a mixed env.

**Two Playground tasks in one jit (works).** `two_task_jit.py`: each task is Ajax's own stack
(`build_env_from_id`, n_envs=2), one `jax.jit` resets both and scans 50 agent steps of both with
random actions, pads observations to O_max and concatenates them `[K, 4, O_max]` with task ids
`[0,0,1,1]`. Results (CPU, `AJAX_NO_COMPILE_CACHE=1`):

| Pair (obs/act) | Trace+lower | Compile | Run (50 steps) |
|---|---|---|---|
| CartpoleBalance alone (5/1) | 2.5 s | 1.0 s | 0.001 s |
| ReacherEasy alone (6/2) | 3.0 s | 1.3 s | 0.002 s |
| both, one jit | 3.6-3.8 s | 2.2-2.4 s | 0.002 s |
| WalkerWalk alone (24/6) | 4.5 s | 2.4 s | 0.022 s |
| CheetahRun alone (17/6) | 9.8 s | 4.6 s | 0.017 s |
| both, one jit | 9.9 s | 7.5 s | 0.050 s |

Finite outputs, padding exactly zero. Different tasks have different MJX models, so their states
are different pytrees: they cannot share one `vmap` (a `lax.switch` under vmap would run every
branch); one jit holds a Python-unrolled list of per-task sub-programs. Compile time and step time add
up over tasks (7.5 s vs 2.4 + 4.6 s; 0.050 s vs 0.039 s), so 19 tasks cost roughly the sum of 19
single-task compiles (about 1-2 min on CPU, extrapolated, not measured).

## 3. Installed Playground DM Control suite (25 tasks, `dmc_facts.py`)

All 25 build (`registry.load(id, {"impl": "jax"})`, 0.2-1.0 s each). Common to every task: default
`episode_length=1000` control steps, config `action_repeat=1` (Ajax ignores the config value and
applies its own `action_repeat` in `EpisodeWrapper`; the TD-MPC2 protocol uses 2, i.e. T = 500 agent
steps), state observations, and termination only on NaN in qpos/qvel (all `done =` sites in
`site-packages/mujoco_playground/_src/dm_control_suite/*.py`), i.e. no task terminates in practice.
mt30 = one of the 19 original mt30 tasks.

| Task | obs | act | ctrl_dt / sim_dt (substeps) | mt30 |
|---|---|---|---|---|
| AcrobotSwingup | 6 | 1 | 0.01 / 0.01 (1) | yes |
| AcrobotSwingupSparse | 6 | 1 | 0.01 / 0.01 (1) | no |
| BallInCup | 8 | 2 | 0.02 / 0.002 (10) | yes (cup-catch) |
| CartpoleBalance | 5 | 1 | 0.01 / 0.01 (1) | yes |
| CartpoleBalanceSparse | 5 | 1 | 0.01 / 0.01 (1) | yes |
| CartpoleSwingup | 5 | 1 | 0.01 / 0.01 (1) | yes |
| CartpoleSwingupSparse | 5 | 1 | 0.01 / 0.01 (1) | yes |
| CheetahRun | 17 | 6 | 0.01 / 0.01 (1) | yes |
| FingerSpin | 9 | 2 | 0.02 / 0.005 (4) | yes |
| FingerTurnEasy | 12 | 2 | 0.02 / 0.005 (4) | yes |
| FingerTurnHard | 12 | 2 | 0.02 / 0.005 (4) | yes |
| FishSwim | 24 | 5 | 0.04 / 0.002 (20) | yes |
| HopperHop | 15 | 4 | 0.02 / 0.005 (4) | yes |
| HopperStand | 15 | 4 | 0.02 / 0.005 (4) | yes |
| HumanoidStand / Walk / Run | 67 | 21 | 0.025 / 0.005 (5) | no |
| PendulumSwingup | 3 | 1 | 0.02 / 0.01 (2) | yes |
| PointMass | 4 | 2 | 0.02 / 0.02 (1) | no |
| ReacherEasy | 6 | 2 | 0.02 / 0.005 (4) | yes |
| ReacherHard | 6 | 2 | 0.02 / 0.005 (4) | yes |
| SwimmerSwimmer6 | 25 | 5 | 0.03 / 0.003 (10) | no |
| WalkerRun / WalkerStand / WalkerWalk | 24 | 6 | 0.025 / 0.0025 (10) | yes (all 3) |

- The 19 mt30 tasks pad to obs 24, act 6 (the paper's mt30 dims). All 25 pad to obs 67, act 21
  (Humanoid), which nearly triples the row size of every other task.
- Not available: the 11 custom mt30 tasks, dog, quadruped (not in this suite).
- Pixels: every config has a `vision` key, but only Cartpole implements it. `WalkerWalk` with
  `vision=True` raises `NotImplementedError: Vision not implemented for PlanarWalker`; `CartpoleBalance`
  with `vision=True` raises `NotImplementedError: refit_bvh only implemented for MuJoCo Warp` on this CPU
  (needs the Warp/GPU renderer). Cartpole vision also changes ctrl_dt to 0.02, episode length to 250
  and adds bound terminations (cartpole.py:74-76, :198-207). Not buildable here.

## 4. Agents: batch and losses, and where a per-task index would flow

Shared off-policy replay: flashbax flat buffer (`buffers/utils.py:13-51`), one row per env
(`add_batch_size=n_envs`); a sample draws the env row and time **uniformly**
(`site-packages/flashbax/buffers/trajectory_buffer.py:213-232`). So with equal envs per task the replay is
balanced per task in expectation (not stratified). The schema (`buffers/utils.py:106-127`) is obs,
action, reward, next_obs, terminated, truncated, raw_obs, is_expert (+ expert, raw_next_obs, actor_carry
fields); `get_batch_from_buffer` (:138-154) and `recurrent.sample_replay` (`agents/recurrent.py:233-257`)
drop the sampled row index, so **no batch knows its env or task** unless a `task` field is stored
(precedent: extension-declared `buffer_fields`, `extensions/base.py:338-340`, `SAC.py:56`).

| Agent | Data per update | Env/task id reachable in the batch? | Loss reduction | Scalars that a per-task version would split |
|---|---|---|---|---|
| SAC (`SAC/core.py`, `train_SAC.py`) | flat replay batch [B] (`train_SAC.py:563-575`) | no (row index dropped) | `mean` inside `critic_loss_fn` (core.py:223), `soft_policy_loss` (:264), actor (`train_SAC.py:445`), temperature (:327) | `log_alpha` one scalar (core.py:59-76) and one target entropy; `gamma: float` (core.py:147) |
| REDQ, ASAC, TD3 | `sample_replay` batch [B] | no | SAC core / own critic, `mean` | REDQ/ASAC alpha as SAC; ASAC reward rate `theta` scalar (`ASAC/state.py:14`) |
| DQN | flat replay (`train_DQN.py:355`) | no | MSE mean | gamma |
| PQN | on-policy rollout (n_steps, n_envs), Q(lambda) targets (`PQN/utils.py:9`), PPO minibatch shuffler (`train_PQN.py:107`) | env axis before the shuffle | mean per minibatch | gamma, lambda |
| PPO | rollout (n_steps, n_envs), geometries time/flat/recurrent (`train_PPO.py:1-15`) | env axis before the shuffle | `surrogate.mean()` (:205), entropy mean (:206) | advantage normalised per minibatch over all tasks (:203); `_compute_gae` has `gamma`, `gae_lambda` **static** (`PPO/utils.py:8`); pooled reward/obs normaliser |
| APO | PPO rollout, GAE with gamma 1 on r - rho | env axis | as PPO | `average_reward` rho and bias `b` scalars, rho updated from the mean over **all envs** (`train_APO.py:240-245`): wrong for tasks with different reward rates |
| AVG | one transition per env, no replay (`train_AVG.py:1-8`) | env axis | TD error / running scale | one running statistic per scalar shared across envs (`train_AVG.py:83`) |
| UDRL | segment replay, command (return, horizon) | episodes per env | NLL/MSE | command target from the buffer's **pooled top-K returns** (`train_UDRL.py:365-370`) |
| APG | differentiable closed-loop rollout over sampled systems; loss `-mean_i sum_k r_k` (`train_APG.py:1-8`, :189-222) | system index i | mean over systems | gymnax / differentiable envs only; a task distribution already = its system class |
| DreamerV3 | sequence windows [B, T+1] (`replay.py:248-275`) | **yes**: `BatchIndex.env [B]` (:105-113) | world-model and actor-critic means | return normaliser percentiles, one scale (`actor_critic.py:212-227`) |
| TD-MPC2 (single) | episode slices [H+1, B] | yes via slot: env = slot mod n_envs (`buffer.py:334-343`) | rho-weighted means | one `RunningScale`; scalar discount |
| TDMPC2MultiTask | dataset slices + `task [B]` | **yes** (`dataset.py:367`) | as TD-MPC2 | per-task discount already; one RunningScale; uniform pooled sampling |

Where a per-task index must flow, by method family:
- **Balanced replay**: sampling sits in each agent's update (`get_batch_from_buffer`, `sample_replay`,
  `EpisodeBuffer.sample`, DreamerV3 `Replay.sample`, `MultiTaskDataset.sample`); no Extension phase
  touches sampling. Equal envs per task already gives expected balance for the flat buffer, DreamerV3
  and the TD-MPC2 ring; exact stratification needs the sampler.
- **Per-task temperature / PopArt / per-task discount / per-task reward rate**: each is a scalar in the
  agent state or config today (alpha, retnorm, RunningScale, rho, theta, gamma). Splitting it needs the
  task id per sample in the loss and a `[K]`-shaped state; `on_target` (`extensions/base.py:364-375`)
  can rescale targets but cannot change the core loss or the actor's alpha.
- **PCGrad / CAGrad / GradNorm**: need per-task losses before the `mean` (computed inside the core loss
  functions above) and a gradient transform between `value_and_grad` and the optimizer step
  (`train_SAC.py:322-324`). Extension phases are additive only: `critic_loss`/`actor_loss` add a
  scalar to the core loss (`extensions/base.py:377-391`); there is no gradient or sampling phase.
- **Architecture (task embedding, multi-head)**: networks see only `obs`; task conditioning outside
  TD-MPC2 would arrive either through the observation (one-hot appended by the env) or a new network
  input. TD-MPC2's `TaskContext` is the only existing pattern.

## 5. Facts bearing on the granularity hypothesis
- A task-set env can reach every agent without agent changes through a registered builder id (eval
  rebuild included); it must step a list of per-task sub-states (no shared vmap), split `n_envs`
  across tasks, pad observations, slice action prefixes, and supply a task signal (one-hot) for
  agents with no `TaskContext`. Compile and step costs add over tasks.
- Things that break silently under such an env, today: pooled observation/reward normalisers, pooled
  PPO advantage normalisation, scalar rho (APO), theta (ASAC), alpha (SAC family), Dreamer's return
  scale, UDRL's pooled command, a single eval mean. These are the layer-3 items; they need the task id
  in the batch, which only DreamerV3, TD-MPC2 and the on-policy env axis carry now.
- The offline TD-MPC2 regime already has its multi-stage driver (`MultiTaskProtocol`, downstream);
  `export_episodes` cannot read episodic runs and the dataset has no terminations.
