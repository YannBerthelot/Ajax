# Multi-task and continual training in Ajax: design

2026-10-11. Code: `clone_BATCH` at b84469b (read only). `map/online/sota/cont §n` = map.md,
lit_online.md, lit_sota.md, lit_continual.md. Checks: `switch_vmap.py` (3.2), `k_task_jit.py` (9).
[U] = unverified. Critic's changes C1-C14 at the end; the designer's draft is `design.designer.md`.

## 0. The answer in one screen

The right granularity is **one object for the tasks, one for when each env slot sees which task, and
one small contract on the agent side**. "Protocol" is kept for multi-stage pipelines.

1. **`TaskSet`** (static data): env id, obs dim, action dim, episode length, optional discount, score
   spec per task. Lifted out of `TDMPC2/multitask.py:57-171` (map §1.2), which then imports it.
2. **`Schedule`** (static data, PR 4): which task env slot s runs at tick t, a pure function of
   `(s, t)`. All-at-once is the env's default; "100 episodes of A, then B" is `sequential(...)`;
   cyclic orders and fixed curricula are orders; only adaptive curricula fall outside (section 3).
3. **The multi-task env** = `TaskSet` (+ `Schedule`), a registered builder id (`create.py:97-122`),
   so every agent and its evaluation reach it with no agent change (map §2, §5). It pads obs, appends
   a one-hot (unless hidden), gives each sub-env its action prefix, puts `task_id` in `info`.
4. **One agent contract**: each stored row carries its `task` (a buffer field, as `buffer_fields`
   already allows, map §4). Per-task statistics (α_i, γ_i, return scale) are **not** a cross-agent
   contract: each is an option of the agent whose published method has it (C4). Single-task
   programs stay exactly as they are.
5. Everything else is an **Extension** (boundary resets; later EWC/L2, plasticity), a **network
   option** (embedding, heads, wider nets), a **descendant agent** for a published method with its
   own losses (BRC), or a **downstream protocol** (TD-MPC2 offline; later demos→online).

**"Is this only true for SAC? What about PPO?"** The per-task piece is algorithm-specific; it
belongs with each agent, not in a shared protocol:

| Agent | Per-task piece | Source |
|---|---|---|
| SAC family | α_i in the Meta-World-era codes (garage, MTRL, CARE); **BRC keeps one α** | online §1.2; C2 |
| PPO | nothing for MT-PPO (garage); PopArt for per-task value scale | online §2, §4 |
| TD-MPC2 | task embedding, per-task γ_i, action masks (all already in its code) | sota §1, map §1.2 |
| BRC | per-task return normalisation, categorical critic, learned embedding | sota §2, C2 |
| APO / ASAC / UDRL | per-task reward rate / command; no published source | map §4 |

"Every agent becomes its published baseline unchanged" holds for PPO (MT-PPO, deviations in 4.5) and
SAC as MTRL's PCGrad config runs it; MT-SAC adds α_i, TD-MPC2 its TaskContext (sota §6.1).

## 1. The layers, corrected

| Layer | Hypothesis | Corrected by the evidence |
|---|---|---|
| L1 interface | one-hot or embedding, padding, masks | Static per-task metadata (dims, T_i, γ_i, score) plus *conditioning*: none (IMPALA, MGDT, MTRL MT-SAC), one-hot, learned embedding, frozen language (CARE, Newt), or a prompt (Gato). Padding is needed across embodiments; action **masks** are a TD-MPC2/Newt choice, BRC pads without them (C2). The test-time view is a separate switch (online §9.1, cont §1.4) |
| L2 data regime | online / offline / distillation | Split in two. (a) **Schedule** inside one run: slot allocation and its time axis (sequential, cyclic, curriculum; cont §5). (b) **Stages** across runs: offline-from-single-task, demos→online, teacher→student, held-out fine-tuning. Distral is joint and online, so it is an agent, not a stage (online §1.1, sota §6.4-5) |
| L3 balancing | replay, α_i, PopArt, γ_i | Per-task *statistics*: α_i, γ_i, reward or return scale (env-side in mtenv and garage; agent-side in BRC and PopArt), balanced collection. Categorical losses are the scale-robust alternative (online §1.5, sota §1). Mostly sidestepped by a sequential schedule (cont §5) |
| L4 interference | PCGrad, CAGrad, GradNorm | Needs per-sample losses with task ids and a new reduction step (online §6). Weak support at the state of the art (sota §6.6). Across time, interference is **forgetting**: EWC, L2, PackNet, CLEAR (cont §3) |
| L5 architecture | trunk, heads, modular, big nets | Moved from modularity to scale plus normalisation (sota §6.7). Plasticity interventions (resets, ReDo, CBP) also sit here and are orthogonal to tasks (cont §2) |
| L6 evaluation (new) | none | Every task every E steps; final vs max over training; normalised score per domain; forgetting and forward transfer need single-task reference runs (online §9.6, cont §1.1, §5) |

## 2. Every method placed (Ajax home: env, agent option, ext, net, new agent, proto, out)

| Method | L1 | L2 | L3 | L4 / retention | L5 | Ajax home | Source |
|---|---|---|---|---|---|---|---|
| MT-PPO / MT-TRPO (TE-PPO) | one-hot (→ latent + inference net) | simult., round robin | none | - | shared MLP (embedding) | env only (TE: net + loss, later) | online §2, §8 |
| MT-SAC (garage; MTRL) | one-hot; none in MTRL | one env per task, shared buffer | α_i, reward norm per task | - | shared MLP | env + SAC option (α_i) | online §1.2, §3 |
| MH-SAC, SAC + task encoder | head per task / learned embedding | as MT-SAC | α_i | - | multi-head / embedding concat | net | online §3 |
| Soft Modularization | one-hot→router | simult. | α_i, softmax loss weights | weights | routed modules | net + reduction phase (deferred) | online §8 |
| CARE, PaCo, MOORE | language / id→w_τ / one-hot | simult., buffers per task (PaCo, MOORE) | α_i (+ mask-out, PaCo) | (vs PCGrad) | encoders / Φw / experts | net (deferred) | sota §2 |
| PCGrad, CAGrad, GradNorm, FAMO | one-hot / any | per-task buffers, stratified (PCGrad) | α_i; GradNorm weights | projection / combination | 6-layer MLP (PCGrad) | reduction phase (deferred) | online §6, sota §2 |
| Distral | per-task nets + π0 | joint online | KL, entropy | - | N+1 policies | new agent (later) | online §1.1 |
| IMPALA multi-task | none | fixed actors per task | reward clipping | - | shared | env (`task_visible=False`) | online §4 |
| PopArt-IMPALA | id for value heads only | as IMPALA | value + advantage norm | - | multi-head value | PPO/APO option (later) | online §4 |
| BRC | learned embedding; pad, no mask | simult., one env per task, shared buffer, 4x batch | one α, return norm, CE critic | none | BroNet ≤1B | new agent (SAC descendant) | sota §2, C2 |
| Newt | language, pad + mask | demos→pretrain→online, 1 env/task | γ_i, CE, BC | - | TD-MPC2 20M | `TDMPC2` online MT (no demos) + demos proto | sota §1, C1 |
| TD-MPC2 MT | embedding, pad + mask | offline from single-task buffers | γ_i, CE, 4x batch | - | 1-317M | `TDMPC2MultiTask` + downstream proto | sota §1, map §1 |
| CDS, Gato, MGDT, MR.Q-MT | one-hot / prompt / none / e(τ) | offline relabel / BC / DQN-Replay / online | - | - | - | out (no CQL, DT, MR.Q) | sota §2-3 |
| Policy Distillation, Actor-Mimic, TD-MPC-Opt | heads / masked shared | teacher→student | KL to teacher | - | trunk + heads | proto (later) | sota §4 |
| Continual World protocol | heads | sequential, 1M-step blocks | sidestepped | buffer + Adam reset | multi-head | env schedule + ext | cont §1.1 |
| L2, EWC, ClonEx-SAC | heads | sequential | - | anchor / Fisher / actor BC | multi-head | ext, each on request (D13) | cont §1.1, §3 |
| CLEAR, Perfect Memory | none | cyclic / sequential, keep buffer | - | V-trace + cloning | - | keep buffer (default); V-trace out | cont §3 |
| PackNet, Progressive nets | id at eval | sequential, K known | - | masks / frozen columns | grows | out for now | cont §3 |
| Resets, ReDo, CBP, S&P, InFeR | - | any (single-task too) | - | - | plasticity | ext (post_update) + probes (D13) | cont §2 |
| RL², MAML | none (in-context) | stationary p(z), trials | - | - | recurrent | off this axis (APG in-context) | cont §4 |

## 3. Sequential (continual) schedules: is "task set + schedule" one axis?

**3.1 Verdict.** Yes for one run's data regime: simultaneous, blocks, cyclic, interleaved and fixed
curricula are all `task(slot, t)` (cont §5). Off the axis, each with its own home: **adaptive
curricula** (read agent metrics; deferred); **hidden drift** like Slippery Ant's friction
(`system_class`, map §2; a hidden *piecewise* switch is `task_visible=False`); **multi-stage
regimes** (4.8); **meta-RL** (cont §4; APG's in-context control); **the metric** (L6, 3.4).

### 3.2 How one compiled program switches tasks (checked)
Different Playground tasks are different pytrees, so they cannot share one vmap (map §2); a switch
over branches is the only way to hold several in one program. A `lax.switch` stays a real branch
only when its index is *unbatched*. The scan index of `build_resumable_train` is unbatched under the
seed vmap by design (`perf_utils.py:66-75`). `switch_vmap.py` (re-run by the critic): a schedule
computed from that index keeps `cond`, so only the active task steps; one computed from a batched
counter (anything in the agent state) becomes `select_n`, which steps **all K tasks every tick**.
- Simultaneous: static slot→task allocation, no switch; cost = sum over tasks (map §2).
- `sequential()`: all slots run `order[(t // block) % len(order)]`. The env carries every task's
  sub-state and switches on the unbatched tick. Compile = sum over tasks; run = one task.
- The tick has to reach `env.step`: `interaction.step(..., tick)` and its training callers
  (`interaction.py:892, :1138`, `row_collector.py:466, :504`, `train_UDRL.py:196`,
  `differentiable.py:141`); off-policy tick = iteration, on-policy iteration × n_steps + j; the
  world models already have `collect(state, tick)` (`loop.py:18-25`). Evaluation needs no tick.

**A switch is an auto-reset into another task.** At a boundary every slot gets `truncation = 1`,
`info["final_obs"]` = the old task's observation, and the new task's fresh reset as obs, so the
bootstrap path (`FinalObsWrapper`) stays correct. Blocks that are multiples of episode length × n_envs
cut no episode (CW: 200 divides 1M, cont §1.1); TD-MPC2 accepts only those (T10, map §1.4).

### 3.3 Task-boundary and task-identity signals (two signals, cont §1.4)
- **Boundary event (training).** A static function of the schedule. An Extension binds the schedule
  (`bind_to_agent(env_args=...)`) and tests `schedule.boundary(ctx.iteration)`, a new *unbatched*
  `ExtensionContext` field. Today's `ctx.step` is the batched collector timestep
  (`loop.py:302-306`), which would run the expensive branch on every update. Only SAC and DreamerV3
  bind today (`train_SAC.py:824`, `train_DreamerV3.py:617`; `TrainLoop.create` does not,
  `loop.py:264`), so PR 4 binds in the shared loop for every agent (C7).
- **Task identity.** `info["task_id"]` goes to the buffer and losses; networks see `task_visible` (D11).
- **At a boundary** (cont §3), with today's phases: `post_update` clears the buffer and resets Adam
  (both in `agent_state`, `state.py:115`), later snapshots an anchor or Fisher; `actor_loss` later
  adds an L2/EWC penalty (`actor_params` is passed, `SAC/core.py:298`); `action` explores uniformly
  for N steps after a switch (CW App A.3).

### 3.4 Evaluation it implies
Every evaluation runs **all K tasks**, whatever the schedule (CW: every 20K steps, 10 episodes;
https://arxiv.org/abs/2105.10919 App A.3, checked), logging `Eval/<task>/return`, mean and normalised
score: the matrix p_i(t). `continual_metrics(logs, block_ends, reference=None)` gives P(T), forgetting
F_i = p_i(end of block i) − p_i(T) and, with single-task reference runs, FT_i (CW eq. 1-3).

## 4. Ajax boundaries

**4.1 `TaskSet`** (`ajax/multitask/tasks.py`; frozen, hashable): `env_ids`, `names`, `obs_dims`,
`action_dims`, `episode_lengths` (agent steps), `action_repeat`, `discounts | None` (None = the
agent's γ), `scores` (e.g. DMC return/10); `obs_dim`/`action_dim` (max), `action_masks`,
`pad_observation`, `from_env_ids(ids, episode_length, action_repeat)` (0.2-1.0 s per task, map §3).

**4.2 Multi-task env** (`ajax/multitask/env.py`, ≈ 1 class), brax-style, over K Ajax sub-stacks (each
built by `build_env_from_id`, keeping its wrappers): `reset` / `step(state, action, tick=None)`;
observations padded at the end, then the one-hot when `task_visible`; sub-env k gets
`action[:, :action_dims[k]]`; `info["task_id"] [n_envs]`; `.task_set` and `.schedule` attributes.
`register_task_set(env_id, task_set, task_visible=True, normalize_reward=False)` registers it (PR 4
adds `schedule=`).

**Normalisation (C5).** `prepare_env` wraps any env in one pooled normaliser (`create.py:441-447`)
and a registered builder sees only `(n_envs, episode_length, action_repeat)` (`create.py:50-95`), so
PR 2 **refuses** env- and agent-side (`obs_norm.py`) normalisation on a task-set id (pooled stats mix
tasks and un-zero the padding, map §2). Per-task reward normalisation as garage MT-SAC and mtenv do
it (reward / running std, online §1.5; Ajax's normaliser with `gamma=None`, `wrappers.py:651-657`)
is the `normalize_reward` field, inside each sub-stack (PR 3). Per-task obs normalisation is deferred:
nothing in section 2 needs it (garage MT-PPO's `normalize` only rescales actions).

**Evaluation** gets an explicit task-set branch: the eval rebuild calls the registered builder with
`n_envs=num_episodes` (`evaluate.py:107-113`) and cannot tell eval from training. The branch builds
`K × num_episodes` slots, simultaneous, with per-task means (today one mean, `evaluate.py:698-699`).

**4.3 `Schedule`** (`ajax/multitask/schedule.py`, PR 4). Fields: `order` (task ids), `block` (agent
steps, summed over envs). Without a schedule the env is simultaneous: equal blocks of slots, refused
when K does not divide n_envs. `sequential(order, steps=None, episodes=None)` converts episodes with
T; it needs one T for all tasks (Playground DMC at a common repeat; all 25 terminate only on NaN, map
§3) and n_envs dividing `episodes`, and refuses otherwise. Methods: `task(slot, t)`, `boundary(t)`,
`block_ends(n)`. Cyclic = `t // block` modulo `len(order)` (CW20 = CW10's order for 20 blocks).

**4.4 Agent contract.** A `task` buffer field, filled from `info["task_id"]`: the flat replay schema
(`buffers/utils.py:106-127`), `get_batch_from_buffer` / `sample_replay`, TD-MPC2's ring and Dreamer's
replay. On-policy rollouts keep the env axis, so their task is per row already.

**4.5 Per agent.** What it is on a simultaneous env unchanged, what its published form needs, and
whether a sequential schedule works.

| Agent | Unchanged on a simultaneous env | Needs for the published form | Sequential |
|---|---|---|---|
| PPO | the **MT-PPO recipe** (one-hot, shared PPO). Recorded deviations: static slot blocks vs garage's round robin at reset (`multi_env_wrapper.py:182-183`); advantages normalised per minibatch (`train_PPO.py:201-202`) vs garage's whole batch (`center_adv`); Ajax's network defaults | nothing; PopArt later (task in batch, value head per task, rescale) | runs |
| SAC, REDQ | MT-SAC with a shared α (BRC's choice, C2) | `task` field + `per_task_alpha` (D4) + per-task reward norm (env field) | runs |
| TD3 | runs; no published MT-TD3 apart from Newt's FastTD3 (language, γ_i, n-step) | - | runs |
| ASAC, APO | **wrong**: θ / ρ pooled over tasks (map §4) | per-task rate (unpublished): refuse | runs (θ, ρ are EMAs) |
| UDRL | **wrong**: command from the pooled top-K returns (`train_UDRL.py:365-370`) | per-task command (unpublished): refuse | only with the buffer cleared at boundaries (C11) |
| AVG | runs; shared running statistics | no published MT form | runs |
| DreamerV3 | runs; one return scale for all tasks | no published MT form read: record it | runs |
| TDMPC2 | runs, the one-hot fed to its encoder: not a published configuration | `task` field in its ring + TaskContext from `env.task_set`, index read from the one-hot, which leaves the encoder input (C1) | episode-aligned blocks |
| TDMPC2MultiTask | offline, unchanged; its dataset holds the TaskSet | - | n/a |
| DQN, PQN / APG | n/a on DMC (discrete) / differentiable envs (`SystemClass` is its task distribution) | - | - |

**4.6 Extensions**: `BoundaryReset(buffer, optimizer, uniform_steps)`, CW's own boundary handling
(App A.3). `L2Anchor`, `EWC` (actor-only, CW §6.1), `PlasticityProbes`, Nikishin / ReDo resets fit
the same phases, one PR each when picked (D13, C8). None touches an agent. **Not expressible today:**
the per-task loss reduction (L4) and a sampling phase (stratified replay, Newt's 50/50 mix); deferred.

**4.7 Network options.** The one-hot in the obs is the only task channel to networks, so acting
paths do not change. A network option decides its use: raw concatenation (MW, garage), an embedding
indexed by it (BRC, MTRL's encoder, TD-MPC2; the one-hot then leaves the input, as garage's MTSAC reads
it back out of the obs, `mtsac.py:181`), or head selection (MH-SAC, CW).
BroNet (residual LayerNorm MLP) goes in `networks/blocks.py`.

**4.8 Protocols** = multi-stage host pipelines chaining ordinary runs. One exists, downstream:
`MultiTaskProtocol` (map §1.5), host-chained on purpose (stages of different shapes: unpadded sources,
a saved dataset, the offline trainer). No core `Protocol` until a third regime (rule 4). A sequential
schedule is **not** a protocol: it compiles end to end (3.2).

## 5. The API a user writes

```python
from ajax import PPO, SAC, TDMPC2MultiTask, BRC          # BRC: PR 7
from ajax.multitask import TaskSet, register_task_set, sequential, BoundaryReset, continual_metrics
DMC = ("CartpoleSwingup", "ReacherEasy", "WalkerWalk", "CheetahRun")
tasks = TaskSet.from_env_ids(DMC, episode_length=1000, action_repeat=2)   # pads to 24 / 6
walker = TaskSet.from_env_ids(("WalkerStand", "WalkerWalk", "WalkerRun"), 1000, 2)  # one body
# (a) MT-PPO and MT-SAC online, all tasks at once (one block of envs per task)
PPO(register_task_set("mt:walker3", walker), n_envs=48).train(seed=[0, 1, 2], n_timesteps=3_000_000)
env = register_task_set("mt:walker3-rn", walker, normalize_reward=True)                # PR 3
SAC(env, n_envs=3, per_task_alpha=True).train(seed=[0, 1, 2], n_timesteps=3_000_000)  # D4
# (b) TD-MPC2's offline protocol from a task list (downstream script, takes the TaskSet)
proto = dataclasses.replace(PROTOCOL, tasks=TaskSet.from_env_ids(MT30_PLAYGROUND, 1000, 2))
run_protocol(proto)   # sources (TDMPC2 per task x seeds) -> pool_tasks -> TDMPC2MultiTask -> report
# (c) SOTA simple recipe: BRC (SAC + BroNet categorical critic + task embedding + return norm)
BRC(register_task_set("mt:dmc4", tasks), n_envs=4).train(seed=[0], n_timesteps=4_000_000)
# (d) Sequential: 100 episodes of each task in order, twice (CW20-style revisit)
schedule = sequential(order=(0, 1, 2, 3), episodes=100)  # 100 x 500 = 50_000 agent steps per block
seq = register_task_set("seq:dmc4", tasks, schedule=schedule)       # n_envs=10: 10 episodes per env
SAC(seq, n_envs=10, extensions=(BoundaryReset(buffer=True, optimizer=True, uniform_steps=10_000),)
    ).train(seed=[0, 1, 2], n_timesteps=2 * 4 * 50_000)            # logs Eval/<task>/ for all 4
continual_metrics(logs, block_ends=schedule.block_ends(2 * 4 * 50_000), reference=single_task_logs)
```

## 6. One source of truth: TD-MPC2's machinery vs the new layer

| Knowledge | Today | After |
|---|---|---|
| per-task dims, T, names | `TDMPC2/multitask.py:TaskSet`, repeated as static fields of `MultiTaskDataset` (map §1.1), as names in `MultiTaskProtocol`, and as the `eval_envs` list | `ajax.multitask.TaskSet`; the dataset holds one (its `.npz` + JSON format v1 still loads, C10); the protocol holds one; eval envs come from `env_ids` |
| obs padding, action prefix masks | `multitask.pad_observation`, `action_masks`, `pool_tasks` padding | move with TaskSet; `pool_tasks` and the env both call it |
| per-task γ | `TaskSet.create` via `core.discount_from_episode_length` | the heuristic stays in TD-MPC2 (paper-specific); TD-MPC2 fills `discounts` from it; other agents use their γ unless given |
| action prefix to the env | `task_planner_policy` slices (offline eval) | the online env slices; TD-MPC2's offline eval path is unchanged (T16) |
| per-task evaluation | `TDMPC2MultiTask` host loop over tasks (map §1.3) | the env's eval branch; moving TDMPC2MultiTask onto it shifts RNG streams, so it waits for D14 |
| embedding table, `TaskContext`, renorm | TD-MPC2 core | **stays**. Generalise on the third consumer (TD-MPC2, BRC, then an MT-SAC encoder), not the second |
| per-sample task | `MultiTaskDataset.sample` returns `task [B]` | online agents: the `task` buffer field, same meaning |

Expand then contract (rule 10): the old `TaskSet` path re-exports the shared class until users move.

## 7. PRs in order (source budget / test budget; one concern each, themed CI batches)

| # | PR | src | tests | Fast tests (two tiny tasks, CPU) |
|---|---|---|---|---|
| 1 | Lift `TaskSet` to `ajax/multitask`, TD-MPC2 and its dataset import it (restructure only, dataset format v1 kept) | +140 / −100 | 60 | existing TD-MPC2 multi-task tests and goldens unchanged; a v1 dataset file loads; `from_env_ids` on CartpoleBalance + ReacherEasy gives (5, 6) / (1, 2) / (500, 500) |
| 1b | Downstream offline pipeline takes a `TaskSet` (`benchmarks/world_models/`) | 40 | 20 | `SMOKE` protocol builds, unchanged settings |
| 2 | Multi-task env (simultaneous only), `register_task_set`, refusals (normalisation, APO, ASAC, UDRL; different action dims only under D5's rule) | 180 | 110 | padding exactly zero; one-hot; `task_id`; each sub-env gets its action prefix (equal to the single-task stack); PPO and SAC 2k steps on CartpoleBalance + CartpoleSwingup finite; refusals raise |
| 2b | Eval branch: K × num_episodes slots, `Eval/<task>/` logs, normalised score | 90 | 50 | per-task means equal K single-task evaluations with the same keys |
| 3 | `task` buffer field, SAC `per_task_alpha` (MTRL's `should_use_disentangled_alpha`), per-task reward normalisation field | 110 | 90 | single-task goldens bitwise (the K = 1 path is the old code); α_i move apart on two tasks; reward-norm statistics separate per task; one slow two-task probe |
| 4 | `Schedule` + `sequential()`, tick threaded into `interaction.step` and its training callers, `ExtensionContext.iteration`, `bind_to_agent` in `TrainLoop` | 220 | 150 | jaxpr keeps `cond` under the seed vmap (`switch_vmap.py` as a test); episode_length 20, blocks of 2 episodes: `task_id` sequence, truncation + `final_obs` at the switch, fresh obs after it; SAC and PPO sequential runs finite; schedule unit tests (cyclic, episodes→steps, refusals) |
| 5 | `BoundaryReset` + `continual_metrics` | 120 | 90 | boundary branch runs once per block; buffer empty and Adam counts zero after it; metrics on a hand-made matrix give CW eq. 1-3 |
| 6 | Online MT `TDMPC2` (Newt without demonstrations, task-ID conditioning; C1): on a task-set env it turns on its TaskContext, indexed by the one-hot | 140 | 100 | update equals `multitask.update` on the same batch; planner masks; encoder input excludes the one-hot; 2-task run finite; refuses `task_visible=False`. `TDMPC2MultiTask` untouched |
| 7 | `BRC` agent (SAC descendant: BroNet, categorical critic via `ajax/distributional.py`, embedding, return norm, one α, padded actions unmasked) | 400 | 200 + probe | critic loss on a fixed batch vs a hand computation; probing test on two tasks |

Deferred to owner decisions: PopArt, the loss-reduction phase (PCGrad, CAGrad, GradNorm, SM), Distral,
distillation and demos→online, language, adaptive curricula, per-task ρ/θ, D13's methods. Source
≈ 1440 lines, measured per PR. PRs 1-2b and 3-5 can each share one CI batch.

## 8. Owner decisions (recommended option first, in bold)

- **D1 central abstraction.** **(a) Task-set env (+ schedule)**; (b) a wrapper per agent; (c) a
  Protocol object per regime.
- **D2 sequential switch.** **(a) Thread the unbatched tick into `env.step`**; (b) a counter in the
  env state (K× env compute under the seed vmap, 3.2); (c) chain K programs on the host by resume.
- **D3 block unit.** **(a) Agent steps, episodes as sugar for equal fixed-length tasks, a switch =
  forced truncation + reset**; (b) episodes per slot (slots drift apart).
- **D4 SAC-family α.** **(a) A SAC hyperparameter `per_task_alpha=False`, as MTRL's flag; MT-SAC sets
  it** (garage, MTRL, CARE use α_i; BRC one α, C2); (b) automatic when K > 1 (BRC, a SAC
  descendant, would have to undo it); (c) a new "temperature" Extension phase.
- **D5 different action dims, model-free.** **(a) Pad, each sub-env takes its prefix, no mask in the
  losses, as BRC's code does (C2); TD-MPC2 keeps its masks**; (b) refuse until each agent has masks;
  (c) same-body sets only. Newt's PPO and FastTD3 handling stays [U].
- **D6 online MT TD-MPC2.** **(a) Yes, in `TDMPC2`, as Newt without demonstrations with task-ID
  conditioning (two published Newt ablations combined; a recorded deviation, since TD-MPC2's code has
  no online multi-task); `TDMPC2MultiTask` stays offline and unchanged**; (b) a `Newt` descendant
  agent now; (c) no online MT TD-MPC2.
- **D7 SOTA recipe.** **(a) BRC** (sota §2); (b) full Newt (needs demos); (c) none yet.
- **D8 gradient surgery (L4).** **(a) Defer** (2025-26 leaders do not use it); (b) build the
  reduction phase now (MTBench reports gains with massively parallel PPO).
- **D9 protocol class.** **(a) None in core until a third multi-stage regime**; (b) one now.
- **D10 evaluation.** **(a) All K tasks at every eval, final values logged, max computed
  downstream**; (b) report the max (MW, CARE; not comparable across papers, online §9.6).
- **D11 test-time task id.** **(a) Same as training (`task_visible`)**; (b) separate now.
- **D12 pooled ρ, θ, UDRL command.** **(a) Refuse under simultaneous, allow sequential (UDRL with
  buffer clearing)**; (b) unpublished per-task versions.
- **D13 continual methods.** **(a) Schedule, `BoundaryReset`, metrics now; EWC, L2, ClonEx,
  plasticity tools one PR each when an experiment needs them**; (b) the whole kit now.
- **D14 TDMPC2MultiTask evaluation.** **(a) Move it onto the eval branch once that has run on
  online agents (numbers shift by RNG only)**; (b) never (two eval paths).
- **D15 default DMC set.** **(a) The 19 mt30 tasks (pad 24 / 6)**; (b) all 25 (Humanoid pads every
  task to 67 / 21: a 1-dim task carries 20 unused action dims under D5).

## 9. Unverified and risks
- **Compile (measured, C9).** One jit stepping 6 Playground tasks (`k_task_jit.py`, CPU, no cache):
  28.6 s trace + 20.6 s compile, env only. So 19 tasks ≈ 2.5 min, ≈ 5 min with the eval env in the
  same program (`loop.py` module doc); more with Humanoid. **GPU**: the K sub-programs run in turn
  each tick; with few envs per task the device may idle (not measured).
- [U]: Newt's PPO/FastTD3 padded actions, IMPALA details, MW v1's MH-SAC, BRC's return normaliser
  code, summarised section numbers. No pixels on this CPU; the 11 custom mt30 tasks are absent.

## Critic's changes
Opened: Newt (https://www.nicklashansen.com/NewtWM/newt.pdf), BRC (https://arxiv.org/abs/2505.23150,
github.com/naumix/BiggerRegularizedCategorical), garage `mtsac.py:142-145, :181` (α_i confirmed),
MTRL `baseline.rst`/`actor.py`, TD-MPC2 `offline_trainer.py` (10 eval episodes per task, mt30/mt80
assert: confirmed), Continual World App A.3 (confirmed), and every Ajax line cited above.
- **C1** Newt lists no "online multi-task TD-MPC2" baseline (lit_online §7 said so): BC, PPO,
  FastTD3, model-based pretraining, 200 single-task TD-MPC2 (Newt §4 "Baselines", App. D). Closest:
  ablations "Ours (no demos)" (Fig. 1, 13) and "Task ID" (Fig. 8). PR 6 / D6 renamed, deviation
  recorded; the one-hot indexes the embedding and leaves the encoder input (reference: embedding only).
- **C2** BRC's code keeps **one** temperature (`jaxrl/networks.py:139-145`; target entropy −A_max/2,
  `agent/brc_learner.py:147`) and pads, handing each env its prefix with no mask (`jaxrl/envs.py:85-119,
  :152`). It also runs mixed bodies (MW+DMC |S| = 39, |A| = 6; HB-Hard; App. E): sota §5's "same
  body" is wrong. D4, D5 changed; PR 3 lost its valid-dims log-prob and per-task target entropy.
- **C3** MTRL MT-SAC has no task input: without an encoder all conditioning flags are off
  (`actor.py:177-195`), though `baseline.rst:90-95` passes them. Confirmed.
- **C4** "Contract B" (scalars become [K] automatically) removed: the literature disagrees (C2).
- **C5** Per-sub-stack normalisation had no code path (`create.py:441-447`, builder signature): PR 2
  refuses it; per-task reward normalisation becomes an env field with MT-SAC.
- **C6** `Schedule` moves to PR 4: one form until `sequential` (rule 4). **C7** Only SAC and DreamerV3
  call `bind_to_agent`; PR 4 binds in `TrainLoop`. **C8** PR 5 = CW's `BoundaryReset` + metrics;
  EWC, L2, probes → D13 (the owner asked for schedules; simplest first).
- **C9** "19 tasks ≈ 1-2 min" → measured K = 6: ≈ 2.5 min env-only for 19, ≈ 5 min with eval; GPU
  serialisation and Humanoid padding added (D15). **C10** PR 1 keeps dataset format v1 loadable.
- **C11** UDRL sequential needs buffer clearing. **C12** "garage MT-PPO exactly" → recipe + three
  deviations (4.5); garage's `normalize(env)` only rescales actions (`mtppo_metaworld_mt10.py:44-46`).
- **C13** `episodes=` refuses unequal T_i; TD-MPC2 only episode-aligned blocks (T10). **C14** PR 2
  split (2b = eval); PR 3, PR 5 shrunk; source total 1610 → 1440 lines.
