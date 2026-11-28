# Architecture

This page explains how Ajax is put together: where the code lives, how a call
to `agent.train()` becomes one compiled program, what the shared training loop
does for every agent, and where extensions, evaluation and logging plug in.
Read it before changing an agent or the loop. To add an agent or an extension,
follow [Contributing](contributing.md) and
[Writing an extension](writing-an-extension.md).

Three ideas shape the code:

- **A run is one compiled program.** Initialisation, every training
  iteration, the evaluations and the logging calls are traced once and run
  inside `jax.lax.scan`. Seeds are a `jax.vmap` over that program, so eight
  seeds cost one compilation.
- **An agent is its algorithm.** Collecting experience, the warm-up before
  learning starts, evaluating, logging and resuming are written once, in the
  shared loop. An agent's folder holds the paper's maths and little else.
- **Research features are extensions.** An agent accepts its own
  hyperparameters plus `extensions=()`. A feature such as behaviour cloning
  before training or an extra loss term is an `Extension` folded into the
  loop, not a keyword on the agent. The exceptions are `expert_policy` and
  `eval_expert_policy`, which every agent but REINFORCE and the world models
  takes, and TD3's `action_pipeline`, for a downstream project.

## The package

```text
src/ajax/
├── __init__.py          the exported agents (__all__); configures the compile cache on import
├── agents/
│   ├── base.py          ActorCritic: environment and network setup, extensions=, train(); run_config
│   ├── families.py      declared agents (fields as hyperparameters) and their families' train functions
│   ├── loop.py          TrainLoop, the shared loop (brax's forced resets included)
│   ├── update_steps.py  critic_step, actor_step, compute_gradients, gradient_step, apply_gradients_clipped_together, run_if, run_actor_if, the target update
│   ├── temperature.py   the entropy temperature every soft actor-critic shares (SoftActorCriticState, update_temperature)
│   ├── minibatch.py     the on-policy rollout split (RolloutSplit), minibatches and epochs
│   ├── returns.py       discounted returns and advantages over an on-policy rollout
│   ├── critics.py       the Q-critics' regression (continuous and discrete actions)
│   ├── centering.py, reset_cost.py   reward centering and the reset cost, for continuing tasks
│   ├── diagnostics.py   KL, explained variance, policy std
│   ├── recurrent.py     replay buffers and R2D2-style sequence replay for memory=, and PPO's networks with or without memory
│   ├── cloning.py       the behaviour-cloning loss and fit that CloningPretrain runs
│   ├── presets.py, sweep.py, devices.py   per-task presets; train(sweep=); train(devices=)
│   └── <AGENT>/         one package per agent (below)
├── extensions/          Extension, ExtensionStack, the capabilities and the shipped extensions
├── environments/
│   ├── create.py        build an env from an id (gymnax, Brax, MuJoCo Playground), add Ajax's layers
│   ├── interaction.py   collect_experience: act, step, store, count
│   ├── action_stage.py  the action extensions' slots around any agent's pipeline (EDGE, JSRL, ValueBox)
│   ├── utils.py         env helpers; the [-1, 1] action map; ExpertInAgentUnits
│   ├── obs_norm.py      agent-side running observation normalisation
│   ├── row_collector.py the world models' collector: one row per env per tick
│   ├── expert_collection.py   rows an expert collects alone (cloning, prefill)
│   ├── gymnasium.py     Gymnasium environments, stepped in Python
│   ├── differentiable.py  closed-loop rollouts you can backpropagate through (APG)
│   └── model_reference.py, system_class.py, pid_terms.py   tracking tasks, parameter distributions, PID error terms
├── networks/            actor and critic networks (deterministic actors, Q-networks), memory blocks, Squash, the world models' layers
├── modules/             the PID actor and PID output head
├── buffers/             replay buffers (flashbax), rings sized to the run, transition datasets
├── logging/             LoggingConfig; the W&B and TensorBoard worker; the printed progress and Evaluations
├── log.py, evaluate.py  when to evaluate and log; the evaluation episodes
├── async_evaluation.py  evaluation in a thread of its own, on a device of its own
├── controller.py        what a step acts on (ActingState) and how the agent acts beyond its actor (Controller)
├── policy.py            how a trained agent acts: agent.policy and the evaluation's action
├── perf_utils.py        build_resumable_train: init or resume, then one jitted scan;
│                        build_chunked_train: the same in chunks, evaluating between them
├── state.py             BaseAgentState, CollectorState and the configuration dataclasses
├── checkpoint.py        save_checkpoint, restore_into
├── random_keys.py       random keys by name
├── precision.py         float32 or bfloat16 compute (train(precision=))
├── wrappers.py          env wrappers: normalisation, action clip, auto-reset, observation noise
├── schedule.py          schedules: a constant or a function of the timestep; warm-up and cosine
├── distributional.py, normalizers.py   symlog, two-hot and return scales (DreamerV3, TD-MPC2)
├── compare.py, ledger.py, reproduction.py   AdaStop comparisons; one step budget per run; the reference comparisons' data
├── render.py            saving an episode as an animation (agent.render)
├── representation/      what RepresentationProbe's recordings need
└── _compile_cache.py    the persistent JAX compile cache
```

A declared agent (every agent but the offline `TDMPC2MultiTask`) is one file,
`<AGENT>/<AGENT>.py`: its hyperparameters as dataclass fields, its `init`,
`update` and equations, its state and metrics. It is built on a family of
`agents/families.py`, which builds its constructor, run config, collection
and train function ([Contributing](contributing.md#adding-an-agent);
[Agent families](../algorithms/index.md#agent-families) lists the agents of
each):

| Family | What the agent writes |
| --- | --- |
| `ReplayAgent` | its `init` and its `update` on a replayed batch |
| `IncrementalAgent` (AVG) | its `init` and its `update` on each step as it comes |
| `OnPolicyAgent` | its `init` and its `update` on a rollout |
| `OwnIterationAgent` (UDRL, APG) | `init`, `iterate`, `report` and `evaluation_metrics`; the loop resumes and logs |
| `OwnReplayAgent` (DreamerV3, TD-MPC2) | `prepare_replay`, `collect`, `count_updates`, `update`, `report` and `evaluation_metrics`, on `loop.off_policy`'s hooks |

Knobs several agents share are mixins of fields (`EntropyTemperature`,
`PolicyUpdateStart`, `ExpertGuided`, ...), and an agent changes what its
family does by overriding a few hooks (`network_settings`,
`evaluation_kwargs`, `uniform_steps`, `keeps_actor_metrics`,
`check_recurrent_options`, ...). The world models' packages also hold their
maths by role: TD-MPC2's networks, one update (`core.py`), its optimisers,
planner, replay and configuration (`state.py`), which the multi-task
`TDMPC2MultiTask` imports instead of copying. That agent trains offline on a
dataset, off the families: its own host loop (`TDMPC2MultiTask.py`) and
train function (`train_TDMPC2MultiTask.py`).

The lineage is explicit: TQC, REDQ, ASAC, RVISAC, CrossQ and AVG import SAC's
equations from `SAC/SAC.py`; APO imports PPO's from `PPO/PPO.py`; the
multi-task TD-MPC2 imports TD-MPC2's. A fix to the parent's maths reaches
every descendant. Some agents keep further modules of their own (TD-MPC2's
`planner.py`). Every agent is importable from the top level:
`from ajax import TD3`.

## From `agent.train()` to a compiled program

```text
agent = SAC("Pendulum-v1", extensions=(...))     DeclaredAgent.__post_init__, ActorCritic's setup
  ├─ self.config = run_config(...)                what the loggers will record: the fields as passed
  ├─ prepare_env                                  the env (episode_length, action_repeat), its params, Ajax's layers
  ├─ EnvironmentConfig, NetworkConfig, OptimizerConfig
  ├─ ExtensionStack(extensions) + checks          capabilities, phases, buffer fields, experts: raise here
  └─ experts read in agent units                  ExpertInAgentUnits, the agent's and the extensions'

agent.train(seed=[0, 1], n_timesteps=..., logging_config=..., initial_state=...)
  ├─ get_make_train()(env_args, ..., total_timesteps, logging_config)
  │    └─ make_train                              its family's, over the agent's init and update, with the extensions
  │         ├─ TrainLoop.create(...)              budget, extensions, logging
  │         └─ loop.off_policy(init, update, ...), loop.on_policy(...) or loop.train(...)
  │              └─ build_resumable_train(...)    jitted train(key, index, initial_state, ...)
  │                 or build_chunked_train(...)   with logging: chunks of ticks, evaluations between
  ├─ jax.vmap over seeds                          one program, every seed
  └─ block_until_ready, stop the logging worker
returns (state, evaluations)                      every leaf has a leading seed axis
```

`ActorCritic` in [`agents/base.py`](../../src/ajax/agents/base.py) does the
work every agent needs at construction. It builds the environment
(`prepare_env`, which every agent hands `episode_length` and `action_repeat`),
groups the static settings into frozen configuration dataclasses, and wraps
the extensions in an `ExtensionStack`. It rejects silent misconfigurations
early: `memory=` on an agent whose `supports_memory` is false, an extension
needing a capability the run lacks or a phase the agent never folds, an
extension reading a replay field the agent does not store, and a stateful
expert called without its state.

`train` then calls the agent's `make_train` (its family's, bound by
`get_make_train()` to the agent's extensions) with the run's settings.
The result is a jitted function of one seed's key. `train` maps it over the
seeds with `jax.vmap` and waits for the result.

The whole run is a single pytree, so this example's state carries both seeds:

```python
import jax

from ajax import PPO

agent = PPO("CartPole-v1", n_envs=2, n_steps=64)
state, out = agent.train(seed=[0, 1], n_timesteps=512, logging_config=None)

print(out)                                # None: no evaluation
print(type(state).__name__)               # PPOState: the whole run, one pytree
print(state.collector_state.timestep)     # [512 512]: env steps, every env counted
print(jax.tree.leaves(state.actor_state.params)[0].shape[0])  # 2: the seed axis
print(agent.config["algo_name"], agent.config["n_steps"])     # PPO 64: the run config
```

### The run config

A declared agent builds its run config from its fields, first thing in
`DeclaredAgent.__post_init__`: each field as passed, and `algo_name`.
DreamerV3 and TD-MPC2 then add, in their setup, the settings they resolve
from those fields. `TDMPC2MultiTask`, the one agent not declared, builds it
as the first line of its constructor: `self.config = run_config(locals(),
self.name, ("dataset", "eval_envs"))`.
[`run_config`](../../src/ajax/agents/base.py) keeps the constructor's
arguments, drops `self`, `__class__` and the names passed in `exclude` (its
dataset and evaluation envs), and adds `algo_name`. With a
`LoggingConfig`, `train` merges `agent.config` into `logging_config.config`,
which W&B and TensorBoard record.

### Agent units and env units

Every continuous agent acts in `[-1, 1]`, whatever the env's action box.
Every env step maps the action onto the box
(`agent_action_to_env` in `environments/utils.py`: `c + h * clip(a, -1, 1)`
with `c` the box's centre and `h` its half-width), so on gymnax Pendulum's
`[-2, 2]` torque the agent reaches the full range. Replay stores the agent's
own action. An expert is written in env units and is read through
`ExpertInAgentUnits`, which maps its answer into agent units with the
inverse map (`env_action_to_agent`, unclipped): the agent's
`expert_policy` and every expert an extension holds (its `expert_fields`) are
converted once, at construction. Replay, losses and extensions see one unit.

## The shared training loop

`TrainLoop` in [`agents/loop.py`](../../src/ajax/agents/loop.py) owns
everything that is the same for every agent. An agent supplies two functions:

- `init(key) -> agent_state`: a fresh state. One-shot pre-training, such as
  behaviour cloning, is an extension's (`CloningPretrain`), on a key the loop
  splits off.
- `update(...) -> (agent_state, aux)`: one update. `aux` holds the update's
  metrics as flax dataclasses nested exactly one level (TD3's
  `AuxiliaryLogs(policy=..., value=...)` is logged as `value/critic_loss` and
  so on).

The loop does the rest, in this order on every iteration:

```text
fresh run:  state = init(key), then apply_init: each extension's initial state, then its pretrain
resumed:    state = the state passed in
iterate:    collect one step per env (off-policy) or a rollout (on-policy)
            if timestep >= learning_starts (on-policy: always):
                state, aux = update(state, experience)
                fold the extensions' post_update
            else:
                aux = the update's metrics, filled with NaN
            evaluate and log every log_frequency steps
```

Because this order lives in one place, a fix to it (when the evaluation runs,
which key an extension draws, how a resumed run continues) reaches every
agent at once. The probes check it end to end: a planted fault that runs
`post_update` before the update instead of after is caught (see
[Testing](testing.md)).

### Three iteration shapes

| Shape | Agents | One iteration | Budget |
| --- | --- | --- | --- |
| `loop.off_policy(init, update, learning_starts, ...)` | the replay agents (<!-- agents: replay -->SAC, TD3, REDQ, CrossQ, TQC, ASAC, RVI-SAC, IQL, DQN, DAgger<!-- /agents -->) and AVG | one step per env (a *tick*); from `learning_starts` on, one `update(state, transition)` | `budget // n_envs` ticks |
| `loop.on_policy(init, update, n_steps, ...)` | the on-policy agents (PPO, APO, REINFORCE, PQN) | an `n_steps` rollout per env from the agent state `agent_state_at_rollout_start`, then `update(agent_state, rollout, agent_state_at_rollout_start)` | the whole rollouts the budget holds, at least one |
| `loop.off_policy(..., collect=, n_updates=, evaluation=)` | DreamerV3, TDMPC2 (`OwnReplayAgent`) | the agent's own `collect(state, tick)`, then `n_updates(tick, state)` updates (a train ratio, a burst), then the agent's own `Evaluation` | `budget // n_envs` ticks, or `num_ticks` |

Before `learning_starts`, the off-policy shape acts uniformly at random
(until `uniform_steps` instead, when an agent's reference acts with its policy
from the start, as TQC's does) and logs the update's metrics as NaN, which the
logger drops. The on-policy shape passes `start` to the update so that a
recurrent agent can replay its rollout from the memory carries it began
with. PPO's update cuts its rollout by a `RolloutSplit`
(`agents/minibatch.py`), which also says where the advantages are computed.

Three agents use the loop's parts without one of these shapes. UDRL and APG
run their own iteration on `TrainLoop.train`, which evaluates, logs and
resumes for them: UDRL a segment, its replay, its command statistics and the
updates, evaluating its command-conditioned policy through an `Evaluation`;
APG a gradient step through a differentiable rollout. The offline
`TDMPC2MultiTask` calls `build_resumable_train` itself, with the loop's
`fresh_state` and its `post_update` fold.

### The update helpers

`compute_gradients(loss, params)` returns a loss's gradients and its aux: the
agent first writes the loss as a small local function of the parameters
alone, its other inputs fixed. PPO and APO step their two networks on it with
`apply_gradients_clipped_together`, which clips both gradients to one global
norm and returns each network's norm before the clip.

Three helpers serve the replay agents: `gradient_step(train_state, loss_fn)`,
one optimiser step; `actor_step(...)`, an actor step that folds the
extensions' `actor_loss` phase; and `critic_step(...)`, a replay critic step
that folds the extensions' `on_target` and `critic_loss` phases. All three
return the gradients' global norm, before the optimiser clips it. SAC, TD3,
TQC, REDQ, ASAC, RVISAC, AVG and IQL use `critic_step`. Its two halves are
public too: `apply_on_target` and `compute_extensions_critic_loss`. An agent
whose target comes out of its own critic pass calls them inside its loss.
SAC, TQC, REDQ, ASAC, RVISAC, CrossQ, TD3, AVG, DAgger and IQL use
`actor_step`, SAC's descendants on SAC's soft policy objective.

Two more keep control flow out of the updates. `run_if(condition, update,
state)` runs `update` where `condition` holds and leaves the state as it is
elsewhere, its metrics then NaN: DQN's target refresh, SAC's warm-ups and the
loop's own `learning_starts` gate. An actor's step goes through
`run_actor_if` instead (a policy delay, `policy_update_start`): a skipped
step returns the metrics of the last one that ran, which the state keeps in
`last_actor_metrics`, so the logs show the last real step. A `Schedule`
(`ajax.schedule`) is a hyperparameter given as a constant or a function of the
timestep, read with `at(schedule, timestep)`; `linear(start, end, steps)` is a
ramp. TD3's exploration noise and PPO's and APO's clip range take one, and
DQN's and PQN's epsilon and SAC's target-entropy ramp are built as one.

An agent whose behaviour policy is its policy plus exploration writes it as
`explore(pi, key, step) -> action` (`ajax.environments.interaction`): from the
policy at the last observations, a key and the timestep, the action to take.
`explore_pipeline(explore, recurrent, n_actions)` turns it into the action
pipeline collection calls, which owns the uniform warm-up actions, a recurrent
actor's carry and a discrete action's stored column. TD3's Gaussian noise
(`make_gaussian_exploration`) and DQN's and PQN's epsilon-greedy
(`epsilon_greedy`) are such functions.

## Agent state

Everything that changes during training lives in one flax dataclass, the
agent state. Nothing changes on the Python objects. This is what lets a run
be one `scan` (the state is the carry), lets seeds be one `vmap` (the state
gains a leading axis), and makes a checkpoint a matter of saving that pytree.

`BaseAgentState` in [`state.py`](../../src/ajax/state.py) holds:

| Field | What it holds |
| --- | --- |
| `rng` | the updates' stream of keys: `get_keys_and_new_state(n)` returns `n` keys and the new state, whose key is the carry, taken first; collection draws on its own stream, `collector_state.rng`, so a draw added on one side leaves the other's keys as they were (`ajax.random_keys`) |
| `actor_state`, `critic_state` | flax `TrainState`s with target parameters, memory carries and the observation normaliser's statistics |
| `collector_state` | the environment state, last observation and episode flags, the replay buffer, `timestep`, the rolling episodic return, a stateful expert's internal state |
| `eval_rng` | the evaluations' key, split apart from `rng` and never advanced: the evaluation at timestep `t` draws from `t` folded into it |
| `n_updates`, `n_logs` | counters; `n_logs` gates the evaluations and numbers them |
| `ext_state` | one entry per extension, `()` without extensions |
| `last_rollout` | the latest rollout, kept only when an extension reads it or the agent is asked to expose it |
| `last_actor_metrics` | the metrics of the actor's last step, kept only by an agent whose actor skips updates (`run_actor_if`), for the skipped updates to log |
| `last_update_metrics` | the metrics of the last update, kept only by a run that updates every few steps, for the evaluations between two updates to log |

Agents subclass it: `SoftActorCriticState` (`agents/temperature.py`) adds
the entropy temperature and is shared by SAC, TQC, REDQ, ASAC, RVISAC,
CrossQ and AVG; the world models carry their model and replay.
`collector_state.timestep` counts environment steps over all parallel
environments (one collection step on four environments adds four). The
`learning_starts` gate, the logging cadence and the training fraction all
read it.

Static settings stay out of the state. `EnvironmentConfig`, `NetworkConfig`,
`OptimizerConfig` and the agent's `<AGENT>Config` are frozen and passed as
static arguments, so they fix shapes and Python-level decisions at trace
time. Extensions are static for the same reason: they must be hashable and
must never change after construction.

## Extensions in the loop

Each phase the stack folds is one method of `ExtensionStack` (`apply_<phase>`,
or `compute_extensions_<network>_loss` for a loss term), and an agent makes
one call to it where the phase applies. The call builds the
`ExtensionContext` (`step`, `rng`, `total_steps`), runs the extensions in the
order given, reads (and, for `pretrain` and `post_update`, writes)
`agent_state.ext_state`, and returns the result. On an empty stack it returns
its input unchanged (a loss term is `0.0`) without building anything, so an
agent without extensions traces nothing extra and never guards a fold. The
phases acting on collection and evaluation are called by the controller
instead.

| Call | Where it is made |
| --- | --- |
| `apply_init` (each `init_state`, then each `pretrain`) | `fresh_state`, fresh runs only (a resumed state already carries both) |
| `apply_post_update` | after every update: the loop's `maybe_update`, `repeat_update` and `on_policy`; APG, UDRL and `TDMPC2MultiTask` in their own iterations |
| `apply_eval_metrics` | at every evaluation, merged into the logged metrics (`compose_eval_metrics` in `log.py`) |
| `apply_reward` | PPO's update, after each rollout and before its targets |
| `apply_on_target`, `compute_extensions_critic_loss` | in the agent's critic step; `critic_step` for SAC, TD3, TQC, REDQ, ASAC, RVISAC, AVG and IQL |
| `compute_extensions_actor_loss` | in the agent's actor step; `actor_step` for the agents listed above |
| the `action` phase | the shared collector's action stage (`ajax.environments.action_stage`), around the agent's pipeline, in the slot each extension names |
| `transform_action`, `actuate` | the controller (`env_args.transforms`): the default pipeline before the choice, the shared collector after it (`expert_collect_kwargs`), evaluation, and `executed_actions` where the critics score executed actions |
| `choose_warmup_action`, `choose_replay_action` | the controller in collection: the warm-up's choice (`env_args.warmups`), the action a replay row stores (`env_args.replay_action_choosers`) |
| `track_memory` | the shared collector, after the environment steps |
| the `observe` phase | the observers the agent lifts out of its stack (`env_args.observers`): the shared collector, at reset and after each step, PPO's forced reset, the expert prefill and evaluation |

`apply_on_target`, `compute_extensions_critic_loss` and
`compute_extensions_actor_loss` return their value and the metrics the phases
returned beside theirs; the agent carries them out of its loss as an aux and
logs them, in its `AuxiliaryLogs`' `extensions` field, under their own names.

The loop's own phases are `LOOP_PHASES`: `pretrain`, `post_update` and
`eval_metrics`, and `observe`, `action`, `eval_action` and `track_memory`,
which only the shared collector and evaluation meet, so an agent collecting
on its own (APG, UDRL, the world models) removes those it does not run. An
agent lists those plus the phases its update folds in
`supported_extension_phases`, and construction rejects any other. Each fold
gets a key of its own, split from one the agent draws nothing else on, so an
extension never shares randomness with the agent's own draws.

[Extension phases](extension-phases.md) covers the phases in detail.

## Evaluation and logging

Three modules share the work:

- [`log.py`](../../src/ajax/log.py) decides **when**. `evaluate_and_log`
  evaluates once every `log_frequency` environment steps, counted against
  `n_logs`, and never past the run's horizon. Agents with their own
  `Evaluation` (the world models, APG, UDRL) use `maybe_eval_and_log`, gated
  on the scan index. Every evaluation draws fresh keys
  (`evaluation_keys`: the evaluation's timestep folded into `eval_rng`'s key
  for the training phase, then split: one key for the episodes, one for the
  extensions' metrics), the same per seed from one run to the next and apart
  from training. `n_logs` counts the evaluations made: a resumed run carries
  it on (`shared_counters`), and an agent's own `Evaluation` reads it to know
  a tick logged.
- [`evaluate.py`](../../src/ajax/evaluate.py) runs **the episodes**:
  `num_episode_test` episodes with the deterministic action, acting as the
  agent's `Controller` says (the observation features, residual RL, gain
  mode). It reports the mean return, the episode length, the policy's
  entropy and, for an agent with a discounted critic, the value bias
  (`Eval/mean bias`: the critic's estimate minus the discounted return
  realised from each evaluated step, in the critic's units). With an expert
  it also rolls the expert out on the same episodes and reports its return.
- [`logging/wandb_logging.py`](../../src/ajax/logging/wandb_logging.py)
  **records**. `LoggingConfig` names the backends and the cadence. The
  compiled program hands each evaluation's metrics to a host callback, which
  forwards them to a separate worker process that passes them to each run's
  backends ([`logging/backends.py`](../../src/ajax/logging/backends.py):
  TensorBoard, Weights & Biases or one of your own). The worker starts only
  when a run has a backend.

What `train` returns follows from the logging config:

- **`logging_config=None`:** no evaluation is traced at all (tracing alone would
  rebuild the evaluation environment), and `train` returns `(state, None)`.
  APG returns its per-update metrics in place of `None`.
- **Not given:** `train` makes a `LoggingConfig` of its own: about ten
  evaluations, no backend, a `ProgressPrinter` as `on_evaluation` and, on a
  terminal or in a notebook, a `ProgressBar` (tqdm) as `on_progress`
  ([`logging/progress.py`](../../src/ajax/logging/progress.py)).
- **A `LoggingConfig`:** the run evaluates whether or not a backend records
  it, and returns `(state, evaluations)`: every logged key's values per seed
  and evaluation. The values are written into rows the loop carries, not
  stacked per iteration, which would cost `seeds x iterations` memory. Rows
  a call does not fill hold -1 (the timestep) or NaN.

With a `LoggingConfig` the run trains in chunks
([`perf_utils.build_chunked_train`](../../src/ajax/perf_utils.py)), as Brax
does: the ticks up to the next evaluation are one jitted scan, and the
evaluation, with its gate and logging, is a program of its own that runs on
the state the chunk ends with. Inside the training program, the evaluation
slowed every tick, by about 4% for SAC on MuJoCo Warp. The ticks after which
the gate can open are known before the run (`TrainLoop.logging_ticks` from
the steps a tick takes, or the agent's `Evaluation.ticks`), so evaluations
come on the same ticks and states as in one scan. Each chunk length compiles
once: one or two lengths, two when `log_frequency` is not a multiple of an
iteration's steps.

With `on_progress` (train's progress bar) each chunk reports from inside its
program (`perf_utils.reporting_chunk`), about a hundred times a call. The
chunk stays one scan of its ticks; a tick that ends a piece, of one length per
chunk (the divisor of its length nearest a hundredth of the call, or the whole
chunk when it has none near, as a prime length), calls `on_progress` from a
host callback under a cond, and the chunk's last report comes after the scan,
once an optimisation barrier has seen the whole state computed (it reads no
tick, and could otherwise come before the chunk runs). The barrier also takes
the count of reports made, which each cond returns once its callback has run,
so the last report comes after the others: callbacks with no data between them
run in any order, and on CI's aarch64 runner a chunk's last report came first,
so the bar, which keeps the furthest, stood still until it. The chunks, their
lengths and the evaluations stay those of a run without it, and on CPU so do
the bits. Cutting the chunks finer instead would report as well, but MuJoCo
Warp captures its CUDA graphs again each time a program runs (two of about
65 ms on SAC Hopper): with a hundred chunks a call, SAC on Hopper trained
about 20% slower. Scanning the pieces one by one inside a chunk, with the
callback or the barrier between them, changed SAC's and TD3's bits on CPU, and
a barrier on every tick AVG's. A host callback inside the training step,
reporting every step, cost 38-63% of the training speed on MuJoCo Warp. APG, which evaluates inside its one training program and returns every
iteration's output, has no chunks to report: its lines keep the time left.

With `LoggingConfig(evaluation_device=...)` the evaluation program keeps its
gate and training metrics but leaves the policy's scores to an
[`AsyncEvaluator`](../../src/ajax/async_evaluation.py): a callback hands it
the state without its replay and the evaluation key, and a thread scores
every seed's policy together on that device and logs the rows. The policy's
evaluation is one function of the state and the key in both log paths
(`ajax.log.evaluate_and_log` and `maybe_eval_and_log`), run inline or handed
over. The loop's train function waits for the evaluator after the last chunk
and writes the scores into the rows.

Two details keep the gates correct under the seed `vmap`. The counters a gate
reads (the timestep, `n_logs`, the scan index) stay unbatched, so
`jax.lax.cond` stays a real branch and evaluates only on logging iterations;
a batched predicate would turn it into a `select` that runs the evaluation on
every iteration. And the logging callback checks its flag itself
(`gated_log_callback`), so it fires only on real evaluations in either
lowering.

## Resuming

[`perf_utils.build_resumable_train`](../../src/ajax/perf_utils.py) builds
every agent's jitted `train(key, index, initial_state, resume_from_state)`.
On a fresh run it calls `init`; on a resumed one it takes the state passed
in, whose buffers XLA reuses (the argument is donated).

`ActorCritic.train(initial_state=state)` continues a run. It checks that
every seed stands at the same timestep, evaluation count and update count
(`shared_counters`; only an agent declaring `per_seed_update_counts`,
TD-MPC2 in its episodic mode, may differ in updates, and keeps them
batched), keeps those counters unbatched, and trains from that timestep to
`timestep + n_timesteps` (the loop's `end_timestep`). Every schedule (the
training fraction, the extensions' `total_steps`, the exploration, the
logging's end) runs to the run's horizon, `train`'s `total_timesteps` (by
default the call's end), so a resumed run continues where it stopped instead
of restarting its schedules, and a run whose calls are all told its horizon
follows the schedules of one call. Agents whose schedules depend on the
absolute tick (the world models, `OwnReplayAgent.resume_iteration_offset`)
also pass an iteration offset. A run split in two, on a new agent instance,
ends in the state one uninterrupted call reaches
(`tests/agents/test_resume_contract.py`).

[`checkpoint.py`](../../src/ajax/checkpoint.py) saves a state's arrays to
disk and grafts them back onto a fresh skeleton; see
[Checkpoints and resuming](../guide/checkpoints.md). A state class that
deletes a data field lists it in its `removed_fields`, a
`ClassVar[frozenset[str]]`: checkpoints saved with the field then restore
without it. A saved field that its class neither has nor lists still fails
to restore. A class that gains a float field lists it in its
`added_fields`: checkpoints saved before it restore it as NaN.

## Removing a public name

A public name or keyword is removed in two steps (CLAUDE.md, rule 10,
"expand, then contract"): first the new form is added and downstream
projects move to it, then the old one goes. A project pinned to a release
keeps its names until it upgrades; the changelog lists what each release
removed.

## Compilation

`import ajax` points JAX at a persistent compile cache
(`~/.cache/ajax/jax_compile_cache`, or `AJAX_JAX_COMPILE_CACHE_DIR`; one
subdirectory per CPU, Python, JAX and jaxlib version; `AJAX_NO_COMPILE_CACHE=1`
turns it off). A second process with the same shapes and code loads the
compiled program instead of compiling it. JAX 0.10 never caches a program
that contains a host callback, so on 0.10 a run with a `LoggingConfig`
recompiles in each new process; JAX 0.11 caches it. CI restores the cache
each job left on the last push to `main`. [Performance](performance.md) has
the numbers.
