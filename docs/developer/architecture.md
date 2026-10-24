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
  loop, not a keyword on the agent. The exceptions left are
  `expert_policy` (SAC, TD3, REDQ, AVG, APO), SAC's options that reach its
  network's input, collection or action pipeline, which extensions do not
  reach yet, and TD3's `action_pipeline` for a downstream project.

## The package

```text
src/ajax/
├── __init__.py          the 14 exported agents; configures the compile cache on import
├── agents/
│   ├── base.py          ActorCritic: environment and network setup, extensions=, train(); run_config
│   ├── families.py      declared agents (fields as hyperparameters) and the replay family's train function
│   ├── loop.py          TrainLoop, the shared loop (brax's forced resets included)
│   ├── steps.py         critic_step, actor_step, gradient_step, joint_step, when, the target update
│   ├── minibatch.py     the on-policy geometry, minibatches and epochs
│   ├── critics.py       the Q-critics' regression (continuous and discrete actions)
│   ├── diagnostics.py   KL, explained variance, policy std
│   ├── recurrent.py     replay buffers and R2D2-style sequence replay for memory=
│   ├── cloning.py       the behaviour-cloning loss and fit that CloningPretrain runs
│   └── <AGENT>/         one package per agent (below)
├── extensions/          Extension, ExtensionStack and the shipped extensions
├── environments/
│   ├── create.py        build an env from an id (gymnax, Brax, MuJoCo Playground), add Ajax's layers
│   ├── interaction.py   collect_experience: act, step, store, count
│   ├── utils.py         env helpers; the [-1, 1] action map; ExpertInAgentUnits
│   ├── obs_norm.py      agent-side running observation normalisation
│   ├── row_collector.py the world models' collector: one row per env per tick
│   ├── differentiable.py  closed-loop rollouts you can backpropagate through (APG)
│   └── model_reference.py, system_class.py, pid_terms.py   tracking tasks, parameter distributions, PID error terms
├── networks/            actor and critic networks (deterministic actors, Q-networks), memory blocks, Squash, the world models' layers
├── modules/             the PID actor and PID output head
├── buffers/             replay buffers (flashbax)
├── logging/             LoggingConfig; the W&B and TensorBoard worker
├── log.py, evaluate.py  when to evaluate and log; the evaluation episodes
├── controller.py      what a step acts on (ActingState) and how the agent acts beyond its actor (Controller)
├── perf_utils.py        build_resumable_train: init or resume, then one jitted scan
├── state.py             BaseAgentState, CollectorState and the configuration dataclasses
├── checkpoint.py        save_checkpoint, restore_into
├── wrappers.py          env wrappers: normalisation, action clip, auto-reset, observation noise
├── deprecation.py       warn_inert and moved_names, for keywords and paths on their way out
├── schedule.py          a warm-up and cosine learning-rate schedule
├── distributional.py, normalizers.py   symlog, two-hot and return scales (DreamerV3, TD-MPC2)
└── _compile_cache.py    the persistent JAX compile cache
```

A declared agent (IQL, CrossQ, TD3, PPO, APO, PQN) is one file, `<AGENT>/<AGENT>.py`: its
hyperparameters as dataclass fields, its `init`, `update` and equations, its
state and metrics, on a family of `agents/families.py` that builds its
constructor, run config, replay and train function
([Contributing](contributing.md#adding-an-agent)); its old `train_<AGENT>.py`
and `state.py` serve their names until 0.5.0. Every other agent package
`src/ajax/agents/<AGENT>/` holds:

| File | What it holds |
| --- | --- |
| `<AGENT>.py` | the public class: the paper's hyperparameters and `extensions=`; its `make_train` and `train_kwargs` |
| `train_<AGENT>.py` | the algorithm, and a `make_train` that hands its initialisation and update to `TrainLoop` |
| `state.py` | `<AGENT>State` and `<AGENT>Config` |
| `core.py` | (SAC, PPO, TDMPC2) the maths that descendants import instead of copying |

The lineage is explicit: ASAC, REDQ and AVG import SAC's `core.py`; APO
imports PPO's; the multi-task TD-MPC2 imports TD-MPC2's. A fix to the parent's
maths reaches every descendant. Some agents keep further modules of their own
(SAC's `action_pipeline.py`, TD-MPC2's `planner.py`).

### Names that moved, keywords that went

A few names now live where their layer is. The old paths are gone, removed
once no downstream project used them rather than kept as aliases.

| Was | Is |
| --- | --- |
| `ajax.agents.obs_norm` | `ajax.environments.obs_norm` |
| `ajax.agents.SAC.utils.SquashedNormal`, then `ajax.networks.distributions.SquashedNormal(loc, scale)` | `ajax.networks.distributions.Squash(distrax.Normal(loc, scale))` |
| `NormalizationInfo` in `ajax.agents.AVG.utils` and `.state` | `RunningMoments` in `ajax.agents.AVG.state` |
| `ajax.plane.get_mode` | `jax.default_backend()` |
| the `_obs_noise_sigma` hook on an env | `ObservationNoiseWrapper` in `ajax.wrappers` |

The keywords below went in October 2026; passing one now raises
`TypeError`.

| Keyword | Agents | Instead |
| --- | --- | --- |
| `pre_train_n_steps`, `actor_cloning_epochs`, `actor_cloning_batch_size`, `actor_cloning_lr`, `skip_actor_pretrain`, `reset_log_std_after_bc`, `reset_actor_head_after_bc` | SAC, TD3, REDQ, APO | `extensions=(CloningPretrain(expert, n_steps=..., epochs=..., batch_size=..., skip_actor=..., reset_log_std=..., reset_actor_head=...),)`, at the actor's learning rate |
| `pid_actor_config` | SAC, TD3, REDQ, ASAC, AVG, PPO, APO | `actor_architecture=PIDPolicy(...)` |
| `action_scale` | SAC | nothing: SAC never read it, and every continuous action is mapped onto the env's bounds |
| `use_expert_guidance` | SAC | nothing: it only switched on diagnostics; `ExpertValueMetrics` logs the critic's view of the expert |
| `reward_shaping_fn`, `use_vtrace_gae` | PPO | nothing: neither ever ran |
| `actor_kernel_init`, `critic_kernel_init`, `encoder_bias_init` | PPO, APO | nothing: no run set them; `encoder_kernel_init`, `actor_bias_init`, `critic_bias_init` and PPO's `mean_kernel_init` stay |
| `n_epochs`, `bc_loss_type` | UDRL | `n_updates_per_iter`; the loss is the log-likelihood |

Every agent is importable from the top level, TD3 and UDRL included:
`from ajax import TD3`.

## From `agent.train()` to a compiled program

```text
agent = SAC("Pendulum-v1", extensions=(...))     ActorCritic.__init__
  ├─ self.config = run_config(locals(), name)     what the loggers will record
  │                                               (a declared agent, IQL: its fields, in DeclaredAgent.__post_init__)
  ├─ prepare_env                                  the env (episode_length, action_repeat), its params, Ajax's layers
  ├─ EnvironmentConfig, NetworkConfig, OptimizerConfig
  ├─ ExtensionStack(extensions) + checks          phases, buffer fields, experts, phi*: raise here
  └─ experts read in agent units                  ExpertInAgentUnits, the agent's and the extensions'

agent.train(seed=[0, 1], n_timesteps=..., logging_config=..., initial_state=...)
  ├─ get_make_train()(env_args, ..., total_timesteps, logging_config)
  │    └─ make_train                              train_<AGENT>.py, with train_kwargs and the extensions
  │                                               (a declared agent: ReplayAgent.make_train over its init and update)
  │         ├─ TrainLoop.create(...)              budget, extensions, logging
  │         └─ loop.off_policy(init, update, ...) or loop.on_policy(...)
  │              └─ build_resumable_train(...)    jitted train(key, index, initial_state, ...)
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
implementing a phase the agent never folds, an extension reading a replay
field the agent does not store, a stateful expert called without its state,
and an extension reading the frozen expert critic phi\* in a stack that never
fits it.

`train` then calls the agent's `make_train` (bound by `get_make_train()` to
the agent's own `train_kwargs` and its extensions) with the run's settings.
The result is a jitted function of one seed's key. `train` maps it over the
seeds with `jax.vmap` and waits for the result.

The whole run is a single pytree, so this example's state carries both seeds:

```python
import jax

from ajax import PPO

agent = PPO("CartPole-v1", n_envs=2, n_steps=64)
state, out = agent.train(seed=[0, 1], n_timesteps=512)

print(out)                                # None: no logging config, no evaluation
print(type(state).__name__)               # PPOState: the whole run, one pytree
print(state.collector_state.timestep)     # [640 640]: env steps, every env counted
print(jax.tree.leaves(state.actor_state.params)[0].shape[0])  # 2: the seed axis
print(agent.config["algo_name"], agent.config["n_steps"])     # PPO 64: the run config
```

The timestep reads 640, not 512, because an on-policy run always makes one
rollout more than the budget holds (see [the loop shapes](#three-iteration-shapes)).

### The run config

Every agent but the declared ones builds its run config the same way, as
the first line of its constructor: `self.config = run_config(locals(),
self.name)`. A declared agent (IQL, CrossQ, TD3, PPO, APO, PQN) builds it from its fields,
first thing in `DeclaredAgent.__post_init__`: each field as passed, and
`algo_name`.
[`run_config`](../../src/ajax/agents/base.py) keeps the constructor's
arguments, drops `self`, `__class__` and any names passed in `exclude`
(`TDMPC2MultiTask` leaves out its dataset and evaluation envs), and adds
`algo_name`. DreamerV3 and TD-MPC2 then add the settings they resolve from
those arguments. With a
`LoggingConfig`, `train` merges `agent.config` into `logging_config.config`,
which is what W&B and TensorBoard record for the run.

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
fresh run:  state = init(key), then fold_init: each extension's initial state, then its pretrain
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
| `loop.off_policy(init, update, learning_starts, ...)` | SAC, TD3, REDQ, ASAC, AVG, DQN | one step per env (a *tick*); from `learning_starts` on, one `update(state, transition)` | `budget // n_envs` ticks |
| `loop.on_policy(init, update, n_steps, ...)` | PPO, APO, PQN | an `n_steps` rollout per env from the state `start`, then `update(state, rollout, start)` | `budget // (n_envs * n_steps) + 1` rollouts |
| `loop.off_policy(..., collect=, n_updates=, evaluation=)` | DreamerV3, TDMPC2 | the agent's own `collect(state, tick)`, then `n_updates(tick, state)` updates (a train ratio, a burst), then the agent's own `Evaluation` | `budget // n_envs` ticks, or `num_ticks` |

Before `learning_starts`, the off-policy shape acts uniformly at random
(until `uniform_steps` instead, when an agent's reference acts with its policy
from the start, as TQC's does) and logs the update's metrics as NaN, which the
logger drops. The on-policy shape
passes `start` to the update so that a recurrent agent can replay its rollout
from the memory carries it began with.

Three agents use the loop's parts without one of these shapes. UDRL runs its
own iteration (a segment, its replay, its command statistics, the updates)
on `TrainLoop.train`, and evaluates its command-conditioned policy every
`log_frequency` steps through an `Evaluation`. APG and the offline
`TDMPC2MultiTask` call `build_resumable_train` themselves, with the loop's
`fresh_state` and its `post_update` fold.

Three helpers serve the replay agents: `gradient_step(train_state, loss_fn)`,
one optimiser step; `actor_step(...)`, an actor step that folds the
extensions' `actor_loss` phase; and `critic_step(...)`, a replay critic step that folds
the extensions' `on_target` and `critic_loss` phases. All three return the
gradients' global norm, before the optimiser clips them. SAC, TD3, REDQ, ASAC,
AVG and IQL use `critic_step`. Its two folds are public too: `fold_on_target`
and `fold_critic_loss`. An agent whose target comes out of its own critic
pass calls them inside its loss.
SAC, TD3, AVG, DAgger and IQL use `actor_step`; ASAC and REDQ use
`soft_actor_step` in `SAC/core.py`, on the same `soft_policy_loss` as SAC.

Two more keep control flow out of the updates. `when(pred, fn, state)` runs
`fn` where `pred` holds and leaves the state as it is elsewhere, its metrics
then NaN (or `fill`): TD3's policy delay, DQN's target refresh, SAC's
warm-ups and the loop's own `learning_starts` gate. A `Schedule`
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
(`gaussian_exploration`) and DQN's and PQN's epsilon-greedy
(`epsilon_greedy`) are such functions.

## Agent state

Everything that changes during training lives in one flax dataclass, the
agent state. Nothing changes on the Python objects. This is what lets a run
be one `scan` (the state is the carry), lets seeds be one `vmap` (the state
gains a leading axis), and makes a checkpoint a matter of saving that pytree.

`BaseAgentState` in [`state.py`](../../src/ajax/state.py) holds:

| Field | What it holds |
| --- | --- |
| `rng` | the training key; every random draw splits it (`next_keys(n)` returns the state and `n` keys) |
| `actor_state`, `critic_state` | flax `TrainState`s with target parameters, memory carries and the observation normaliser's statistics |
| `collector_state` | the environment state, last observation and episode flags, the replay buffer, `timestep`, the rolling episodic return, a stateful expert's internal state |
| `eval_rng` | the evaluations' key, split apart from `rng` and never advanced: evaluation `n` draws from `fold_in(eval_rng, n)` |
| `n_updates`, `n_logs` | counters; `n_logs` gates the evaluations and numbers them |
| `ext_state` | one entry per extension, `()` without extensions |
| `last_rollout` | the latest rollout, kept only when the agent is asked to expose it |

Agents subclass it: `SoftACState` adds SAC's temperature and is shared by
SAC, ASAC, REDQ and AVG; the world models carry their model and replay.
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

Each phase the stack folds is one method of `ExtensionStack`,
`fold_<phase>`, and an agent makes one call to it where the phase applies
(SAC's action pipeline dispatches `action` itself). The call builds the
`ExtensionContext` (`step`, `rng`, `total_steps`), runs the extensions in
the order given, reads (and, for `pretrain` and `post_update`, writes)
`agent_state.ext_state`, and returns the result. On an empty stack it returns
its input unchanged (a loss term is `0.0`) without building anything, so an
agent without extensions traces nothing extra and never guards a fold.

| Call | Where it is made |
| --- | --- |
| `fold_init` (each `init_state`, then each `pretrain`) | `fresh_state`, fresh runs only (a resumed state already carries both) |
| `fold_post_update` | after every update: the loop's `maybe_update`, `repeat_update` and `on_policy`; APG, UDRL and `TDMPC2MultiTask` in their own iterations |
| `fold_eval_metrics` | at every evaluation, merged into the logged metrics (`compose_eval_metrics` in `log.py`) |
| `fold_on_target`, `fold_critic_loss` | in the agent's critic step; `critic_step` for SAC, TD3, REDQ, ASAC, AVG and IQL |
| `fold_actor_loss` | in the agent's actor step; `actor_step` for SAC, TD3, AVG, DAgger and IQL |
| the `action` phase | SAC's action pipeline only, with an expert policy |
| `transform_action`, `actuate` | the controller (`env_args.transforms`): the action pipeline before the choice, the shared collector after it (`expert_collect_kwargs`), evaluation, and `executed_actions` where the critics score executed actions |
| the `observe` phase | the observers the agent lifts out of its stack (`env_args.observers`): the shared collector, at reset and after each step, PPO's forced reset, the expert prefill and evaluation |

`fold_on_target`, `fold_critic_loss` and `fold_actor_loss` return their
value and the metrics the phases returned beside theirs; the agent carries
them out of its loss as an aux and logs them, in its `AuxiliaryLogs`'
`extensions` field, under their own names.

The loop's own phases are `LOOP_PHASES`: `pretrain`, `post_update`,
`eval_metrics` and `observe`, which an agent collecting on its own (APG,
UDRL, the world models) removes; an agent lists those plus the phases its
update folds in `supported_extension_phases`, and construction rejects any
other. Each fold gets a key of its own, split from one the agent draws
nothing else on, so an extension never shares randomness with the agent's
own draws.

[Writing an extension](writing-an-extension.md) covers the phases in detail.

## Evaluation and logging

Three modules share the work:

- [`log.py`](../../src/ajax/log.py) decides **when**. `evaluate_and_log`
  evaluates once every `log_frequency` environment steps, counted against
  `n_logs`, and never past the run's horizon. Agents with their own
  `Evaluation` (the world models, APG, UDRL) use `maybe_eval_and_log`, gated
  on the scan index. Every evaluation draws fresh keys
  (`evaluation_keys`: `split(fold_in(eval_rng, n_logs))`, one key for the
  episodes, one for the extensions' metrics), the same per seed from one run
  to the next and apart from training.
- [`evaluate.py`](../../src/ajax/evaluate.py) runs **the episodes**:
  `num_episode_test` episodes with the deterministic action, acting as the
  agent's `Controller` says (SAC's expert in the actor's input, residual RL,
  gain mode), reporting the
  mean return, the episode length, the policy's entropy and, for the
  discounted agents with a critic (SAC, TD3, REDQ, AVG, DQN, PQN, PPO), the
  value bias (`Eval/mean bias`: the critic's estimate minus the discounted
  return realised from each evaluated step, in the critic's units). With an expert it also rolls the expert out on the same episodes
  at each evaluation and reports its return.
- [`logging/wandb_logging.py`](../../src/ajax/logging/wandb_logging.py)
  **records**. `LoggingConfig` names the backend (`use_wandb`,
  `use_tensorboard`) and the cadence. The compiled program hands each
  evaluation's metrics to a host callback, which forwards them to a separate
  worker process that writes to Weights & Biases or TensorBoard. The worker
  starts only when a backend is enabled.

What `train` returns follows from the logging config:

- **No `LoggingConfig`:** no evaluation is traced at all (tracing alone would
  rebuild the evaluation environment), and `train` returns `(state, None)`.
  APG returns its per-update metrics in place of `None`.
- **A `LoggingConfig`:** the run evaluates whether or not a backend records
  it, and returns `(state, evaluations)`: every logged key's values per seed
  and evaluation. The values are written into rows the scan carries, not
  stacked per iteration, which would cost `seeds x iterations` memory. Rows
  a call does not fill hold -1 (the timestep) or NaN.

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
TD-MPC2 in its episodic mode, may differ in updates, and keeps them batched), keeps those
counters unbatched, and trains from that timestep to `timestep +
n_timesteps` (the loop's `end_timestep`). Every schedule (the training
fraction, the extensions' `total_steps`, the exploration, the logging's end)
runs to the run's horizon, `train`'s `total_timesteps` (by default the
call's end; the loop's `total_timesteps`), so a resumed run continues where
it stopped instead of restarting its schedules, and a run whose calls are
all told its horizon follows the schedules of one call. Agents whose
schedules depend on the absolute tick (the world models) also pass an
iteration offset. A run split in two, on a new agent instance, ends in the
state one uninterrupted call reaches (`tests/agents/test_resume_contract.py`).
[`checkpoint.py`](../../src/ajax/checkpoint.py) saves a state's arrays to disk
and grafts them back onto a fresh skeleton; see
[Checkpoints and resuming](../guide/checkpoints.md). A state class that
deletes a data field lists it in its `removed_fields`, a
`ClassVar[frozenset[str]]`: checkpoints saved with the field then restore
without it. A saved field that its class neither has nor lists still fails
to restore. A class that gains a float field lists it in its
`added_fields`: checkpoints saved before it restore it as NaN.

## Keywords on their way out

A public keyword is removed in two steps (CLAUDE.md, rule 10, "expand, then
contract"): first the new form is added and downstream projects move to it,
then the old one goes. A keyword that no longer does anything but that
callers still pass stays accepted meanwhile, and says so with
[`ajax.deprecation.warn_inert`](../../src/ajax/deprecation.py): a
`FutureWarning` for each such keyword set to something other than `None`,
pointing at the caller's line and giving the reason.

```python
import warnings

from ajax import SAC

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    SAC("Pendulum-v1", jsrl_curriculum=True)  # SAC counts episode steps itself
print([str(w.message) for w in caught if w.category is FutureWarning])
# ["SAC's jsrl_curriculum has no effect and will be removed: SAC counts each
#   env's steps in its episode when an extension reads them (JSRLCurriculum)."]
```

Today it covers SAC's `jsrl_curriculum` (SAC now counts episode steps
whenever an extension reads them) and `LoggingConfig`'s `horizon`, which
nothing reads. Keywords whose downstream callers have moved are removed
outright, without a warning.

## Compilation

`import ajax` points JAX at a persistent compile cache
(`~/.cache/ajax/jax_compile_cache`, or `AJAX_JAX_COMPILE_CACHE_DIR`; one
subdirectory per CPU, Python, JAX and jaxlib version; `AJAX_NO_COMPILE_CACHE=1`
turns it off). A second process with the same shapes and code loads the
compiled program instead of compiling it. JAX never caches a program that
contains a host callback, so a run with a `LoggingConfig` recompiles in each
new process. CI restores the cache each job left on the last push to `main`.
[Performance](performance.md) has the numbers.
