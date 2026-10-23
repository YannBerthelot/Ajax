# Writing an extension

An extension is a research feature that plugs into an agent's training loop
without changing the agent: an expert to learn from, an extra loss term, a
change to the TD target, a measurement logged at each evaluation. This page
explains the contract an extension follows and walks through examples that
run. [Extensions](../guide/extensions.md) in the user guide lists the ones
Ajax ships.

## The rule: a feature is an extension, not a keyword

An agent's constructor accepts its algorithm's own hyperparameters plus
`extensions=()`, and nothing else. The documented exceptions are
`expert_policy` (SAC, TD3, REDQ, AVG, APO), whose return each evaluation
logs and which SAC also learns from; SAC's options that reach its network's
input, its collection or its action pipeline, which extensions do not reach
yet ([SAC](../algorithms/sac.md#expert-options)); and TD3's
`action_pipeline`, kept for a downstream project. Do not add to them. If a
new feature seems to need a keyword on an
agent (`use_critic_blend=True`, an expert's weight), write an extension
instead. The agent must not know that any particular extension exists, and
must not hold any extension's hyperparameters.

The reason is maintenance. Each feature written into an agent grows that
agent's file and couples it to one paper. SAC once grew to about a hundred
constructor keywords that way, before those features moved into extensions;
behaviour cloning was the last to move (`CloningPretrain`). Kept separate,
an extension works with every agent that folds its phases, and an agent's
file stays readable as its paper's pseudocode.

## The contract

An extension subclasses [`Extension`](../../src/ajax/extensions/base.py) and
follows six rules.

1. **A frozen dataclass.** Write it as `@dataclass(frozen=True)`, with its
   settings as fields. Extensions are static arguments of the compiled
   program, so they must be hashable and must never change. Two extensions
   with equal fields hash equal; an equivalent run on a new agent traces the
   same program again, which the persistent compile cache then loads instead
   of compiling (unless the run logs).
2. **State lives in the agent state.** Anything that changes during training
   is a pytree returned by `init_state` and stored in
   `agent_state.ext_state`, one entry per extension. Never store it on
   `self`.
3. **Override only what you need.** Every phase has a default that changes
   nothing (the identity, `0.0`, `{}`, `None`). An extension is exactly as
   invasive as the phases it overrides, and an agent without extensions
   traces nothing extra.
4. **Self-contained.** The agent passes context in; the extension decides
   what to do with it. When the extension needs something the agent builds
   (its replay buffer, its discount), it takes it through `bind_to_agent`,
   not through a new agent keyword.
5. **Raise on a missing operand.** When a phase needs a batch key, or a
   value the agent should have bound, and does not get it, it raises a
   `ValueError` naming what is missing. It never returns its input or `0.0`
   instead: that would let an agent that does not pass the key silence the
   extension for a whole run.
6. **Declare what you read beyond the batch.** A replay field the agent
   stores only on request, an expert, the frozen expert critic phi\*: each
   has a class attribute or method (below), and the agent checks them when
   it is built.

## The phases

An agent folds its `ExtensionStack` through each phase in the order the
extensions were given.

| Phase | Returns | How several extensions combine | Folded by |
| --- | --- | --- | --- |
| `init_state(agent_state, rng)` | the extension's state (`()`: none) | one entry each | every agent, fresh runs only |
| `pretrain(agent_state, ext_state, ctx)` | `(agent_state, ext_state)` | in order, each sees the last one's agent state | every agent, fresh runs only |
| `reward(agent_state, ext_state, batch, rewards, ctx)` | `(rewards, ext_state, metrics)`: the rollout's new rewards | in order, each transforms the last one's rewards | PPO, after each rollout, before its targets |
| `on_target(agent_state, ext_state, batch, target, ctx)` | the new target, or `(target, metrics)` (below) | in order, each transforms the last one's target | SAC, TD3, REDQ, ASAC, AVG, DQN, PQN, PPO, APO |
| `critic_loss(agent_state, ext_state, batch, ctx)` | a term added to the critic loss, or `(term, metrics)` | summed | SAC, TD3, REDQ, ASAC, AVG, DQN, PQN, PPO, APO |
| `actor_loss(agent_state, ext_state, batch, ctx)` | a term added to the actor loss, or `(term, metrics)` | summed | SAC, TD3, REDQ, ASAC, AVG, PPO, APO, APG, UDRL |
| `action(agent_state, ext_state, obs, rng, ctx)` | an action, or `None` to defer | the last non-`None` wins | SAC with an expert policy (below) |
| `eval_action(agent_state, ext_state, obs, rng, ctx)` | an action, or `None` | the last non-`None` wins | no agent yet |
| `transform_action(actions, inputs)` | the executed action, from the policy's and the expert's (`ajax.controller.TransformInputs`): before the action is chosen; the replay stores it and the critics score it | one per agent | SAC, TD3, REDQ, TQC, CrossQ, ASAC, AVG, PPO, APO and IQL with an expert policy, in collection, their losses and evaluation |
| `actuate(actions, inputs)` | `(env_actions, expert_state)`: the chosen action executed through the expert; the replay stores the chosen one, and `policy_action_dim(env_args, expert)` gives its width where it is not the env's | one per agent | SAC with an expert policy (gain mode), in collection and evaluation |
| `post_update(agent_state, ext_state, ctx)` | `(agent_state, ext_state)` | in order | every agent, after each update |
| `eval_metrics(agent_state, ext_state, rng, ctx)` | a dict of metrics | merged | every agent, at each evaluation |
| `observe(inputs, state)` | `(columns, state)`: the `(n_envs, observation_width(env_args))` columns appended to the observation `inputs` (`ajax.controller.ObserveInputs`) describes | in column order (`observes_normalised` first) | the shared collector and evaluation, whenever an observation arrives |

Each phase the stack folds is one `ExtensionStack` method, `fold_<phase>`
(`fold_init` runs `init_state` then `pretrain`), and an agent calls it once
where the phase applies; the controller dispatches `action`,
`transform_action` and `actuate` itself.
Each agent lists the phases it folds in
`supported_extension_phases`. An extension implementing any other phase is
rejected when the agent is built, instead of being silently ignored.
DreamerV3, TD-MPC2 and its multi-task variant fold only the loop's phases
(`pretrain`, `post_update`, `eval_metrics`).

`init_state` and `pretrain` run on a fresh run only. A resumed run's state
already carries the extension's state and whatever pretraining produced.

### Logging what a phase computes

`on_target`, `critic_loss` and `actor_loss` may return `(value, metrics)`
instead of their value: `metrics` is a dict of scalars named
`<extension>/<metric>`, logged at each evaluation with the agent's own
training metrics, and like them the last update's or, for PPO, APO, PQN and
UDRL, the mean over the update's epochs, minibatches or steps. They are not
differentiated: the agent returns them beside its loss, as it returns its
own metrics. Returning the value alone logs nothing. Two extensions logging
the same name raise when the agent is traced.

```python
def on_target(self, agent_state, ext_state, batch, target, ctx):
    wins = ...  # where the expert's next action scores higher
    return jnp.where(wins, expert_target, target), {"ibrl/expert_wins": wins.mean()}
```

`eval_metrics` stays the place for what can be read off the agent state at
an evaluation (a schedule, a counter kept in the extension's state).

### What a phase receives

`ctx` is an `ExtensionContext`:

- `ctx.step`: the collector's timestep, traced. Use it in `jnp.where`, not
  in a Python `if`.
- `ctx.rng`: a key for this call.
- `ctx.total_steps`: the run's horizon, a plain Python `int`. Python-level
  decisions (a schedule's length) can use it at trace time. On a resumed run
  it is the whole run's horizon, so schedules continue rather than restart.

`batch` is a dictionary. Every agent folding a phase puts in it the keys
`COMMON_BATCH_KEYS` lists, in
[`extensions/base.py`](../../src/ajax/extensions/base.py); each agent
declares the others in its `phase_batch_keys` (`ajax.PPO.phase_batch_keys`,
for instance), and `tests/extensions/test_phase_contract.py` checks every
agent's batches against its declaration. The replay agents on the shared
steps (`critic_step`, `actor_step`) pass the sets `ajax.agents.loop` lists:

| Phase | Every agent folding it | The shared replay steps add |
| --- | --- | --- |
| `on_target` | `observations`, `next_observations`, `rewards`, `gamma` | `actions`, `dones`, `reward_scale`, `raw_next_observations`, `next_a_expert`, `next_action` (`TARGET_KEYS`) |
| `critic_loss` | `observations`, `targets` | `actions`, `next_observations`, `critic_params`, `critic_state`, `actor_state`, `return_to_go` (`CRITIC_LOSS_KEYS`) |
| `actor_loss` | `actor_params` | `actor_state`, `pi_mean`, `agent_loss`, and the rows the agent passes: `observations`, `raw_observations`, `actions`, `a_expert`, `critic_state` (`REPLAY_ACTOR_KEYS`) |

The other agents build their own batches: PPO's and APO's `on_target` carry
`terminated`, `truncated`, `gae` and `values`, DQN's and PQN's
`critic_loss` their `q_state`, APG's `actor_loss` its `rollout` and
`returns`, PPO's `reward` the rollout's `observations`, `raw_observations`
and `actions`.

What the keys mean: `gamma` is `None` for the average-reward agents, ASAC
and APO. `dones` marks where the target cut the bootstrap. `targets` is the
target `on_target` returned, held constant. The critic and actor parameters
(`critic_params`, `q_state`, `actor_params`) are the ones being
differentiated. `raw_observations` is `None` in a replay agent's run without
an expert, whose replay does not keep them, and so are the stored fields
(`next_action`, `return_to_go`, ...) a run does not store. An agent may add
keys of its own (SAC's `on_target` batch carries `rng_key`, the target's own
key), and declares them.

## A worked example

This extension clips the TD target to `[low, high]` for the first part of
the run, then lets it go. It counts the updates it was active for and logs
the count at each evaluation. It touches four phases: `init_state` for its
counter, `on_target` for the clip, `post_update` to count, and
`eval_metrics` to report.

```python
from dataclasses import dataclass

import jax.numpy as jnp

from ajax import TD3
from ajax.extensions import Extension
from ajax.logging.wandb_logging import LoggingConfig


@dataclass(frozen=True)
class ClippedTarget(Extension):
    """Clips the TD target to [low, high] for the first `fraction` of the
    run, and counts the updates it was active for."""

    low: float
    high: float
    fraction: float = 0.5
    name: str = "clipped-target"

    def active(self, ctx):
        # ctx.step is traced (the collector's timestep); ctx.total_steps
        # is a plain int, the run's horizon.
        return ctx.step < self.fraction * ctx.total_steps

    def init_state(self, agent_state, rng):
        return jnp.asarray(0, jnp.int32)

    def on_target(self, agent_state, ext_state, batch, target, ctx):
        clipped = jnp.clip(target, self.low, self.high)
        return jnp.where(self.active(ctx), clipped, target)

    def post_update(self, agent_state, ext_state, ctx):
        return agent_state, ext_state + self.active(ctx).astype(jnp.int32)

    def eval_metrics(self, agent_state, ext_state, rng, ctx):
        # A float: between evaluations the logger fills metrics with NaN.
        return {"ClippedTarget/active updates": ext_state.astype(jnp.float32)}


agent = TD3(
    "Pendulum-v1",
    learning_starts=200,
    extensions=(ClippedTarget(low=-50.0, high=0.0),),
)
logging = LoggingConfig(config={}, log_frequency=500, use_wandb=False)
state, evaluations = agent.train(seed=[0, 1], n_timesteps=2000, logging_config=logging)
print(state.ext_state[0])                              # [800 800]
print(evaluations["ClippedTarget/active updates"])     # [[301. 800. 800. 800.] ...]
```

Reading it part by part:

- **Fields.** `low`, `high` and `fraction` are the extension's settings.
  `name` labels it in error messages. The dataclass is frozen, so the agent
  can pass it to `jax.jit` as a static argument.
- **State.** `init_state` returns the counter. The agent stores it at
  `state.ext_state[0]`, the first extension's slot, with a leading seed axis
  like every other leaf. That is why `print(state.ext_state[0])` shows one
  count per seed.
- **`on_target`** receives the target the agent computed and returns a new
  one. It must not change the agent state or its own state; the phases that
  can (`pretrain`, `post_update`) return both. On the replay agents the
  result is held constant (`stop_gradient`) before the critic regresses on
  it.
- **`post_update`** runs after every update. Updates start at
  `learning_starts` (timestep 200) and the clip is active while the timestep
  is below 1,000, so each seed counts the 800 updates from timestep 200 to
  999. The first evaluation, at timestep 500, reads 301.
- **`eval_metrics`** adds a key to every evaluation. With a `LoggingConfig`,
  `train` returns it per seed and per evaluation, and W&B or TensorBoard
  record it when one is enabled.

## Reading the batch, and raising when a key is missing

A phase that reads the batch takes what it needs and raises when it is not
there. This critic-loss term is an L2 penalty on the critic's parameters, as
DDPG put on its critic (Lillicrap et al., 2016). It reads `critic_params`,
which the actor-critic agents pass; DQN and PQN pass their Q-network as
`q_state` instead, so on them it raises when training traces the fold:

```python
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ajax import DQN, TD3
from ajax.extensions import Extension


@dataclass(frozen=True)
class CriticWeightDecay(Extension):
    """coef * the sum of the critic's squared parameters, added to its loss."""

    coef: float = 1e-4
    name: str = "critic-weight-decay"

    def critic_loss(self, agent_state, ext_state, batch, ctx):
        params = batch.get("critic_params")
        if params is None:
            raise ValueError(
                "CriticWeightDecay needs the differentiated critic parameters"
                " ('critic_params') in the critic-loss batch; DQN and PQN pass"
                " their Q-network as 'q_state' instead."
            )
        return self.coef * sum(jnp.sum(jnp.square(p)) for p in jax.tree.leaves(params))


TD3("Pendulum-v1", learning_starts=200, extensions=(CriticWeightDecay(),)).train(
    seed=0, n_timesteps=400
)  # trains, with the term in its critic loss

try:
    DQN("CartPole-v1", learning_starts=200, extensions=(CriticWeightDecay(),)).train(
        seed=0, n_timesteps=400
    )
except ValueError as error:
    print(error)  # CriticWeightDecay needs the differentiated critic parameters ...
```

DQN folds `critic_loss`, so it accepts the extension at construction; the
missing key shows when the fold is traced. Every shipped extension
implementing `on_target`, `critic_loss` or `actor_loss` is checked to raise
on an empty batch (`tests/extensions/test_no_silent_noop.py`).

## What an extension declares

An extension says what it needs with class attributes and methods the
agent checks at construction, against what its algorithm has
(`capabilities`) and what its code passes on (the phases it folds, the
context it binds, its batches and replay fields;
`ajax.extensions.capabilities`):

| Declaration | What it says | What the agent does |
| --- | --- | --- |
| `requires` (class attribute, a frozenset) and `needs()` | the capabilities it needs beyond those its phases imply (`on_target`, `critic_loss`: `critic`; `actor_loss`: `actor`; `action`: `off_policy`; `transform_action`: `continuous_actions`) | refuses it when its run lacks one (`MissingCapability`, naming each); `needs()` returns them all and may depend on the extension's options |
| `provides` (class attribute) | the capabilities it gives its stack | adds them to the run's: `MCPretrain` provides `phi_star`, which its readers require |
| `context` (class attribute) | the agent-context keys it cannot work without | refuses it when its `extension_context`, the keys its `make_train` binds, lacks one (`NotWired`) |
| `pretrain_context` (class attribute) | the `ExtensionContext` fields its `pretrain` reads (`env_args`, `update`, `buffer`) | refuses it when its loop does not fill one: its `pretrain_context` (`NotWired`) |
| `batch_keys` (class attribute) and `reads()` | per phase, the batch keys it reads beyond those every agent folding the phase passes | refuses it when the agent's `phase_batch_keys` lacks one of them (`NotWired`) |
| `buffer_fields` (class attribute, a frozenset) | replay fields beyond the agent's own that the extension reads, such as `raw_next_obs`, the environment's own next observation | stores them only when an extension declares them; rejects the ones it cannot store (SAC, TD3, REDQ, ASAC and AVG store `raw_next_obs`) |
| `expert_fields` (class attribute) and `experts()` | the fields holding an expert, written in env units, and the experts the extension calls itself | reads every expert in `expert_fields` in agent units (`ExtensionStack.with_experts`); keeps the raw observations and expert flags in replay for a run with an expert; refuses a stateful expert (one with `init_state`) that `experts()` would call without its state |
| `reads_agent_expert_actions` (class attribute) | the extension reads the expert actions the agent stored, when its expert is the agent's `expert_policy` | lets a stateful expert through for it, on an agent that stores those actions (`stores_expert_actions`: SAC, TD3, REDQ, ASAC, AVG, APO) |
| `reads_step_in_episode` (class attribute) | the extension reads each env's step within its episode | SAC counts it only then (`JSRLCurriculum` reads it) |
| `action_slot` (class attribute) | where in SAC's action pipeline an `action` override applies: `"pre_warmup"` or `"post_warmup"` | SAC dispatches `action` only to extensions with a slot, and only with an `expert_policy` |

`EDGEExploration` does not require `phi_star`: it reads phi\* when
`MCPretrain` ran and the live critic otherwise. Its value gates require a Q
critic (its `needs()` depends on its `gate`).

These checks run when the agent is built:

```python
from dataclasses import dataclass
from typing import ClassVar

from ajax import PQN, SAC, TD3
from ajax.extensions import Extension


@dataclass(frozen=True)
class ActorNudge(Extension):
    """A term on the actor's loss (a stand-in: it adds nothing)."""

    name: str = "actor-nudge"

    def actor_loss(self, agent_state, ext_state, batch, ctx):
        return 0.0


@dataclass(frozen=True)
class NextObservationReader(Extension):
    """Reads the environment's own next observation from replay."""

    name: str = "next-observation-reader"
    buffer_fields: ClassVar[frozenset[str]] = frozenset({"raw_next_obs"})


@dataclass(frozen=True)
class PhiStarReader(Extension):
    """Reads the frozen expert critic phi*, which only MCPretrain fits."""

    name: str = "phi-star-reader"
    requires: ClassVar[frozenset[str]] = frozenset({"phi_star"})


for build in (
    lambda: PQN("CartPole-v1", extensions=(ActorNudge(),)),  # PQN has no actor
    lambda: TD3("Pendulum-v1", extensions=(NextObservationReader(),)),  # SAC stores it
    lambda: SAC("Pendulum-v1", extensions=(PhiStarReader(),)),  # no MCPretrain
):
    try:
        build()
    except ValueError as error:
        print(error)
# PQN cannot run ActorNudge ('actor-nudge'), which needs actor (a policy network trained by gradient).
# TD3 has what its extensions need, but does not pass it on yet: NextObservationReader ('next-observation-reader'): TD3 stores no ['raw_next_obs'] yet.
# SAC cannot run PhiStarReader ('phi-star-reader'), which needs phi_star (add MCPretrain(expert_policy=...), which fits phi*, to the extensions).
```

If the agent you target does not fold a phase you need, adding the fold to
that agent is a change of its own, made in the agent's update, and the agent
still must not know about your extension. One more limit: SAC, TD3, REDQ,
ASAC and PPO built with `memory=` reject every extension when training
starts; APG takes both.

## Taking context from the agent: `bind_to_agent`

Some extensions need what the agent builds: its environment, optimiser,
replay buffer or discount. The agent hands this context to every extension
through `ExtensionStack.bind_to_agent(**context)`. Each extension's
`bind_to_agent` returns a new frozen instance holding what it needs and
ignores the rest. The default returns the extension unchanged.

```python
import dataclasses
from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp

from ajax import SAC
from ajax.extensions import Extension
from ajax.logging.wandb_logging import LoggingConfig


@dataclass(frozen=True)
class EffectiveHorizon(Extension):
    """Logs 1 / (1 - gamma), the agent's effective planning horizon."""

    gamma: Optional[float] = None  # filled by bind_to_agent unless given
    name: str = "effective-horizon"

    def bind_to_agent(self, **agent_context):
        gamma = agent_context.get("gamma")
        if self.gamma is not None or gamma is None:
            return self  # a value the user passed wins
        return dataclasses.replace(self, gamma=gamma)

    def eval_metrics(self, agent_state, ext_state, rng, ctx):
        if self.gamma is None:
            raise ValueError("EffectiveHorizon needs gamma: pass it, or use an agent that binds it")
        return {"Debug/effective horizon": jnp.asarray(1.0 / (1.0 - self.gamma))}


agent = SAC("Pendulum-v1", gamma=0.95, learning_starts=200, extensions=(EffectiveHorizon(),))
logging = LoggingConfig(config={}, log_frequency=500, use_wandb=False)
_, evaluations = agent.train(seed=0, n_timesteps=1000, logging_config=logging)
print(evaluations["Debug/effective horizon"])  # [[20. 20.]]
```

Five agents call `bind_to_agent`, when training starts:

- SAC passes `env_args`, `actor_optimizer_args`, `network_args`,
  `critic_optimizer_args`, `num_critics`, `buffer`, `gamma`, `reward_scale`,
  `extensions` (the whole stack), `expert_policy`, `recurrent`,
  `update_value_fn` and `update_target_fn`. Its observation features reach
  extensions through `env_args.observers`.
- TD3, REDQ and APO pass `env_args`, `actor_optimizer_args` and
  `expert_policy`, which `CloningPretrain` needs.
- DreamerV3 passes `env_args`, `agent_config`, `gamma` and
  `total_timesteps`.

Follow the shipped extensions' pattern. Keep a value the user passed
explicitly, as `MCPretrain` and `PhiRefresh` do. And raise with a clear
message when the context you need is missing, as `PhiRefresh` and
`CloningPretrain` do on an agent that did not bind it, rather than running
with `None`. On an agent that binds nothing, `EffectiveHorizon` above raises
the same way.

## Randomness

`init_state` and `eval_metrics` give each extension a key of its own, the
`rng` argument. The other phases pass one `ctx` to every extension of the
stack, so two extensions drawing from `ctx.rng` in the same phase draw the
same numbers: split it with a constant of your own (`jax.random.fold_in`)
when that matters. SAC's action pipeline passes the same key as `rng` to
every `action` extension too.

Each fold's key is one no other fold gets and one the agent draws nothing
else on (`tests/extensions/test_phase_contract.py` checks both on every
agent). `critic_step` and `actor_step` derive their folds' keys from the
step's key with `extension_key(key)` (`ajax.agents.loop`), so an agent hands
them the key its step draws on and never splits one for the extensions. When
you add a fold to an agent outside them, fold on `extension_key` of a key the
step holds, never a split of it: under JAX's partitionable threefry
`fold_in(k, i)` is `split(k, n)[i]`, which is why `extension_key` folds in a
tag far past any split's length. An agent's own random stream is then the
same with or without extensions.

## Pitfalls

- **A loss term needs the differentiated parameters.** `critic_loss` and
  `actor_loss` run inside the agent's `value_and_grad`. A term computed from
  `agent_state.critic_state.params` has no gradient. Use the parameters the
  batch passes (`critic_params`, `actor_params`, or `q_state` for DQN and
  PQN). The probes plant this exact mistake and catch it, and the phase
  contract checks that a term moves the network it names, and only that one.
- **Integer metrics.** Between evaluations the logger fills `eval_metrics`
  with NaN of the metric's dtype. An integer cannot hold NaN: it becomes 0,
  which reads as a real value. Return floats.
- **State on `self`.** A Python attribute is read once, when the program is
  traced, and never again. Whatever must change during training belongs in
  `init_state`'s pytree.
- **`on_target` cannot pass a gradient.** The target is held constant. To
  change what the critic learns through a gradient, use `critic_loss`.
- **Shapes.** A replay agent's target has a batch axis; a rollout agent's
  has time and environment axes. Read the shapes from the agent's fold call,
  or stay shape-agnostic as `ClippedTarget` does.

## Shipping it

1. Put the class in `src/ajax/extensions/<topic>.py`, next to related ones:
   `expert.py` (learning from an expert), `exploration.py`, `target_mods.py`
   (target modifiers), `pretrain.py`, `instrumentation.py` (measurements),
   `ensemble.py`. `ensemble.py`'s `KernelRepulsion`, a single `critic_loss`
   term with its paper's citation, is a good model.
2. Test it in `tests/extensions/`: the feature does what its paper says (a
   metric meets its closed form, a term moves the loss the right way, a
   target shifts by the expected amount on a hand-built batch).
3. Give it the rows the generic tests ask for, which fail until it has them:
   a builder in `EXTENSIONS` of `tests/agents/test_option_smoke.py` (every
   extension of `ajax.extensions` is stacked into an agent that folds it and
   trained briefly), and, if it implements `on_target`, `critic_loss` or
   `actor_loss`, a row in `BUILD` of `tests/extensions/test_no_silent_noop.py`.
4. If it adds a learning behaviour, add a probe case in
   `tests/probing/test_extensions.py` and plant the faults it should catch
   (see [Testing](testing.md)).
5. Document it in the user guide's [Extensions](../guide/extensions.md)
   page, with the paper it comes from.
