# Writing an extension

An extension is a research feature that plugs into an agent's training loop
without changing the agent: an expert to learn from, an extra loss term, a
change to the TD target, a measurement logged at each evaluation. This page
explains the contract an extension follows and walks through one that runs.
[Extensions](../guide/extensions.md) in the user guide lists the ones Ajax
ships.

## The rule: a feature is an extension, not a keyword

An agent's constructor should accept its algorithm's own hyperparameters
plus `extensions=()`, and nothing else. The expert options some agents still
carry (`expert_policy`, cloning, SAC's
[expert options](../algorithms/sac.md#expert-options)) are the documented
exception: they reach the network's input and the action pipeline, which
extensions do not reach yet. Do not add to them. If a new feature seems to
need a keyword on an agent (`use_critic_blend=True`, an expert's weight),
write an extension instead. The agent must not know that any particular extension
exists, and must not hold any extension's hyperparameters.

The reason is maintenance. Each feature written into an agent grows that
agent's file and couples it to one paper. SAC once grew to about a hundred
constructor keywords that way, before those features moved into extensions.
Kept separate, an extension works with every agent that folds its phases, and
an agent's file stays readable as its paper's pseudocode.

## The contract

An extension subclasses [`Extension`](../../src/ajax/extensions/base.py) and
follows four rules.

1. **A frozen dataclass.** Write it as `@dataclass(frozen=True)`, with its
   settings as fields. Extensions are static arguments of the compiled
   program, so they must be hashable and must never change. Two extensions
   with equal fields hash equal, so an equivalent run reuses the compiled
   program.
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

## The phases

An agent folds its `ExtensionStack` through each phase in the order the
extensions were given.

| Phase | Returns | How several extensions combine | Folded by |
| --- | --- | --- | --- |
| `init_state(agent_state, rng)` | the extension's state (`()`: none) | one entry each | every agent, fresh runs only |
| `pretrain(agent_state, ext_state, ctx)` | `(agent_state, ext_state)` | in order, each sees the last one's agent state | every agent, fresh runs only |
| `on_target(agent_state, ext_state, batch, target, ctx)` | the new target | in order, each transforms the last one's target | SAC, TD3, REDQ, ASAC, AVG, DQN, PQN, PPO, APO |
| `critic_loss(agent_state, ext_state, batch, ctx)` | a term added to the critic loss | summed | SAC, TD3, REDQ, ASAC, AVG, DQN, PQN, PPO, APO |
| `actor_loss(agent_state, ext_state, batch, ctx)` | a term added to the actor loss | summed | SAC, TD3, REDQ, ASAC, AVG, PPO, APO, APG, UDRL |
| `action(agent_state, ext_state, obs, rng, ctx)` | an action, or `None` to defer | the last non-`None` wins | SAC with an expert policy (below) |
| `eval_action(agent_state, ext_state, obs, rng, ctx)` | an action, or `None` | the last non-`None` wins | no agent yet |
| `post_update(agent_state, ext_state, ctx)` | `(agent_state, ext_state)` | in order | every agent, after each update |
| `eval_metrics(agent_state, ext_state, rng, ctx)` | a dict of metrics | merged | every agent, at each evaluation |

Each agent lists the phases it folds in `supported_extension_phases`. An
extension implementing any other phase is rejected when the agent is built,
instead of being silently ignored. DreamerV3, TD-MPC2 and its multi-task
variant fold only the loop's phases (`pretrain`, `post_update`,
`eval_metrics`).

`init_state` and `pretrain` run on a fresh run only. A resumed run's state
already carries the extension's state and whatever pretraining produced.

### What a phase receives

`ctx` is an `ExtensionContext`:

- `ctx.step`: the collector's timestep, traced. Use it in `jnp.where`, not
  in a Python `if`.
- `ctx.rng`: a key for this call.
- `ctx.total_steps`: the run's horizon, a plain Python `int`. Python-level
  decisions (a schedule's length) can use it at trace time. On a resumed run
  it is the whole run's horizon, so schedules continue rather than restart.

`batch` is a dictionary, and its keys differ by agent. Every `on_target`
batch carries `observations`, `next_observations`, `rewards` and `gamma`.
The episode-end flags differ: the off-policy agents (SAC, TD3, REDQ, ASAC,
AVG, DQN) pass `dones`, the rollout agents (PPO, APO, PQN) pass `terminated`
and `truncated`. `gamma` is `None` for the average-reward agents, ASAC and
APO. The off-policy actor-critics' `critic_loss` batch carries
`observations`, `actions`, `critic_params` and `critic_state`; DQN and PQN
pass their Q-network as `q_state` instead. Before relying on any other key, read the agent's fold
call in its `train_<AGENT>.py` (search for `fold_on_target`,
`fold_critic_loss`, `fold_actor_loss`), or
[`critic_step`](../../src/ajax/agents/loop.py) for TD3, REDQ, ASAC and AVG.

## A worked example

This extension clips the TD target to `[low, high]` for the first part of
the run, then lets it go. It counts the updates it was active for and logs
the count at each evaluation. It touches four phases: `init_state` for its
counter, `on_target` for the clip, `post_update` to count, and
`eval_metrics` to report.

```python
from dataclasses import dataclass

import jax.numpy as jnp

from ajax.agents.TD3.TD3 import TD3
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

## Checking where it applies

An agent that never folds a phase rejects an extension implementing it, at
construction:

```python
from dataclasses import dataclass

from ajax import PQN
from ajax.extensions import Extension


@dataclass(frozen=True)
class ActorNudge(Extension):
    name: str = "actor-nudge"

    def actor_loss(self, agent_state, ext_state, batch, ctx):
        return 0.0


try:
    PQN("CartPole-v1", extensions=(ActorNudge(),))
except ValueError as error:
    print(error)  # PQN does not support these extension phases: ... ['actor_loss'] ...
```

PQN has no actor, so an actor-loss term would never be applied. If the agent
you target does not fold a phase you need, adding the fold to that agent is a
change of its own, made in the agent's update, and the agent still must not
know about your extension. Two further limits: SAC folds `action` only with
an `expert_policy`, and only for extensions that declare an `action_slot`
class variable (`"pre_warmup"` or `"post_warmup"`, where in the action
pipeline the override applies); and SAC, TD3, REDQ and ASAC built with
`memory=` reject every extension when training starts.

## Taking context from the agent: `bind_to_agent`

Some extensions need what the agent builds: its environment, network
configuration, replay buffer or discount. The agent hands this context to
every extension through `ExtensionStack.bind_to_agent(**context)`. Each
extension's `bind_to_agent` returns a new frozen instance holding what it
needs and ignores the rest. The default returns the extension unchanged.

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

Today only SAC and DreamerV3 call `bind_to_agent`, when training starts:

- SAC passes `env_args`, `network_args`, `critic_optimizer_args`,
  `num_critics`, `buffer`, `gamma`, `reward_scale`, `use_train_frac`,
  `augment_obs_with_expert_action`, `action_dim`, `extensions` (the whole
  stack), `expert_policy`, `recurrent`, `update_value_fn` and
  `update_target_fn`.
- DreamerV3 passes `env_args`, `agent_config`, `gamma` and
  `total_timesteps`.

Follow the shipped extensions' pattern. Keep a value the user passed
explicitly, as `MCPretrain` and `PhiRefresh` do. And fail with a clear
message when the context you need is missing, as `ExpertValueMetrics` does
when SAC has no expert, rather than running with `None`.

## Randomness

In `init_state`, `action`, `eval_action` and `eval_metrics` the stack gives
each extension a key of its own, the `rng` argument. The other phases pass
one `ctx` to every extension of the stack, so two extensions drawing from
`ctx.rng` in the same phase draw the same numbers: split it with a constant
of your own (`jax.random.fold_in`) when that matters.

Agents split a key for the extensions only when the stack is not empty, so
the framework did not change the random stream of an agent trained without
extensions. Keep it that way when you add a fold to an agent: split under
`if stack:`.

## Pitfalls

- **A loss term needs the differentiated parameters.** `critic_loss` and
  `actor_loss` run inside the agent's `value_and_grad`. A term computed from
  `agent_state.critic_state.params` has no gradient. Use the parameters the
  batch passes (`critic_params`, or `q_state` for DQN and PQN). The probes
  plant this exact mistake and catch it.
- **Integer metrics.** Between evaluations the logger fills
  `eval_metrics` with NaN of the metric's dtype, and NaN does not fit an
  integer. Return floats.
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
2. Test it in `tests/extensions/`: the feature fires and does what its paper
   says (a metric appears, a term moves the loss the right way, a target
   shifts by the expected amount on a hand-built batch).
3. If it adds a learning behaviour, add a probe case in
   `tests/probing/test_extensions.py` and plant the faults it should catch
   (see [Testing](testing.md)).
4. Document it in the user guide's [Extensions](../guide/extensions.md)
   page, with the paper it comes from.
