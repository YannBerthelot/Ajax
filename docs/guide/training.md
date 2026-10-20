# Training an agent

Every agent trains the same way: build it with an environment and its
hyperparameters, then call `train`. This page covers the arguments every
agent shares, what `train` does with them and what it gives back.

## Building an agent

An agent's constructor takes the environment first, then its paper's
hyperparameters, then `extensions=()`:

```python
from ajax import SAC

agent = SAC(
    "Pendulum-v1",  # an environment id, or a prebuilt environment
    n_envs=4,  # parallel copies of the environment
    actor_architecture=("256", "relu", "256", "relu"),
    critic_architecture=("256", "relu", "256", "relu"),
    gamma=0.99,
    learning_starts=1_000,
)
```

The defaults follow each paper or its reference implementation; the
[algorithm pages](../algorithms/index.md) list them and every deviation.
Most agents accept the arguments below; DQN and PQN take one `learning_rate`
and one `architecture` for their Q-network instead of an actor and a critic.

| Argument | What it sets | Page |
| --- | --- | --- |
| `n_envs` | parallel environment copies per seed | below |
| `actor_learning_rate`, `critic_learning_rate` | a number, or a schedule of the optimiser step | below |
| `actor_architecture`, `critic_architecture` | hidden layers, or a PID policy for the actor | [Networks](networks.md) |
| `env_params` | gymnax environment parameters | [Environments](environments.md) |
| `episode_length`, `action_repeat` | episode length and repeat of a Brax or Playground id | [Environments](environments.md) |
| `normalize_observations`, `normalize_rewards` | running normalisation | [Environments](environments.md) |
| `max_grad_norm` | global-norm gradient clipping (`None`: off); twin and ensemble critics are each clipped on their own | |
| `weight_decay` | decoupled (AdamW) weight decay of the network optimisers, multiplied by the learning rate: each step multiplies the parameters by `1 - lr * weight_decay` (`0`: plain Adam) | |
| `memory` | a recurrent memory block | [Memory](memory.md) |
| `extensions` | research features: experts, exploration, diagnostics | [Extensions](extensions.md) |

An agent refuses what it cannot do when you build it, not halfway through a
run: `memory` on an agent without memory support, SAC on a discrete action
space, `action_repeat` on a gymnax environment, an extension that needs a
step the agent's loop never runs or data its replay does not keep (the
[extension checks](extensions.md#what-is-checked-and-when) list them all).

Each agent also records its run configuration in `agent.config`: every
constructor argument by name (`TDMPC2MultiTask` leaves out its dataset and
evaluation environments), plus `algo_name`. The loggers store it with each
run ([Logging](logging.md)).

## `train`

```python
from ajax import SAC

agent = SAC("Pendulum-v1", learning_starts=1_000)
state, out = agent.train(
    seed=[1, 2, 3],  # one run per seed, all in one program
    n_timesteps=2_000,  # environment steps this call trains for
    num_episode_test=10,  # test episodes per evaluation
    logging_config=None,  # None: no evaluation, and out is None
    initial_state=None,  # or a state to continue from
    total_timesteps=None,  # the run's horizon; None: this call's end
)
```

### The budget

`n_timesteps` counts environment steps summed over the `n_envs` copies. An
off-policy agent collects one step in each copy per iteration, so 100,000
steps with `n_envs=4` are 25,000 iterations. An on-policy agent (PPO, APO,
PQN) collects a rollout of `n_steps` per copy per iteration and trains in
whole rollouts: it runs `n_timesteps // (n_envs * n_steps) + 1` of them, so
it can stop a little past the budget.

The off-policy agents act uniformly at random and do not update until
`learning_starts` steps are collected: 10,000 by default for SAC, TD3, REDQ
and ASAC, 1,000 for DQN, 0 for AVG. A short test run must lower it, or the
agent never trains.

### Seeds

`seed` is one integer or a list. Each seed starts from `jax.random.PRNGKey(seed)`,
and `train` runs all of them at once with `jax.vmap`: one compilation, one
program, every seed in step. More seeds cost little extra time until the
device is full ([Performance](performance.md) has the measurements), so run
the seeds you need in one call rather than one call per seed.

Every array in the returned state has the seed axis first. Seed `i`'s part
is `jax.tree.map(lambda x: x[i], state)`.

The same seed on the same machine and library versions gives the same run.
Two programs compiled apart, or on different hardware, may round
differently, so do not expect identical numbers across machines.

### Parallel environments

`n_envs` copies of the environment step together inside each seed. They
collect more data per iteration, and each copy starts its own episodes. For
an off-policy agent each iteration still makes one update (or the number its
own settings ask for), so at a fixed budget more copies mean fewer updates.
The defaults differ by agent:

| `n_envs` default | Agents |
| --- | --- |
| 1 | SAC, ASAC, REDQ, TD3, IQL, AVG, DQN, UDRL, DAgger, TDMPC2 |
| 4 | PPO, APO |
| 8 | PQN, APG |
| 16 | DreamerV3 |

### Evaluation

With a `logging_config`, the agent is evaluated every `log_frequency`
steps on `num_episode_test` episodes, run in parallel in a separate copy of
the environment. The policy acts deterministically: the mean action of a
continuous policy, the most likely one of a discrete policy.

Each evaluation draws fresh initial states and noise. Its key is folded
from the number of evaluations made so far, out of an evaluation key split
apart from the training key when the run starts. So two evaluations of a
seed see different episodes, the same seed always sees the same ones, and
evaluating repeats none of the random draws the training makes. Every agent
follows this, the world models included.

UDRL evaluates through the same code, with the command (desired return and
horizon) it would collect with; DreamerV3 and TD-MPC2 act through their own
latent policy or planner. Their pages say what they log.

### What `train` returns

`train` returns `(state, out)`.

- **Without a `logging_config`**, `out` is `None`. The evaluation is not
  even compiled into the program.
- **With one**, `out` holds the evaluations: a dictionary from each logged
  name to an array of shape `(seeds, evaluations)`. `out["timestep"]` says
  when each evaluation ran. A call that evaluates fewer times than its
  maximum pads the rest with -1 (the timestep) and NaN (everything else).
- **APG** returns its per-update metrics instead. See [APG](../algorithms/apg.md).

The logged names are listed in [Logging](logging.md).

`state` is the agent's whole state: `actor_state` and `critic_state` (the
networks, their optimisers and parameters), `collector_state` (the
environments, the replay buffer, the step counter `timestep`), the random
keys (`rng` for training, `eval_rng` for evaluation), the evaluation count
`n_logs`, and each extension's state.

## Continuing a run

Pass a returned state back as `initial_state` to train further:

```python
from ajax import PPO

agent = PPO("CartPole-v1", n_envs=8, n_steps=128)
state, _ = agent.train(seed=[0, 1], n_timesteps=8_192)
state, _ = agent.train(seed=[0, 1], n_timesteps=8_192, initial_state=state)
print(state.collector_state.timestep)  # [18432 18432]
```

The second call continues from the first call's timestep. Its schedules
(the extensions', the training fraction, the exploration, the logging's
end) continue where they stopped instead of restarting, and run to the
horizon `total_timesteps`. By default that is the call's own end, the
timestep reached plus `n_timesteps`, so a schedule set as a fraction of
training, such as DQN's epsilon decay, has already finished at the end of a
first call. To split a run and keep the schedules of one call, pass the
whole run's horizon to every call. A call whose `n_timesteps` would take it
past the horizon is refused, so the last call trains the rest. An on-policy
call trains in whole rollouts plus one and ends up to a rollout past its
budget (the last call past the horizon, as one call does), so take the rest
from the timestep reached:

```python
total = 16_384
state, _ = agent.train(seed=[0, 1], n_timesteps=8_192, total_timesteps=total)
reached = int(state.collector_state.timestep[0])  # 9216: one rollout past
state, _ = agent.train(
    seed=[0, 1],
    n_timesteps=total - reached,
    initial_state=state,
    total_timesteps=total,
)
print(state.collector_state.timestep)  # [17408 17408], as one call of 16_384
```

To resume in another process, save the state first:
[Checkpoints and resuming](checkpoints.md).

The call may reuse the memory of the state you pass in. Read or save
anything you need from it before the call.

## Training offline

A replay agent (SAC, TD3, REDQ, ASAC, DQN, IQL) can train on a
[transition dataset](../api/buffers.md) alone, with no environment step.
`train_offline(dataset, n_updates)` puts the dataset's rows in the replay
and in the normalisers, then runs the agent's own update `n_updates` times.
With a `logging_config` it evaluates every `log_frequency` updates and logs
the updates done as the `timestep`. The state it returns is back at
timestep 0, so `train(initial_state=state, ...)` then fine-tunes it online,
its replay keeping the dataset's rows:

```python
import jax
import jax.numpy as jnp

from ajax import TD3
from ajax.buffers.dataset import record
from ajax.logging.wandb_logging import LoggingConfig


def pd_expert(obs):
    theta = jnp.arctan2(obs[..., 1], obs[..., 0])
    return jnp.clip(-(2.0 * theta + 0.5 * obs[..., 2]), -2.0, 2.0)[..., None]


agent = TD3("Pendulum-v1", learning_starts=0)  # online: update from the first step
data = record(pd_expert, agent.env_args, jax.random.PRNGKey(0), n_steps=1_000)
logging = LoggingConfig(config={}, log_frequency=500, use_wandb=False)
state, offline = agent.train_offline(data, n_updates=1_000, seed=[0], logging_config=logging)
print(offline["timestep"])  # [[ 500 1000]]: updates done
state, online = agent.train(seed=[0], n_timesteps=1_000, initial_state=state)
```

The replay samples a row with its successor, so the last row of each
environment's stream (each env's share of the dataset) is not sampled
offline. It runs the agent's own update, with its extensions' terms:
[TD3+BC](../algorithms/td3.md#td3bc-offline) is TD3 with a cloning term, and
[IQL](../algorithms/iql.md) an agent of its own.
`PriorData(dataset, offline_updates=N)` runs the same updates inside the
pre-training of an ordinary `train` call, without evaluations.

## Learning-rate schedules

A learning rate can be a function of the optimiser step, such as an optax
schedule:

```python
import optax

from ajax import PPO

# Decays linearly to zero over the first 1,000 optimiser steps.
schedule = optax.linear_schedule(init_value=3e-4, end_value=0.0, transition_steps=1_000)
agent = PPO(
    "CartPole-v1",
    n_envs=8,
    n_steps=128,
    actor_learning_rate=schedule,
    critic_learning_rate=schedule,
)
state, _ = agent.train(seed=0, n_timesteps=8_192)
```

The step counts gradient updates, not environment steps: here 9 rollouts of
10 epochs of 2 minibatches, 180 steps. `ajax.schedule.warmup_cosine_schedule`
builds the linear warm-up then cosine decay used by APG's in-context
controller.

## The shared training loop

Most agents run on one loop, `TrainLoop` in
[`agents/loop.py`](../../src/ajax/agents/loop.py). An agent brings only its
algorithm: how to build a fresh state and how to make one update. The loop
does everything else, the same way for every agent:

```
fresh run:  build the agent's state, then each extension's state
            and its one-shot pre-training (if any)
resumed:    take the state passed as initial_state
repeat:     collect one step per environment (off-policy)
              or one rollout (on-policy)
            once learning_starts steps are collected:
              update, then let the extensions act after the update
            every log_frequency steps: evaluate and log
```

The world models (DreamerV3, TD-MPC2) act through their own planner or
latent policy and decide how many updates to make per step, but they plug
those pieces into the same loop, in the same order. UDRL runs its own
iteration (a segment, its replay of segments, its command statistics, its
updates) inside the loop's train function. APG and the offline
TDMPC2MultiTask, whose iterations differ more, reuse the loop's pieces
(the fresh state, the extensions' folds, the evaluation) rather than the
loop itself.

Why one loop: collection, the `learning_starts` gate, the extensions,
evaluation, logging and resuming then behave identically across agents and
are fixed in one place. A bug fixed there is fixed for every agent, and an
extension written for one agent works on the others.

The whole loop is one `jax.lax.scan` inside one `jax.jit`, vmapped over the
seeds. Python runs once, to build the program; after that the CPU or GPU runs
the entire training without returning to Python, except to hand the logs
to the [logging worker](logging.md).
[Architecture](../developer/architecture.md) describes the loop for
contributors.
