# Training an agent

Every agent trains the same way: build it with an environment and its
hyperparameters, then call `train`. This page covers the arguments every
agent shares, what `train` does with them and what it returns.

## Building an agent

The constructor takes the environment first, then the algorithm's
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

The defaults are each paper's or its reference implementation's; the
[algorithm pages](../algorithms/index.md) list them and every deviation.
Most agents take the arguments below. Some take one `learning_rate` instead
of two (CrossQ, APG, DQN, PQN and the world models); DQN and PQN take one
`architecture` for their Q-network, and CrossQ an `actor_width` and a
`critic_width`.

| Argument | What it sets | See |
| --- | --- | --- |
| `n_envs` | parallel copies of the environment, per seed | [below](#parallel-environments) |
| `actor_learning_rate`, `critic_learning_rate` | a number, or a schedule of the optimiser step | [below](#learning-rate-schedules) |
| `actor_architecture`, `critic_architecture` | the hidden layers, or a PID policy for the actor | [Networks](networks.md) |
| `env_params` | gymnax environment parameters | [Environments](environments.md#your-own-environment) |
| `episode_length`, `action_repeat` | the episode length and action repeat of a Brax or Playground id | [Environments](environments.md#brax-and-playground) |
| `normalize_observations`, `normalize_rewards` | running normalisation | [Environments](environments.md#normalisation) |
| `max_grad_norm` | global-norm gradient clipping, `None` for none: each network on its own, and each critic of a pair or an ensemble too; PPO and APO clip their two networks together, as their reference codes do | |
| `weight_decay` | decoupled (AdamW) weight decay: each step multiplies the parameters by `1 - learning_rate * weight_decay`; `0` is plain Adam | |
| `memory` | a recurrent memory block | [Memory](memory.md) |
| `extensions` | research features: experts, exploration, diagnostics | [Extensions](extensions.md) |

An agent refuses what it cannot do when you build it, not halfway through a
run: `memory` on an agent without memory support, SAC on a discrete action
space, `action_repeat` on a gymnax environment, an extension it cannot run
(the [extension checks](extensions.md#what-is-checked-and-when) list them
all).

`print(agent)` shows the call that builds the agent, with every argument
that differs from its default. `agent.config` holds every constructor
argument by name, plus `algo_name`; each logged run stores it
([Logging](logging.md#the-run-configuration)).

### Presets

An agent's defaults are its reference code's general configuration. Where
the reference tunes some values for one task (a temperature for Humanoid, a
discount for Swimmer, a whole configuration for Atari), the agent keeps them
in a preset named after the task:

```python
from ajax import TQC

agent = TQC.preset("Hopper", "hopper")  # brax's Hopper
print(agent.top_quantiles_to_drop_per_net)  # 5, the authors' Hopper setting
print(agent.config["preset"])  # Hopper: the run config records it

# The arguments you give win over the preset's.
agent = TQC.preset("Hopper", "hopper", top_quantiles_to_drop_per_net=3)
```

- A preset is named after its task, not an environment id, so it applies to
  the Gymnasium, Brax and Playground versions of the task alike. An unknown
  name raises and lists the agent's presets.
- PPO also has one named after a library: `PPO.preset("SB3", env_id)` is
  Stable-Baselines3's PPO defaults in Ajax's names
  ([PPO, "Presets"](../algorithms/ppo.md#presets)).
- A preset sets every value its reference varies across tasks, even one
  that equals the default today, and the agent's fields only: the
  environment's own settings (Atari's frame skip or reward clipping) and a
  dataset's preprocessing stay yours.

Each agent module keeps its presets in a `PRESETS` table, each entry with
the values and where they come from:

```python
from ajax.agents.TQC.TQC import PRESETS

print(PRESETS["Hopper"].citation)  # bayesgroup/tqc_pytorch@65114eb README.md:42-48
```

| Agent | Presets | Fields a preset sets | Source |
| --- | --- | --- | --- |
| [ASAC](../algorithms/asac.md#presets) | `Ant`, `HalfCheetah`, `Hopper`, `Humanoid`, `InvertedDoublePendulum`, `InvertedPendulum`, `Pusher`, `Reacher`, `Swimmer`, `Walker2d` | `alpha_init`, `tau_theta`, `buffer_size`, `learning_starts` | the authors' tuned runs |
| [CrossQ](../algorithms/crossq.md#presets) | `Swimmer` | `gamma` | the authors' `train.py` |
| [DreamerV3](../algorithms/dreamerv3.md#presets) | `DMC` | `model_size`, `train_ratio`, `n_envs`, `action_repeat` | `configs.yaml` |
| [DQN](../algorithms/dqn.md#presets) | `Atari` | `learning_rate`, `gamma`, `buffer_size`, `batch_size`, `learning_starts`, `train_frequency`, `target_update_interval`, `epsilon_start`, `epsilon_end`, `epsilon_decay_start`, `epsilon_decay_frac` | Mnih et al.'s `run_gpu` |
| [IQL](../algorithms/iql.md#presets) | `AntMaze` | `expectile`, `temperature` | `antmaze_config.py` |
| [PPO](../algorithms/ppo.md#presets) | `Atari`, `SB3` | `n_envs`, `n_steps`, `batch_size`, `n_epochs`, `num_minibatches`, `actor_learning_rate`, `critic_learning_rate`, `gamma`, `gae_lambda`, `clip_range`, `clip_value_loss`, `normalize_advantage`, `ent_coef`, `vf_coef`, `max_grad_norm`, `adam_eps`, `normalize_observations`, `normalize_rewards`, `share_encoder` | ppo2's `defaults.py`; Stable-Baselines3's PPO defaults |
| [PQN](../algorithms/pqn.md#presets) | `CartPole`, `Atari` | `architecture`, `n_envs`, `n_steps`, `n_epochs`, `num_minibatches`, `learning_rate`, `learning_rate_decay`, `epsilon_end`, `epsilon_decay_frac`, `q_lambda`, `reward_scale` | purejaxql's configs |
| [REDQ](../algorithms/redq.md#presets) | `Hopper`, `HalfCheetah`, `Walker2d`, `Ant`, `Humanoid` | `target_entropy_per_dim` | MBPO's target entropies |
| [REINFORCE](../algorithms/reinforce.md#presets) | `rtg_pg` | `baseline`, `normalize_advantage`, `gamma`, `n_steps`, `actor_architecture`, `actor_learning_rate` | Spinning Up's `2_rtg_pg.py` |
| [TD3](../algorithms/td3.md#presets) | `D4RL` (TD3+BC) | `normalize_observations`, `extensions` | TD3_BC's `main.py` |
| [TDMPC2](../algorithms/tdmpc2.md#presets) | `MuJoCo` | `discount_max`, `rho` | the latest code's `envs/mujoco.py` |
| [TQC](../algorithms/tqc.md#presets) | `Hopper`, `HalfCheetah`, `Walker2d`, `Ant`, `Humanoid` | `top_quantiles_to_drop_per_net` | the authors' README |

Each agent's page gives the values.

## `train`

```python
from ajax import SAC

agent = SAC("Pendulum-v1", learning_starts=1_000)
state, out = agent.train(seed=[1, 2, 3], n_timesteps=2_000, logging_config=None)
```

| Argument | Default | What it does |
| --- | --- | --- |
| `seed` | `42` | one seed or a list of them: a run per seed, all in one program ([Seeds](#seeds)) |
| `n_timesteps` | `1_000_000` | the environment steps this call trains for ([The budget](#the-budget)) |
| `num_episode_test` | `10` | the episodes of each evaluation ([Evaluation](#evaluation)) |
| `logging_config` | not given | when to evaluate and where to log ([What `train` returns](#what-train-returns)) |
| `initial_state` | `None` | a state to continue from ([Continuing a run](#continuing-a-run)) |
| `total_timesteps` | `None` | the whole run's horizon, which every schedule runs to; `None` is this call's end |
| `sweep` | `None` | configurations to train in the same program ([Hyperparameter sweeps](#hyperparameter-sweeps)) |
| `devices` | `None` | devices to split the runs over ([Performance](performance.md#seeds-over-several-devices)) |
| `precision` | `None` | what the networks compute in: `"float32"`, `"bfloat16"` or `"auto"` ([Performance](performance.md#bfloat16)) |
| `on_ids_ready` | `None` | a function given the runs' ids before training starts ([Logging](logging.md#backends)) |

### The budget

`n_timesteps` counts environment steps, summed over the `n_envs` copies. An
agent trains in whole iterations:

| Agents | One iteration collects | Iterations in `n_timesteps` |
| --- | --- | --- |
| the on-policy agents (PPO, APO, REINFORCE, PQN) and UDRL | `n_steps` in each copy | `n_timesteps // (n_envs * n_steps)`, at least one |
| APG | `horizon` steps in each copy | `n_timesteps // (n_envs * horizon)`, at least one |
| the others (SAC, TD3, DQN, ...) | one step in each copy | `n_timesteps // n_envs` |

So 100,000 steps of SAC with `n_envs=4` are 25,000 iterations. A budget
that is not whole iterations stops at the last whole one, with a warning; a
budget below one iteration trains that one iteration and records no
evaluation.

An off-policy agent acts uniformly at random, without updating, until it
has collected `learning_starts` steps. A short test run must lower it, or
the agent never trains; a call that ends before it warns.

| `learning_starts` default | Agents |
| --- | --- |
| 25,000 | TD3 |
| 10,000 | SAC, ASAC, RVI-SAC, DQN |
| 5,000 | REDQ, CrossQ |
| 257 | TQC, whose policy acts from the first step |
| 0 | AVG, IQL |

### Seeds

`seed` is one integer or a list. Each seed starts from
`jax.random.PRNGKey(seed)`, and `train` runs all of them at once with
`jax.vmap`: one compilation, one program, every seed in step. More seeds
cost little extra time until the device is full
([Performance](performance.md)), so run the seeds you need in one call.

Everything `train` returns has the seed axis first, whatever the number of
seeds: with `seed=0`, `evaluations.returns` has shape `(1, evaluations)`,
and that run's curve is `evaluations.returns[0]`. A script then keeps
working when one seed becomes several. The last line `train` prints gives
the shape, and so does `print(evaluations)`. Seed `i`'s part of the state
is `jax.tree.map(lambda x: x[i], state)`.

The scores of [`ajax.compare`](comparing-agents.md) take that array as it
is, and so do libraries that expect a row per run, such as
[rliable](https://github.com/google-research/rliable).

A seed gives the same run on the same machine and library versions. Two
programs compiled apart, or on other hardware, may round differently.

### Hyperparameter sweeps

`sweep` trains several configurations of the agent in the same program,
each with every seed:

```python
state, evaluations = agent.train(
    seed=[1, 2],
    n_timesteps=2_000,
    sweep={"actor_learning_rate": [1e-4, 3e-4], "tau": [0.005, 0.02]},
)
agent.members[0]  # {'seed': 1, 'actor_learning_rate': 0.0001, 'tau': 0.005}

# Each configuration with its rows, one per seed: here returns is (2, evaluations).
for configuration, returns in agent.by_configuration(evaluations.returns):
    print(configuration, "final return", returns[:, -1].mean())
```

- **What trains.** A dict of value lists trains its grid, here 4
  configurations, so 8 members; a list of dicts trains those
  configurations. The returned arrays have a row per member, configuration
  by configuration, each over the seeds, as `agent.members` lists them.
- **What can vary.** The numbers the agent's training reads as numbers,
  such as SAC's learning rates, `gamma` and `tau`, or PPO's `ent_coef` and
  `clip_range`: `print(agent.sweepable_fields())` lists them, and a refused
  field's message gives the reason. A setting that changes array shapes or
  the program (the network sizes, `n_envs`, `batch_size`, `buffer_size`)
  takes one `train` call per value.
- **Memory.** Each member keeps its own replay buffer, so memory grows with
  the members.
- **Logs.** Without a `logging_config`, each evaluation prints a line for
  all the runs and one per configuration under it. With one, each member is
  a logged run of its own, with its values in its run config.
- **Resuming.** A sweep continues from its state with the same `seed` and
  `sweep`.

The sweep compiles once. Each member trains as its separate run would, up
to floating-point rounding, which grows over a run as it does between two
machines: on CrossQ, which has no target network to damp it, the two can
part visibly within a hundred updates. To tell whether one configuration
beats another, give each one's per-seed scores (`returns[:, -1]` above) to
the tests of [Comparing agents](comparing-agents.md).

An agent of your own lists the fields a sweep may vary in a `sweepable`
class attribute, inheriting its base classes', and refuses an inherited one
in `unsweepable`, with its reason. A field qualifies when a member changing
it trains as a separate run built with the value does
(`tests/agents/test_sweep_every_agent.py`).

### Parallel environments

`n_envs` copies of the environment step together inside each seed. They
collect more data per iteration, and each copy runs its own episodes. An
off-policy agent still makes one update per iteration (or the number its
own settings ask for), so at a fixed budget more copies mean fewer updates.
The defaults differ by agent:

| `n_envs` default | Agents |
| --- | --- |
| 1 | PPO, REINFORCE, SAC, ASAC, RVI-SAC, REDQ, CrossQ, TQC, TD3, IQL, AVG, DQN, UDRL, DAgger, TD-MPC2 |
| 8 | APG |
| 10 | APO |
| 16 | DreamerV3 |
| 128 | PQN |

### Evaluation

With a `logging_config`, the agent is evaluated every `log_frequency` steps
on `num_episode_test` episodes, run in parallel in a separate copy of the
environment. The policy acts deterministically: the mean action of a
continuous policy, the most likely one of a discrete policy. UDRL acts with
the command (desired return and horizon) it would collect with; DreamerV3
and TD-MPC2 act through their own latent policy or planner, and their pages
say what they log.

Each evaluation draws fresh initial states and noise, from a key of its own
derived from the number of evaluations so far. Two evaluations of a seed
see different episodes, the same seed always sees the same ones, and the
evaluations repeat none of the training's random draws.

### What `train` returns

`train` returns `(state, out)`. What `out` holds depends on
`logging_config`:

| `logging_config` | Evaluations | Printed | `out` |
| --- | --- | --- | --- |
| not given | about ten: every tenth of `n_timesteps`, rounded up to whole iterations | a line per evaluation: the mean return over the seeds, the lowest and highest; under them, on a terminal or in a notebook, a progress bar with the rate and the time left, else the time left at the end of each line from the second | the evaluations |
| `None` | none, not even compiled into the program: the fastest run | nothing | `None` |
| a `LoggingConfig` | every `log_frequency` steps, logged to its backends | one line at the start: how often it evaluates and where it logs | the evaluations |

The evaluations are a dictionary from each logged name
([Logging](logging.md#what-is-logged)) to an array with a row per seed and
a column per evaluation. Two are also attributes: `out.returns`, the mean
evaluation return, and `out.timesteps`, the step of each evaluation (one
row; `out["timestep"]` has one per seed). A call that evaluates fewer times
than its maximum pads the rest with -1 (the timestep) and NaN (everything
else). APG returns its per-update metrics instead
([APG](../algorithms/apg.md)).

`state` is the agent's whole state, each leaf with the seed axis first:

| Field | What it holds |
| --- | --- |
| `actor_state`, `critic_state` | the networks: their parameters and optimiser states. DQN's and PQN's Q-network is in `actor_state`, and their `critic_state` holds no parameters |
| `collector_state` | the environments, the replay buffer, the observation statistics and the step counter `timestep` |
| `rng`, `eval_rng` | the training and evaluation keys |
| `n_logs` | the number of evaluations so far |
| `ext_state` | each extension's state |

To see the networks an agent built, print the shapes of their parameters
(the first axis is the seeds'):

```python
import jax

from ajax import PPO

agent = PPO("CartPole-v1", actor_architecture=("32", "relu", "32", "relu"))
state, _ = agent.train(seed=0, n_timesteps=2_048, logging_config=None)
print(jax.tree.map(lambda array: array.shape, state.actor_state.params))
```

## Continuing a run

Pass a returned state back as `initial_state` to train further, with the
same seeds:

```python
from ajax import PPO

agent = PPO("CartPole-v1", n_envs=8, n_steps=128)
state, _ = agent.train(seed=[0, 1], n_timesteps=8_192)
state, _ = agent.train(seed=[0, 1], n_timesteps=8_192, initial_state=state)
print(state.collector_state.timestep)  # [16384 16384]
```

The second call continues from the first call's timestep. Its schedules
(the training fraction, the exploration, the extensions', the logging's
end) continue too, and run to the horizon `total_timesteps`. By default
that is the call's own end, so a schedule set as a fraction of training,
such as DQN's epsilon decay, has finished by the end of the first call. To
split a run and keep the schedules of one call, pass the whole run's
horizon to every call; a call that would train past it is refused.

```python
total = 16_384
state, _ = agent.train(seed=[0, 1], n_timesteps=8_192, total_timesteps=total)
state, _ = agent.train(
    seed=[0, 1], n_timesteps=8_192, initial_state=state, total_timesteps=total
)
print(state.collector_state.timestep)  # [16384 16384], as one call of 16_384
```

An on-policy agent trains whole rollouts, as its original code does
([The budget](#the-budget)): give each call a whole number of them (8,192
steps are 8 rollouts of 8 × 128 here), and the split run ends where one
call does.

!!! warning "Read the state before passing it back"
    The call may reuse the memory of the state you pass in. Read or save
    anything you need from it before the call.

To resume in another process, save the state first:
[Checkpoints and resuming](checkpoints.md).

## Training offline

A [replay agent](../algorithms/index.md#agent-families)
(<!-- agents: offline -->SAC, TD3, REDQ, TQC, CrossQ, ASAC, RVI-SAC, DQN, IQL,
DAgger<!-- /agents -->) can train on a
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
from ajax.logging import LoggingConfig


def pd_expert(obs):
    theta = jnp.arctan2(obs[..., 1], obs[..., 0])
    return jnp.clip(-(2.0 * theta + 0.5 * obs[..., 2]), -2.0, 2.0)[..., None]


agent = TD3("Pendulum-v1", learning_starts=0)  # online: update from the first step
data = record(pd_expert, agent.env_args, jax.random.PRNGKey(0), n_steps=1_000)
logging = LoggingConfig(log_frequency=500, backends=())
state, offline = agent.train_offline(data, n_updates=1_000, seed=[0], logging_config=logging)
print(offline["timestep"])  # [[ 500 1000]]: updates done
state, online = agent.train(seed=[0], n_timesteps=1_000, initial_state=state)
```

- **Recording a dataset.** `record` takes any function from observations to
  actions in the environment's units. A trained agent's `predict` is one,
  so `record(lambda obs: agent.predict(state, obs), agent.env_args, key,
  n_steps)` records its policy. `export(state, seed_index=0)`, from the
  same module, gives the whole episodes a replay agent's own replay holds.
- **The update.** `train_offline` runs the agent's own update, with its
  extensions' terms: [TD3+BC](../algorithms/td3.md#td3bc-offline) is TD3
  with a cloning term, and [IQL](../algorithms/iql.md) an agent of its own.
- **Offline updates before online training.** `PriorData(dataset,
  offline_updates=N)` runs the same updates inside the pre-training of an
  ordinary `train` call, without evaluations.

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

The step counts gradient updates, not environment steps: here 8 rollouts
of 10 epochs of 2 minibatches, 160 steps.
`ajax.schedule.warmup_cosine_schedule` builds the linear warm-up then
cosine decay of APG's in-context controller.

## The shared training loop

Most agents run on one loop, `TrainLoop` in
[`agents/loop.py`](../../src/ajax/agents/loop.py). An agent brings only its
algorithm, how to build a fresh state and how to make one update, and the
loop does the rest the same way for every agent:

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

Collection, the `learning_starts` gate, the extensions, evaluation, logging
and resuming then behave the same for every agent, and an extension written
for one works on the others. The world models plug their planner or latent
policy into the same loop, and UDRL its own iteration; APG and the offline
TDMPC2MultiTask reuse the loop's pieces (the fresh state, the extensions,
the evaluation) rather than the loop itself.

The whole loop is one `jax.lax.scan` inside one `jax.jit`, vmapped over the
seeds. Python runs once, to build the program; the CPU or GPU then runs the
entire training, handing only the logs back to the
[logging worker](logging.md#the-logging-worker).
[Architecture](../developer/architecture.md) describes the loop for
contributors.
