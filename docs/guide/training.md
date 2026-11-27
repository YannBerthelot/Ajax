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

### Presets

An agent's defaults are its reference code's general configuration. Where
the reference tunes some values for one task (a temperature for Humanoid, a
discount for Swimmer, a whole configuration for Atari), the agent keeps them
in a preset, named after the task:

```python
from ajax import TQC

agent = TQC.preset("Hopper", "hopper")  # brax's Hopper
print(agent.top_quantiles_to_drop_per_net)  # 5, the authors' Hopper setting
print(agent.config["preset"])  # Hopper: the run config records it

# The arguments you give win over the preset's.
agent = TQC.preset("Hopper", "hopper", top_quantiles_to_drop_per_net=3)
```

A preset is named after its task, not an environment id, so it applies to
the gymnasium, Brax and Playground versions of the task alike. PPO also has
one named after a library: `PPO.preset("SB3", env_id)` is Stable-Baselines3's
PPO defaults in Ajax's names, for settings brought from SB3 or SBX
([PPO, "Presets"](../algorithms/ppo.md#presets)). An unknown
name raises and lists the agent's presets. Each agent module keeps its
presets in a `PRESETS` table, each entry the values and where they come
from:

```python
from ajax.agents.TQC.TQC import PRESETS

print(PRESETS["Hopper"].citation)  # bayesgroup/tqc_pytorch@65114eb README.md:42-48
```

A preset sets every value its reference varies across tasks, even one that
equals the default today. It sets the agent's fields only: the environment's
own settings (Atari's frame skip or reward clipping) and a dataset's
preprocessing stay yours.

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

Each agent's page gives the values. A test compares this table with the
`PRESETS` tables (`tests/test_docs_agent_sets.py`).

## `train`

```python
from ajax import SAC

agent = SAC("Pendulum-v1", learning_starts=1_000)
state, out = agent.train(
    seed=[1, 2, 3],  # one run per seed, all in one program
    n_timesteps=2_000,  # environment steps this call trains for
    num_episode_test=10,  # test episodes per evaluation
    logging_config=None,  # None: no evaluation; not given: about ten, printed
    initial_state=None,  # or a state to continue from
    total_timesteps=None,  # the run's horizon; None: this call's end
)
```

### The budget

`n_timesteps` counts environment steps summed over the `n_envs` copies. An
off-policy agent collects one step in each copy per iteration, so 100,000
steps with `n_envs=4` are 25,000 iterations. An on-policy agent (PPO, APO,
REINFORCE, PQN) collects a rollout of `n_steps` per copy per iteration and trains in
whole rollouts: it runs `n_timesteps // (n_envs * n_steps)` of them, and at
least one. A budget that is not whole rollouts stops at the last whole one,
with a warning; a budget below one rollout runs that one rollout and
records no evaluation.

The off-policy agents act uniformly at random and do not update until
`learning_starts` steps are collected: 10,000 by default for SAC, ASAC,
RVI-SAC and DQN, 25,000 for TD3, 5,000 for REDQ and CrossQ, 257 for TQC
(whose policy acts from the first step), 0 for AVG. A short test run must
lower it, or the agent never trains; a call that ends before it warns.

### Seeds

`seed` is one integer or a list. Each seed starts from `jax.random.PRNGKey(seed)`,
and `train` runs all of them at once with `jax.vmap`: one compilation, one
program, every seed in step. More seeds cost little extra time until the
device is full ([Performance](performance.md) has the measurements), so run
the seeds you need in one call rather than one call per seed.

Everything `train` returns has the seed axis first, whatever the number of
seeds: with `seed=0`, `evaluations.returns` has shape `(1, evaluations)`, and
that run's curve is `evaluations.returns[0]`. The shape does not depend on
how the seed is written, so a script keeps working when one seed becomes
several. The last line `train` prints gives the shape, and so does
`print(evaluations)`. Seed `i`'s part of the state is
`jax.tree.map(lambda x: x[i], state)`.

To summarise several seeds, the scores of
[`ajax.compare`](comparing-agents.md) take that array as it is
(`final_return`, `last_third`, `auc`), and so do libraries that expect one
row per run, such as [rliable](https://github.com/google-research/rliable)
for interquartile means and their intervals.

The same seed on the same machine and library versions gives the same run.
Two programs compiled apart, or on different hardware, may round
differently, so do not expect identical numbers across machines.

### Hyperparameter sweeps

`sweep` trains several configurations of the agent in the same program, each
with every seed:

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

To decide whether one configuration is better than another, give each one's
per-seed scores to the tests of [Comparing agents](comparing-agents.md):
`returns[:, -1]` above is one score per seed.

A dict of value lists trains its grid, here 4 configurations, so 8 members;
a list of dicts trains those configurations. The returned arrays have one
leading row per member, configuration by configuration, each over the
seeds, as `agent.members` lists them. Each member trains as a separate run
of its configuration with its seed would, up to floating-point rounding, and
the sweep compiles once. Rounding grows over a run as it does between two
machines: on CrossQ, which has no target network to damp it, a member and
its separate run can part visibly within a hundred updates, while the
member stays equal to a sweep of the agent built with its values.

Only the agent's sweepable fields can vary (`agent.sweepable_fields()`):
numbers its training reads as numbers, such as SAC's learning rates, `gamma`
and `tau`, or PPO's `ent_coef` and `clip_range`. Every agent on the shared
training loop declares some; `print(agent.sweepable_fields())`
lists them, and a refused field's message gives the agent's reason where it
has one. A setting that changes array shapes or the program, such as the
network sizes, `n_envs`, `batch_size` or `buffer_size`, takes one `train`
call per value. Each member keeps its own replay buffer, so memory grows with
the members.

Without a `logging_config`, each evaluation prints a line for all the runs
and one per configuration under it. With one, each member is a logged run
of its own, with its values in its run config. A sweep continues from its state with the same
`seed` and `sweep`.

An agent of your own declares the fields it adds in a `sweepable` class
attribute, and takes those of its base classes with them (the temperature's
come with `EntropyTemperature`); `unsweepable` refuses an inherited one, with
the reason its message gives. A field belongs there once a member changing it
trains as a separate run built with the value does, which
`tests/agents/test_sweep_every_agent.py` checks for every declared field.

### Parallel environments

`n_envs` copies of the environment step together inside each seed. They
collect more data per iteration, and each copy starts its own episodes. For
an off-policy agent each iteration still makes one update (or the number its
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

- **With no `logging_config` given**, the run evaluates about ten times
  (every tenth of `n_timesteps`, rounded up to a whole number of the agent's
  iterations, such as PPO's rollouts), prints a line for each (the mean return over the seeds, with the lowest
  and highest; from the second, the time left) and `out` holds the evaluations.
- **With `logging_config=None`**, `out` is `None` and nothing is printed.
  The evaluation is not even compiled into the program: the fastest run.
- **With a `LoggingConfig`**, the run evaluates every `log_frequency` steps,
  logs to its backends, prints one line when it starts (how often it
  evaluates and where it logs), and `out` holds the evaluations.

The evaluations are a dictionary from each logged name to an array of shape
`(seeds, evaluations)`, with two entries by name: `out.returns` (the mean
evaluation return) and `out.timesteps` (when each evaluation ran, also
`out["timestep"]` per seed). A call that evaluates fewer times than its
  maximum pads the rest with -1 (the timestep) and NaN (everything else).
- **APG** returns its per-update metrics instead. See [APG](../algorithms/apg.md).

The logged names are listed in [Logging](logging.md).

`state` is the agent's whole state: `actor_state` and `critic_state` (the
networks, their optimisers and parameters), `collector_state` (the
environments, the replay buffer, the step counter `timestep`), the random
keys (`rng` for training, `eval_rng` for evaluation), the evaluation count
`n_logs`, and each extension's state. An agent with one network keeps it
in `actor_state`: DQN's and PQN's Q-network is there, and their
`critic_state` holds no parameters. To see the networks an agent built,
print the shapes of their parameters (the first axis is the seeds'):

```python
import jax

from ajax import PPO

agent = PPO("CartPole-v1", actor_architecture=("32", "relu", "32", "relu"))
state, _ = agent.train(seed=0, n_timesteps=2_048, logging_config=None)
print(jax.tree.map(lambda array: array.shape, state.actor_state.params))
```

## Continuing a run

Pass a returned state back as `initial_state` to train further:

```python
from ajax import PPO

agent = PPO("CartPole-v1", n_envs=8, n_steps=128)
state, _ = agent.train(seed=[0, 1], n_timesteps=8_192)
state, _ = agent.train(seed=[0, 1], n_timesteps=8_192, initial_state=state)
print(state.collector_state.timestep)  # [16384 16384]
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
agent (PPO, APO, REINFORCE, PQN) trains in whole rollouts of `n_envs x n_steps` steps,
as its original code does: give each call a whole number of them (8,192 is
8 rollouts of 1,024 here) and the calls end where one call does. A budget
that is not whole rollouts stops at the last whole one, with a warning; a
budget under one rollout trains one.

```python
total = 16_384
state, _ = agent.train(seed=[0, 1], n_timesteps=8_192, total_timesteps=total)
state, _ = agent.train(
    seed=[0, 1], n_timesteps=8_192, initial_state=state, total_timesteps=total
)
print(state.collector_state.timestep)  # [16384 16384], as one call of 16_384
```

To resume in another process, save the state first:
[Checkpoints and resuming](checkpoints.md).

The call may reuse the memory of the state you pass in. Read or save
anything you need from it before the call.

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

`record` takes any function from observations to actions in the
environment's units. A trained agent's `predict` is one, so
`record(lambda obs: agent.predict(state, obs), agent.env_args, key, n_steps)`
records its policy. `export(state, seed_index=0)`, from the same module,
gives the whole episodes a replay agent's own replay holds.

`train_offline` runs the agent's own update, with its extensions' terms:
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
