# IQL

Implicit Q-Learning learns from a fixed dataset without ever asking its
critic about actions the data does not contain, which offline critics
overrate. A state value V(s) is fitted as an upper expectile of the critic
over the dataset's own actions, which approximates the best of them
without a max. The critic then regresses on `r + γ V(s')`, and the actor
copies the dataset's actions, each weighted by how much better than V it
was (advantage-weighted regression).

**Paper:** Kostrikov, Nair and Levine, *Offline Reinforcement Learning with
Implicit Q-Learning*, ICLR 2022
([arXiv:2110.06169](https://arxiv.org/abs/2110.06169)). The defaults and
the update order are the authors' JAX code's,
`ikostrikov/implicit_q_learning` at `09d7002`.

## When to use it

IQL suits learning from a dataset with no simulator to explore in, such as
logged runs or a plant's history; it can then fine-tune online with the
same update. For a minimal alternative, see [TD3+BC](td3.md#td3bc-offline):
TD3 with a cloning term.

| Feature | IQL |
| --- | --- |
| Discrete actions | no |
| Continuous actions | yes, a Gaussian on a `tanh` mean, its samples clipped to `[-1, 1]` and mapped onto the environment's bounds |
| Memory (`memory=`) | no |
| Environments | gymnax, Brax, MuJoCo Playground |
| Image observations | no |
| PID policy | yes, `actor_architecture=PIDPolicy(...)` ([Networks](../guide/networks.md)) |
| Several seeds in one call | yes, `train(seed=[...])` |
| Offline training | yes, `train_offline` ([Training](../guide/training.md#training-offline)), then online with `train(initial_state=...)` |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `observe`, `on_target`, `critic_loss`, `actor_loss`, `action`, `eval_action`, `transform_action`, `actuate`, `choose_warmup_action`, `choose_replay_action`, `track_memory` ([which extensions run](../guide/compatibility.md)) |
| Resume and checkpoints | yes; a run split by a checkpoint equals the run left whole |

## Example

```python
import jax
import jax.numpy as jnp

from ajax import IQL
from ajax.buffers.dataset import record
from ajax.logging import LoggingConfig


def pd_expert(obs):
    theta = jnp.arctan2(obs[..., 1], obs[..., 0])
    return jnp.clip(-(2.0 * theta + 0.5 * obs[..., 2]), -2.0, 2.0)[..., None]


agent = IQL("Pendulum-v1", buffer_size=10_000)
data = record(pd_expert, agent.env_args, jax.random.PRNGKey(0), n_steps=2_000)
logging = LoggingConfig(log_frequency=1_000, backends=())
state, offline = agent.train_offline(data, n_updates=3_000, seed=[0], logging_config=logging)
print(offline["Eval/episodic mean reward"])
state, online = agent.train(seed=[0], n_timesteps=1_000, initial_state=state)
```

Evaluations also log the value's, the actor's and the critics' losses
(`value/value_loss`, `value/actor_loss`, `value/critic_loss`), the mean
advantage `q - V(s)` (`value/advantage`) and weight (`value/weight`).

## Key hyperparameters

The defaults are the authors' locomotion configuration. The [API](#api)
section lists every argument.

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | `1` | parallel environments, online |
| `expectile` | `0.7` | the expectile V fits (0.5: the mean; towards 1: the best action in the data); 0.9 on AntMaze |
| `temperature` | `3.0` | the advantage weights' inverse temperature, `min(exp(temperature (q - V)), 100)`; 10 on AntMaze |
| `actor_learning_rate`, `critic_learning_rate`, `value_learning_rate` | `3e-4` | Adam step sizes (epsilon `1e-8`, optax's) |
| `actor_learning_rate_decay` | `"cosine"` | the actor's rate decays to 0 by a cosine over the run's updates; `None` keeps it constant ([Notes](#notes)) |
| `actor_architecture`, `critic_architecture` | `("256", "relu", "256", "relu")` | hidden layers, also the value's |
| `gamma` | `0.99` | discount |
| `tau` | `0.005` | Polyak rate of the target critics |
| `buffer_size` | `1_000_000` | replay capacity, which must hold the dataset |
| `batch_size` | `256` | rows per update |
| `learning_starts` | `0` | online: steps before the first update (the replay holds the dataset already) |
| `reward_scale` | `1.0` | multiplies the rewards in the critics' target |
| `max_grad_norm` | `None` | gradient clipping |
| `normalize_observations`, `normalize_rewards` | `False` | running normalisation ([Environments](../guide/environments.md)) |
| `episode_length`, `action_repeat` | `None`, `1` | simulator steps per episode and per agent step, on Brax and Playground; `None` is 1,000 steps, or a `mujoco:` task's own time limit ([Environments](../guide/environments.md#brax-and-playground)) |
| `expert_policy` | `None` | an expert the run tracks online, its actions stored for the extensions that read them (`ImitationLoss`, `IBRL`, `LCBGatedBootstrap`), beside the observation features and `ResidualPolicy`; each evaluation logs its return next to the agent's |
| `extensions` | `()` | research features ([Extensions](../guide/extensions.md)) |

## Presets

`IQL.preset("AntMaze", env_id)` takes the authors' AntMaze configuration,
`expectile=0.9` and `temperature=10` (`configs/antmaze_config.py:15-16`;
[Presets](../guide/training.md#presets)). The reference also preprocesses
each dataset's rewards, which a preset cannot do:

- on AntMaze it subtracts 1 from them (`train_offline.py:68-69`): shift
  your dataset's rewards before training;
- on the locomotion tasks it scales them by 1000 over the range of the
  dataset's episode returns (`:35-52`): pass
  `reward_scale=1000 / (max_return - min_return)`, computed from your
  dataset.

## How it works

Each update samples one replay batch and runs, in the reference's order:

1. **Value.** V(s) regresses on `q = min_i Q_target_i(s, a)` at the
   batch's own actions, by the expectile loss `|expectile − 1(q < V)| (q −
   V)²`.
2. **Actor.** It maximises `E[w log π(a|s)]` with `w = min(exp(temperature
   (q − V(s))), 100)`, with V the value just updated. The policy is a
   Gaussian on a `tanh` mean, with one learned log-std per action
   dimension, clipped to `[−5, 2]`. Online, its samples act clipped to
   `[−1, 1]` and the replay keeps them clipped, as the reference does. The
   evaluations act with its mean.
3. **Critics.** Two Q critics regress on `reward_scale · r + γ (1 −
   terminated) V(s')`, their squared errors summed.
4. **Targets.** The target critics move by Polyak averaging at rate `tau`.

## Differences from the reference

The update follows the reference's equations and order
([How it works](#how-it-works)); no conformance check runs it against the
reference's code. `normalize_rewards`, off by default, is Ajax's: it
divides the rewards by a running standard deviation that counts the
dataset's rows and each step collected, as on the other agents.

## Notes

- **Learning-rate schedule.** The actor's cosine decay is the reference's
  offline training's. A decayed rate carries into a
  `train(initial_state=...)` that fine-tunes the state: to fine-tune at a
  constant rate, as the reference's fine-tuning configuration does, build
  the agent with `actor_learning_rate_decay=None`.
- **The value's input.** With `normalize_observations=True`, V reads the
  normalised observations, as the actor and the critics do.
- **Initialisation.** As in the authors' code: orthogonal kernels with gain
  `sqrt(2)`.

## Results

The repository records no learning curves for IQL yet. A probe
(`tests/probing/test_offline.py`) trains it offline on logged one-step
rows: V must reach the expectile of their returns, and the actor the
advantage-weighted mean of their actions, with the weights clipped.
[Performance](../guide/performance.md) compares its speed with the other
agents'.

## API

::: ajax.agents.IQL.IQL.IQL
