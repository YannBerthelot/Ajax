# IQL

Implicit Q-Learning learns from a fixed dataset without ever asking its
critic about actions the data does not contain, which offline RL's critics
overrate. A state value V(s) is fitted as an upper expectile of the critic
over the dataset's own actions, which approximates the best of them without
a max. The critic then regresses on `r + γ V(s')`. The actor copies the
dataset's actions, each weighted by how much better than V it was
(advantage-weighted regression).

**Paper:** Kostrikov, Nair and Levine, *Offline Reinforcement Learning with
Implicit Q-Learning*, ICLR 2022
([arXiv:2110.06169](https://arxiv.org/abs/2110.06169)). The defaults and
the update order are the authors' JAX code, ikostrikov/implicit_q_learning
at `09d7002` (`learner.py`, `critic.py`, `actor.py`, `policy.py`,
`configs/`).

## Can I use?

| Feature | IQL |
| --- | --- |
| Continuous actions | yes; the actor's mean is tanh-bounded |
| Discrete actions | no |
| Offline training | yes, `train_offline(dataset, n_updates)` |
| Online fine-tuning | yes, `train(initial_state=...)` after it, the same update |
| Memory (`memory=`) | no |
| Environments | gymnax, Brax, MuJoCo Playground |
| Several seeds in one call | yes, `train(seed=[...])` |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `on_target`, `critic_loss`, `actor_loss` |

## Example

```python
import jax
import jax.numpy as jnp

from ajax import IQL
from ajax.buffers.dataset import record
from ajax.logging.wandb_logging import LoggingConfig


def pd_expert(obs):
    theta = jnp.arctan2(obs[..., 1], obs[..., 0])
    return jnp.clip(-(2.0 * theta + 0.5 * obs[..., 2]), -2.0, 2.0)[..., None]


agent = IQL("Pendulum-v1", buffer_size=10_000)
data = record(pd_expert, agent.env_args, jax.random.PRNGKey(0), n_steps=2_000)
logging = LoggingConfig(config={}, log_frequency=1_000, use_wandb=False)
state, offline = agent.train_offline(data, n_updates=3_000, seed=[0], logging_config=logging)
print(offline["Eval/episodic mean reward"])
state, online = agent.train(seed=[0], n_timesteps=1_000, initial_state=state)
```

Evaluations log, next to the return, the losses of the value, the actor
and the critics (`value/value_loss`, `value/actor_loss`,
`value/critic_loss`), the mean advantage `q - V(s)` on the batch
(`value/advantage`) and the mean advantage weight (`value/weight`).

## Key hyperparameters

| Argument | Default | What it does |
| --- | --- | --- |
| `expectile` | 0.7 | the expectile V fits (0.5: the mean; towards 1: the best action in the data). The reference uses 0.7 on locomotion, 0.9 on antmaze |
| `temperature` | 3.0 | the advantage weights' inverse temperature, `min(exp(temperature (q - V)), 100)`; 10 on antmaze |
| `gamma`, `tau` | 0.99, 0.005 | discount; Polyak rate of the target critics |
| `actor_learning_rate`, `critic_learning_rate`, `value_learning_rate` | 3e-4 | Adam step sizes (eps 1e-8, optax's) |
| `actor_architecture`, `critic_architecture` | `("256", "relu", "256", "relu")` | hidden layers, also the value's |
| `batch_size`, `buffer_size` | 256, 1,000,000 | replay batch; capacity, which must hold the dataset |
| `reward_scale` | 1.0 | multiplies the rewards in the critic's target |
| `learning_starts` | 0 | online: steps before the first update (the replay holds the dataset already) |
| `normalize_observations` | False | running observation normalisation of the actor and the critics |
| `expert_policy` | `None` | an expert the collector tracks (online): its actions feed the observation features and a residual policy (`ResidualPolicy`), and the evaluation reports its return next to the agent's |
| `extensions` | `()` | research features (see [Extensions](../guide/extensions.md)) |

## How it works

Each update samples one replay batch and runs, in the reference's order:

1. **Value.** V(s) regresses on `q = min_i Q_target_i(s, a)` at the
   batch's own actions, by the expectile loss `|expectile − 1(q < V)| (q −
   V)²`.
2. **Actor.** It maximises `E[w log π(a|s)]` with `w = min(exp(temperature
   (q − V(s))), 100)`, with V the value just updated. The policy is a
   Gaussian on a tanh mean, with one learned log-std per action dimension,
   clipped to [−5, 2]. The evaluations act with its mean.
3. **Critics.** Two Q critics regress on `reward_scale · r + γ (1 −
   terminated) V(s')`.
4. **Targets.** The target critics move by Polyak averaging at rate `tau`.

## Notes

- **Learning-rate schedule.** For offline training the reference decays the
  actor's learning rate to 0 by a cosine over its updates. For that, pass
  `actor_learning_rate=optax.cosine_decay_schedule(3e-4, n_updates)`.
  Fine-tuning runs keep it constant, as here by default.
- **Rewards.** The reference scales a locomotion dataset's rewards by 1000
  over its range of episode returns, and subtracts 1 from antmaze's. Compute
  the factor from your dataset and pass it as `reward_scale` (or shift the
  dataset's rewards).
- **Critics' loss.** Their mean squared error over both critics, half the
  reference's sum. Adam's step is the same but for its eps, whose weight
  against the gradient doubles.
- **The value's input.** V reads the raw observation. With
  `normalize_observations=True` only the actor and the critics normalise
  theirs.

## Results

No learning curves are published for IQL yet. Unit tests
(`tests/agents/IQL`) check the expectile loss and the reference's defaults.
The probe (`tests/probing/test_offline.py`) trains IQL offline on logged
one-step rows: V must reach the expectile of their returns, and the actor
the advantage-weighted mean of their actions, with the weights clipped.

## API

::: ajax.agents.IQL.IQL.IQL
