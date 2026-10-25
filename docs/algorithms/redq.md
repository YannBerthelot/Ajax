# REDQ

Randomized Ensembled Double Q-learning is SAC with many critics and many
critic updates per environment step. Each critic target takes the minimum
over a random subset of the target critics, which keeps over-estimation in
check even at a high update-to-data ratio, and the actor maximises the
ensemble's mean value. Each environment step costs far more compute than
SAC's, in exchange for many more critic updates per step.

**Paper.** Chen et al., *Randomized Ensembled Double Q-Learning*, 2021
([arXiv:2101.05982](https://arxiv.org/abs/2101.05982)). The update follows its
Algorithm 1; where the authors' code (`watchernyu/REDQ` at `7b5d1bf`) is more
specific, Ajax follows the code (see [Notes](#notes)).

## Can I use?

| | REDQ |
| --- | --- |
| Continuous actions (`Box`) | Yes |
| Discrete actions | No |
| Learning | Off-policy, replay buffer |
| Vectorised environments (`n_envs`) | Yes |
| Recurrent memory (`memory=`) | Yes, without extensions or `expert_policy` |
| PID policy (`actor_architecture=PIDPolicy(...)`) | Yes |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `observe`, `on_target`, `critic_loss`, `actor_loss`, `transform_action`, `actuate`, `choose_warmup_action` |

## Example

```python
# doc: skip (20 critic steps of 10 critics per update: minutes on a CPU)
from ajax import REDQ
from ajax.logging.wandb_logging import LoggingConfig

# 10 critics, targets from the min over 2 of them, 20 critic steps per update.
agent = REDQ("Pendulum-v1", n_envs=4, learning_starts=1_000)
logging = LoggingConfig(config={}, log_frequency=2_000, use_wandb=False)
state, evaluations = agent.train(seed=[0, 1], n_timesteps=4_000, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])  # one row per seed
```

An SVGD-style variant, which pushes the critics apart in function space, is
the `KernelRepulsion` extension, an Ajax research feature:

```python
from ajax import REDQ
from ajax.extensions.ensemble import KernelRepulsion

agent = REDQ("Pendulum-v1", extensions=(KernelRepulsion(coef=0.1),))
```

## Key hyperparameters

Defaults are read from `REDQ.__init__`.

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | `1` | parallel environments |
| `actor_learning_rate`, `critic_learning_rate`, `alpha_learning_rate` | `3e-4` | Adam step sizes (Adam's epsilon is torch's 1e-8) |
| `actor_architecture`, `critic_architecture` | `("256", "relu", "256", "relu")` | hidden layers (plain MLPs) |
| `gamma` | `0.99` | discount |
| `buffer_size` | `1_000_000` | replay capacity |
| `batch_size` | `256` | replay minibatch |
| `learning_starts` | `10_000` | steps of uniform random actions before the first update |
| `tau` | `0.005` | Polyak rate of the target critics |
| `num_critics` | `10` | critics in the ensemble |
| `subset_size` | `2` | target critics the target takes the minimum over |
| `num_critic_updates` | `20` | critic steps per update, each on a fresh minibatch |
| `alpha_init` | `1.0` | initial entropy weight, as in the authors' code |
| `target_entropy_per_dim` | `-1.0` | target entropy, times the action dimension; `"mbpo"` for MBPO's per-task value |
| `reward_scale` | `1.0` | multiplies the rewards in the target |
| `max_grad_norm` | `None` | gradient clipping |
| `normalize_observations`, `normalize_rewards` | `False` | running normalisation |
| `reward_normalization_gamma` | `None` | with `normalize_rewards`, divide by the spread of the discounted return |
| `memory`, `burn_in`, `sequence_length`, `stored_state` | `None`, `8`, `16`, `False` | recurrent memory (see [Memory](../guide/memory.md)) |
| `expert_policy` | `None` | an expert whose return the evaluation reports next to the agent's (`Eval/expert bias`) |
| `episode_length`, `action_repeat` | `None`, `1` | simulator steps per episode (1,000 when `None`) and per agent step, on Brax and Playground |
| `extensions` | `()` | research features (see [Extensions](../guide/extensions.md)) |

Cloning an expert into the actor before training is the `CloningPretrain`
extension; a cloning term during training is `ImitationLoss`
([Expert guidance](../guide/expert-guidance.md)).

## Notes

- **One update.** `num_critic_updates` critic steps, each on a fresh replay
  minibatch and each followed by a target update; then one actor step and
  one temperature step on the last minibatch.
- **Following the authors' code.** Four choices come from `watchernyu/REDQ`
  rather than the paper's Algorithm 1. The actor step reads the critics as
  they were before the last critic step, since the code computes the actor
  loss before that step's optimizer update; the saved state keeps the
  updated critics. The critic loss is the sum over the critics of each one's
  mean squared error. `alpha_init=1.0` matches the code's `log_alpha`
  starting at 0; the code then uses `alpha=0.2` until its first temperature
  step (its constructor sets the temperature twice), and so does Ajax: the
  first update's critic targets and actor loss read 0.2.
- **Initialisation.** As the authors' code: every layer of the actor and
  the critics, the mean and log-std layers included, starts Xavier-uniform
  with a zero bias, each critic of the ensemble drawn on its own. With
  `memory=`, REDQ keeps its previous initialisation until Ni et al.'s is
  audited.
- **Target entropy.** The default is `-1` per action dimension.
  `target_entropy_per_dim="mbpo"` takes MBPO's per-task value instead, the
  authors' default, for the Brax ids `hopper` (-1), `halfcheetah` (-3),
  `walker2d` (-3), `ant` (-4) and `humanoid` (-2); any other environment
  raises.
- **Update-to-data ratio.** Each iteration steps every env once, then makes
  one update, so the ratio of critic steps to environment steps is
  `num_critic_updates / n_envs`: 20 with the defaults.
- **Cost.** On the CPU speed benchmark below, REDQ executed 47 steps per
  second, against SAC's 2,218 on the same preset.
- **Actions and evaluation.** As SAC: a `tanh`-squashed Gaussian in
  `[-1, 1]`, mapped onto the environment's bounds; evaluation uses the mean
  action. Its log-probability subtracts `log(1 - tanh(u)^2 + 1e-6)`, as the
  authors' code (`core.py:232`). The critics read actions in the environment's units, as the
  authors' code's do (`act_limit * tanh`, `core.py:238`): on a box other
  than `[-1, 1]` Adam then steps their action inputs as the authors' does.
- **Time limits.** A transition cut by a time limit bootstraps on its final
  observation; only a termination stops the bootstrap.
- **Diagnostics.** REDQ logs how spread out its ensemble is
  (`ensemble_q_std`, `mean_pairwise_q_dist`), computed without gradient.
- **Shared maths.** REDQ imports SAC's maths (`ajax.agents.SAC.core`) instead
  of copying it.

## Results

The repository records no learning curves for REDQ yet. A conformance
check (`tests/conformance/test_redq_original.py`) runs one training step of
Ajax's REDQ and of watchernyu/REDQ from the same state, with the same
batches, subsets and noises, and compares every critic, target, the actor,
the temperature and their optimisers, the first step (temperature 0.2)
included. Unit tests
(`tests/agents/REDQ` and the contracts every agent shares) check its pieces
against exact contracts, without learning: the summed critic loss, the
random subset, the critics the actor reads and the target entropy. The
probes (`tests/probing`) train it briefly on tiny environments with known
answers: its targets, discounting, time limits, memory, extension phases,
and a run split by a checkpoint matching the run left whole. On the CPU
speed benchmark (`benchmarks/agent_bench.py`: Pendulum, 4 envs, 8,000 steps,
updates from step 1,000) REDQ ran at 47 steps per second at main `a8a13c2`
(see
[PERFORMANCE_REPORT.md](https://github.com/YannBerthelot/Ajax/blob/main/PERFORMANCE_REPORT.md)).

## API

::: ajax.agents.REDQ.REDQ.REDQ
