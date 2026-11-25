# REINFORCE

REINFORCE is the plain policy gradient. It runs the policy, then makes every
action it took more likely in proportion to how well things went after it.
It reuses nothing: each rollout gives one gradient step and is discarded.
That makes it slower to learn than [PPO](ppo.md), which takes several steps
on each rollout, but it is the simplest on-policy method and the one the
others build on. By default it subtracts a learned value from the returns
(a baseline), which lowers the variance of its gradient without biasing it.

**Paper:** Williams, *Simple statistical gradient-following algorithms for
connectionist reinforcement learning*, Machine Learning 8, 1992. Ajax
follows Spinning Up's Vanilla Policy Gradient (`openai/spinningup` at
`038665d`, `spinup/algos/pytorch/vpg/vpg.py`), which adds GAE to Williams's
learned baseline, and its plain reward-to-go example (`2_rtg_pg.py`) for
the version without a baseline. The returns are the shared on-policy code
(`agents/returns.py`); REINFORCE's own is in `agents/REINFORCE/REINFORCE.py`.

## Can I use?

| Feature | REINFORCE |
| --- | --- |
| Discrete actions | yes |
| Continuous actions | yes, a Gaussian, clipped and mapped onto the environment's bounds |
| Memory (`memory=`) | no |
| Environments | gymnax, Brax, MuJoCo Playground |
| Image observations | no |
| Several seeds in one call | yes, `train(seed=[...])` |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `observe` |
| Resume and checkpoints | yes |

## Example

```python
from ajax import REINFORCE
from ajax.logging import LoggingConfig

agent = REINFORCE("CartPole-v1")
logging = LoggingConfig(log_frequency=8_000, backends=())
state, evaluations = agent.train(seed=[1, 2], n_timesteps=16_000, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])

# Spinning Up's plain version: reward-to-go weights, no baseline
# (Presets, in Training).
plain = REINFORCE.preset("rtg_pg", "CartPole-v1")
```

This budget shows the calls; CartPole takes longer to learn.

## Key hyperparameters

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | 1 | environments stepped in parallel |
| `n_steps` | 4000 | steps per environment in each rollout; one policy step per rollout |
| `baseline` | True | subtract a learned value V(s) (GAE); False weights each step by its reward-to-go alone |
| `gamma` | 0.99 | discount |
| `gae_lambda` | 0.97 | GAE's lambda, with a baseline |
| `actor_learning_rate` | 3e-4 | the policy's Adam step size |
| `critic_learning_rate` | 1e-3 | the critic's Adam step size |
| `value_steps` | 80 | full-batch Adam steps of the critic per rollout |
| `normalize_advantage` | True | standardise the advantages over the rollout |
| `actor_architecture`, `critic_architecture` | `("64", "tanh", "64", "tanh")` | hidden layers |
| `log_std_init` | -0.5 | the continuous policy's initial log-std, one learned value per action dimension |
| `max_grad_norm` | None | clips each network's gradient to this global norm |
| `normalize_observations`, `normalize_rewards` | False | running normalisation |
| `extensions` | `()` | research features (see [Extensions](../guide/extensions.md)) |

`n_epochs` and `num_minibatches` are fixed at 1: a second pass over a
rollout would no longer follow the policy that collected it.

## Presets

`REINFORCE.preset("rtg_pg", env_id)` is Spinning Up's plain reward-to-go
policy gradient (`spinup/examples/pytorch/pg_math/2_rtg_pg.py:24-55`;
[Presets](../guide/training.md#presets)): no baseline and no normalisation,
undiscounted returns (`gamma=1`), rollouts of 5,000 steps, one hidden layer
of 32 `tanh` units, and Adam at `1e-2`. The script collects whole episodes;
Ajax skips the steps of the episode each rollout cuts (see the notes).

## How it works

Each update reads the rollout it was just given:

1. **Returns.** For every step, the discounted sum of the rewards from that
   step to the end of its episode (the reward-to-go). Where the episode was
   cut rather than ended (the rollout's last step, a time limit), the
   critic's value of the next state stands in for the rewards that are
   missing.
2. **Advantages.** With a baseline, GAE with `gae_lambda` on the critic's
   values; without, the reward-to-go itself. They are standardised over the
   rollout.
3. **Policy step.** One Adam step down `-mean(log pi(a_t | s_t) A_t)`.
4. **Critic steps.** With a baseline, `value_steps` Adam steps down the
   squared error between V(s_t) and the returns, all on the whole rollout.

Both steps read the advantages and returns computed before either moves.

## Notes

- **Episodes cut by the rollout, without a baseline.** Spinning Up's
  reward-to-go example collects whole episodes only. Ajax's rollout has a
  fixed length, so the last episode of each environment is usually cut.
  Without a critic nothing can stand in for its missing rewards, so its
  steps get no gradient. Make `n_steps` several episode lengths long, or a
  large share of each rollout is skipped (`policy/weighted_fraction` logs
  the share used).
- **Time limits.** With a baseline, a time limit bootstraps from the critic
  as Spinning Up's VPG does at its `max_ep_len`. Without one, it ends the
  return, as in Spinning Up's example.
- **Initialisation.** Every layer starts as torch's `nn.Linear`, kernel and
  bias uniform in `±1/sqrt(fan_in)`, as in the reference.
- **Optimiser.** Two Adams with torch's epsilon `1e-8`, no gradient clipping,
  as in the reference.
- **Differences from the reference.** The advantages are divided by their
  standard deviation plus `1e-8`, so a rollout of equal advantages stays
  finite; the reference divides by the bare deviation. The logged
  `policy/approx_kl` is the one policy step's; the reference logs it before
  the step, where it is 0. The reference runs 4 processes of 1,000 steps by
  default from its command line; `n_envs=4, n_steps=1000` reproduces that.
- **Against the reference.** On CartPole, over 200,000 steps and 96 seeds,
  Ajax's final returns are similar to Spinning Up's VPG (within ±25). On
  Pendulum neither learns at that budget; the comparison is inconclusive
  ([Results](../reproduction/results.md)).

## API

::: ajax.agents.REINFORCE.REINFORCE.REINFORCE
