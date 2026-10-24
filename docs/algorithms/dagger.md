# DAgger

DAgger learns to imitate an expert. It needs no reward and no critic. The
agent lets the expert act at first, then acts itself. Whoever acts, every
state it reaches is labelled with the action the expert would take there,
and the policy is fitted to all the labels gathered so far. Copying only the
expert's own trajectories (behaviour cloning) never shows the policy the
states its own mistakes lead to. DAgger collects exactly those.

**Paper:** Ross, Gordon and Bagnell, *A Reduction of Imitation Learning and
Structured Prediction to No-Regret Online Learning*, AISTATS 2011
([arXiv:1011.0686](https://arxiv.org/abs/1011.0686)), Algorithm 3.1. The
authors released no code. The defaults follow the paper and
HumanCompatibleAI/imitation at `e5ef188`
(`src/imitation/algorithms/dagger.py` and `bc.py`).

## Can I use?

| Feature | DAgger |
| --- | --- |
| Discrete actions | yes; the expert returns its action index as an integer `(n, 1)` column |
| Continuous actions | yes, tanh-bounded, mapped onto the environment's bounds |
| Stateful experts | yes; their state is tracked along each trajectory |
| Memory (`memory=`) | no |
| Environments | gymnax, Brax, MuJoCo Playground |
| Several seeds in one call | yes, `train(seed=[...])` |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `observe`, `actor_loss` |

## Example

```python
import jax.numpy as jnp

from ajax import DAgger
from ajax.logging.wandb_logging import LoggingConfig


def pd_expert(obs):
    """A PD controller on Pendulum's raw observation [cos θ, sin θ, θ̇]."""
    theta = jnp.arctan2(obs[..., 1], obs[..., 0])
    return jnp.clip(-(2.0 * theta + 0.5 * obs[..., 2]), -2.0, 2.0)[..., None]


agent = DAgger("Pendulum-v1", expert_policy=pd_expert, n_envs=2)
logging = LoggingConfig(config={}, log_frequency=1_000, use_wandb=False)
state, evaluations = agent.train(seed=[1, 2], n_timesteps=2_000, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])         # the policy's return
print(evaluations["Eval/episodic mean expert reward"])  # the expert's, same episodes
```

Evaluations log the policy's return, the expert's on the same episodes
(`Eval/episodic mean expert reward`), their difference (`Eval/expert bias`),
the fitting loss (`policy/imitation_loss`), the iteration's mixing
probability (`dagger/beta`) and the share of environments the expert drove
at the last step (`dagger/expert_frac`).

## Key hyperparameters

| Argument | Default | What it does |
| --- | --- | --- |
| `expert_policy` | required | the expert to imitate, as in [Expert guidance](../guide/expert-guidance.md#the-expert) |
| `beta_decay` | 0.0 | the expert acts with probability `beta_i = beta_decay ** (i - 1)` in iteration `i`; 0 lets it act in the first iteration only |
| `iteration_steps` | 500 | timesteps (over all environments) per iteration |
| `n_envs` | 1 | environments stepped in parallel |
| `buffer_size` | 1,000,000 | labelled states kept |
| `batch_size` | 32 | labelled states per gradient step |
| `actor_learning_rate` | 1e-3 | Adam step size (eps 1e-8, torch's) |
| `actor_architecture` | `("32", "tanh", "32", "tanh")` | hidden layers |
| `max_grad_norm` | None | gradient clipping by global norm |
| `normalize_observations` | False | running observation normalisation |
| `episode_length`, `action_repeat` | None, 1 | simulator steps per episode and per agent step, on Brax and Playground |
| `extensions` | `()` | research features (see [Extensions](../guide/extensions.md)) |

## How it works

At each step, each environment executes the expert's action with probability
`beta_i`, and the policy's deterministic action otherwise: per environment
and per step, as the reference's collector mixes them. The replay keeps
every visited state with the expert's action there, whoever acted. It is the
paper's aggregated dataset D. Once the replay holds a batch, each step makes
one gradient step on a replay batch. The loss is the squared error between
the policy's action and the expert's (both in `[-1, 1]`), or the
cross-entropy of the expert's action for discrete actions.

`beta_decay=0` is the paper's parameter-free choice, `beta_i = I(i = 1)`.
The paper reports that it "often performs best in practice" (Sec. 3). Any
value in (0, 1] gives its exponential schedule. The guarantee needs only
that the mean of the `beta_i` goes to 0.

## DAgger and `ImitationLoss`

A replay agent (SAC, TD3, REDQ, ASAC) given an `expert_policy` and
`ImitationLoss` with the same expert stores the expert's action at every
state it visits and pulls its actor towards those actions on replay batches.
That is DAgger's aggregation with `beta_i = I(i = 1)`, plus the agent's
reinforcement-learning loss, its critic and its exploration noise. Use
DAgger to imitate without a reward, or as the imitation baseline. Use
`ImitationLoss` to improve on the expert. On PPO, APO and AVG the term reads
only the latest rollout or step, so there is no aggregated dataset.

## Notes

- **Online fit.** The paper refits the policy on D after each iteration
  (Follow-the-Leader). Ajax takes one gradient step per collection step, an
  online learner. The paper's analysis holds for any no-regret online
  learner (Sec. 4).
- **The last policy.** The paper returns the best policy of the iterations,
  chosen on validation. Ajax returns the last one; the evaluations show each.
- **Bounded memory.** The replay drops its oldest rows past `buffer_size`.
  D keeps them all.
- **Deterministic policy, squared error.** The reference's behaviour cloning
  fits a Gaussian policy by likelihood, with an entropy bonus. Ajax's policy
  is deterministic, as the paper's is, so it fits by squared error.
- **Rewards are not used.** They are recorded, and the evaluations measure
  them, but the policy learns from the expert's actions alone. The critic
  built beside the actor is unused.
- **The expert's next action.** The replay stores the expert's action at each
  state only. Its action at the next state (`next_a_expert`), which only
  critic modules read, stays zero.

## Results

No learning curves are published for DAgger yet. Unit tests
(`tests/agents/DAgger`) check the `beta` schedule, the mixing and the labels
on every stored row. The probes (`tests/probing/test_imitation.py`) train it
on a four-state chain where one state is reached only by the learner's
mistakes. The policy must take the expert's action there, which cloning the
expert's own states never teaches. A discrete twin checks the cross-entropy
fit.

## API

::: ajax.agents.DAgger.DAgger.DAgger
