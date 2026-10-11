# Expert guidance

Often a controller already exists for the task: a tuned PID, a model
predictive controller, a hand-written rule, an earlier policy. Starting
reinforcement learning from scratch throws that knowledge away. Expert
guidance lets an agent learn from such an *expert*: copy it at the start,
let it act while the agent is still poor, use its values in the critic's
target, and then improve on it.

In Ajax, an expert is any function you pass to an agent. SAC uses it the
most: it fills the replay buffer with the expert's experience, can clone it
before training, and lets the expert [extensions](extensions.md) shape
collection, the critic and the actor. TD3, REDQ, ASAC and APO can imitate it
through an extra actor-loss term.

## The expert

An expert is a callable that maps observations to actions. Ajax accepts two
forms:

- **Stateless:** `expert(obs) -> action`.
- **Stateful**, for a controller with memory such as a PID's integral: an
  object with `init_state(n_envs)`, which returns the zero state for `n_envs`
  environments, and `expert(state, obs) -> (action, new_state)`. Ajax keeps
  the state during collection and resets it at each episode's end.
  TargetGym's PID experts (`target_gym.experts.pid.FunctionalExpertPolicy`)
  follow this form.

Four conventions matter:

1. **The expert reads raw environment units.** When it acts (prefill,
   cloning, warm-up, collection, evaluation), it receives the environment's
   own observation, before any observation normalisation. An expert is
   designed in the plant's units, so `normalize_observations=True` or SAC's
   `normalize_obs_running` change what the networks see, not what the
   expert sees. One exception is listed below.
2. **Its action is executed as is.** Return actions in the range the agent
   itself acts in: SAC's actions lie in `[-1, 1]`.
3. **It works on batches.** Ajax calls it on one observation per environment,
   and on replay batches. Index the last axis (`obs[..., 0]`), not the
   first, and keep the action's last axis (`[..., None]` for one action).
4. **It is hashable.** The expert is part of the compiled program: a plain
   function or a frozen dataclass works.

Extensions that query the expert on replayed observations (`ImitationLoss`,
`IBRL`, `LCBGatedBootstrap`, `CriticBlend`, `PhiRefresh`) call it as
`expert(obs)`: they need the stateless form. `ImitationLoss` passes the raw
observation. The other four pass the replayed *next* observation as the
networks see it: with observation normalisation on, the expert does not get
raw units there, so do not combine them with it.

## Which agents take an expert

| Agent | `expert_policy=` | Expert extensions |
| --- | --- | --- |
| [SAC](../algorithms/sac.md) | replay prefill, warm-up actions, cloning before training, the expert's evaluation return | all but `ImitationLoss` |
| [TD3](../algorithms/td3.md), [REDQ](../algorithms/redq.md), [APO](../algorithms/apo.md) | cloning before training, the expert's evaluation return | `ImitationLoss` |
| [ASAC](../algorithms/asac.md) | not accepted | `ImitationLoss` |
| [AVG](../algorithms/avg.md) | the expert's evaluation return only | none |

With an expert, evaluations also log `Eval/episodic mean expert reward`, the
expert's own evaluation return, and `Eval/expert bias`, the agent's return
minus the expert's (positive once the agent does better). SAC
takes a separate `eval_expert_policy` when the expert to compare against is
not the one to learn from.

None of this works with `memory=` yet: recurrent SAC, TD3 and REDQ refuse an
expert, cloning and extensions.

## Example

SAC on Pendulum with a PD controller that is good near the top and poor
elsewhere. Three extensions build on it: `MCPretrain` fits a critic to the
expert's returns, `OnlineBC` pulls the actor towards the expert early on, and
`EDGEExploration` lets the expert act while the critic rates it higher than
the policy.

```python
import jax.numpy as jnp
from ajax import SAC
from ajax.extensions.expert import OnlineBC
from ajax.extensions.exploration import EDGEExploration
from ajax.extensions.pretrain import MCPretrain
from ajax.logging.wandb_logging import LoggingConfig


def pd_expert(obs):
    """A PD controller on Pendulum's raw observation [cos θ, sin θ, θ̇]."""
    theta = jnp.arctan2(obs[..., 1], obs[..., 0])
    torque = -(2.0 * theta + 0.5 * obs[..., 2])
    return jnp.clip(torque, -1.0, 1.0)[..., None]


agent = SAC(
    "Pendulum-v1",
    expert_policy=pd_expert,
    expert_buffer_n_steps=1_000,  # expert steps written to the replay before training
    learning_starts=1_000,
    extensions=(
        # The frozen expert critic, fitted to ~1,000 steps of the expert's returns.
        MCPretrain(pd_expert, n_mc_steps=100, n_mc_episodes=10, n_steps=200),
        OnlineBC(bc_coef=1.0, critic_warmup_frac=0.2),
        EDGEExploration(gate="argmax", decay_frac=0.3),
    ),
)
logging = LoggingConfig(config={}, log_frequency=1_000, use_wandb=False)
state, evaluations = agent.train(seed=0, n_timesteps=2_000, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])
print(evaluations["Eval/expert bias"])       # agent minus expert
print(evaluations["edge/live_expert_frac"])  # share of envs the expert drove
```

At this budget the agent has barely started; the example shows the wiring.
Give `MCPretrain` a few hundred thousand expert steps and train for tens of
thousands of steps for a real run (its defaults collect about a million).

## What SAC does with an expert, step by step

### Before training

1. **Replay prefill.** SAC runs the expert for `expert_buffer_n_steps` steps
   in each environment (default 20,000) and writes them to the replay buffer,
   flagged as expert rows. Every row also stores the expert's action at `s`
   and `s′`, which several extensions read later. Set it to 0 to skip.
2. **Cloning.** With `pre_train_n_steps > 0`, SAC collects that many expert
   steps per environment and trains the actor to reproduce them for
   `actor_cloning_epochs` epochs, in batches of `actor_cloning_batch_size`,
   at the actor's learning rate (`actor_cloning_lr` is accepted but not
   read today). The cloned actor also hands its observation statistics to
   the agent's observation normaliser. TD3, REDQ and APO take the same
   options. Two more, SAC's only, reset part of the cloned actor afterwards:
   `reset_log_std_after_bc` restores its exploration noise,
   `reset_actor_head_after_bc` keeps only the cloned hidden layers.
3. **Critic pre-training**, by extension. `MCPretrain` runs the expert, fits a
   separate critic, the *frozen expert critic*, to its Monte-Carlo returns,
   and records the range of values it predicts. Unlike a Bellman target, a
   Monte-Carlo return does not lean on an untrained critic, so it is
   unbiased. `BellmanPretrain` instead runs ordinary critic updates on the
   prefilled buffer; it is kept for ablations.

### Warm-up

Until `learning_starts`, SAC does not use its policy. At each step, with
probability `expert_fraction` (default 0.7), every environment takes the
expert's action; otherwise, a uniform random one. With `ResidualPolicy`, the
warm-up is uniform only.

### Collection

Collection extensions decide, at each step, whether the expert or the policy
acts. SAC runs them in two places:

- **Before the warm-up choice**, `EDGEExploration` (a value-based gate) and
  `JSRLCurriculum` (the expert starts each episode, for a shrinking number of
  steps; it needs `SAC(jsrl_curriculum=True)`).
- **After it**, `ValueBox`: the expert acts wherever its value exceeds a
  threshold that rises over training.

`ResidualPolicy` changes what the policy's action means: SAC executes
`clip(a_E + scale · a_π, −1, 1)`, in collection, in the critic's target, in
the actor's loss and in evaluation. The policy then learns a correction to
the expert (residual RL).

Steps where the expert acted are flagged in the buffer;
`value/expert_frac_in_buffer` logs their share of each training batch.

### Critic

Target extensions change what the critic learns towards. `IBRL` and
`LCBGatedBootstrap` compare the expert's and the policy's next actions.
`CriticBlend` and `MCVarianceCorrection` lean on the frozen expert critic, so
they need `MCPretrain` in the same stack. `PhiRefresh` refits the frozen
expert critic during training on the rows where the expert acted.

### Actor

`OnlineBC` adds a cloning term to the actor's loss, weighted by the expert's
value at each state and fading out over `critic_warmup_frac` of training. It
needs `MCPretrain`; without it, it adds nothing.

### Observation

Three SAC options change the networks' input, because they change its size:

- `augment_obs_with_expert_action=True` appends the expert's action to the
  observation, so the policy can condition on the expert's suggestion.
- `augment_obs_with_expert_state=True` appends a stateful expert's internal
  state (a PID's integral, say).
- `use_train_frac=True` appends the fraction of training done.

### Evaluation

Besides the expert's return and the bias above, `ExpertValueMetrics` logs
the critic's value at the expert's actions (`value/q_expert_mean`) and its
gap to the value of the actions taken (`value/q_gap`). `EDGEExploration` logs
`edge/live_expert_frac`.

## Choosing the pieces

| You want to | Use |
| --- | --- |
| Start the replay with good experience | `expert_buffer_n_steps` (on by default) |
| Start the actor as a copy of the expert | `pre_train_n_steps` |
| Keep the actor near the expert early on | `OnlineBC` (SAC), `ImitationLoss` (TD3, REDQ, ASAC, APO) |
| Let the expert act while the agent is poor | `EDGEExploration`, `JSRLCurriculum`, `ValueBox` |
| Learn a correction to the expert | `ResidualPolicy` |
| Give the critic a head start | `MCPretrain`, then `CriticBlend`, `MCVarianceCorrection`, `PhiRefresh` |
| Make the target account for the expert | `IBRL`, `LCBGatedBootstrap` |
| Measure how the critic rates the expert | `ExpertValueMetrics` |

Some pieces need others. The agent refuses most wrong combinations when it is
built; the others add nothing or fail when training starts.

| Extension | Needs |
| --- | --- |
| `EDGEExploration`, `JSRLCurriculum`, `ValueBox`, `ResidualPolicy` | SAC with an `expert_policy` and without `use_pid_policy` (refused when built otherwise) |
| `JSRLCurriculum` | also `SAC(jsrl_curriculum=True)` |
| `OnlineBC`, `CriticBlend`, `MCVarianceCorrection` | `MCPretrain` in the stack (otherwise they add nothing) |
| `ValueBox`, `PhiRefresh` | `MCPretrain` in the stack |
| `BellmanPretrain`, `ExpertValueMetrics` | SAC with an `expert_policy` (they raise otherwise) |

The extensions that hold an expert themselves (`MCPretrain`, `PhiRefresh`,
`IBRL`, `LCBGatedBootstrap`, `CriticBlend`, `ImitationLoss`) take it as their
first argument. Pass the same expert you give SAC: Ajax does not check that
they match.

## Imitation for TD3, REDQ, ASAC and APO

These agents take `ImitationLoss`, the behaviour-cloning term of TD3+BC
(Fujimoto and Gu, 2021) with a fixed expert as the target:
`coef · mean((μ(s) − a_E(s))²) / 4`, where `μ(s)` is the actor's mean
action. Dividing by 4 maps the squared distance of two actions in `[-1, 1]`
to `[0, 1]`. Unlike `OnlineBC`, it is unweighted and needs no expert critic.

```python
import jax.numpy as jnp
from ajax.agents.TD3.TD3 import TD3
from ajax.extensions.expert import ImitationLoss


def pd_expert(obs):
    theta = jnp.arctan2(obs[..., 1], obs[..., 0])
    return jnp.clip(-(2.0 * theta + 0.5 * obs[..., 2]), -1.0, 1.0)[..., None]


agent = TD3("Pendulum-v1", learning_starts=500, extensions=(ImitationLoss(pd_expert, coef=1.0),))
state, _ = agent.train(seed=0, n_timesteps=2_000)
```

## Learning a PID's gains: SAC's gain mode

With a stateful PID expert that exposes `learnable_fields`, `anchor_gains`
and `step_with_gains` (TargetGym's `FunctionalExpertPolicy` after
`register_learnable_gains`), `SAC(use_pid_policy=True)` learns to schedule
the expert's gains instead of acting directly. The actor outputs one number
`a` in `[-1, 1]` per learnable gain; the gain applied is `anchor · 10^a`,
between a tenth and ten times the expert's own. The PID computes the action.
This mode bypasses the collection extensions and refuses `ResidualPolicy`.

It is different from the [PID policy](networks.md), in which the actor
itself is a PID whose gains are learned, for any continuous-action agent.

## Limits and caveats

- **No memory.** Recurrent SAC, TD3 and REDQ refuse an expert, cloning and
  extensions.
- **SAC is the hub.** The expert extensions read quantities only SAC's
  update provides; the [catalogue](extensions.md#catalogue) gives each one's
  scope.
- **Stateless queries.** The extensions listed under [The
  expert](#the-expert) call `expert(obs)` on replayed observations, four of
  them in the networks' units rather than raw ones.
- **`CriticBlend` drops the last observation column** before querying the
  expert on the next observation. It inherited this from the option it
  replaced, written for observations that end with an appended column
  (`use_train_frac`); check that it matches your setup.
- **Budgets add up.** The prefill, cloning and `MCPretrain` collections run
  before the first training step and are not counted in `n_timesteps`.

## See also

- [Extensions](extensions.md): the phases and the full catalogue.
- [SAC](../algorithms/sac.md): every SAC option.
- [Networks and the PID policy](networks.md).
