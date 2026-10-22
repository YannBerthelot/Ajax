# TD3

Twin Delayed DDPG learns a deterministic policy and two critics. It fixes
DDPG's over-estimation with three changes: the target takes the smaller of
two target critics, the target action gets clipped noise (target policy
smoothing), and the actor and the targets update only every `policy_delay`
critic steps. It explores by adding Gaussian noise to its actions and learns
off-policy from a replay buffer.

**Paper.** Fujimoto et al., *Addressing Function Approximation Error in
Actor-Critic Methods*, 2018 ([arXiv:1802.09477](https://arxiv.org/abs/1802.09477)).
The update follows its Algorithm 1 as the authors' code (`sfujim/TD3`)
makes it, and the defaults are that code's as updated in 2019 (see
[Notes](#notes)).

## Can I use?

| | TD3 |
| --- | --- |
| Continuous actions (`Box`) | Yes |
| Discrete actions | No |
| Learning | Off-policy, replay buffer |
| Vectorised environments (`n_envs`) | Yes |
| Recurrent memory (`memory=`) | Yes, without extensions or `expert_policy` |
| PID policy (`actor_architecture=PIDPolicy(...)`) | Yes |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `on_target`, `critic_loss`, `actor_loss` |

## Example

```python
from ajax import TD3
from ajax.logging.wandb_logging import LoggingConfig

agent = TD3("Pendulum-v1", n_envs=4, learning_starts=1_000)
logging = LoggingConfig(config={}, log_frequency=2_000, use_wandb=False)
state, evaluations = agent.train(seed=[0, 1], n_timesteps=4_000, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])  # one row per seed
```

This budget only shows the call; Pendulum needs more steps to be solved.

## Key hyperparameters

Defaults are read from `TD3.__init__`. They are the authors' code as
updated in 2019, not the paper's: see [Notes](#notes).

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | `1` | parallel environments |
| `actor_learning_rate`, `critic_learning_rate` | `3e-4` | Adam step sizes (Adam's epsilon is torch's 1e-8) |
| `actor_architecture`, `critic_architecture` | `("256", "relu", "256", "relu")` | hidden layers (plain MLPs) |
| `gamma` | `0.99` | discount |
| `buffer_size` | `1_000_000` | replay capacity |
| `batch_size` | `256` | replay minibatch |
| `learning_starts` | `25_000` | steps of uniform random actions before the first update |
| `tau` | `0.005` | Polyak rate of the target networks |
| `num_critics` | `2` | critics; the target takes their minimum |
| `policy_delay` | `2` | critic steps per actor and target update |
| `exploration_noise` | `0.1` | standard deviation of the behaviour noise; a float or a schedule of the timestep |
| `target_policy_noise`, `target_noise_clip` | `0.2`, `0.5` | target smoothing noise and its clip |
| `reward_scale` | `1.0` | multiplies the rewards in the target |
| `max_grad_norm` | `None` | gradient clipping |
| `normalize_observations`, `normalize_rewards` | `False` | running normalisation |
| `reward_normalization_gamma` | `None` | with `normalize_rewards`, divide by the spread of the discounted return |
| `memory`, `burn_in`, `sequence_length`, `stored_state` | `None`, `8`, `16`, `False` | recurrent memory (see [Memory](../guide/memory.md)) |
| `expert_policy` | `None` | an expert whose return the evaluation reports next to the agent's (`Eval/expert bias`) |
| `action_pipeline` | `None` | replaces the collection-time action choice (used by a downstream project) |
| `episode_length`, `action_repeat` | `None`, `1` | simulator steps per episode (1,000 when `None`) and per agent step, on Brax and Playground |
| `extensions` | `()` | research features (see [Extensions](../guide/extensions.md)) |

Cloning an expert into the actor before training is the `CloningPretrain`
extension; a cloning term during training, of the TD3+BC form, is `ImitationLoss`
([Expert guidance](../guide/expert-guidance.md)).

## Notes

- **Defaults.** The authors updated their code's hyperparameters in 2019
  "to improve performance": 256-256 layers, learning rate `3e-4`, batch
  256 and a 25,000-step warm-up. Ajax uses these. The paper's run used
  400-300 layers, `1e-3`, batch 100 and 10,000 steps; pass them to
  reproduce it.
- **Actor.** A deterministic MLP ending in `tanh`. As in the authors' code,
  every layer of the actor and the critics, the actor's last included,
  starts as torch's `nn.Linear` does: kernel and bias uniform in
  `±1/sqrt(fan_in)` (not DDPG's `±3e-3` last layer, which TD3 keeps with
  `memory=` until Ni et al.'s initialisation is audited). Actions lie in
  `[-1, 1]`, and Ajax maps them onto the environment's bounds, so the noise
  levels are in those `[-1, 1]` units: on gymnax Pendulum (torque in
  `[-2, 2]`) an `exploration_noise` of 0.1 is a torque noise of standard
  deviation 0.2.
- **Critics.** They read the action in the environment's units, `c + h a`
  with the box's centre `c` and half-range `h`, as the authors' critics read
  `max_action * tanh`; on a `[-1, 1]` box the map is the identity. Adam's
  steps on a critic's action inputs depend on those units. Their loss sums
  the critics' squared errors, as the authors' code does.
- **Behaviour policy.** `clip(mu(s) + N(0, exploration_noise), -1, 1)` after
  the warm-up; uniform random actions before `learning_starts`. Evaluation
  uses `mu(s)` without noise.
- **Target.** `r + gamma (1 - terminated) min_i Q'_i(s', a')`, with
  `a' = clip(mu'(s') + clip(N(0, target_policy_noise), -c, c), -1, 1)` and
  `c = target_noise_clip`.
- **Actor loss.** `-Q_1(s, mu(s))`: the first critic only.
- **Updates per step.** Each iteration steps every env once, then makes one
  critic step; every `policy_delay`-th one also steps the actor and the
  targets, the first actor step coming with the `policy_delay`-th update, as
  in the authors' code.
- **Time limits.** A transition cut by a time limit bootstraps on its final
  observation; only a termination stops the bootstrap.
- **PID policy.** With `actor_architecture=PIDPolicy(...)` the actor is a
  PID whose output goes through `tanh`; see [Networks](../guide/networks.md).
- **Memory.** Sequence replay also burns in the target actor's memory, since
  TD3's bootstrap action comes from it.

## TD3+BC: offline

TD3+BC (Fujimoto and Gu, *A Minimalist Approach to Offline Reinforcement
Learning*, NeurIPS 2021, [arXiv:2106.06860](https://arxiv.org/abs/2106.06860))
is TD3 trained on a fixed dataset with a behaviour-cloning term in the
actor's loss. Its reference code is sfujim/TD3_BC at `8791ad7`. In Ajax it
is TD3, `ImitationLoss` with the dataset's own actions as the target, and
the offline mode:

```python
from ajax import TD3
from ajax.extensions.expert import ImitationLoss

agent = TD3(
    "Pendulum-v1",
    normalize_observations=True,  # the dataset's statistics, as the paper's
    learning_starts=0,
    extensions=(ImitationLoss(None, target="replay", alpha=2.5),),
)
# state, evaluations = agent.train_offline(dataset, n_updates=1_000_000)
```

The paper's actor loss is `-λ mean Q(s, π(s)) + mean((π(s) - a)²)`, with
`λ = α / mean|Q(s, π(s))|` and α = 2.5 (`TD3_BC.py:141-143`), the actions
compared in the environment's units. With `alpha`, `ImitationLoss` makes
TD3's actor loss exactly that: it adds `(λ - 1)` times TD3's own `-mean Q`
and the squared difference of the actions mapped onto the env's box. The
other settings are TD3's defaults, which match the reference's. The states are standardised with the dataset's
statistics. The reference divides by the standard deviation plus `1e-3`,
Ajax by `sqrt(var + 1e-8)`; the two differ on a column the dataset holds
constant. The [offline mode](../guide/training.md#training-offline)
evaluates every `log_frequency` updates, as the reference does every 5,000.
A conformance check (`tests/conformance/test_td3bc_sfujim.py`) makes one
update of each from the same state, batch and noise, on Pendulum and
MountainCarContinuous, and compares them.

## ReBRAC: offline

ReBRAC (Tarasov, Kurenkov, Nikulin and Kolesnikov, *Revisiting the
Minimalist Approach to Offline Reinforcement Learning*, NeurIPS 2023,
[arXiv:2305.09836](https://arxiv.org/abs/2305.09836)) is TD3+BC with design
choices of its own. The main ones are a behaviour-cloning penalty in the
critic's target as well, LayerNorm in the critic, deeper networks and
larger batches. Its reference code is tinkoff-ai/ReBRAC at `694385b`. In
Ajax it is TD3 with TD3+BC's actor term and `TargetBC`, the target
penalty, trained with the offline mode:

```python
from ajax import TD3
from ajax.extensions.expert import ImitationLoss
from ajax.extensions.offline import TargetBC

hidden = ("256", "relu")
agent = TD3(
    "Pendulum-v1",
    actor_architecture=hidden * 3,
    critic_architecture=("256", "relu", "layernorm") * 3,
    actor_learning_rate=1e-3,
    critic_learning_rate=1e-3,
    batch_size=1024,
    learning_starts=0,
    extensions=(
        ImitationLoss(None, target="replay", alpha=1.0),  # actor_bc_coef 1
        TargetBC(beta=1.0),  # critic_bc_coef 1
    ),
)
# state, evaluations = agent.train_offline(dataset, n_updates=1_000_000)
```

`TargetBC` subtracts `γ (1 − d) β Σ (ã′ − a′)²` from the target, with `ã′`
the next actions TD3's target scored (its target policy plus clipped noise)
and `a′` the dataset's next action (`rebrac.py:134-144`). The replay keeps
`a′` as each row's episode's next row's action. A row ending its episode,
or a collected one, has none, so it keeps its target. The coefficients
`actor_bc_coef` and `critic_bc_coef` are tuned per dataset in the paper.

Deviations from the reference:

- TD3's actor follows its first critic, and the cloning term's scale is
  that critic's `mean|Q|`. ReBRAC uses the minimum over the critics for
  both.
- The cloning term averages the squared error over the action
  dimensions. ReBRAC sums it, so pass `coef` equal to the action dimension
  to match.
- TD3 keeps its reference's initialisation. ReBRAC's layers start their
  biases at 0.1 and its heads uniform in ±1e-3 (actor) and ±3e-3 (critic).

## Results

The repository records no learning curves for TD3 yet. A conformance
check (`tests/conformance/test_td3_sfujim.py`) runs one update of Ajax's
TD3 and of sfujim/TD3 from the same state, batch and noise, on
MountainCarContinuous and Pendulum, and compares the change each makes to
every parameter and optimiser moment. Unit tests
(`tests/agents/TD3` and the contracts every agent shares) check its pieces
against exact contracts, without learning, such as its smoothed target. The
probes (`tests/probing`) train it briefly on tiny environments with known
answers: its target smoothing, discounting, time limits, memory, extension
phases, and a run split by a checkpoint matching the run left whole. On the
CPU speed benchmark (`benchmarks/agent_bench.py`: Pendulum, 4 envs, 8,000
steps, updates from step 1,000) TD3 ran at 2,659 steps per second of
execution (1,961 with compilation) at main `a8a13c2` (see
[PERFORMANCE_REPORT.md](https://github.com/YannBerthelot/Ajax/blob/main/PERFORMANCE_REPORT.md)).

## API

::: ajax.agents.TD3.TD3.TD3
