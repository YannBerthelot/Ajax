# TD3

Twin Delayed DDPG learns a deterministic policy and two critics. It curbs
DDPG's over-estimation in three ways: the target takes the smaller of the
two target critics' values, the target action gets clipped noise (target
policy smoothing), and the actor and the targets update only every
`policy_delay` critic steps. It explores by adding Gaussian noise to its
actions.

**Paper:** Fujimoto, van Hoof and Meger, *Addressing Function Approximation
Error in Actor-Critic Methods*, ICML 2018
([arXiv:1802.09477](https://arxiv.org/abs/1802.09477)). Ajax follows the
authors' code, `sfujim/TD3` at `34770cc`.

## When to use it

TD3 learns off-policy from a replay buffer, as [SAC](sac.md) does, but its
policy is deterministic: one action per state, explored around with noise
of a size you set. It is also the base of the offline methods TD3+BC and
ReBRAC below. SAC learns a stochastic policy and tunes its exploration
itself.

| Feature | TD3 |
| --- | --- |
| Discrete actions | no |
| Continuous actions | yes, a deterministic `tanh` policy, mapped onto the environment's bounds |
| Memory (`memory=`) | yes, on replayed sequences, without `expert_policy` or most extensions ([Memory](../guide/memory.md)) |
| Environments | gymnax, Brax, MuJoCo Playground |
| Image observations | no |
| PID policy | yes, `actor_architecture=PIDPolicy(...)` ([Networks](../guide/networks.md)) |
| Several seeds in one call | yes, `train(seed=[...])` |
| Offline training | yes, `train_offline` ([Training](../guide/training.md#training-offline)); also [TD3+BC](#td3bc-offline) and [ReBRAC](#rebrac-offline) |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `observe`, `on_target`, `critic_loss`, `actor_loss`, `action`, `eval_action`, `transform_action`, `actuate`, `choose_warmup_action`, `choose_replay_action`, `track_memory` ([which extensions run](../guide/compatibility.md)) |
| Resume and checkpoints | yes; a run split by a checkpoint equals the run left whole |

## Example

```python
from ajax import TD3
from ajax.logging import LoggingConfig

agent = TD3("Pendulum-v1", n_envs=4, learning_starts=1_000)
logging = LoggingConfig(log_frequency=2_000, backends=())
state, evaluations = agent.train(seed=[0, 1], n_timesteps=4_000, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])  # one row per seed
```

This budget only shows the call; Pendulum needs more steps to be solved.

## Key hyperparameters

The defaults are the authors' code's as they updated it in 2019. The
paper's runs used 400-300 layers, learning rates of `1e-3`, batches of 100
and 10,000 warm-up steps; pass these to reproduce them. The [API](#api)
section lists every argument.

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | `1` | parallel environments |
| `actor_learning_rate`, `critic_learning_rate` | `3e-4` | Adam step sizes (epsilon `1e-8`, torch's) |
| `actor_architecture`, `critic_architecture` | `("256", "relu", "256", "relu")` | hidden layers |
| `gamma` | `0.99` | discount |
| `tau` | `0.005` | Polyak rate of the target networks |
| `buffer_size` | `1_000_000` | replay capacity |
| `batch_size` | `None` | rows per update: 256, or 64 sequences with `memory=` |
| `learning_starts` | `25_000` | steps of uniform random actions before the first update |
| `num_critics` | `2` | critics; the target takes their minimum |
| `policy_delay` | `None` | critic steps per actor and target update: 2, or 1 with `memory=` |
| `exploration_noise` | `0.1` | standard deviation of the behaviour noise: a number, or a schedule of the timestep |
| `target_policy_noise`, `target_noise_clip` | `0.2`, `0.5` | target smoothing noise and its clip |
| `reward_scale` | `1.0` | multiplies the rewards in the target |
| `max_grad_norm` | `None` | gradient clipping |
| `normalize_observations`, `normalize_rewards` | `False` | running normalisation ([Environments](../guide/environments.md)) |
| `memory`, `burn_in`, `sequence_length`, `stored_state` | `None`, `0`, `64`, `False` | recurrent memory and its sequence replay ([Memory](../guide/memory.md)) |
| `episode_length`, `action_repeat` | `None`, `1` | simulator steps per episode and per agent step, on Brax and Playground; `None` is 1,000 steps, or a `mujoco:` task's own time limit ([Environments](../guide/environments.md#brax-and-playground)) |
| `expert_policy` | `None` | an expert the run tracks; each evaluation logs its return next to the agent's (`Eval/expert bias`) |
| `extensions` | `()` | research features ([Extensions](../guide/extensions.md)) |

Cloning an expert into the actor before training is the `CloningPretrain`
extension; a cloning term during training, of the TD3+BC form, is
`ImitationLoss` ([Expert guidance](../guide/expert-guidance.md)).

## Presets

`TD3.preset("D4RL", env_id)` is TD3+BC as its authors run it on the D4RL
datasets (`sfujim/TD3_BC` at `8791ad7`, `main.py:56-57`;
[Presets](../guide/training.md#presets)): `ImitationLoss` cloning the
dataset's actions at `alpha=2.5`, and `normalize_observations=True`. Train
it with `train_offline` ([TD3+BC](#td3bc-offline)).

## Differences from the reference

None in the update: one update of Ajax's TD3 and of `sfujim/TD3` from the
same state agree to within float32 rounding ([Results](#results)). TD3+BC
and ReBRAC list their own differences below.

## Notes

- **Actor.** A deterministic MLP ending in `tanh`. Its actions lie in
  `[-1, 1]`, and so do the noise levels: on gymnax Pendulum (torque in
  `[-2, 2]`) an `exploration_noise` of 0.1 is a torque noise of standard
  deviation 0.2.
- **Initialisation.** As in the authors' code, every layer of the actor
  and the critics, the actor's last included, starts as torch's
  `nn.Linear` does: kernel and bias uniform in `±1/sqrt(fan_in)` (not
  DDPG's `±3e-3` last layer).
- **Critics.** They read the action in the environment's units, `c + h a`
  with the box's centre `c` and half-range `h`, as the authors' critics
  read `max_action * tanh`. Their loss sums the critics' squared errors.
- **Behaviour policy.** `clip(mu(s) + N(0, exploration_noise), -1, 1)`
  after the warm-up, uniform random actions before `learning_starts`.
  Evaluation uses `mu(s)` without noise.
- **Target.** `r + gamma (1 - terminated) min_i Q'_i(s', a')`, with
  `a' = clip(mu'(s') + clip(N(0, target_policy_noise), -c, c), -1, 1)` and
  `c = target_noise_clip`.
- **Actor loss.** `-Q_1(s, mu(s))`: the first critic only.
- **Updates per step.** Each iteration steps every environment once, then
  makes one critic step; every `policy_delay`-th one also steps the actor
  and the targets, the first actor step coming with the
  `policy_delay`-th update, as in the authors' code.
- **Time limits.** A transition cut by a time limit bootstraps on its final
  observation; only a termination stops the bootstrap.
- **Memory.** With `memory=`, TD3 is Ni et al. (2022)'s recurrent TD3:
  their networks and initialisation, an actor maximising `min(Q_1, Q_2)` at
  every update (`policy_delay` 1), and critics reading the `[-1, 1]`
  actions. Sequence replay also burns in the target actor's memory, which
  picks the bootstrap action.

## TD3+BC: offline

TD3+BC (Fujimoto and Gu, *A Minimalist Approach to Offline Reinforcement
Learning*, NeurIPS 2021, [arXiv:2106.06860](https://arxiv.org/abs/2106.06860))
is TD3 trained on a fixed dataset with a behaviour-cloning term in the
actor's loss. Its reference code is `sfujim/TD3_BC` at `8791ad7`. In Ajax
it is TD3 with `ImitationLoss`, the dataset's own actions as the target,
trained with the [offline mode](../guide/training.md#training-offline):

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

`TD3.preset("D4RL", env_id)` builds the same agent ([Presets](#presets)).

The paper's actor loss is `-λ mean Q(s, π(s)) + mean((π(s) - a)²)`, with
`λ = α / mean|Q(s, π(s))|` and α = 2.5 (`TD3_BC.py:141-143`), the actions
compared in the environment's units. With `alpha`, `ImitationLoss` makes
TD3's actor loss exactly that. The other settings are TD3's defaults, which
match the reference's. The offline mode evaluates every `log_frequency`
updates, as the reference does every 5,000. A conformance check
(`tests/conformance/test_td3bc_sfujim.py`) makes one update of each from
the same state, batch and noise, and compares them. One difference: both
standardise the states with the dataset's statistics, but the reference
divides by the standard deviation plus `1e-3` and Ajax by
`sqrt(var + 1e-8)`, which differ on a column the dataset holds constant.

## ReBRAC: offline

ReBRAC (Tarasov, Kurenkov, Nikulin and Kolesnikov, *Revisiting the
Minimalist Approach to Offline Reinforcement Learning*, NeurIPS 2023,
[arXiv:2305.09836](https://arxiv.org/abs/2305.09836)) is TD3+BC with
design choices of its own: chiefly a behaviour-cloning penalty in the
critic's target as well, LayerNorm in the critic, deeper networks and
larger batches. Its reference code is `tinkoff-ai/ReBRAC` at `694385b`. In
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
the next actions TD3's target scored (its target policy plus clipped
noise) and `a′` the dataset's next action (`rebrac.py:134-144`). The
replay keeps `a′` as the action of the next row of the same episode; a row
ending its episode, or a collected one, has none and keeps its target. The
paper tunes `actor_bc_coef` and `critic_bc_coef` per dataset.

Differences from the reference, where a conformance check
(`tests/conformance/test_rebrac_original.py`) finds the critics' steps
alike:

- TD3's actor follows its first critic, and the cloning term's scale is
  that critic's `mean|Q|`. ReBRAC uses the minimum over the critics for
  both.
- ReBRAC moves the target actor towards the actor as it was before its
  step; TD3 towards the stepped actor.
- The cloning term averages the squared error over the action dimensions,
  where ReBRAC sums it: pass `coef` equal to the action dimension to match.
- TD3 keeps its reference's initialisation. ReBRAC's layers start their
  biases at 0.1 and its heads uniform in ±1e-3 (actor) and ±3e-3 (critic).

## Results

The [reproduction results](../reproduction/results.md) set TD3 beside
Stable-Baselines3's on Pendulum, with learning curves. A conformance check
(`tests/conformance/test_td3_sfujim.py`) makes one update of Ajax's TD3
and of `sfujim/TD3` from the same state, batch and noise, on
MountainCarContinuous and Pendulum: every parameter and optimiser moment
changes alike, to within float32 rounding. The probes (`tests/probing`)
train it briefly on tiny environments with known answers.
[Performance](../guide/performance.md) compares its speed with the other
agents'.

## API

::: ajax.agents.TD3.TD3.TD3
