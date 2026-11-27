# PPO

Proximal Policy Optimization collects a rollout from `n_envs` parallel
environments, then takes several passes of minibatch gradient steps on it.
A clipped objective keeps each update close to the policy that collected
the data, so the same rollout can be reused for `n_epochs` passes.

**Paper:** Schulman, Wolski, Dhariwal, Radford and Klimov, *Proximal Policy
Optimization Algorithms*, 2017 ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347)).
Ajax follows its reference code, OpenAI Baselines' ppo2 (`ea25b9e`).

## When to use it

PPO is Ajax's general-purpose on-policy agent: discrete or continuous
actions, memory, image observations and many parallel environments. It
suits tasks where environment steps are cheap, as in JAX simulators. Where
steps are scarce, an off-policy agent such as [SAC](sac.md) reuses each one
more. With demonstrations and no reward, see
[GAIL](#gail-imitation-by-a-discriminator) below.

| Feature | PPO |
| --- | --- |
| Discrete actions | yes |
| Continuous actions | yes, a Gaussian; `squash=True` bounds it with `tanh` |
| Memory (`memory=`) | yes: GRU, LSTM, transformer, Mamba, without `expert_policy` or most extensions ([Memory](../guide/memory.md)) |
| Environments | gymnax, Brax, MuJoCo Playground |
| Image observations | yes, a CNN encoder (`cnn_image_shape`) |
| PID policy | yes, `actor_architecture=PIDPolicy(...)` ([Networks](../guide/networks.md)) |
| Several seeds in one call | yes, `train(seed=[...])` |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `observe`, `reward`, `on_target`, `critic_loss`, `actor_loss`, `action`, `eval_action`, `transform_action`, `actuate`, `track_memory` ([which extensions run](../guide/compatibility.md)) |
| Resume and checkpoints | yes; a run split by a checkpoint equals the run left whole |

## Example

```python
from ajax import PPO
from ajax.logging import LoggingConfig

agent = PPO("CartPole-v1", n_envs=4, n_steps=128, batch_size=32)
logging = LoggingConfig(log_frequency=4_096, backends=())
state, evaluations = agent.train(seed=[1, 2], n_timesteps=20_480, logging_config=logging)
print(evaluations["Eval/episodic mean reward"])  # one row per seed, one column per evaluation
```

With memory, the update replays the rollout in contiguous chunks of
`bptt_length` steps, each started from the memory the policy had there:

```python
from ajax import PPO
from ajax.networks.memory import MemoryConfig

agent = PPO("CartPole-v1", n_envs=4, n_steps=128, batch_size=32,
            memory=MemoryConfig(kind="gru", hidden_size=32), bptt_length=32)
state, _ = agent.train(seed=0, n_timesteps=4_096)
```

## Key hyperparameters

The defaults are ppo2's MuJoCo configuration.

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | `1` | environments stepped in parallel |
| `n_steps` | `2048` | steps per environment in each rollout |
| `batch_size` | `64` | sets the number of minibatches, not their size (see the notes) |
| `num_minibatches` | `0` | when positive, the number of minibatches, as in Brax |
| `n_epochs` | `10` | passes over the rollout per update, each reshuffled |
| `actor_learning_rate`, `critic_learning_rate` | `3e-4` | Adam step sizes, each a float or a schedule |
| `adam_eps` | `1e-5` | Adam's epsilon |
| `gamma` | `0.99` | discount |
| `gae_lambda` | `0.95` | GAE's lambda |
| `clip_range` | `0.2` | the ratio clip; a float or a schedule of the timestep |
| `ent_coef` | `0` | entropy bonus weight |
| `vf_coef` | `0.5` | critic loss weight, on half the squared error |
| `clip_value_loss` | `None` | clip the value loss (see the notes); `None` clips except with `unroll_length` |
| `max_grad_norm` | `0.5` | one gradient clip by global norm over both networks; `None` turns it off |
| `normalize_advantage` | `True` | standardise the advantages in each minibatch |
| `actor_architecture`, `critic_architecture` | `("64", "tanh", "64", "tanh")` | hidden layers, as width and activation strings |
| `share_encoder` | `None` | one encoder for the actor and the critic; `None` shares it with images only |
| `normalize_observations`, `normalize_rewards` | `True` | running normalisation, as ppo2's `VecNormalize` wrapper |
| `observation_clip` | `10.0` | bound of the normalised observations (rewards are clipped at 10 too) |
| `reward_normalization_gamma` | `None` | with `normalize_rewards`, the discount of the return whose spread divides the rewards: `gamma` if None, 0 for the single reward's |
| `memory`, `bptt_length` | `None` | memory block, and the length of the replayed chunks (see the notes) |
| `unroll_length` | `None` | Brax's minibatching: fragments of this length, advantages per minibatch (feedforward only) |
| `log_std_state_independent`, `log_std_init` | `True`, `0.0` | one learned log standard deviation per action dimension, and its initial value |
| `log_std_bounds` | `None` | the `(low, high)` clip of that log standard deviation; `None` leaves it unclipped |
| `squash` | `False` | bound the Gaussian with `tanh` |
| `actor_bias_init`, `critic_bias_init` | `None` | initialisers by name ([Networks](../guide/networks.md)) |
| `encoder_kernel_init` | `"orthogonal(np.sqrt(2))"` | the hidden layers' initialiser; None is the same one |
| `cnn_image_shape`, `cnn_extra_obs_dim`, `cnn_spec` | `None`, `0`, `None` | convolutional encoder for flat image observations |
| `episode_length`, `action_repeat` | `None`, `1` | simulator steps per episode and per agent step, on Brax and Playground; `None` is 1,000 steps, or a `mujoco:` task's own time limit ([Environments](../guide/environments.md#brax-and-playground)) |
| `expose_recent_rollout` | `False` | keep the last rollout on the state even when no extension reads it |
| `expert_policy` | `None` | an expert the collector tracks: its actions feed the observation features and a residual policy (`ResidualPolicy`), and the evaluation reports its return next to the agent's |
| `extensions` | `()` | research features (see [Extensions](../guide/extensions.md)) |

`squash`, `mean_kernel_init`, `num_resets_per_eval` and `num_evals`
reproduce choices of Brax's PPO. They are off by default, as is
`encoder_layer_norm` (a LayerNorm on the encoder's output).

## Presets

`PPO.preset("Atari", env_id)` takes ppo2's Atari configuration
(`ppo2/defaults.py:15-22`; [Presets](../guide/training.md#presets)): 8
environments of 128 steps, 4 minibatches, 4 epochs, learning rates
`2.5e-4`, `clip_range=0.1`, `ent_coef=0.01` and one network shared by the
actor and the critic. The 8 environments are the PPO paper's actors, where
ppo2 runs one per CPU. Atari's frame skip, frame stack and reward clipping
are the environment's.

`PPO.preset("SB3", env_id)` is Stable-Baselines3's PPO defaults (2.9.0,
`ppo/ppo.py:84-95`) in Ajax's names, for settings brought from SB3 or SBX.
Ajax's own defaults are ppo2's, which differ from SB3's in three ways that
change results; the preset sets each to SB3's:

| | Ajax's default (ppo2) | SB3 and SBX | The preset sets |
|---|---|---|---|
| Value loss | clipped around the rollout's value | not clipped (`clip_range_vf=None`) | `clip_value_loss=False` |
| Observations and rewards | normalised (ppo2's `VecNormalize`) | as the environment gives them | `normalize_observations=False`, `normalize_rewards=False` |
| `vf_coef` | weighs half the squared error | weighs the whole squared error | `vf_coef=1.0` for SB3's 0.5 |

Three things stay yours to set, or cannot be matched:

- **Minibatches.** SB3's `batch_size=64` is a minibatch's size. In Ajax it
  gives 32 minibatches whatever `n_envs` is (see the notes), which is SB3's
  64 samples with one environment only. With `n_envs=N`, pass
  `num_minibatches=n_steps * N // 64` for minibatches of 64.
- **SBX's networks.** SBX starts its layers as flax does
  (`encoder_kernel_init="lecun_normal()"` matches the hidden ones) and clips
  each network's gradient on its own; Ajax, as SB3, starts them orthogonal
  and clips the two together.
- **Adam.** Ajax's PPO steps TensorFlow 1's form of Adam, as ppo2; SB3 and
  SBX step the usual one. The two differ over the first few thousand steps.

## Differences from the reference

- **The learning rate is constant.** ppo2's MuJoCo configuration decays it
  linearly to 0; pass a schedule as `actor_learning_rate` and
  `critic_learning_rate` to do the same.
- **With memory, the actor and the critic keep separate networks.** ppo2
  and PureJaxRL share one network between them; a shared memory failed
  Ajax's memory probe on masked CartPole.

## Notes

- **`batch_size` sets the number of minibatches, not their size.** Unless
  `num_minibatches` is positive, the rollout of `n_steps * n_envs` samples
  splits into `max(n_steps, batch_size) // min(n_steps, batch_size)`
  minibatches. With the defaults that is 32 minibatches of `64 * n_envs`
  samples. In Stable-Baselines3, `batch_size` is the minibatch size.
- **Minibatches.** The advantages are computed once on the whole rollout,
  and its samples are shuffled flat into new minibatches at every epoch, as
  in ppo2 and the paper's Algorithm 1. `unroll_length` switches to Brax's
  PPO: the rollout is cut into fragments of that length that bootstrap at
  their ends, each minibatch holds whole fragments, and its advantages are
  recomputed with the current critic. `unroll_length=n_steps` keeps each
  environment's rollout as one fragment. A length that does not split the
  rollout evenly into the minibatches is refused at construction.
- **Actions.** A continuous PPO samples an unbounded Gaussian. Ajax clips
  the sample to `[-1, 1]` and maps that range onto the environment's
  bounds, so on gymnax Pendulum (torque in `[-2, 2]`) one unit of the
  Gaussian is two units of torque. Evaluation uses the mean action.
- **Exploration noise.** The Gaussian learns one log standard deviation
  per action dimension, whatever the observation, starting at 0 (a standard
  deviation of 1) and never clipped, as in ppo2, CleanRL and PureJaxRL.
  `log_std_bounds` adds a clip, and `log_std_state_independent=False`
  computes the log standard deviation from the observation instead. A PID
  policy's log standard deviation starts at `log_std_init` and is clipped
  to `log_std_bounds`.
- **Networks.** The actor and the critic are separate networks, as ppo2's
  MuJoCo configuration copies them; with images they share one encoder, as
  ppo2's Atari policies do. Hidden layers start orthogonal with gain
  `sqrt(2)`, the policy's output layer with gain 0.01 and the value layer
  with gain 1, as in ppo2.
- **Optimiser.** As ppo2's one optimiser, the two networks' gradients are
  clipped together to one global norm, then each takes an Adam step; Adam
  acts on each coordinate, so this is one Adam over the whole model. It is
  TensorFlow 1's Adam (`optimizer="tf1_adam"`), which adds its epsilon
  before the bias correction: at the first step that epsilon weighs 32
  times more than optax's, and about the same after a few thousand steps.
  With memory or `unroll_length`, PPO steps optax's Adam, as PureJaxRL and
  Brax do.
- **Value loss.** As in ppo2, each squared error is the larger of its own
  and that of the value clipped to `clip_range` around the value that
  collected the rollout, so the loss stops pulling a value once it has
  moved `clip_range` towards its target. With `unroll_length` it is not
  clipped, as in Brax.
- **Entropy bonus.** On a squashed policy it is a one-sample estimate of
  the executed action's entropy, as in Brax.
- **Normalisation.** With `normalize_observations=True` the loss sees each
  observation normalised with the statistics of the step that collected it,
  as SB3's `VecNormalize` and CleanRL do.
- **Extensions.** PPO folds the loss and target phases, so instrumentation
  such as `CriticDiagnostics` works on it (PPO keeps its latest rollout for
  it). Extensions that replace the collected action (the `action` phase)
  need an off-policy agent, and PPO refuses them. See
  [Extensions](../guide/extensions.md).
- **Memory.** A recurrent PPO minibatches whole chunks of `bptt_length`
  steps. By default the number of minibatches sets that length: each
  environment's whole rollout when `n_envs` is a multiple of the number of
  minibatches, as ppo2, CleanRL and PureJaxRL do, otherwise the fewest
  equal chunks that split (64 steps with the defaults). Chunks that do not
  split are refused at construction.
  [Memory](../guide/memory.md#the-libraries-disagree-on-the-length)
  compares the libraries: to reproduce theirs, choose `num_minibatches` to
  divide `n_envs`.

## GAIL: imitation by a discriminator

GAIL (Ho and Ermon, *Generative Adversarial Imitation Learning*, NeurIPS 2016,
[arXiv:1606.03476](https://arxiv.org/abs/1606.03476)) trains a policy to
imitate demonstrations without their rewards. A discriminator learns to tell
the policy's rollouts from the demonstrations, and the policy is rewarded
where it is mistaken for the expert. In Ajax it is PPO with the
`AdversarialReward` extension, which folds the `reward` phase:

```python
import jax
import jax.numpy as jnp

from ajax import PPO
from ajax.buffers.dataset import record
from ajax.extensions.adversarial import AdversarialReward


def pd_expert(obs):
    theta = jnp.arctan2(obs[..., 1], obs[..., 0])
    return jnp.clip(-(2.0 * theta + 0.5 * obs[..., 2]), -2.0, 2.0)[..., None]


env_args = PPO("Pendulum-v1").env_args
demonstrations = record(pd_expert, env_args, jax.random.PRNGKey(0), n_steps=1_000)
agent = PPO("Pendulum-v1", extensions=(AdversarialReward(demonstrations),))
```

After each rollout, the rewards become `-log(1 - D(s, a))` from the current
discriminator, and the discriminator takes `n_updates` (2) cross-entropy
steps on the rollout's rows and as many demonstration rows. The evaluations
still report the environment's return. It follows HumanCompatibleAI/imitation
at `e5ef188` (`algorithms/adversarial/gail.py`): its 32-32 ReLU reward
network, torch's Adam at 1e-3, and PPO in place of the paper's TRPO. The
discriminator reads the action the environment runs, clipped to `[-1, 1]`,
as the reference records the clipped action SB3 steps with. It differs
from the reference in two ways: it reads `(s, a)` without the running
normalisation of the reference's examples, and it trains on the whole
rollout, where the reference draws `demo_batch_size` rows.

## Results

The [reproduction results](../reproduction/results.md) set PPO beside
Stable-Baselines3's and SBX's on CartPole, and Brax's on inverted_pendulum,
with learning curves; all three are similar. The probes (`tests/probing`) train it briefly on tiny
environments with known answers: exact values of small problems, episode
ends, memory, extension phases, and a run split by a checkpoint matching
the run left whole. [Performance](../guide/performance.md) compares its
speed with the other agents'.

## API

::: ajax.agents.PPO.PPO.PPO
