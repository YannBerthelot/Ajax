# Networks and the PID policy

An agent's networks are set by its constructor: the hidden layers as a tuple
of strings, initialisers by name, optional normalisation and an image
encoder. The actor can also be a PID controller whose gains the agent learns.

## Hidden layers

An architecture is a tuple read left to right. A number adds a dense layer
of that width; a name adds an activation:

```python
from ajax import PPO

agent = PPO(
    "CartPole-v1",
    n_envs=8,
    n_steps=128,
    actor_architecture=("64", "tanh", "64", "tanh"),
    critic_architecture=("256", "relu", "256", "relu"),
    encoder_kernel_init="orthogonal(np.sqrt(2))",  # the hidden layers' kernels
    critic_bias_init="0.0",  # a number gives a constant initialiser
    encoder_layer_norm=True,  # a LayerNorm after the hidden layers
)
state, _ = agent.train(seed=0, n_timesteps=2_048)
```

The activations are `relu`, `tanh`, `leaky_relu`, `swish`, `silu` and `mish`;
a callable works too. The agent adds its output heads after the hidden
layers: the policy's mean and standard deviation, the value, the Q-values.
DQN and PQN take a single `architecture` for their Q-network.

Each agent's default follows its paper or reference implementation:

| Default | Agents |
| --- | --- |
| `("256", "relu", "256", "relu")` | SAC, ASAC, REDQ |
| `("400", "relu", "300", "relu")` | TD3 |
| `("256", "leaky_relu", "256", "leaky_relu")` | AVG |
| `("128", "tanh", "128", "tanh")` | PPO |
| `("64", "tanh", "64", "tanh")` | APO |
| `("128", "relu", "128", "relu")` | DQN, PQN, UDRL, APG (its controller) |

DreamerV3 and TD-MPC2 build their own networks to their papers'
specifications: see [World models](world-models.md).

### Initialisers

Kernels start from an orthogonal initialiser with gain 1 and biases from
zero, except the policy's mean, whose kernel starts from
`orthogonal(0.01)`, a near-zero first action (TD3's deterministic actor
starts its last layer uniform in `±3e-3`, as DDPG does), and APO's hidden
layers, which start as its reference code's (`"torch_linear"`, below).
Initialisers are named by string:

- `"orthogonal(0.01)"`, `"lecun_uniform"`, or any other factory in
  `flax.linen.initializers`, with its argument in brackets (`np`, `jnp` and
  `math` may appear in it, as in `"orthogonal(np.sqrt(2))"`);
- `"zeros"`, and `"trunc_normal_fan_in(1.0)"` (DreamerV3's initialiser);
- a number such as `"0.0"` or `"-1.0"`, for a constant.

PPO and APO take three of them: `encoder_kernel_init` for the hidden
layers' kernels, `actor_bias_init` for the policy head's bias and
`critic_bias_init` for the value head's. PPO also takes `mean_kernel_init`
(below). `encoder_kernel_init="torch_linear"` starts each hidden layer as
torch's `nn.Linear` does: its kernel and its bias both uniform in
`±1/sqrt(fan_in)`, the fan-in being the layer's input width.

### Layer normalisation

The hidden layers form an encoder that the output heads read.
`encoder_layer_norm=True` adds a LayerNorm on its output. It is off by
default, because the SAC, TD3, PPO and DQN papers use plain MLPs. PPO takes
it as an argument. SAC, ASAC, REDQ, APO and DQN, and TD3's critics (not its
actor), read it from the network configuration, which you can set before
training:

```python
from ajax import SAC

agent = SAC("Pendulum-v1", learning_starts=500)
agent.network_args = agent.network_args.replace(encoder_layer_norm=True)
state, _ = agent.train(seed=0, n_timesteps=1_000)
```

UDRL turns it on for its policy and APG's controller always keeps a
LayerNorm on its encoder, as Ajax's earlier networks did. PQN puts one after
every hidden layer, as its paper does, whatever the setting. AVG normalises
its penultimate features instead, as its paper does, which overrides the
setting.

### The policy head

PPO exposes the policy head's options. By default its Gaussian learns one
log standard deviation per action dimension, whatever the observation,
starting at 0 (a standard deviation of 1) and never clipped
(`log_std_bounds=None`), as PPO's reference code does: OpenAI Baselines at
`ea25b9e` (ppo2's
`common/distributions.py:96-106`; ppo1's `ppo1/mlp_policy.py:36-39`, with
`gaussian_fixed_var=True`), CleanRL at `fe8d8a0`
(`ppo_continuous_action.py:129`) and PureJaxRL at `31756b1`
(`ppo_continuous_action.py:41`). `log_std_state_independent=False` with
`log_std_init=-1.0` builds the head PPO had before, SAC's: a log standard
deviation computed from the observation, which `log_std_bounds=(-20.0,
2.0)` clips as SAC's is. APO sets its reference code's head (`xtma/apo`,
built on the authors' rlpyt fork): one learned log standard deviation per
dimension, starting at 0 and unclipped by default (`log_std_bounds=None`),
around a mean passed through `tanh` (`tanh_mean`), with an unsquashed
sample; its mean layer is `lecun_uniform`. The other agents clip theirs, a
PID policy's included, to `(-20.0, 2.0)`.

| Argument | Default (PPO) | Effect |
| --- | --- | --- |
| `log_std_state_independent` | `True` | one learned log standard deviation per action dimension; `False` computes it from the observation |
| `log_std_init` | `0.0` | the initial log standard deviation |
| `log_std_bounds` | `None` | the `(low, high)` clip of the log standard deviation, whichever head (or PID policy) learns it, `low < high` (the agent, its network config and the actor refuse others); `None` leaves it unclipped |
| `mean_kernel_init` | `None`: `orthogonal(0.01)`, a near-zero first mean | the initialiser of the mean's output layer |
| `squash` | `False` | `tanh` on the sampled action, with the matching log-probability correction |

## Image observations

PPO, DQN, PQN and UDRL can read images through a convolutional encoder.
Ajax flattens image observations, so give the image's shape
`(height, width, channels)` as `cnn_image_shape`; the encoder reshapes the
flat vector back. `cnn_extra_obs_dim` counts extra scalar entries after the
image, which join the image features after the convolutions. `cnn_spec`
sets the convolutions:

```python
from ajax import PQN
from ajax.networks.networks import CNNSpec

# MinAtar observations are 10 x 10 images with 4 channels.
agent = PQN(
    "Breakout-MinAtar",
    cnn_image_shape=(10, 10, 4),
    cnn_spec=CNNSpec(channels=(16, 32), kernel_sizes=(3, 3), strides=(1, 1), feature_dim=128),
)
state, _ = agent.train(seed=0, n_timesteps=2_048)
```

`CNNSpec()` with no arguments is the default encoder: two ReLU convolutions
of 16 and 32 channels (kernels 4 and 3, stride 2, `"SAME"` padding), then a
128-wide dense layer, all at flax's default initialisation. The image is
read in height, width, channels order. `padding` takes flax's padding
(`"VALID"` for none), `layer_norm=True` puts a LayerNorm after every layer,
before its ReLU, and `kernel_init` names the initialiser of every layer, as
the network's other `*_kernel_init` settings do.

## The PID policy

Many control tasks are run by PID controllers, which engineers can read,
tune and certify. The PID policy makes the actor such a controller: its
action is

```
u = Kp e + Ki I + Kd D
```

where `e`, `I` and `D` are the tracking error, its integral and its
difference, which [`PIDTermsWrapper`](environments.md#pid-error-terms) puts
first in the observation. The agent learns the gains. Pass a `PIDPolicy` as
the `actor_architecture` of any continuous-action agent: SAC, TD3, REDQ,
ASAC, AVG, PPO or APO.

```python
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from ajax import SAC
from ajax.environments.pid_terms import PIDTermsWrapper
from ajax.modules.pid_actor import PIDPolicy, pid_gains


@struct.dataclass
class State(environment.EnvState):
    x: jax.Array  # the controlled value
    r: jax.Array  # its set-point


@struct.dataclass
class Params(environment.EnvParams):
    max_steps_in_episode: int = 10


class Setpoint(environment.Environment):
    """An integrator, x <- x + 4u, that must reach a random set-point r."""

    @property
    def default_params(self):
        return Params()

    def reset_env(self, key, params):
        r = jax.random.uniform(key, minval=-1.0, maxval=1.0)
        state = State(time=0, x=jnp.float32(0.0), r=r)
        return self.get_obs(state), state

    def step_env(self, key, state, action, params):
        x = state.x + 4.0 * jnp.clip(action[0], -1.0, 1.0)
        state = state.replace(x=x, time=state.time + 1)
        return self.get_obs(state), state, -((state.r - x) ** 2), jnp.array(False), {}

    def get_obs(self, state, params=None, key=None):
        return jnp.stack([state.x, state.r])

    def observation_space(self, params):
        return spaces.Box(-jnp.inf, jnp.inf, (2,))

    def action_space(self, params):
        return spaces.Box(-1.0, 1.0, (1,))


# The observation becomes [e, I, D, x, r], with e = r - x.
env = PIDTermsWrapper(Setpoint(), current_idx=(0,), target_idx=(1,), dt=1.0)
agent = SAC(
    env,
    env_params=Params(),
    n_envs=4,
    actor_architecture=PIDPolicy(n_errors=1, gains="fixed"),
    critic_architecture=("64", "relu", "64", "relu"),
    actor_learning_rate=3e-3,
    critic_learning_rate=1e-3,
    gamma=0.9,
    learning_starts=500,
)
state, _ = agent.train(seed=[0, 1], n_timesteps=4_000, num_episode_test=4)
gains = pid_gains(state.actor_state.params)  # {"kp", "ki", "kd"}, one value per seed
print({name: value.squeeze() for name, value in gains.items()})
```

The best controller here is `Kp = 1/4, Ki = Kd = 0`, which reaches the
set-point in one step; the repository's tests check that SAC and PPO find it
with longer training.

`PIDPolicy` has two kinds of gains:

- **`gains="fixed"`** makes the gains the actor's only parameters: a linear
  map with no bias and no hidden layer. The trained policy deploys as a
  plain PID with the gains `pid_gains` reads off it, in the agent's units
  (below) (Lawrence et al., 2022, tune PID gains with reinforcement learning
  the same way).
- **`gains="observation"`** predicts the gains from the observation with an
  MLP of `architecture`, as gain scheduling does. Read them for given
  observations with `actor.apply_fn(params, obs, method="gains")`, an array
  of shape `(..., 3, action_dim, n_errors)`.

`n_errors` is the number of errors the wrapper tracks; with several, each
action dimension gets its own gain on each error. The gains start at zero:
the untrained controller commands the centre of the action box, which is
no action on a symmetric box. The stochastic agents explore with a
learned standard deviation that does not depend on the observation,
clipped as the agent's other heads are (`log_std_bounds`: none in PPO and
APO by default, `[-20, 2]` in the others).

Each agent's own output rule applies to `u`. SAC, TD3, REDQ, ASAC and AVG
squash it with `tanh` and APO takes `tanh(u)` as its mean, so the deployed
controller is a saturated PID; APO samples unsquashed around `tanh(u)`,
PPO around `u` (unless `squash=True`), and their samples are clipped to
`[-1, 1]`. Like every action, `u` is in the agent's units and is
then mapped onto the environment's bounds
([Actions](environments.md#actions)): the deployed controller is
`c + h · sat(u)`, with `c` the centre of the action box, `h` its half-width
and `sat` the agent's `tanh` or clip. The learned gains are therefore
normalised: a PID written in the environment's units has gains `h · K`
(for small signals, where `tanh` is close to linear) and the offset `c`.
Observation normalisation leaves the
error terms raw, so the gains act on errors in the environment's units,
while the gain network of `gains="observation"` reads the rest of the
observation normalised.

A PID policy needs a continuous action space and takes no `memory`: its
memory is the integral and the difference in the observation.

Two other PID features are separate from this one. APG's in-context
controller ends with a learnable PID layer on its network's output: see
[APG](../algorithms/apg.md). SAC's `use_pid_policy` lets SAC tune the gains of
an expert controller: see [Expert guidance](expert-guidance.md).
