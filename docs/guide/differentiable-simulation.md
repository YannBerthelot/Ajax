# Differentiable simulation

Reinforcement learning usually treats the environment as a black box: the
agent sees rewards, never how they depend on its actions, and estimates that
dependence from many trials. When the simulator is written in JAX, the
dependence can be computed exactly instead. Run the controller in closed
loop, sum the rewards, and `jax.grad` gives the gradient of that sum with
respect to the controller's parameters, back through every step of the
physics.

This is the *analytic policy gradient*. It works well when the dynamics are
smooth and the horizon is short to moderate: there is no critic to learn and
no noise to average out, so each update is precise. It works poorly through
contacts and other discontinuities, and over long horizons, where gradients
can explode or vanish.

Ajax supports it at three levels: environments whose transitions keep their
gradients, families of systems and tracking tasks to train on, and the
[APG](../algorithms/apg.md) agent, which includes the in-context controller
of Busetto et al., 2024.

## Differentiable environments

Ajax's collectors cut gradients at every environment step, as reinforcement
learning expects. gymnax (from version 1.0) also detaches its transitions by
default; a few environments let you switch that back on.
`with_transition_gradients(env)` does so, through any Ajax wrappers around
the environment, and raises for an environment that cannot, instead of
silently returning zero gradients. Today these are Pendulum,
MountainCarContinuous, PointRobot, Reacher and Swimmer.

`closed_loop_rollout` runs a policy against such an environment for a fixed
number of steps with no gradient cut anywhere. The policy is any function
`(carry, obs, resets) -> (action, carry)`, where `carry` is its memory
(`None` for a memoryless one) and `resets` marks episode starts. Here, plain
gradient ascent tunes the two gains of a PD controller over a family of
pendulums:

```python
import gymnax
import jax
import jax.numpy as jnp
from ajax.environments.differentiable import closed_loop_rollout, with_transition_gradients
from ajax.environments.system_class import UniformPerturbation

env, params = gymnax.make("Pendulum-v1")
env = with_transition_gradients(env)
systems = UniformPerturbation(params, fields=("m", "l"), scale=0.2)  # mass, length +-20 %


def mean_return(gains, key):
    def pd(carry, obs, resets):
        theta = jnp.arctan2(obs[:, 1], obs[:, 0])
        torque = -(gains[0] * theta + gains[1] * obs[:, 2])
        return jnp.clip(torque, -2.0, 2.0)[:, None], carry

    system_key, rollout_key = jax.random.split(key)
    batch = systems.sample(system_key, 16)  # 16 pendulums, one per parallel env
    rollout, _ = closed_loop_rollout(pd, None, rollout_key, env, batch, horizon=50)
    return rollout.reward.sum(axis=0).mean()  # rollout leaves are (horizon, envs, ...)


gains = jnp.array([1.0, 0.1])
step = jax.jit(jax.value_and_grad(mean_return))
for i in range(20):
    value, grad = step(gains, jax.random.PRNGKey(i))
    gains = gains + 1e-3 * grad  # ascend the return
print(value, gains)
```

`closed_loop_rollout` hands the policy's actions to the environment as they
are, so the policy above returns torques in the environment's units; APG
maps its controller's `[-1, 1]` actions onto the bounds itself (below).

This path is gymnax only: Brax and MuJoCo Playground environments take no
per-step parameters and have their own gradient machinery. For those,
`differentiable_reset=True` in `build_env_from_id` lets gradients flow
through the automatic reset as well (see [Environments](environments.md)).

## Families of systems

A controller tuned for one plant may fail on its neighbours. A *system class*
is a distribution over an environment's physical parameters, so training can
cover a whole family:

- `UniformPerturbation(nominal, fields, scale)` draws each listed parameter as
  `p · (1 + U(−scale, scale))`, independently, leaving the others nominal;
  a vector or matrix field is scaled by one factor per system. Busetto et
  al. build their meta-dataset this way, with `scale=0.05`. A misspelt
  field raises when the class is built.
- `FixedSystem(nominal)` always returns the nominal system; it is the first
  stage of a curriculum.
- Subclass `SystemClass` and implement `sample(rng, n)` for anything else.

`sample(rng, n)` returns parameters with a leading axis of `n`, one system
per parallel environment. Ajax's gymnax `reset` and `step` notice that axis
and simulate each environment with its own system.

## Tracking tasks: model-reference control

In model-reference control, the controller makes a plant's measured output
`y` follow the output of a *reference model* driven by a reference signal
`r`: say, a first-order lag towards each new setpoint. Three pieces turn any
gymnax environment into such a task:

- `StepReference(horizon, min_value, max_value, min_duration, max_duration)`
  draws piecewise-constant references: steps of random height held for a
  random number of steps.
- `LinearReferenceModel` is the reference model, `x' = A x + B r`,
  `y_d = C x + D r`. `LinearReferenceModel.first_order()` defaults to the
  paper's model.
- `ModelReferenceWrapper(env, reference, model, output_fn=...)` is the task
  as the controller sees it: the observation is the tracking error and the
  previous action, `[e, u_prev]` (optionally followed by the plant's own
  observation), and the reward is `−‖y_d − y‖²`. The undiscounted return is
  minus the matching cost, and `sqrt(−return / horizon)` is the paper's
  M-RMSE. The error is computed from the observation the plant returns, so
  sensor noise on the plant (an
  [`ObservationNoiseWrapper`](environments.md#sensor-noise) under the task)
  reaches the controller and the reward alike.

All three are plain `jnp`, so the task stays differentiable when the plant
is. `ajax.wrappers.InitialStateWrapper(env, fn)` sets each episode's initial
state, for tasks that fix it, as the paper does.

## Training a controller with APG

`APG` puts these together. Each update samples `n_envs` systems from the
system class, runs the controller on them for `horizon` steps, and takes one
optimiser step on minus the mean return. One update consumes
`n_envs · horizon` of `n_timesteps`. The controller is deterministic and its
actions are bounded with `tanh` (`squash=True`), then mapped onto the
environment's bounds, as every agent's are
([Actions](environments.md#actions)); an unsquashed controller is clipped to
`[-1, 1]` first. The map is differentiable, so gradients flow through it.

A controller made of a small MLP and a learnable PID output layer, trained to
track step references across pendulums whose mass and length vary by 30 %:

```python
import gymnax
import jax.numpy as jnp
from ajax import APG
from ajax.agents.APG import PIDHeadConfig
from ajax.environments.model_reference import (
    LinearReferenceModel,
    ModelReferenceWrapper,
    StepReference,
)
from ajax.environments.system_class import UniformPerturbation
from ajax.wrappers import InitialStateWrapper

plant, params = gymnax.make("Pendulum-v1")
params = params.replace(g=0.0)  # no gravity: a pure inertia to steer


def at_rest(key, state, params):  # every episode starts at angle 0, at rest
    return state.replace(theta=jnp.asarray(0.0), theta_dot=jnp.asarray(0.0))


task = ModelReferenceWrapper(
    InitialStateWrapper(plant, at_rest),
    StepReference(horizon=100, min_value=-0.5, max_value=0.5, min_duration=20, max_duration=50),
    LinearReferenceModel.first_order(a=0.9, b=0.1, c=1.0, d=0.0),
    output_fn=lambda obs: jnp.arctan2(obs[1], obs[0]),  # the measured output: the angle
)
systems = UniformPerturbation(params, fields=("m", "l"), scale=0.3)

agent = APG(
    task,
    system_class=systems,
    n_envs=8,
    horizon=100,
    actor_architecture=("64", "relu"),
    pid=PIDHeadConfig(),  # close the controller with a learnable PID layer
)
state, aux = agent.train(seed=0, n_timesteps=8 * 100 * 200)  # 200 updates
print(aux.m_rmse[0, 0], aux.m_rmse[0, -1])  # (seeds, updates): falls from about 1.7 to 0.45
```

`train` always returns every update's `loss`, `matching_loss`, `m_rmse` and
`timestep`, with a seed axis first. With a `logging_config`, it also
evaluates on freshly drawn systems every `log_frequency` steps. The
[APG page](../algorithms/apg.md) lists every hyperparameter.

Two defaults depart from a plain implementation, both for stability on this
test bed: gradients are clipped to a global norm of 1 (`max_grad_norm`;
without it, training across a system class diverged), and actions are
`tanh`-bounded.

## The in-context controller

Busetto, Breschi, Forgione, Piga and Formentin, *One controller to rule them
all*, 2024 ([arXiv:2411.06482](https://arxiv.org/abs/2411.06482)), train one
controller for a whole class of systems. It never learns which system it
faces explicitly; it infers it from the history of tracking errors and
inputs it has seen in the current episode, as a transformer reads its
context. `APG.contextual_controller(task, system_class)` builds that
configuration:

- a decoder-only transformer over the episode's history (8 layers, 4 heads,
  width 128, context 100), with a linear input embedding
  (`actor_architecture=()`), closed by a PID output layer;
- one system per update (`n_envs=1`) and a horizon of 100;
- Adam with betas (0.9, 0.95) (AdamW once `weight_decay` is above its
  default of 0), a 5,000-update linear warm-up, then cosine decay to a
  tenth of the learning rate.

Ajax departs from the reference code in three places, each backed by runs on
the pendulum tracking test bed: actions are `tanh`-bounded, gradients are
clipped, and the PID layer starts as the identity (gains 1, 0, 0) rather than
all ones. Those runs predate the map of actions onto the environment's
bounds, when the pendulum's torque was capped at 1 of its 2. A fourth
difference is not backed by runs: the transformer's LayerNorms keep a bias
and an epsilon of 1e-6, where the reference's have no bias and 1e-5
(buswayne/in-context-controller at 7eb094ce17: `transformer_onestep.py:33`
and `:156`, `train_onestep_lin_CL.py:103`).

The paper trains in stages of growing diversity (its Algorithm 2): one system
first, then the class. `train_curriculum` chains `CurriculumStage(agent,
n_timesteps)` objects through Ajax's resume path, carrying the parameters
over and, by default, restarting the optimiser and its schedule at each
stage (`reset_optimizer=True`). The stages must share `n_envs`, the network
and the output head; each may have its own task and system class. The paper
trains each stage to a loss tolerance; Ajax gives each a fixed budget.

```python
# doc: skip (compiles an 8-layer transformer and trains 700,000 steps)
import gymnax
from ajax import APG
from ajax.agents.APG import CurriculumStage, train_curriculum
from ajax.environments.system_class import FixedSystem, UniformPerturbation

# `task` as in the APG example above
_, params = gymnax.make("Pendulum-v1")
systems = UniformPerturbation(params, fields=("m", "l"), scale=0.05)
stages = [
    CurriculumStage(APG.contextual_controller(task, FixedSystem(params)), n_timesteps=200_000),
    CurriculumStage(APG.contextual_controller(task, systems), n_timesteps=500_000),
]
results = train_curriculum(stages, seed=0)  # one (state, aux) per stage
```

## The PID output layer

`PIDOutputHead` closes a network with a learnable PID. The network emits a
signal `z_k`, and the layer applies

`u_k = K_P z_k + K_I Σ_{j≤k} z_j + K_D (z_k − z_{k−1})`

with learnable gains, one triple per action dimension. The integral term
helps remove steady-state error on step references, and Busetto et al.
report faster training with it. Like a memory block, the layer carries a
state (the running sum and the previous signal), reset at episode starts.

`PIDHeadConfig` sets it up for APG: `use_p`, `use_i`, `use_d` switch each
term on (a disabled term has no parameter); `kp_init=1`, `ki_init=0`,
`kd_init=0` start it as the identity; `anti_windup` stops the integral from
growing while the output is beyond that magnitude in the same direction
(about 3 suits a `tanh` bound, where `tanh` is within 0.5 % of its limit).

Ajax has two PID components, which do different jobs:

| | PID output layer (`pid=PIDHeadConfig()`) | PID policy ([Networks](networks.md)) |
| --- | --- | --- |
| Acts on | the network's own output signal | the tracking-error terms in the observation |
| Role | the last layer of a larger controller | the whole actor |
| Agents | APG | SAC, TD3, REDQ, ASAC, AVG, PPO, APO |
| Deploys as | the network plus the layer | a plain PID with the learned gains |

## Your own controller

`APG(controller_factory=...)` trains any Flax module you build, given the
agent's context (`env_args`, `network_args`, `action_dim`, `pid`, `squash`).
The module must follow the `Controller` contract in
[`ajax/agents/APG/networks.py`](../../src/ajax/agents/APG/networks.py): a
static `stateful` attribute, `initialize_carry(rng, batch_size)` when
stateful, and a call that returns a deterministic `distrax` distribution
(plus the new carry when stateful). This lets downstream projects train
their own architectures with the same trainer.

APG also takes extensions in the `actor_loss` phase, whose dictionary holds
the `rollout`, the controller's `actor_params` and the per-system
`returns`, plus the phases every agent runs (`pretrain`, `post_update`,
`eval_metrics`). See [Extensions](extensions.md).

## See also

- [APG](../algorithms/apg.md): hyperparameters, notes and results.
- [Environments](environments.md): building environments and wrappers.
- [`ajax/environments/differentiable.py`](../../src/ajax/environments/differentiable.py),
  [`system_class.py`](../../src/ajax/environments/system_class.py) and
  [`model_reference.py`](../../src/ajax/environments/model_reference.py):
  the source, with the paper's equations in the docstrings.
