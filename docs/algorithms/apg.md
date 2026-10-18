# APG

Analytic Policy Gradient trains a deterministic controller by differentiating
the closed-loop return through the simulator itself. Each update runs the
controller for `horizon` steps on a batch of systems and follows the exact
gradient of the summed reward with respect to the controller's parameters,
back through every step of the physics. There is no critic, no replay buffer
and no exploration noise, so it needs an environment whose transitions carry
gradients.

**Papers:** APG is the general method of training a policy by
backpropagation through a differentiable simulator. `APG.contextual_controller`
is the configuration of Busetto, Breschi, Forgione, Piga and Formentin, *One
controller to rule them all*, 2024
([arXiv:2411.06482](https://arxiv.org/abs/2411.06482)): a transformer over the
tracking history closed by a PID layer, trained across a family of systems.

## Can I use?

| Feature | APG |
| --- | --- |
| Discrete actions | no |
| Continuous actions | yes, tanh-bounded by default (`squash=True`), mapped onto the environment's bounds |
| Memory (`memory=`) | yes: GRU, LSTM, transformer, Mamba |
| Environments | gymnax environments with transition gradients: Pendulum, MountainCarContinuous, PointRobot, Reacher, Swimmer |
| Families of systems | yes, `system_class=` |
| Several seeds in one call | yes, `train(seed=[...])` |
| Extension phases | `pretrain`, `post_update`, `eval_metrics`, `actor_loss` |
| Resume and checkpoints | yes; a split run equals the uninterrupted one (probe, and the resume contract with memory) |

Brax and Playground environments are not supported: they take no per-step
parameters and have their own gradient path. An environment that cannot
expose its gradients raises instead of training on zero gradients.

## Example

Train one controller for a family of pendulums whose mass and length vary:

```python
import gymnax
from ajax import APG
from ajax.environments.system_class import UniformPerturbation

_, params = gymnax.make("Pendulum-v1")
systems = UniformPerturbation(params, fields=("m", "l"), scale=0.1)  # +-10 %
agent = APG("Pendulum-v1", n_envs=16, horizon=50, system_class=systems)
state, aux = agent.train(seed=[1, 2], n_timesteps=80_000)
print(aux.matching_loss.shape)     # (seeds, updates): 80,000 / (16 * 50) = 100 updates
print(aux.matching_loss[:, ::20])  # minus the closed-loop return
```

`train` returns `(state, aux)` whatever the logging: `aux` holds every
update's `loss`, `matching_loss` (minus the mean closed-loop return),
`m_rmse` and `timestep`. With a `logging_config`, every `log_frequency` steps
it also evaluates the controller on `num_episode_test` systems, drawn
afresh at each evaluation, and sends `Eval/episodic mean reward` and
`Eval/m_rmse` to the logging backend. `aux.evaluations` then holds what each
update logged, NaN on the updates that did not evaluate, with or without a
backend.

The in-context controller of Busetto et al., trained in stages on a
reference-tracking task (`train_curriculum`, the paper's Algorithm 2):

```python
# doc: skip (compiles a transformer and trains 700,000 steps; tests/agents/APG/test_APG.py covers the curriculum)
import gymnax
import jax.numpy as jnp
from ajax.agents.APG import APG, CurriculumStage, train_curriculum
from ajax.environments.model_reference import LinearReferenceModel, ModelReferenceWrapper, StepReference
from ajax.environments.system_class import FixedSystem, UniformPerturbation
from ajax.wrappers import InitialStateWrapper

plant, params = gymnax.make("Pendulum-v1")
fixed_start = lambda key, s, p: s.replace(theta=jnp.asarray(0.0), theta_dot=jnp.asarray(0.0))
plant = InitialStateWrapper(plant, fixed_start)  # the paper keeps the initial conditions fixed
task = ModelReferenceWrapper(  # track a reference model's step response
    plant,
    StepReference(horizon=100, min_value=-0.5, max_value=0.5, min_duration=20, max_duration=50),
    LinearReferenceModel.first_order(a=0.9, b=0.1, c=1.0, d=0.0),
    output_fn=lambda obs: jnp.arctan2(obs[1], obs[0]),
)
systems = UniformPerturbation(params, fields=("m", "l"), scale=0.05)
stages = [  # the nominal pendulum first, then the whole family
    CurriculumStage(APG.contextual_controller(task, FixedSystem(params), env_params=params), n_timesteps=200_000),
    CurriculumStage(APG.contextual_controller(task, systems, env_params=params), n_timesteps=500_000),
]
*_, (state, aux) = train_curriculum(stages, seed=0)  # one (state, aux) per stage: keep the last
```

## Key hyperparameters

| Argument | Default | What it does |
| --- | --- | --- |
| `n_envs` | 8 | systems drawn and simulated in parallel per update |
| `horizon` | 100 | closed-loop steps differentiated through |
| `learning_rate` | 5e-4 | AdamW step size |
| `weight_decay` | 0.0 | AdamW weight decay (0 is plain Adam) |
| `beta_1`, `beta_2` | 0.9, 0.999 | Adam's moment rates (its epsilon is torch's 1e-8) |
| `lr_schedule` | None | `"warmup_cosine"`: linear warmup over `warmup_steps` updates, then cosine decay |
| `warmup_steps`, `lr_end_fraction` | 0, 0.1 | warmup length, and the final rate as a fraction of `learning_rate` |
| `max_grad_norm` | 1.0 | gradient clipping by global norm (`None` turns it off) |
| `actor_architecture` | `("128", "relu", "128", "relu")` | MLP in front of the memory or output; `()` for a linear embedding |
| `memory` | None | memory block (`MemoryConfig` or a dict) |
| `pid` | None | `PIDHeadConfig`: close the controller with a learnable PID layer |
| `squash` | True | bound actions to [-1, 1] with tanh; without it they are clipped there |
| `system_class` | None | distribution over the environment's parameters; `None` trains on `env_params` only |
| `controller_factory` | None | builds a custom controller module (see `Controller` in `ajax.agents.APG.networks`) |
| `episode_length`, `action_repeat` | None, 1 | accepted as by every agent; on gymnax environments, the only ones APG takes, the episode length is the environment's and the repeat must be 1 |

`APG.contextual_controller(env, system_class, ...)` sets the paper's values:
8 layers, 4 heads, width 128, context 100, one system per update
(`n_envs=1`), horizon 100, betas (0.9, 0.95) with `weight_decay=0` (plain
Adam), 5,000 warmup updates then cosine decay to a tenth of the rate, no
gradient clipping (`max_grad_norm=None`), and a PID output layer. Its `final_norm=False` rebuilds a controller saved before
the transformer ended with its final LayerNorm (see
[Memory](../guide/memory.md)). Such a controller's parameters also map onto
the new network, which computes the same outputs:
`controller_params_with_final_norm(params, memory)` moves the controller's
LayerNorm, which played that final norm's part, into the transformer. It maps
the parameters only, so a resumed training state must restart its optimizer.

## Notes

- **Budget.** One update uses `n_envs * horizon` steps of `n_timesteps`.
- **Actions.** Squashed or clipped, the controller's `[-1, 1]` action is
  mapped onto the environment's bounds inside the differentiated rollout, so
  on gymnax Pendulum (torque in `[-2, 2]`) the controller reaches the whole
  range.
- **Deviations from Busetto et al.**, each backed by runs on the pendulum
  tracking task (see the `APG` docstrings): actions are tanh-bounded (the
  reference leaves them unbounded); gradients are clipped, since BPTT over 100
  steps across a family of systems diverged without it; the PID layer starts
  as the identity (`K_P = 1`, `K_I = K_D = 0`) instead of all three gains at 1,
  which trained markedly worse under the bounded action (fresh-system M-RMSE
  0.33 against 0.25). These runs predate the mapping of actions onto the
  environment's bounds: the pendulum's torque was then capped at 1 of its
  2. Not backed by runs: the transformer's LayerNorms, its final one
  included, keep a bias and an epsilon of 1e-6, where the reference's have
  no bias and 1e-5 (buswayne/in-context-controller at 7eb094ce17:
  `transformer_onestep.py:33` and `:156`, `train_onestep_lin_CL.py:103`).
  The paper trains each curriculum stage to a loss tolerance; Ajax gives
  each stage a fixed `n_timesteps`.
- **Initialisation.** A transformer memory starts as the reference's GPT:
  kernels from N(0, 0.02²), the residual projections at 0.02/√(2 `n_layers`),
  zero biases (`transformer_onestep.py:162-166`, `:189-195`;
  `MemoryConfig(init="nanogpt")`). Other memory kinds keep flax's.
- **Curriculum.** `train_curriculum` resumes each stage from the previous
  one's parameters and, by default, restarts the optimizer and its schedule
  at each stage. All stages must share `n_envs`, the network and the PID
  layer.
- **Tasks.** `ajax.environments.model_reference` turns a plant into a
  reference-tracking task: `ModelReferenceWrapper`, a `StepReference` signal
  and a `LinearReferenceModel` whose response the plant must match. On such a
  task the loss is the paper's closed-loop matching cost and `m_rmse` its
  M-RMSE.
- [Differentiable simulation](../guide/differentiable-simulation.md) covers
  system classes, transition gradients and the reference-tracking wrapper.

## Results

Unit tests (`tests/agents/APG`) check its pieces against exact contracts,
without learning, including the curriculum. The probes (`tests/probing`)
train it briefly on tiny environments with known answers: it reaches the
exact optimum of a small problem whose best action is reachable only
through the transition's gradient, and a run split by a checkpoint equals
the run left whole. On the CPU speed benchmark
(`benchmarks/agent_bench.py`: Pendulum tracking, 4 environments, 8,000
steps) APG ran at 34,958 steps per second of execution (5,339 with
compilation) at main `a8a13c2`
([PERFORMANCE_REPORT.md](https://github.com/YannBerthelot/Ajax/blob/main/PERFORMANCE_REPORT.md)).

## API

::: ajax.agents.APG.APG.APG

::: ajax.agents.APG.curriculum.CurriculumStage

::: ajax.agents.APG.curriculum.train_curriculum

::: ajax.agents.APG.networks.PIDHeadConfig

::: ajax.agents.APG.networks.Controller

::: ajax.agents.APG.networks.controller_params_with_final_norm
