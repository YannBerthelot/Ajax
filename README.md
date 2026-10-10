# Ajax: reinforcement-learning agents in JAX

Ajax is a library of reinforcement-learning agents written in [JAX](https://github.com/jax-ml/jax),
each following a published paper. A whole training run, environment included, compiles into one
program, so several seeds train side by side on one CPU or GPU. Research features (learning from an
expert, exploration rules, diagnostics) plug into any agent as *extensions*, not agent options.

## Agents

| Agent | Paper | Actions | Memory |
| --- | --- | --- | --- |
| `SAC` | Haarnoja et al., *Soft Actor-Critic*, 2018 ([arXiv:1801.01290](https://arxiv.org/abs/1801.01290)) | continuous | yes |
| `TD3` | Fujimoto et al., *Addressing Function Approximation Error in Actor-Critic Methods*, 2018 ([arXiv:1802.09477](https://arxiv.org/abs/1802.09477)) | continuous | yes |
| `REDQ` | Chen et al., *Randomized Ensembled Double Q-Learning*, 2021 ([arXiv:2101.05982](https://arxiv.org/abs/2101.05982)) | continuous | yes |
| `ASAC` | Adamczyk et al., *Average-Reward Soft Actor-Critic*, 2025 ([arXiv:2501.09080v2](https://arxiv.org/abs/2501.09080v2)) | continuous | yes |
| `AVG` | Vasan et al., *Deep Policy Gradient Methods Without Batch Updates, Target Networks, or Replay Buffers*, 2024 ([arXiv:2411.15370](https://arxiv.org/abs/2411.15370)) | continuous | no |
| `PPO` | Schulman et al., *Proximal Policy Optimization Algorithms*, 2017 ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347)) | both | yes |
| `APO` | Ma et al., *Average-Reward Reinforcement Learning with Trust Region Methods*, 2021 ([arXiv:2106.03442](https://arxiv.org/abs/2106.03442)) | both | no |
| `DQN` | Mnih et al., *Human-level control through deep reinforcement learning*, Nature 2015 | discrete | no |
| `PQN` | Gallici et al., *Simplifying Deep Temporal Difference Learning*, 2024 | discrete | no |
| `UDRL` | Schmidhuber, *Reinforcement Learning Upside Down*, 2019 ([arXiv:1912.02875](https://arxiv.org/abs/1912.02875)) | both | no |
| `APG` | Analytic policy gradient through a differentiable simulator; `APG.contextual_controller` is Busetto et al., *One controller to rule them all*, 2024 ([arXiv:2411.06482](https://arxiv.org/abs/2411.06482)) | continuous | yes |
| `DreamerV3` | Hafner et al., *Mastering Diverse Domains through World Models*, 2023, Nature 2025 ([arXiv:2301.04104v2](https://arxiv.org/abs/2301.04104v2)) | both | built in |
| `TDMPC2` | Hansen, Su & Wang, *TD-MPC2: Scalable, Robust World Models for Continuous Control*, ICLR 2024 ([arXiv:2310.16828](https://arxiv.org/abs/2310.16828)) | continuous | no |
| `TDMPC2MultiTask` | The same paper's multi-task agent: one model trained offline on the data of several tasks | continuous | no |

*Memory*: whether the agent accepts `memory=` (see [Memory](#memory)); DreamerV3's world model is
recurrent by design. `TD3` and `UDRL` are not exported from `ajax`: import them from
`ajax.agents.TD3.TD3` and `ajax.agents.UDRL.UDRL`. DreamerV3 and TD-MPC2 follow the paper-era
official code; versions, specifications and deviations are in [docs/world_models](docs/world_models/README.md).

## Environments

An agent takes an environment id or a prebuilt environment. Ids are looked up in [gymnax](https://github.com/RobertTLange/gymnax)
(`"Pendulum-v1"`), then [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
(`"CheetahRun"`), then [Brax](https://github.com/google/brax) (`"ant"`); `n_envs` copies run in
parallel. To build one differently, pass the agent the result of
`ajax.environments.create.build_env_from_id`: `fresh_reset=False` (Playground) restarts each copy
from the same first state, as older Ajax runs did; `differentiable_reset=True` (Brax, Playground)
lets `jax.grad` flow through the reset itself (say, to physics parameters), at one reset per step.

## Install

Ajax uses [uv](https://docs.astral.sh/uv/) and Python 3.11 to 3.13. On Apple-silicon Macs it
installs JAX for CPU; elsewhere `jax[cuda]`, which falls back to the CPU when there is no GPU.

```bash
git clone https://github.com/YannBerthelot/Ajax.git
cd Ajax
uv sync   # creates .venv with the locked dependencies; activate it or prefix commands with `uv run`
```

## Quickstart

```python
from ajax import SAC
from ajax.logging.wandb_logging import LoggingConfig

agent = SAC("Pendulum-v1")
# Evaluate every 5,000 steps; use_wandb=False keeps the results in memory only.
logging = LoggingConfig(config={}, log_frequency=5_000, use_wandb=False)
state, evaluations = agent.train(seed=[1, 2, 3], n_timesteps=50_000, logging_config=logging)

print(evaluations["timestep"][0])                # when each evaluation ran
print(evaluations["Eval/episodic mean reward"])  # one row per seed, one column per evaluation
```

`evaluations` maps each logged name to an array of shape `(seeds, evaluations)`;
`"Eval/episodic mean reward"` is the mean return of `num_episode_test` (default 10) test episodes.
Without `logging_config`, `train` does not evaluate and returns `(state, None)` (APG returns every
update's metrics instead). With `use_wandb=True` (the default) or `use_tensorboard=True`, the
records also go to Weights & Biases or TensorBoard.

## Extensions

An extension adds a feature to an agent's training without changing the agent: an extra loss term,
another way to pick actions while collecting, an expert to learn from, extra measurements. Pass
them as a tuple, `extensions=`; each holds its own settings. [CONTRIBUTING.md](CONTRIBUTING.md) shows how to write one.

```python
from ajax import PPO
from ajax.extensions.instrumentation import ConditioningMetrics
from ajax.logging.wandb_logging import LoggingConfig

# Measures the critic's health (effective rank, dormant units, norms) at each evaluation, on the
# latest rollout for an on-policy agent: expose_recent_rollout=True keeps it.
agent = PPO("CartPole-v1", expose_recent_rollout=True, extensions=(ConditioningMetrics(),))
logging = LoggingConfig(config={}, log_frequency=8_192, use_wandb=False)
state, evaluations = agent.train(seed=[1, 2], n_timesteps=32_768, logging_config=logging)
print(evaluations["Cond/critic_srank"])
```

| Module (`ajax.extensions.`) | Extensions | What for |
| --- | --- | --- |
| `expert` | `ExpertGuidance`, `ExpertObsAugmentation`, `OnlineBC`, `ImitationLoss`, `ResidualPolicy`, `JSRLCurriculum` | learning from a fixed expert policy |
| `exploration` | `EDGEExploration` | letting the expert act during collection while it looks better |
| `target_mods` | `IBRL`, `LCBGatedBootstrap`, `CriticBlend`, `MCVarianceCorrection`, `ValueBox` | using the expert in the critic's target (`ValueBox`: in the action) |
| `pretrain` | `MCPretrain`, `BellmanPretrain`, `PhiRefresh` | pre-training a critic on expert data |
| `ensemble` | `KernelRepulsion` | keeping an ensemble's critics apart |
| `instrumentation` | `ConditioningMetrics`, `BiasVoreDecomposition`, `DiagnosticSnapshots`, `BiasVorePenalty` | measuring the critic (and one critic-loss term) |

## Memory

When the current observation is not enough (a hidden velocity, a cue seen earlier), the agent must
remember. `memory=` gives `PPO`, `SAC`, `TD3`, `REDQ`, `ASAC` and `APG` a memory block before their
output heads; carrying it, resetting it at episode ends and replaying it in training are automatic.

```python
from ajax import PPO, SAC
from ajax.networks.memory import MemoryConfig

agent = PPO("CartPole-v1", memory=MemoryConfig(kind="gru", hidden_size=64))
agent = SAC("Pendulum-v1", memory={"kind": "lstm", "hidden_size": 64})  # a dict works too
```

`kind` is `"gru"`, `"lstm"`, `"transformer"` (attention over the episode's last `window` steps) or
`"mamba"` (a selective state-space model). PPO trains on whole rollouts or on `bptt_length`-step
chunks. Replay agents train on stored sequences, as in R2D2 (Kapturowski et al. 2019): `burn_in`
steps warm the memory up, then `sequence_length` steps are learned from; `stored_state=True`
starts each sequence from the memory the actor had when it collected it.

## Differentiable simulation: APG

`APG` has no critic and no replay buffer: it runs the controller in closed loop and follows the
gradient of the return through the simulator, so it needs a gymnax environment that exposes those
gradients (Pendulum, MountainCarContinuous, PointRobot, Reacher, Swimmer), and can train across a
*system class*, a distribution over the environment's physical parameters. `APG.contextual_controller`
is the transformer-plus-PID controller of Busetto et al. 2024; `train_curriculum` runs the paper's
staged training (Algorithm 2) on a reference-tracking task:

```python
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
    LinearReferenceModel.first_order(a=0.9, b=0.1, c=1.0, d=0.0),  # slower than the paper's M: 50 ms steps
    output_fn=lambda obs: jnp.arctan2(obs[1], obs[0]),
)
systems = UniformPerturbation(params, fields=("m", "l"), scale=0.05)
stages = [  # the nominal system first, then the whole class
    CurriculumStage(APG.contextual_controller(task, FixedSystem(params), env_params=params), n_timesteps=200_000),
    CurriculumStage(APG.contextual_controller(task, systems, env_params=params), n_timesteps=500_000),
]
(state, aux), *_ = train_curriculum(stages, seed=0)
```

## How training is organised

The replay agents, the on-policy agents and the world models run on one shared loop, `TrainLoop`
in [src/ajax/agents/loop.py](src/ajax/agents/loop.py). An agent supplies only its algorithm (how to
initialise its state and how to update it); the loop does the rest the same way for all: collecting
experience, waiting for `learning_starts`, applying the extensions, evaluating and logging. APG,
UDRL and the offline `TDMPC2MultiTask`, whose iterations differ, reuse the loop's pieces. `train`
runs the seeds side by side with `jax.vmap`: every array in the returned state has a seed axis first.
To continue a run, pass its state back: `agent.train(seed=seeds, n_timesteps=more, initial_state=state)`
trains `more` further steps. `ajax.checkpoint.save_checkpoint` and `restore_into` save a state and
load it into a new agent's 0-step state, so another process can resume; the probing tests check
that a run split this way equals the run left whole.

## Testing

*Unit tests* check pieces in isolation (losses, buffers, networks, wrappers, logging) and that every
agent and extension runs. *Probing tests* (`tests/probing`, older ones in `tests/agents/test_probing.py`)
check that agents learn the right thing: each trains an agent on a tiny problem whose answer is known
exactly (a value, a best action, a count) and tells it apart from named wrong answers, such as a
discount applied twice.

```bash
uv run pytest                  # every test
uv run pytest -m "not slow"    # skip the long trainings
uv run pytest tests/probing    # the probing tests
make ci                        # what CI runs
```

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs on CPU for every pull request and
push to `main`: pre-commit (ruff, mypy), the tests not marked slow with at least 70% coverage, the
slow tests, and the probing tests. A pull request merges only when all of them pass.

## Project layout

```
src/ajax/
├── agents/        one folder per agent: <AGENT>.py (the class), train_<AGENT>.py (the algorithm);
│                  base.py (ActorCritic, train), loop.py (TrainLoop), recurrent.py (sequence replay)
├── extensions/    the Extension base class and the extensions above
├── environments/  environment creation, collection, differentiable rollouts, system classes
├── networks/      actor and critic networks, memory blocks (modules/: learnable PID layers)
├── buffers/, logging/   replay buffers (flashbax); Weights & Biases and TensorBoard logging
└── checkpoint.py, evaluate.py, log.py, wrappers.py   checkpoints, evaluation, logged metrics, env wrappers
tests/             unit tests; tests/probing holds the probing tests
benchmarks/        per-agent speed benchmark (agent_bench.py) and recorded baselines
```

## Contributing

Run `uv run pre-commit install` once after `uv sync`. [CONTRIBUTING.md](CONTRIBUTING.md) covers adding
an agent or an extension; [CLAUDE.md](CLAUDE.md) lists the checks every commit must pass.

## License and citation

MIT, see [LICENSE](LICENSE). To cite Ajax:

```bibtex
@misc{ajax2025,
  title        = {Ajax: Reinforcement Learning Agents in Jax},
  author       = {Yann Berthelot},
  year         = {2025},
  url          = {https://github.com/YannBerthelot/Ajax},
}
```
