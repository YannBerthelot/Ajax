# Agents in JAX (AJAX): A JAX-Based Library for Modular and Efficient RL Agents

AJAX is a high-performance reinforcement learning library built entirely on **JAX**. It provides a modular, composable framework for implementing and training RL agents, enabling **massive speedups** for parallel experiments on **GPUs / TPUs**.

---

## Features

| Feature                               | AJAX               |
| ------------------------------------- | ------------------ |
| End-to-end JAX implementation         | :heavy_check_mark: |
| Composable hook API (no flag soup)    | :heavy_check_mark: |
| GPU / TPU acceleration                | :heavy_check_mark: |
| TensorBoard + Weights & Biases        | :heavy_check_mark: |
| Truncation / termination handling     | :heavy_check_mark: |
| Recurrent network support             | :soon:             |

### Available Agents

| Agent | Paper |
| ----- | ----- |
| **SAC**   | Haarnoja et al., *Soft Actor-Critic*, 2018 — [arXiv:1801.01290](https://arxiv.org/abs/1801.01290) |
| **ASAC**  | Adamczyk et al., *Average-Reward Soft Actor-Critic*, 2025 — [arXiv:2501.09080](https://arxiv.org/pdf/2501.09080v2) |
| **REDQ**  | Chen et al., *Randomized Ensembled Double Q-Learning*, 2021 — [arXiv:2101.05982](https://arxiv.org/abs/2101.05982) |
| **AVG**   | Vasan et al., *Deep Policy Gradient Methods Without Batch Updates, Target Networks, or Replay Buffers*, 2024 — [arXiv:2411.15370](https://arxiv.org/abs/2411.15370) |
| **PPO**   | Schulman et al., *Proximal Policy Optimization*, 2017 — [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) |
| **APO**   | Ma et al., *Average-Reward Reinforcement Learning with Trust Region Methods*, 2021 — [arXiv:2106.03442](https://arxiv.org/abs/2106.03442) |

### Environment Compatibility
- **Gymnax**, **Brax**, and **MuJoCo Playground** (with full termination vs truncation handling).
- Parallel environments via `n_envs`.
- Env lookup is by id: a gymnax id (e.g. `"Pendulum-v1"`) routes to gymnax, a playground id (e.g. `"HopperHop"`, `"CheetahRun"`, `"Go1JoystickFlatTerrain"`) routes to playground, and a brax id (e.g. `"ant"`, `"halfcheetah"`, `"humanoid"`) routes to brax. Brax and playground have disjoint env sets — both backends are kept side-by-side rather than one superseding the other.
- Terminal observations on truncation are preserved in `state.info["final_obs"]` via an Ajax-supplied `FinalObsWrapper`, so PPO/SAC value bootstrap is correct at time-limit truncations.

### Replay Buffer
- Trajectory storage and sampling via **flashbax**.

### Optimizations
- Memory-efficient updates using `donate_argnums`.
- JIT compilation with static hook callables — one compilation per unique feature configuration.

---

## Installation

```bash
git clone https://github.com/YannBerthelot/Ajax.git
cd Ajax
poetry install
poetry shell
```

Poetry is required. Install it via `curl -sSL https://install.python-poetry.org | python3 -` if needed.

---

## Quickstart

```python
from ajax import SAC

agent = SAC(env_id="Pendulum-v1", n_envs=1)
agent.train(seed=[1, 2, 3], n_timesteps=int(1e6))
```

Every agent accepts the same base arguments (`env_id`, `n_envs`, `gamma`, architectures, …) plus agent-specific hyperparameters.

### Composable hooks

Agents expose `Optional[Callable]` hooks that let you override behavior without subclassing. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full list and semantics.

```python
from ajax import SAC
from ajax.agents.SAC.train_SAC import make_action_pipeline

pipeline = make_action_pipeline(
    expert_policy=my_expert,
    recurrent=False,
    env_args=env_args,
    use_expert_guided_exploration=True,
    total_timesteps=1_000_000,
)
agent = SAC(env_id="Pendulum-v1", action_pipeline=pipeline)
```

---

## Project Structure

```
src/ajax/
├── agents/
│   ├── base.py              # Shared ActorCritic base class
│   ├── cloning.py           # Behavioral-cloning utilities (actor + critic pretrain)
│   ├── SAC/, ASAC/, REDQ/, AVG/, PPO/, APO/
│   │   ├── <AGENT>.py       # Public class (config, __init__, get_make_train)
│   │   ├── train_<AGENT>.py # make_train, update steps, loss functions
│   │   └── state.py         # Agent-specific flax.struct.dataclass state
├── buffers/                 # flashbax-based replay buffer helpers
├── environments/            # Env creation, interaction loops, collect_experience
├── logging/                 # wandb / tensorboard logging
├── modules/                 # Composable pieces (expert, exploration, pretrain, pid_actor)
├── networks/                # Actor / Critic / ScannedRNN
├── state.py                 # Shared config dataclasses
├── wrappers.py              # Env wrappers (AutoReset, Normalize, Noise, …)
├── evaluate.py, log.py      # Eval loop and metric logging
└── schedule.py              # Scalar schedules (constant, linear, exponential, polynomial)

tests/                       # Unit + probing tests (see tests/agents/test_probing.py)
```

Top-level scripts (experiment runners; see [pipeline.py](pipeline.py)):
- `pipeline.py` — orchestrates hyperparameter search → ablation → noise study → plots.
- `gpu_launcher.py` — launches experiments one-per-GPU.
- `sac_hyperparam_search.py` — TPE-based SAC hyperparameter search.
- `ablation_study.py`, `noisy_expert_study.py` — research experiments on the `Plane` env.
- `plot_sweep.py` — plotting utilities.
- `task_configs.py` — per-task pipeline config (currently `Plane`).

---

## Running Tests

```bash
poetry run pytest                                          # all tests
poetry run pytest tests/agents/test_probing.py             # cross-agent behavioral tests
poetry run pytest tests/modules/test_hook_composition.py   # hook API contract
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for how the test suite is structured.

---

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Adding a new agent
- Adding a new composable module (hook)
- Style / CI requirements

---

## License

MIT. See [LICENSE](LICENSE).

---

## Citation

```bibtex
@misc{ajax2025,
  title        = {Ajax: Reinforcement Learning Agents in Jax},
  author       = {Yann Berthelot},
  year         = {2025},
  url          = {https://github.com/YannBerthelot/Ajax},
}
```
