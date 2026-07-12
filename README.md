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
| Recurrent networks (PPO, SAC, ASAC, REDQ, TD3) | :heavy_check_mark: |

### Available Agents

| Agent | Paper |
| ----- | ----- |
| **SAC**   | Haarnoja et al., *Soft Actor-Critic*, 2018 — [arXiv:1801.01290](https://arxiv.org/abs/1801.01290) |
| **ASAC**  | Adamczyk et al., *Average-Reward Soft Actor-Critic*, 2025 — [arXiv:2501.09080](https://arxiv.org/pdf/2501.09080v2) |
| **REDQ**  | Chen et al., *Randomized Ensembled Double Q-Learning*, 2021 — [arXiv:2101.05982](https://arxiv.org/abs/2101.05982) |
| **AVG**   | Vasan et al., *Deep Policy Gradient Methods Without Batch Updates, Target Networks, or Replay Buffers*, 2024 — [arXiv:2411.15370](https://arxiv.org/abs/2411.15370) |
| **PPO**   | Schulman et al., *Proximal Policy Optimization*, 2017 — [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) |
| **APO**   | Ma et al., *Average-Reward Reinforcement Learning with Trust Region Methods*, 2021 — [arXiv:2106.03442](https://arxiv.org/abs/2106.03442) |
| **TD3**   | Fujimoto et al., *Addressing Function Approximation Error in Actor-Critic Methods*, 2018 — [arXiv:1802.09477](https://arxiv.org/abs/1802.09477) |
| **UDRL**  | Schmidhuber, *Reinforcement Learning Upside Down: Don't Predict Rewards, Just Map Them to Actions*, 2019 — [arXiv:1912.02875](https://arxiv.org/abs/1912.02875) |

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

### Recurrent networks (memory)

PPO, SAC, ASAC, REDQ and TD3 support memory-augmented actors and critics
through a single hyperparameter — the network becomes
`encoder → memory → heads` and all hidden-state plumbing (collection,
training, evaluation, episode-boundary resets) is handled internally:

```python
from ajax import PPO, ASAC
from ajax.networks.memory import MemoryConfig

agent = PPO("CartPole-v1", memory=MemoryConfig(kind="gru", hidden_size=64))
agent = ASAC("Pendulum-v1", memory={"kind": "lstm", "hidden_size": 64})  # dict works too
agent = PPO("CartPole-v1", memory=MemoryConfig(kind="mamba", hidden_size=64))
agent = PPO(
    "CartPole-v1",
    memory=MemoryConfig(kind="transformer", hidden_size=64, window=32, num_heads=4),
)
```

Available kinds and their profiles:

| Kind | Carry per env | Training over T |
| ---- | ------------- | --------------- |
| `"gru"` / `"lstm"` | O(hidden) | sequential (`nn.scan`) |
| `"transformer"` (sliding-window attention) | O(window · hidden) | parallel attention with episode-segment masks |
| `"mamba"` (selective SSM) | O(d_state · hidden) | parallel `associative_scan` |

Shared knobs: `num_layers` (stacked blocks) and `gradient_checkpoint`
(rematerialize activations on long sequences). All kinds are reset-aware
(no information leaks across episode boundaries, forward or backward) and
step-wise acting is numerically identical to sequence-mode training —
enforced by the equivalence tests in `tests/networks/test_memory.py`.
`tests/agents/PPO/test_memory_probe.py` verifies end-to-end that each kind
actually uses its memory: on velocity-masked CartPole, feedforward PPO
plateaus near 40 return while every memory kind exceeds 450/500.

- **PPO** trains with truncated BPTT over full rollouts: each epoch uses the
  whole `(n_steps, n_envs)` sequence from the rollout-start hidden states
  (`batch_size` is ignored in recurrent mode to keep sequences intact).
- **SAC / ASAC / REDQ / TD3** switch their replay buffer to trajectory
  storage and train on sampled sequences R2D2-style (shared machinery in
  `ajax/agents/recurrent.py`): carries are warmed up from zero over
  `burn_in` steps under `stop_gradient`, then `sequence_length` steps are
  trained with BPTT. TD3 additionally burns in a target-actor carry for
  its bootstrap action. Expert-guidance features are rejected loudly when
  combined with memory.
- Other agents (AVG, APO, UDRL) raise `NotImplementedError` when `memory`
  is set rather than silently ignoring it.

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
│   ├── SAC/, ASAC/, REDQ/, AVG/, PPO/, APO/, TD3/, UDRL/
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
