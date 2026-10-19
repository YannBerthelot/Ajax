<h1 align="center">Ajax</h1>

<h3 align="center">
  Reinforcement-learning agents in JAX, compiled end to end with the environment,<br/>
  including average-reward, expert-guided and world-model agents.
</h3>

<p align="center"><i>A whole training run, environment and every seed included, one compiled program.</i></p>

<p align="center">
  <a href="https://yannberthelot.github.io/Ajax/"><img alt="Documentation" src="https://img.shields.io/badge/docs-yannberthelot.github.io%2FAjax-blue"></a>
  <a href="https://github.com/YannBerthelot/Ajax/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/YannBerthelot/Ajax/ci.yml?branch=main&label=CI"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"></a>
  <img alt="JAX" src="https://img.shields.io/badge/JAX-jit%20%7C%20vmap%20%7C%20grad-orange">
</p>

Ajax is a library of reinforcement-learning agents written in
[JAX](https://github.com/jax-ml/jax). Compared with
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) and its JAX
version [SBX](https://github.com/araffin/sbx), it differs in four ways:

- **[End to end on the accelerator](docs/guide/performance.md)**: the
  environment and the training loop compile into one JAX program, as in
  [PureJaxRL](https://github.com/luchris429/purejaxrl),
  [Rejax](https://github.com/keraJLi/rejax),
  [Stoix](https://github.com/EdanToledo/Stoix) and
  [Brax](https://github.com/google/brax)'s training agents, with no Python
  step between them, on a CPU or a GPU.
- **Average-reward agents**: [ASAC](docs/algorithms/asac.md) and
  [APO](docs/algorithms/apo.md) maximise the reward per step, with no
  discount, for tasks that never end.
- **[Expert-guided agents](docs/guide/expert-guidance.md)**: learn from a
  controller you already have, or from recorded transitions, by cloning,
  imitation, residual policies, hand-over or prior data in the replay
  ([methods below](#learning-from-an-expert-or-from-data)).
- **[World models](docs/guide/world-models.md)**: DreamerV3 and TD-MPC2,
  following the code behind their papers' results.

[Why Ajax](docs/getting-started/why-ajax.md) shows each with an example and
compares Ajax with SB3, SBX, PureJaxRL, Brax, Stoix and Rejax, including
when to use them instead.

It has fifteen agents, from SAC and PPO to DreamerV3 and TD-MPC2, each
implementing a published paper, with the places where it departs from that
paper written down. Several seeds train side by side in one program, on a CPU
or a GPU. Research features (learning from an expert, exploration rules,
diagnostics) are *extensions*, passed as `extensions=(...)` rather than added
as new arguments to the agents. An agent refuses, when built, an extension
it cannot run; the [compatibility table](docs/guide/compatibility.md) shows
which agent runs which.

---

## Installation

Ajax installs from source with [uv](https://docs.astral.sh/uv/), on Python 3.11
to 3.13:

```bash
git clone https://github.com/YannBerthelot/Ajax.git
cd Ajax
uv sync   # creates .venv with the locked dependencies
```

uv installs the versions in `uv.lock`, which CI tests, with JAX for the CPU;
on a Linux machine with an NVIDIA GPU, add CUDA with `uv sync --extra cuda`
(a project that depends on Ajax asks for `ajax[cuda]`).

pip works too, with JAX pinned to the locked version. pip does not read
`uv.lock`, and it would pick JAX 0.11, on which the flax version Ajax uses
fails when training starts:

```bash
pip install "ajax @ git+https://github.com/YannBerthelot/Ajax.git" "jax==0.7.2"
```

[Installation](docs/getting-started/installation.md) covers the extras
(`cuda`, `compare`, `reproduction`) and the dependency groups.

## Quickstart

```python
from ajax import PPO
from ajax.logging.wandb_logging import LoggingConfig

agent = PPO("CartPole-v1")  # 4 environments x 2048-step rollouts: one update per 8,192 steps
# Evaluate after every update; use_wandb=False keeps the results in memory only.
logging = LoggingConfig(config={}, log_frequency=8_192, use_wandb=False)
state, evaluations = agent.train(seed=[1, 2, 3], n_timesteps=40_960, logging_config=logging)

print(evaluations["timestep"][0])                # when each evaluation ran
print(evaluations["Eval/episodic mean reward"])  # one row per seed, one column per evaluation
```

The three seeds train together, and the run takes about fifteen seconds on a
laptop CPU, compilation included. `evaluations` maps each logged name to an
array of shape `(seeds, evaluations)`; `"Eval/episodic mean reward"` is the
mean return of 10 test episodes (`num_episode_test`). With `use_wandb=True` (the default) or
`use_tensorboard=True`, the same records also go to Weights & Biases or
TensorBoard. To train further, pass the state back:
`agent.train(seed=[1, 2, 3], n_timesteps=more, initial_state=state)`.

Features are arguments of the same agents:

```python
from ajax import PPO, SAC
from ajax.extensions.instrumentation import ConditioningMetrics

# A recurrent actor and critic, for when one observation is not enough.
agent = SAC("Pendulum-v1", memory={"kind": "lstm", "hidden_size": 64})

# A research feature as an extension: the critic's health at each evaluation.
agent = PPO("CartPole-v1", expose_recent_rollout=True, extensions=(ConditioningMetrics(),))

# Brax and MuJoCo Playground environments by id, here 8 copies in parallel.
agent = PPO("ant", n_envs=8)
```

The [quickstart guide](docs/getting-started/quickstart.md) goes on from here.

---

## Agents

| Agent | Kind | Paper | Actions | Memory |
| --- | --- | --- | --- | --- |
| [`SAC`](docs/algorithms/sac.md) | off-policy | Haarnoja et al., *Soft Actor-Critic*, 2018 ([arXiv:1801.01290](https://arxiv.org/abs/1801.01290)) | continuous | yes |
| [`ASAC`](docs/algorithms/asac.md) | off-policy, average reward | Adamczyk et al., *Average-Reward Soft Actor-Critic*, 2025 ([arXiv:2501.09080v2](https://arxiv.org/abs/2501.09080v2)) | continuous | yes |
| [`REDQ`](docs/algorithms/redq.md) | off-policy | Chen et al., *Randomized Ensembled Double Q-Learning*, 2021 ([arXiv:2101.05982](https://arxiv.org/abs/2101.05982)) | continuous | yes |
| [`TD3`](docs/algorithms/td3.md) | off-policy | Fujimoto et al., *Addressing Function Approximation Error in Actor-Critic Methods*, 2018 ([arXiv:1802.09477](https://arxiv.org/abs/1802.09477)) | continuous | yes |
| [`AVG`](docs/algorithms/avg.md) | incremental, no replay | Vasan et al., *Deep Policy Gradient Methods Without Batch Updates, Target Networks, or Replay Buffers*, 2024 ([arXiv:2411.15370](https://arxiv.org/abs/2411.15370)) | continuous | no |
| [`IQL`](docs/algorithms/iql.md) | offline | Kostrikov, Nair & Levine, *Offline Reinforcement Learning with Implicit Q-Learning*, ICLR 2022 ([arXiv:2110.06169](https://arxiv.org/abs/2110.06169)) | continuous | no |
| [`DQN`](docs/algorithms/dqn.md) | off-policy, value-based | Mnih et al., *Human-level control through deep reinforcement learning*, Nature 2015 | discrete | no |
| [`PQN`](docs/algorithms/pqn.md) | on-policy, value-based | Gallici et al., *Simplifying Deep Temporal Difference Learning*, 2024 | discrete | no |
| [`PPO`](docs/algorithms/ppo.md) | on-policy | Schulman et al., *Proximal Policy Optimization Algorithms*, 2017 ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347)) | both | yes |
| [`APO`](docs/algorithms/apo.md) | on-policy, average reward | Ma et al., *Average-Reward Reinforcement Learning with Trust Region Methods*, 2021 ([arXiv:2106.03442](https://arxiv.org/abs/2106.03442)) | both | no |
| [`UDRL`](docs/algorithms/udrl.md) | supervised, command-conditioned | Schmidhuber, *Reinforcement Learning Upside Down*, 2019 ([arXiv:1912.02875](https://arxiv.org/abs/1912.02875)) | both | no |
| [`APG`](docs/algorithms/apg.md) | differentiable simulation | Analytic policy gradient through the simulator; `APG.contextual_controller` is Busetto et al., *One controller to rule them all*, 2024 ([arXiv:2411.06482](https://arxiv.org/abs/2411.06482)) | continuous | yes |
| [`DreamerV3`](docs/algorithms/dreamerv3.md) | world model | Hafner et al., *Mastering Diverse Domains through World Models*, 2023, Nature 2025 ([arXiv:2301.04104v2](https://arxiv.org/abs/2301.04104v2)) | both | built in |
| [`TDMPC2`](docs/algorithms/tdmpc2.md) | world model, planning | Hansen, Su & Wang, *TD-MPC2: Scalable, Robust World Models for Continuous Control*, ICLR 2024 ([arXiv:2310.16828](https://arxiv.org/abs/2310.16828)) | continuous | no |
| [`TDMPC2MultiTask`](docs/algorithms/tdmpc2.md) | world model, offline | The same paper's multi-task agent: one model trained offline on the data of several tasks | continuous | no |

*Memory* says whether the agent accepts `memory=`; DreamerV3's world model is
recurrent by design. Every agent is importable from `ajax`
(`from ajax import TD3`). The [algorithms overview](docs/algorithms/index.md)
helps choose one.

---

## Learning from an expert or from data

Each method is an extension or an agent option. The last column lists the
agents that accept it, as the [compatibility table](docs/guide/compatibility.md),
generated from the code, does.

| Method | In Ajax | Agents |
| --- | --- | --- |
| Behaviour cloning before training | `CloningPretrain` | SAC, TD3, REDQ, ASAC, AVG, APO |
| TD3+BC's actor term (Fujimoto and Gu, 2021); kickstarting (Schmitt et al., 2018) with `anneal_frac` | `ImitationLoss` | SAC, TD3, REDQ, ASAC, AVG, APO; PPO, APG, UDRL with a stateless expert; continuous actions only |
| Q-filtered behaviour cloning (Nair et al., 2018) | `ImitationLoss(q_filter=True)` | SAC, TD3, REDQ, ASAC, AVG |
| IBRL targets (Hu et al., 2023) | `IBRL` | SAC, TD3, REDQ, AVG |
| Residual RL (Johannink et al., 2019) | `ResidualPolicy` | SAC |
| Jump-start RL hand-over (Uchendu et al., 2023) | `JSRLCurriculum` | SAC |
| Expert transitions in the replay before training | `expert_buffer_n_steps` | SAC |
| Prior data in the replay, as RLPD (Ball et al., 2023) without its symmetric sampling, then optional offline updates | `PriorData` with a transition dataset (`ajax.buffers.dataset`) | SAC, TD3, REDQ, ASAC, DQN |
| Offline training on a transition dataset, then online fine-tuning | `agent.train_offline(dataset, n_updates)` | SAC, TD3, REDQ, ASAC, DQN, IQL |
| Implicit Q-learning, offline then online (IQL, Kostrikov et al., 2022) | [`IQL`](docs/algorithms/iql.md) with `train_offline` | its own agent |
| TD3+BC, offline (Fujimoto and Gu, 2021) | [`TD3` with `ImitationLoss(None, target="replay", alpha=2.5)`](docs/algorithms/td3.md#td3bc-offline), trained with `train_offline` | TD3; the term on SAC, REDQ, ASAC too |
| Offline multi-task training on pooled datasets (TD-MPC2, Hansen et al., 2024) | `TDMPC2MultiTask` | its own agent |
| Features developed with Ajax, with no external reference: EDGE exploration, LCB-gated targets, a value-gated hand-over, an expert critic fitted to Monte-Carlo returns and the extensions that read it | `EDGEExploration`, `LCBGatedBootstrap`, `ValueBox`, `MCPretrain`, ... | SAC (LCB: SAC, TD3, REDQ) |

Not in Ajax yet: interactive imitation such as DAgger, adversarial imitation
such as GAIL, and the offline RL algorithms CQL and Cal-QL.
[Expert guidance](docs/guide/expert-guidance.md) explains the expert's
conventions and how the pieces combine.

---

## What else is in the box

- **[Extensions](docs/guide/extensions.md)**: twenty of them, for expert
  guidance, exploration, critic-target changes, actor and critic
  pre-training, prior data, ensemble diversity and critic diagnostics, passed
  as `extensions=(...)`; the [compatibility table](docs/guide/compatibility.md)
  says which agents run each one.
- **[Expert guidance](docs/guide/expert-guidance.md)**: the expert reads the
  environment's raw observations and answers in the environment's action
  units; an expert with a state, such as a PID's integral, gets it back at
  every step.
- **[Recurrent memory](docs/guide/memory.md)**: a GRU, LSTM, transformer or
  Mamba block for PPO, SAC, TD3, REDQ, ASAC and APG, with R2D2-style sequence
  replay (burn-in, stored state) for the replay agents.
- **[Differentiable simulation](docs/guide/differentiable-simulation.md)**: APG
  back-propagates the return through gymnax dynamics, across *system classes*
  (distributions over physical parameters), with Busetto et al.'s in-context
  controller and its staged curriculum.
- **[World models](docs/guide/world-models.md)**: DreamerV3 and TD-MPC2
  (single- and multi-task), following the paper-era official code, with
  [specifications, deviations and parity tests](docs/world_models/README.md)
  against that code.
- **[Networks and the PID policy](docs/guide/networks.md)**: layers and
  initialisers named by strings, a CNN encoder for image observations (PPO,
  DQN, PQN, UDRL), and a PID controller as the actor of SAC, TD3, REDQ, ASAC,
  AVG, PPO or APO, with learned fixed gains that deploy as a plain PID, or
  gains scheduled on the observation. Where an agent starts from its
  reference code's initialisation and optimiser settings, its algorithm page
  says so and cites the code.
- **[Environments](docs/guide/environments.md)**: gymnax, Brax and MuJoCo
  Playground by id, `n_envs` copies in parallel, terminations kept apart from
  time limits (a truncated episode still bootstraps), `episode_length` and
  `action_repeat` on every agent, and wrappers for normalisation, observation
  noise, fixed initial states, model-reference tracking and PID error terms.
  Every continuous agent acts in `[-1, 1]`, mapped onto the environment's
  action bounds.
- **[Logging](docs/guide/logging.md)**: `train` returns every evaluation as
  arrays, with training diagnostics (gradient norms, PPO's approximate KL,
  explained variance, ...) and each extension's own metrics, and a background
  worker sends them to Weights & Biases or TensorBoard.
- **[Comparing agents](docs/guide/comparing-agents.md)**: `ajax.compare` runs
  AdaStop (Mathieu et al., 2024), which trains more seeds only while the
  answer is unclear, and `ajax.compare.estimate` gives one agent's metrics
  confidence intervals that stay valid however many seeds are added.
- **[Reproduction](docs/reproduction/index.md)**: Ajax agents trained beside
  a reference implementation (Stable-Baselines3, Brax, the authors' code) on
  the same JAX environment, run on demand from
  [`benchmarks/reference`](benchmarks/reference/README.md), under a written
  protocol; the results so far cover PPO and PQN.
- **[Checkpoints and resuming](docs/guide/checkpoints.md)**: pass a state back
  to train further, or save it and resume in another process; the tests
  check that a split run matches the run left whole.
- **[Tests](docs/developer/testing.md)**: unit tests check one component
  against an exact contract with no learning, written once for every agent or
  extension that shares the contract. Probes are tiny generic environments
  with known answers: each agent trains on them briefly, and each probe checks
  one capability of the whole agent. A catalogue of planted bugs lists the
  probes that should catch each one; CI checks that every bug still applies
  to the code, and planting them to run the probes is done by hand. A
  mismatch of convention with a reference implementation is left to code
  review.
- **[Performance](docs/guide/performance.md)**: seeds vectorised with
  `jax.vmap`, a persistent compile cache, one per machine and JAX version, and
  a per-agent benchmark script, run by hand, that flags slowdowns over 10%
  against a recorded baseline.

---

## Documentation

| | |
| --- | --- |
| **[Getting started](docs/getting-started/installation.md)** | Installation and a guided quickstart |
| **[User guide](docs/guide/training.md)** | Training, environments, networks, memory, logging, checkpoints, performance |
| **[Research features](docs/guide/extensions.md)** | Extensions, compatibility, expert guidance, comparing agents, differentiable simulation, world models |
| **[Reproduction](docs/reproduction/index.md)** | Ajax against reference implementations: protocol, specifications, results |
| **[Algorithms](docs/algorithms/index.md)** | One page per agent: the paper, when to use it, an example, its parameters and deviations |
| **[API reference](https://yannberthelot.github.io/Ajax/api/agents/)** | Generated from the docstrings |
| **[Developer guide](docs/developer/architecture.md)** | Architecture, contributing, writing an extension, testing, performance |
| **[World model design notes](docs/world_models/README.md)** | DreamerV3 and TD-MPC2 specifications, deviations and validation |

The site, with search and navigation, is at
**[yannberthelot.github.io/Ajax](https://yannberthelot.github.io/Ajax/)**. The
links above point at the Markdown in this repository, which also reads on
GitHub.

---

## Related projects

- **[TargetGym](https://github.com/YannBerthelot/TargetGym)**: JAX environments
  for reaching and holding a setpoint, with tuned PID and MPC baselines. A
  scenario of Ajax's `benchmarks/perf_bench.py` trains on one of its aircraft
  tasks (`uv sync --group plane`).
- **[ProbingEnvironments](https://github.com/YannBerthelot/ProbingEnvironments)**:
  small environments with known answers, used by Ajax's probing tests.
- **[gymnax](https://github.com/RobertTLange/gymnax)**,
  **[Brax](https://github.com/google/brax)** and
  **[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)**:
  the environment libraries Ajax trains on.
- **[flashbax](https://github.com/instadeepai/flashbax)**: the replay buffers of
  SAC, ASAC, REDQ, TD3, DQN and IQL (UDRL and the world models keep their own).

---

## Contributing

Bug reports, new agents and new extensions are welcome. After installing, run
`uv run pre-commit install` once, so ruff and mypy check every commit.
[CONTRIBUTING.md](CONTRIBUTING.md) explains where the code lives, how to add an
agent or an extension, and how the tests check them. CI runs pre-commit, the
tests with coverage, the slow tests, the probing tests and a strict build of
the documentation site on every pull request; `make ci` runs the same locally
(about an hour of CPU).

## Citation

```bibtex
@misc{ajax2025,
  title        = {Ajax: Reinforcement Learning Agents in Jax},
  author       = {Yann Berthelot},
  year         = {2025},
  url          = {https://github.com/YannBerthelot/Ajax},
}
```

## License

MIT, see [LICENSE](LICENSE).
