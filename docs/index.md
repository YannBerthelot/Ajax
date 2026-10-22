---
hide:
  - toc
---

# Ajax

<p align="center">
  <b>Reinforcement-learning agents in JAX, compiled end to end with the environment,<br/>
  including average-reward, expert-guided and world-model agents.</b><br/>
  A whole training run, environment and every seed included, one compiled program.
</p>

Compared with [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
and its JAX version [SBX](https://github.com/araffin/sbx), Ajax differs in
four ways:

- **[End to end on the accelerator](guide/performance.md)**: the environment
  and the training loop compile into one JAX program, as in
  [PureJaxRL](https://github.com/luchris429/purejaxrl),
  [Rejax](https://github.com/keraJLi/rejax),
  [Stoix](https://github.com/EdanToledo/Stoix) and
  [Brax](https://github.com/google/brax)'s training agents, with no Python
  step between them, on a CPU or a GPU.
- **Average-reward agents**: [ASAC](algorithms/asac.md) and
  [APO](algorithms/apo.md) maximise the reward per step, with no discount, for
  tasks that never end.
- **[Expert-guided agents](guide/expert-guidance.md)**: learn from a
  controller you already have, or from recorded transitions, by cloning,
  imitation, residual policies, hand-over, prior data in the replay or
  offline training.
  [Choosing the pieces](guide/expert-guidance.md#choosing-the-pieces) names
  each method and its paper: TD3+BC, Q-filtered cloning, kickstarting,
  residual RL, JSRL, IBRL, RLPD's prior data, IQL, CQL, Cal-QL, ReBRAC,
  GAIL, DAgger.
- **[World models](guide/world-models.md)**: DreamerV3 and TD-MPC2, following
  the code behind their papers' results.

[Why Ajax](getting-started/why-ajax.md) shows each with an example and
compares Ajax with SB3, SBX, PureJaxRL, Brax, Stoix and Rejax, including
when to use them instead.

Ajax has seventeen agents, from SAC and PPO to DreamerV3 and TD-MPC2. Each
implements a published paper, and its page lists where it departs from it.
Several seeds train side by side in one program, on a CPU or a GPU. Research
features (learning from an expert, exploration rules, diagnostics) are
[extensions](guide/extensions.md), passed as `extensions=(...)` rather than
added as new arguments to the agents. An agent refuses, when built, an
extension it cannot run; [Compatibility](guide/compatibility.md) shows which
agent runs which.

```bash
git clone https://github.com/YannBerthelot/Ajax.git
cd Ajax
uv sync
```

```python
from ajax import PPO
from ajax.logging.wandb_logging import LoggingConfig

agent = PPO("CartPole-v1")  # 4 environments x 2048-step rollouts: one update per 8,192 steps
# Evaluate after every update; use_wandb=False keeps the results in memory only.
logging = LoggingConfig(config={}, log_frequency=8_192, use_wandb=False)
state, evaluations = agent.train(seed=[1, 2, 3], n_timesteps=40_960, logging_config=logging)

print(evaluations["Eval/episodic mean reward"])  # one row per seed, one column per evaluation
```

[Quickstart →](getting-started/quickstart.md){ .md-button .md-button--primary }
[Why Ajax →](getting-started/why-ajax.md){ .md-button }
[Choose an algorithm →](algorithms/index.md){ .md-button }

## The agents

| Kind | Agents |
| --- | --- |
| Off-policy actor-critic | [SAC](algorithms/sac.md), [ASAC](algorithms/asac.md) (average reward), [REDQ](algorithms/redq.md), [CrossQ](algorithms/crossq.md), [TD3](algorithms/td3.md) |
| Offline, then online | [IQL](algorithms/iql.md) |
| Incremental, no replay | [AVG](algorithms/avg.md) |
| Value-based, discrete actions | [DQN](algorithms/dqn.md), [PQN](algorithms/pqn.md) |
| On-policy | [PPO](algorithms/ppo.md), [APO](algorithms/apo.md) (average reward) |
| Supervised, command-conditioned | [UDRL](algorithms/udrl.md) |
| Imitation of an expert, no reward | [DAgger](algorithms/dagger.md) |
| Differentiable simulation | [APG](algorithms/apg.md) |
| World models | [DreamerV3](algorithms/dreamerv3.md), [TD-MPC2 and its multi-task variant](algorithms/tdmpc2.md) |

## What else is in the box

- [Extensions](guide/extensions.md): expert guidance, exploration, critic-target
  changes, actor and critic pre-training, offline RL, adversarial imitation
  and diagnostics, each usable with the agents that run the training phases
  it touches.
- [Training offline](guide/training.md#training-offline): a replay agent
  trains on a transition dataset before it acts, then fine-tunes online.
- [Expert guidance](guide/expert-guidance.md): learn from an existing
  controller, which reads the environment's raw observations and answers in
  its action units.
- [Memory](guide/memory.md): GRU, LSTM, transformer or Mamba blocks, with
  R2D2-style sequence replay.
- [PID policy](guide/networks.md): a PID controller as the actor, whose learned
  gains deploy as a plain PID.
- [Differentiable simulation](guide/differentiable-simulation.md): APG through
  gymnax dynamics, across distributions of physical parameters.
- [World models](guide/world-models.md): DreamerV3 and TD-MPC2, following the
  paper-era official code.
- [Environments](guide/environments.md): gymnax, Brax and MuJoCo Playground by
  id, with wrappers for normalisation, observation noise, reference tracking
  and more; every continuous agent's `[-1, 1]` actions are mapped onto the
  environment's bounds.
- [Logging](guide/logging.md) and [checkpoints](guide/checkpoints.md):
  evaluations as arrays, W&B or TensorBoard, resuming in another process.
- [Performance](guide/performance.md): every seed in one program, a persistent
  compile cache, one per machine and JAX version.
- [Comparing agents](guide/comparing-agents.md): AdaStop, which trains more
  seeds only while the answer is unclear, and confidence intervals for one
  agent's metrics.
- [Reproduction](reproduction/index.md): Ajax agents trained beside reference
  implementations on the same JAX environment, under a written protocol.

## Documentation

<div class="grid cards" markdown>

- **Use it**

    [Installation](getting-started/installation.md) ·
    [Quickstart](getting-started/quickstart.md) ·
    [Why Ajax](getting-started/why-ajax.md) ·
    [Training an agent](guide/training.md) ·
    [Algorithms](algorithms/index.md)

- **Look it up**

    [API reference](api/agents.md) ·
    [Training loop](api/training-loop.md) ·
    [Extensions API](api/extensions.md) ·
    [Compatibility](guide/compatibility.md)

- **Change it**

    [Architecture](developer/architecture.md) ·
    [Contributing](developer/contributing.md) ·
    [Writing an extension](developer/writing-an-extension.md) ·
    [Testing and probing](developer/testing.md) ·
    [Performance checks](developer/performance.md)

- **Check the world models**

    [Design notes](world_models/README.md) ·
    [Deviations](world_models/deviations.md) ·
    [Validation](world_models/VALIDATION.md)

</div>

Ajax is MIT-licensed. [TargetGym](https://github.com/YannBerthelot/TargetGym),
its sibling project, provides JAX environments for holding a setpoint.
