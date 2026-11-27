---
hide:
  - toc
---

# Ajax

Ajax is a library of reinforcement-learning agents written in JAX. A
training run, with its environment and all its seeds, compiles into one
program that runs on a CPU or a GPU. These pages show how to install it,
train and use its agents, and extend them.

```bash
pip install ajax-rl  # imports as ajax; Python 3.12 or newer
```

```python
from ajax import PPO

agent = PPO("CartPole-v1", n_envs=8, n_steps=128)
state, evaluations = agent.train(seed=[1, 2, 3], n_timesteps=40_960)  # prints a line per evaluation

print(evaluations.returns)  # one row per seed, one column per evaluation
```

[Get started →](getting-started/installation.md){ .md-button .md-button--primary }
[Coming from SB3 →](getting-started/coming-from-sb3.md){ .md-button }
[Choose an algorithm →](algorithms/index.md){ .md-button }

## Why Ajax

Ajax has twenty agents, each following the code behind its paper's results.
Next to [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) and
its JAX version [SBX](https://github.com/araffin/sbx), it offers:

- **[Speed](getting-started/why-ajax.md#one-program-for-the-whole-run)**: the
  environment steps inside the compiled program, beside the agent, with no
  Python between two steps.
- **[Many runs in one call](getting-started/why-ajax.md#one-program-for-the-whole-run)**:
  every seed, and every value of a hyperparameter sweep, trains in the same
  program.
- **[Variations that combine](getting-started/why-ajax.md#variations-that-combine)**:
  a memory, an image encoder, a PID policy or an
  [extension](guide/extensions.md) is an argument of the same agent, and an
  agent refuses, when built, one it cannot run.
- **[Checked against the original code](getting-started/why-ajax.md#checked-against-the-original-code)**:
  unit tests of each update's maths, probes with known answers, and
  conformance tests against the original implementations.
- **[Average reward](getting-started/why-ajax.md#average-reward-agents)**,
  **[world models](getting-started/why-ajax.md#world-models)** and
  **[offline training](guide/training.md#training-offline)** beside the
  usual discounted, model-free agents.
- **[Learning from an expert](getting-started/why-ajax.md#learning-from-an-expert-or-from-data)**:
  start from a controller you already have, or from recorded transitions.

[Why Ajax](getting-started/why-ajax.md) explains each, with examples,
compares Ajax with SB3, SBX, PureJaxRL, Brax, Stoix, Rejax, TorchRL and
Tianshou, and says when one of them fits better.

## Getting started

1. [Installation](getting-started/installation.md): pip, the extras, a GPU.
2. [Quickstart](getting-started/quickstart.md): train an agent, read its
   results, act with it and save it.
3. [Why Ajax](getting-started/why-ajax.md): what Ajax offers, and when
   another library fits better.
4. [Coming from SB3](getting-started/coming-from-sb3.md): Stable-Baselines3's
   calls and defaults beside Ajax's.

## The agents

| Kind | Agents |
| --- | --- |
| Off-policy actor-critic | [SAC](algorithms/sac.md), [ASAC](algorithms/asac.md) and [RVI-SAC](algorithms/rvisac.md) (average reward), [REDQ](algorithms/redq.md), [CrossQ](algorithms/crossq.md), [TQC](algorithms/tqc.md), [TD3](algorithms/td3.md) |
| Offline, then online | [IQL](algorithms/iql.md) |
| Incremental, no replay | [AVG](algorithms/avg.md) |
| Value-based, discrete actions | [DQN](algorithms/dqn.md), [PQN](algorithms/pqn.md) |
| On-policy | [PPO](algorithms/ppo.md), [APO](algorithms/apo.md) (average reward), [REINFORCE](algorithms/reinforce.md) (the plain policy gradient) |
| Supervised, command-conditioned | [UDRL](algorithms/udrl.md) |
| Imitation of an expert, no reward | [DAgger](algorithms/dagger.md) |
| Differentiable simulation | [APG](algorithms/apg.md) |
| World models | [DreamerV3](algorithms/dreamerv3.md), [TD-MPC2 and its multi-task variant](algorithms/tdmpc2.md) |

Each agent's page gives its paper, its defaults and where it departs from
the original.

## What else is in the box

| Feature | What it gives |
| --- | --- |
| [Presets](guide/training.md#presets) | the settings an agent's reference code tunes for a task: `Agent.preset(task, env_id)` |
| [Hyperparameter sweeps](guide/training.md#hyperparameter-sweeps) | `train(sweep=...)`: several configurations, each with every seed, in one program |
| [Expert guidance](guide/expert-guidance.md) | cloning, imitation, prior data and hand-over, from a controller in the environment's own units |
| [Memory](guide/memory.md) | GRU, LSTM, transformer or Mamba blocks, with R2D2-style sequence replay |
| [Networks](guide/networks.md) | hidden layers, initialisers, CNN encoders, a shared encoder, [layers per head](guide/networks.md#layers-per-head) |
| [The PID policy](guide/networks.md#the-pid-policy) | a PID controller as the actor, whose learned gains deploy as a plain PID |
| [Environments](guide/environments.md) | gymnax, Brax and MuJoCo Playground by id, wrappers, and any Gymnasium environment (`"gymnasium:<id>"`) |
| [Gymnasium's MuJoCo tasks in JAX](guide/environments.md#gymnasiums-mujoco-tasks-in-jax) | `"mujoco:<id>"`, and [MuJoCo Warp](guide/environments.md#mujoco-warp-on-a-gpu) physics on an NVIDIA GPU |
| [Normalisation](guide/environments.md#normalisation) | running observation statistics, and rewards divided by the spread of the discounted return |
| [Differentiable simulation](guide/differentiable-simulation.md) | APG through gymnax dynamics, across distributions of physical parameters |
| [Logging](guide/logging.md) | evaluations as arrays, Weights & Biases or TensorBoard, [evaluation on another device](guide/logging.md#evaluating-on-another-device) |
| [Checkpoints](guide/checkpoints.md) | save a run and resume it in another process |
| [Performance](guide/performance.md) | seeds, CPU or GPU, and the persistent compile cache |
| [Comparing agents](guide/comparing-agents.md) | AdaStop, which trains more seeds only while the answer is unclear, and confidence intervals |
| [Reproduction](reproduction/index.md) | Ajax agents trained beside reference implementations, under a written protocol |

## Documentation

<div class="grid cards" markdown>

- **Use it**

    [Training an agent](guide/training.md) ·
    [Train, evaluate, deploy](guide/deploy.md) ·
    [Environments](guide/environments.md) ·
    [Extensions](guide/extensions.md) ·
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

Ajax is MIT-licensed. Its sibling project
[TargetGym](https://github.com/YannBerthelot/TargetGym) provides JAX
environments for holding a setpoint.
