<h1 align="center">Ajax</h1>

<p align="center">
  <b>Fast, modular reinforcement learning in JAX.</b><br/>
  The environment, the agent and every seed run together in one program, on a CPU or a GPU, and agents are built from parts that snap together.
</p>

<p align="center">
  <a href="https://yannberthelot.github.io/Ajax/"><img alt="Documentation" src="https://img.shields.io/badge/docs-yannberthelot.github.io%2FAjax-blue"></a>
  <a href="https://github.com/YannBerthelot/Ajax/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/YannBerthelot/Ajax/ci.yml?branch=main&label=CI"></a>
  <a href="https://pypi.org/project/ajax-rl/"><img alt="PyPI" src="https://img.shields.io/pypi/v/ajax-rl"></a>
  <a href="https://github.com/YannBerthelot/Ajax/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="https://yannberthelot.github.io/Ajax/latest/getting-started/coming-from-sb3/">Coming from SB3</a> ·
  <a href="#why-ajax">Why Ajax</a> ·
  <a href="#agents">Agents</a> ·
  <a href="#learning-from-an-expert-or-from-data">Expert and data</a> ·
  <a href="https://yannberthelot.github.io/Ajax/">Documentation</a>
</p>

## Install

```bash
pip install ajax-rl                    # imports as ajax; Python 3.12 or newer
pip install "ajax-rl[playground]"      # adds Brax and MuJoCo Playground environments
pip install "ajax-rl[gymnasium-mjx]"   # adds Gymnasium's MuJoCo tasks (HalfCheetah-v5, ...) in JAX
pip install "ajax-rl[cuda]"            # adds CUDA, on Linux with an NVIDIA GPU
```

`ajax-rl[brax]` adds Brax alone; [Installation](https://yannberthelot.github.io/Ajax/latest/getting-started/installation/)
covers the other extras and `uv sync` from a clone.

## Quickstart

```python
from ajax import PPO

agent = PPO("CartPole-v1", n_envs=8, n_steps=128)
state, evaluations = agent.train(seed=[1, 2, 3], n_timesteps=40_960)

print(evaluations.timesteps)  # when each evaluation ran
print(evaluations.returns)    # one row per seed, one column per evaluation
print(agent.evaluate(state, episodes=20).mean(axis=1))  # each seed's score now
```

`train` prints a line per evaluation, with the mean return over the seeds.
The three seeds train together, in under a minute on a laptop CPU,
compilation included.

`agent.predict(state, obs)` returns the trained policy's action for an
observation, and `agent.save(state, path)` and `ajax.load(path)` keep it
for later ([Train, evaluate, deploy](https://yannberthelot.github.io/Ajax/latest/guide/deploy/)).

Coming from Stable-Baselines3? [The calls side by side](https://yannberthelot.github.io/Ajax/latest/getting-started/coming-from-sb3/).

`"CartPole-v1"` here is [gymnax](https://github.com/RobertTLange/gymnax)'s
CartPole, written in JAX. That is the intended use: with a JAX environment
(gymnax, Brax, MuJoCo Playground) the environment, the agent and every seed
compile into one program that stays on the CPU or the GPU from start to
end. Gymnasium's MuJoCo tasks also run in JAX, with the `gymnasium-mjx`
extra: `SAC("mujoco:HalfCheetah-v5")`
([Environments](https://yannberthelot.github.io/Ajax/latest/guide/environments/#gymnasiums-mujoco-tasks-in-jax)).
Any [Gymnasium](https://gymnasium.farama.org) environment runs too, with
`PPO("gymnasium:CartPole-v1")`; it steps in Python, so training is slower
and a GPU does not help
([Environments](https://yannberthelot.github.io/Ajax/latest/guide/environments/#gymnasium-environments)).

A notebook trains, acts and saves step by step: [`notebooks/quickstart.ipynb`](https://github.com/YannBerthelot/Ajax/blob/main/notebooks/quickstart.ipynb)
([open in Colab](https://colab.research.google.com/github/YannBerthelot/Ajax/blob/main/notebooks/quickstart.ipynb)).
Complete scripts are in [`examples/`](https://github.com/YannBerthelot/Ajax/tree/main/examples).

## Why Ajax

Ajax has twenty agents, each from a published paper. Compared with
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) and
[SBX](https://github.com/araffin/sbx), it offers:

- **[Faster](#speed)**: the environment steps beside the agent on a CPU or a GPU, with no Python in the loop, and every seed and environment copy trains in the same program.
- **[More experiments per run](#research-in-a-few-lines)**: several seeds and several values of a hyperparameter train in one compiled program, so ten seeds of three learning rates are one call.
- **[Variations in one argument, parts that compose](#research-in-a-few-lines)**: recurrent PPO is PPO with a memory, and memories, CNN encoders, a PID policy and forty-four extensions (cloning, an expert's transitions in the replay, diagnostics) add up in one constructor. The [compatibility table](https://yannberthelot.github.io/Ajax/latest/guide/compatibility/) shows which go together.
- **[Trustworthy](https://yannberthelot.github.io/Ajax/latest/developer/testing/)**: each agent follows the code behind its paper's results, and its page lists where it departs. Unit tests check each update's maths, probes with known answers check that agents learn, and conformance tests compare updates with the original implementations. Comparisons of learning results with reference libraries are in progress ([Reproduction](https://yannberthelot.github.io/Ajax/latest/reproduction/results/)).
- **Multi-paradigm**: discounted and average-reward agents ([ASAC](https://yannberthelot.github.io/Ajax/latest/algorithms/asac/), [RVI-SAC](https://yannberthelot.github.io/Ajax/latest/algorithms/rvisac/), [APO](https://yannberthelot.github.io/Ajax/latest/algorithms/apo/)), model-free and model-based, online and offline.
- **[Expert-guided](https://yannberthelot.github.io/Ajax/latest/guide/expert-guidance/)**: cloning, imitation, prior data and offline training.
- **[World models](https://yannberthelot.github.io/Ajax/latest/guide/world-models/)**: DreamerV3 and TD-MPC2, which their papers run on over a hundred tasks with one set of hyperparameters.

[Why Ajax](https://yannberthelot.github.io/Ajax/latest/getting-started/why-ajax/) compares it with SB3, SBX,
PureJaxRL, Brax, Stoix, Rejax, TorchRL and Tianshou, including when to use
them instead.

## Speed

The environment, the agent and every seed compile into one program, so many seeds train at once on one device. With 10 seeds, one NVIDIA L40S ran SAC about 7 times as fast as four CPU cores, and CrossQ and REDQ over 50 times ([speeds](https://yannberthelot.github.io/Ajax/latest/guide/performance/#cpu-or-gpu)).

<!-- speed-by-family:start -->
With 10 seeds on four CPU cores, against the fastest other library, Ajax trained SAC and TD3 at 0.8 to 1.0 times, PPO at 0.2 to 0.8 times, DQN and PQN at 0.9 to 1.1 times its speed. On one NVIDIA L40S, against the JAX libraries, Ajax trained SAC at 5.6 to 7.8 times, PPO at 0.8 to 1.3 times, DQN and PQN at 0.8 to 0.9 times its speed. On HalfCheetah, Ajax, Brax and PureJaxRL step MJX, the others C MuJoCo ([figure and settings](https://yannberthelot.github.io/Ajax/latest/guide/performance/#against-other-libraries)).
<!-- speed-by-family:end -->

## Research in a few lines

Several seeds and several values of a hyperparameter train in one compiled program: here 30 runs, 10 seeds for each of 3 learning rates.

```python
from ajax import PPO

agent = PPO("CartPole-v1", n_envs=8, n_steps=128)
state, evaluations = agent.train(
    seed=list(range(10)),
    n_timesteps=40_960,
    sweep={"actor_learning_rate": [1e-4, 3e-4, 1e-3]},
)
for configuration, returns in agent.by_configuration(evaluations.returns):
    print(configuration, returns[:, -1].mean())  # mean final return over the 10 seeds
```

A variation is one argument, and parts add up in one constructor:

```python
import jax.numpy as jnp

from ajax import PPO, SAC
from ajax.extensions.instrumentation import CriticDiagnostics
from ajax.extensions.pretrain import CloningPretrain, ExpertPrefill

# A variation is one argument: recurrent PPO is PPO with a memory.
agent = PPO("CartPole-v1", memory={"kind": "lstm", "hidden_size": 64})


def pd_expert(obs):
    """A PD controller on Pendulum's angle, a torque in the env's range."""
    theta = jnp.arctan2(obs[..., 1], obs[..., 0])
    return jnp.clip(-(2.0 * theta + 0.5 * obs[..., 2]), -2.0, 2.0)[..., None]


# Parts add up in one constructor: the actor cloned from the expert before
# training, the expert's transitions in the replay, and at each evaluation
# whether the critic's features are degrading (how many directions they
# span, how many units went silent).
agent = SAC(
    "Pendulum-v1",
    extensions=(
        CloningPretrain(pd_expert, n_steps=1_000),
        ExpertPrefill(n_steps=1_000, expert=pd_expert),
        CriticDiagnostics(),
    ),
)
```

## Agents

- **Off-policy actor-critic**: [SAC](https://yannberthelot.github.io/Ajax/latest/algorithms/sac/), [TD3](https://yannberthelot.github.io/Ajax/latest/algorithms/td3/), [REDQ](https://yannberthelot.github.io/Ajax/latest/algorithms/redq/), [CrossQ](https://yannberthelot.github.io/Ajax/latest/algorithms/crossq/) (batch-renormalised, no target networks), [TQC](https://yannberthelot.github.io/Ajax/latest/algorithms/tqc/) (truncated quantile critics), [AVG](https://yannberthelot.github.io/Ajax/latest/algorithms/avg/) (no replay), [ASAC](https://yannberthelot.github.io/Ajax/latest/algorithms/asac/) and [RVISAC](https://yannberthelot.github.io/Ajax/latest/algorithms/rvisac/) (RVI-SAC; average reward), [IQL](https://yannberthelot.github.io/Ajax/latest/algorithms/iql/) (offline)
- **On-policy**: [PPO](https://yannberthelot.github.io/Ajax/latest/algorithms/ppo/), [APO](https://yannberthelot.github.io/Ajax/latest/algorithms/apo/) (average reward), [REINFORCE](https://yannberthelot.github.io/Ajax/latest/algorithms/reinforce/) (the plain policy gradient)
- **Value-based**: [DQN](https://yannberthelot.github.io/Ajax/latest/algorithms/dqn/), [PQN](https://yannberthelot.github.io/Ajax/latest/algorithms/pqn/)
- **World models**: [DreamerV3](https://yannberthelot.github.io/Ajax/latest/algorithms/dreamerv3/), [TDMPC2](https://yannberthelot.github.io/Ajax/latest/algorithms/tdmpc2/), [TDMPC2MultiTask](https://yannberthelot.github.io/Ajax/latest/algorithms/tdmpc2/) (multi-task, offline)
- **Supervised**: [DAgger](https://yannberthelot.github.io/Ajax/latest/algorithms/dagger/) (imitation of an expert), [UDRL](https://yannberthelot.github.io/Ajax/latest/algorithms/udrl/) (command-conditioned)
- **Differentiable simulation**: [APG](https://yannberthelot.github.io/Ajax/latest/algorithms/apg/)

DQN and PQN take discrete actions; PPO, APO, REINFORCE, DAgger, UDRL and
DreamerV3 take both; the others continuous ones. <!-- agents: offline -->SAC, TD3, REDQ, TQC, CrossQ, ASAC, RVI-SAC,
DQN, IQL and DAgger<!-- /agents --> also train offline (`train_offline`). Each name opens the agent's page, with its paper;
the [compatibility table](https://yannberthelot.github.io/Ajax/latest/guide/compatibility/) lists every extension
per agent.

## Learning from an expert or from data

- **Before training**: behaviour cloning (`CloningPretrain`), expert transitions in the replay (`ExpertPrefill`), prior data as in RLPD (`PriorData`).
- **During training**: TD3+BC's actor term, kickstarting and Q-filtered cloning (`ImitationLoss`), IBRL (`IBRL`), residual RL (`ResidualPolicy`), JSRL hand-over (`JSRLCurriculum`), GAIL (`AdversarialReward`), and the `DAgger` agent.
- **Instead of online data**: `agent.train_offline(dataset, n_updates)`, then online fine-tuning if wanted: IQL, CQL and Cal-QL (`ConservativeQ`, `CalQL`), ReBRAC (`TargetBC`), TD3+BC; TD-MPC2's multi-task agent (`TDMPC2MultiTask`) on pooled datasets.

[Choosing the pieces](https://yannberthelot.github.io/Ajax/latest/guide/expert-guidance/#choosing-the-pieces) gives
each method's paper and the agents that take it.

## Also in the box

<table>
<tr>
<td width="33%" valign="top"><a href="https://yannberthelot.github.io/Ajax/latest/guide/extensions/"><b>Extensions</b></a><br/>Forty-four research features, passed as <code>extensions=(...)</code></td>
<td width="33%" valign="top"><a href="https://yannberthelot.github.io/Ajax/latest/guide/memory/"><b>Recurrent memory</b></a><br/>GRU, LSTM, transformer or Mamba, with sequence replay</td>
<td width="33%" valign="top"><a href="https://yannberthelot.github.io/Ajax/latest/guide/differentiable-simulation/"><b>Differentiable simulation</b></a><br/>APG back-propagates the return through gymnax dynamics</td>
</tr>
<tr>
<td valign="top"><a href="https://yannberthelot.github.io/Ajax/latest/guide/environments/"><b>Environments</b></a><br/>gymnax, Brax and MuJoCo Playground by id, many copies at once</td>
<td valign="top"><a href="https://yannberthelot.github.io/Ajax/latest/guide/networks/"><b>Networks and the PID policy</b></a><br/>A CNN encoder; a PID controller as the actor</td>
<td valign="top"><a href="https://yannberthelot.github.io/Ajax/latest/guide/logging/"><b>Logging</b></a><br/>Every evaluation as arrays; Weights &amp; Biases or TensorBoard</td>
</tr>
<tr>
<td valign="top"><a href="https://yannberthelot.github.io/Ajax/latest/guide/comparing-agents/"><b>Comparing agents</b></a><br/>AdaStop, which adds seeds only while the answer is unclear</td>
<td valign="top"><a href="https://yannberthelot.github.io/Ajax/latest/guide/checkpoints/"><b>Checkpoints</b></a><br/>Save a run and resume it in another process</td>
<td valign="top"><a href="https://yannberthelot.github.io/Ajax/latest/developer/testing/"><b>Tests</b></a><br/>Unit tests, probes with known answers, conformance tests</td>
</tr>
</table>

The [documentation site](https://yannberthelot.github.io/Ajax/) has the user
guide, one page per agent, the [API reference](https://yannberthelot.github.io/Ajax/latest/api/agents/)
and the [developer guide](https://yannberthelot.github.io/Ajax/latest/developer/architecture/).

## Related projects

[gymnax](https://github.com/RobertTLange/gymnax),
[Brax](https://github.com/google/brax) and
[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
provide the environments, [flashbax](https://github.com/instadeepai/flashbax)
most replay buffers. [TargetGym](https://github.com/YannBerthelot/TargetGym)
has setpoint-tracking tasks with PID and MPC baselines, and
[ProbingEnvironments](https://github.com/YannBerthelot/ProbingEnvironments) the
small environments with known answers that Ajax's probes train on.

## Contributing, citing, license

Bug reports, new agents and new extensions are welcome; see
[CONTRIBUTING.md](https://github.com/YannBerthelot/Ajax/blob/main/CONTRIBUTING.md). Ajax is MIT-licensed ([LICENSE](https://github.com/YannBerthelot/Ajax/blob/main/LICENSE)).

```bibtex
@misc{ajax2025,
  title        = {Ajax: Reinforcement Learning Agents in Jax},
  author       = {Yann Berthelot},
  year         = {2025},
  url          = {https://github.com/YannBerthelot/Ajax},
}
```
