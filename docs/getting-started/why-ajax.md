# Why Ajax

[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (SB3) is a
PyTorch library of reinforcement-learning agents, with more agents in
[sb3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib).
[SBX](https://github.com/araffin/sbx) is its JAX version.
[PureJaxRL](https://github.com/luchris429/purejaxrl) is a set of single-file
JAX implementations. Ajax differs from them in four ways:

1. [End to end on the accelerator](#end-to-end-on-the-accelerator): the
   environment and the training loop compile into one JAX program.
2. [Average-reward agents](#average-reward-agents): ASAC and APO, for tasks
   that never end.
3. [Expert-guided agents](#expert-guided-agents): agents that learn from a
   controller you already have.
4. [World models](#world-models): DreamerV3 and TD-MPC2.

This page shows each one, [compares the four libraries](#comparison) and
says [when another library fits better](#when-to-use-something-else). The
examples run on a laptop CPU in seconds, except the world-model one.

## End to end on the accelerator

`train` compiles the whole run into one JAX program: the environment's
steps, the replay buffer or rollout, the updates and the evaluations. The
seeds run side by side in that same program, through `jax.vmap`. The loop
does not come back to Python between two steps, so the program can run on a
GPU the way it runs on a CPU, without a round trip to the host at each
step. This is PureJaxRL's approach, and every Ajax agent trains this way
except `TDMPC2MultiTask`, which learns offline from a dataset and evaluates
its tasks from Python between compiled chunks of updates. It needs an
environment written in JAX: [gymnax, Brax, MuJoCo Playground or
TargetGym](../guide/environments.md). Logging and a few extensions do call
back to the host, to hand over their records, save snapshots or print
progress.

```python
import jax

from ajax import PPO

print(jax.default_backend())  # "cpu" here; "gpu" when JAX finds one
# 8 seeds, each with 8 copies of CartPole, compiled and run as one program.
agent = PPO("CartPole-v1", n_envs=8, n_steps=128)
state, _ = agent.train(seed=list(range(8)), n_timesteps=16_384)
print(state.collector_state.timestep)  # one count per seed
```

The same script runs on a GPU when JAX is installed with CUDA support and
finds one ([Installation](installation.md)). This section describes how the
program is built, not a measured gain: Ajax's tests and per-agent benchmark
run on the CPU, and no measurement compares a run on a GPU with the same run
on a CPU. [Performance](../guide/performance.md) has the costs that were
measured: compilation, seeds, each agent's speed on a CPU.

## Average-reward agents

Most agents maximise a discounted return: a reward `k` steps ahead counts
`gamma**k` times as much as one now. For a task that never ends, such as
holding a setpoint or running a plant, the discount is not part of the
problem; it only shortens how far ahead the agent looks. An average-reward
agent maximises the reward per step over the long run instead, and has no
discount factor. Ajax has two, each following its paper:
[ASAC](../algorithms/asac.md) (Adamczyk et al., 2025), SAC for this
criterion, and [APO](../algorithms/apo.md) (Ma et al., 2021), PPO for it.
Each keeps its estimate of the reward per step on the state.

```python
from ajax import APO, ASAC

# On-policy: PPO's clipped update on the reward minus the reward per step.
apo = APO("Pendulum-v1", n_envs=4, n_steps=256)
state, _ = apo.train(seed=[0, 1], n_timesteps=4_096)
print(state.average_reward)  # one estimate per seed

# Off-policy: soft actor-critic on the same criterion.
asac = ASAC("Pendulum-v1", learning_starts=1_000)
state, _ = asac.train(seed=[0, 1], n_timesteps=2_000)
print(state.theta)  # its reward-rate estimate, entropy term included
```

These budgets show the calls; Pendulum needs far more steps to be solved.

## Expert-guided agents

Often a controller already exists for the task: a tuned PID, a model
predictive controller, an earlier policy. In Ajax an *expert* is a JAX
function from observations to actions (or a stateful one, such as a PID
carrying its integral), and an agent can learn from it instead of starting
from scratch. SAC uses it the most: it fills the replay buffer with the
expert's experience, lets the expert act during warm-up, can clone it before
training, and takes the expert [extensions](../guide/extensions.md): online
behaviour cloning, residual policies, JSRL hand-over, IBRL targets and EDGE
exploration, among others. TD3, REDQ and APO can also clone it before
training, and TD3, REDQ, ASAC and APO can add an imitation term to the
actor's loss. An agent given `expert_policy=` also logs the expert's own
return at each evaluation, so the curves show whether the agent has overtaken
it. [Expert guidance](../guide/expert-guidance.md) covers every option.

```python
import jax.numpy as jnp

from ajax import SAC
from ajax.agents.TD3.TD3 import TD3
from ajax.extensions.expert import ImitationLoss
from ajax.logging.wandb_logging import LoggingConfig


def pd_expert(obs):
    """A PD controller on Pendulum's raw observation [cos θ, sin θ, θ̇]."""
    theta = jnp.arctan2(obs[..., 1], obs[..., 0])
    return jnp.clip(-(2.0 * theta + 0.5 * obs[..., 2]), -1.0, 1.0)[..., None]


# SAC writes 1,000 expert steps to its replay buffer before training.
sac = SAC("Pendulum-v1", expert_policy=pd_expert, expert_buffer_n_steps=1_000,
          learning_starts=1_000)
logging = LoggingConfig(config={}, log_frequency=1_000, use_wandb=False)
state, evaluations = sac.train(seed=0, n_timesteps=2_000, logging_config=logging)
print(evaluations["Eval/expert bias"])  # the agent's return minus the expert's

# TD3 pulls its actor towards the expert's actions with an extra loss term.
td3 = TD3("Pendulum-v1", learning_starts=1_000, extensions=(ImitationLoss(pd_expert),))
state, _ = td3.train(seed=0, n_timesteps=2_000)
```

## World models

A world-model agent learns a model of the environment, a network that
predicts what happens next and what reward follows, and uses it to act.
[DreamerV3](../algorithms/dreamerv3.md) learns its policy inside the model,
from imagined trajectories; [TD-MPC2](../algorithms/tdmpc2.md) plans with
the model at every step; `TDMPC2MultiTask` trains one model offline on the
data of several tasks. Each follows the code that produced its paper's
results: the [deviations register](../world_models/deviations.md) lists
where Ajax departs from it, and tests compare Ajax's components with it.
DreamerV3 and TD-MPC2 run in one compiled program with the environment, like
the other agents; `TDMPC2MultiTask` steps no environment while it trains.
[World models](../guide/world-models.md) explains them in plain words.

```python
# doc: skip (about half a minute on a laptop CPU, compilation included)
from ajax import DreamerV3

# The smallest model and a low train ratio, for a first look on a CPU.
agent = DreamerV3("CartPole-v1", model_size="1m", n_envs=4, train_ratio=32,
                  batch_size=4, batch_length=16)
state, _ = agent.train(seed=0, n_timesteps=1_000)
```

Both agents are much heavier than the model-free ones: use a GPU for
anything beyond a check.

## Comparison

What each library offers, as its own pages describe it. Every cell about
another library links to the page it was read from, on 11 October 2026; the
Ajax cells link to its documentation.

| | Ajax | SB3 with sb3-contrib | SBX | PureJaxRL |
| --- | --- | --- | --- | --- |
| Environment stepping inside the compiled program | Yes, for JAX environments ([Environments](../guide/environments.md)) | No: Gymnasium environments ([custom envs][sb3-env]), stepped from Python ([VecEnv][sb3-vec]) | No: Gymnasium environments, through SB3's API ([README][sbx]); its agents subclass SB3's, whose collection loop steps them from Python ([source][sbx-src]) | Yes: the whole pipeline, environment included, is JAX ([README][pjx]) |
| Vectorised environments | `n_envs` copies stepped as one operation, seeds with `jax.vmap` ([Performance](../guide/performance.md)) | `DummyVecEnv` (one process) and `SubprocVecEnv` (one process per copy) ([VecEnv][sb3-vec]) | Through the SB3 API it shares ([README][sbx]) | `vmap` over environment copies ([`ppo.py`][pjx-ppo]) and over seeds ([README][pjx]) |
| Average-reward agents | [ASAC](../algorithms/asac.md), [APO](../algorithms/apo.md) | None listed ([algorithms][sb3-algos], [contrib][contrib]) | None listed ([README][sbx]) | None listed ([files][pjx-files]) |
| Expert-guided agents | SAC with its expert extensions; an imitation term for TD3, REDQ, ASAC and APO ([Expert guidance](../guide/expert-guidance.md)) | None listed in SB3 ([algorithms][sb3-algos]) or sb3-contrib ([contrib][contrib]); the guide points to the separate `imitation` package: behaviour cloning, DAgger, AIRL, GAIL, learning from human preferences ([imitation][sb3-imitation]) | None listed ([README][sbx]) | None listed ([README][pjx]) |
| World models | [DreamerV3](../algorithms/dreamerv3.md), [TD-MPC2 and its multi-task agent](../algorithms/tdmpc2.md) | None listed ([algorithms][sb3-algos], [contrib][contrib]) | None listed ([README][sbx]) | None listed ([files][pjx-files]) |
| Recurrent memory | GRU, LSTM, transformer or Mamba for PPO, SAC, TD3, REDQ, ASAC and APG; DreamerV3 is recurrent by design ([Memory](../guide/memory.md)) | RecurrentPPO, an LSTM PPO, in sb3-contrib ([contrib][contrib]) | None listed ([README][sbx]) | A GRU PPO, `ppo_rnn.py`, and an experimental S5 PPO in `experimental/s5` ([files][pjx-files]) |
| Differentiable-simulation agent | [APG](../algorithms/apg.md) | None listed ([algorithms][sb3-algos], [contrib][contrib]) | None listed ([README][sbx]) | None listed ([files][pjx-files]) |
| Algorithms | 14 agents, TD-MPC2's multi-task agent counted apart ([Algorithms](../algorithms/index.md)) | 7 in SB3's core (A2C, DDPG, DQN, HER, PPO, SAC, TD3) ([algorithms][sb3-algos]) and 7 in sb3-contrib (ARS, CrossQ, MaskablePPO, QR-DQN, RecurrentPPO, TQC, TRPO) ([contrib][contrib]) | 9 entries: SAC and SAC-N, TQC, DroQ, PPO, DQN, TD3, DDPG, CrossQ, SimBa ([README][sbx]) | 3: PPO (`ppo.py`, `ppo_continuous_action.py`, `ppo_rnn.py`, `ppo_minigrid.py`, `experimental/s5/ppo_s5.py`), DQN (`dqn.py`), DPO (`dpo_continuous_action.py`) ([files][pjx-files]) |

"None listed" means the page cited does not list one; it is not a claim
about every fork or companion project.

[sb3-algos]: https://stable-baselines3.readthedocs.io/en/master/guide/algos.html
[sb3-vec]: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
[sb3-env]: https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
[sb3-imitation]: https://stable-baselines3.readthedocs.io/en/master/guide/imitation.html
[sb3]: https://github.com/DLR-RM/stable-baselines3
[contrib]: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib
[sbx]: https://github.com/araffin/sbx
[pjx]: https://github.com/luchris429/purejaxrl
[pjx-files]: https://github.com/luchris429/purejaxrl/tree/main/purejaxrl
[pjx-ppo]: https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo.py
[sbx-src]: https://github.com/araffin/sbx/blob/master/sbx/common/on_policy_algorithm.py

## When to use something else

- **Your environment is not written in JAX.** A Gymnasium environment that
  steps in Python, an external simulator or a real system cannot run inside
  Ajax's compiled program ([Environments](../guide/environments.md)). SB3
  and SBX take environments with the Gymnasium interface, and SB3 ships a
  checker for it ([custom envs][sb3-env], [README][sbx]).
- **You want a mature, stable library.** SB3 describes itself as stable, with
  development focused on bug fixes and maintenance, and lists documentation,
  high code coverage and type hints among its features ([README][sb3]). It
  installs with `pip`, and the RL Baselines3 Zoo adds tuned hyperparameters,
  pre-trained agents and training scripts ([README][sb3]). Ajax installs from
  source ([Installation](installation.md)).
- **Your observations are dictionaries.** SB3 supports single-level `Dict`
  observation spaces ([algorithms][sb3-algos]); Ajax's networks read one flat
  vector, and it flattens image observations into one
  ([Networks](../guide/networks.md)).
- **You want SB3's interface with JAX networks.** SBX shares SB3's API and
  works with the RL Zoo ([README][sbx]). It calls itself a proof of concept
  ([README][sbx]).
- **You want to read or change one algorithm in one file.** PureJaxRL's
  implementations are single files, not a library meant to be imported
  ([README][pjx]). Its README reports its own speed comparisons with PyTorch
  implementations, including on a GPU; Ajax has not measured a GPU speed-up.
- **You need an algorithm Ajax does not have,** such as A2C, TRPO, TQC,
  CrossQ or invalid-action masking: SB3, sb3-contrib or SBX have them
  ([algorithms][sb3-algos], [contrib][contrib], [README][sbx]).
