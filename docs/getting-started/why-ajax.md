# Why Ajax

This page is for deciding whether Ajax suits your work. It explains what
Ajax offers, with examples, compares it with other libraries, and says
[when one of them fits better](#when-to-use-something-else). The examples
run on a laptop CPU in under a minute each, except the world-model one.

## One program for the whole run

`train` compiles the whole run into one JAX program: the environment's
steps, the replay buffer or the rollout, the updates and the evaluations.
The seeds run side by side in that program, through `jax.vmap`, and so do
the configurations of a hyperparameter sweep. No step goes back to Python,
so the program runs on a CPU or a GPU alike:

```python
import jax

from ajax import PPO

print(jax.default_backend())  # "cpu" here; "gpu" when JAX finds one
agent = PPO("CartPole-v1", n_envs=8, n_steps=128)
# 4 seeds of 2 clip ranges: 8 runs, each with 8 copies of CartPole, in one program.
state, evaluations = agent.train(
    seed=[0, 1, 2, 3], n_timesteps=16_384, sweep={"clip_range": [0.1, 0.2]}
)
print(evaluations.returns.shape)  # a row per run, a column per evaluation
```

A GPU gains most on agents whose updates take the time (many critics, wide
networks, world models) and little on those spending it on one simulator
step per seed ([CPU or GPU](../guide/performance.md#cpu-or-gpu)). The same
script runs on a GPU once JAX has CUDA ([Installation](installation.md#cpu-or-gpu)).

This needs an environment written in JAX: gymnax, Brax, MuJoCo Playground,
Gymnasium's MuJoCo tasks on MJX or your own
([Environments](../guide/environments.md)); a Gymnasium environment steps in
Python. Only logging leaves the program, to hand over each evaluation.
`TDMPC2MultiTask` alone trains otherwise: offline, evaluating its tasks from
Python between compiled chunks of updates.

## Variations that combine

A variant of an algorithm is an argument of its agent, not another class:
recurrent PPO is PPO with `memory=` ([Memory](../guide/memory.md)), images
go through a CNN encoder with `cnn_image_shape=`, and a `PIDPolicy` as
`actor_architecture` makes the actor a PID controller
([Networks](../guide/networks.md)). Research features (cloning an expert, an
extra loss term, a diagnostic) are [extensions](../guide/extensions.md),
passed as `extensions=(...)`, and they add up: the TD3
[below](#learning-from-an-expert-or-from-data) clones an expert, then keeps
an imitation term in its loss. An agent refuses, when built, an extension it
cannot run, and says why ([Compatibility](../guide/compatibility.md)).

## Checked against the original code

Each agent starts from the code behind its paper's results, with that
code's defaults, initialisation and order of updates; its
[algorithm page](../algorithms/index.md) cites the code and lists where Ajax
departs from it. Its defaults can therefore differ from another library's
([Coming from SB3](coming-from-sb3.md#the-defaults-are-the-originals)).
Three kinds of tests check the agents ([Testing](../developer/testing.md)):

- unit tests, on each update's maths with hand-computed values;
- probes, which train an agent on a small problem with a known answer and
  name the mistake behind a wrong one, such as a discount applied twice;
- [conformance checks](../developer/testing.md#conformance-checks), which
  run one training step of Ajax and of the original implementation from the
  same state and compare them.

Comparisons of whole training runs with reference implementations are in
progress ([Reproduction](../reproduction/index.md)).

## Average-reward agents

Most agents maximise a discounted return: a reward `k` steps ahead counts
`gamma**k` times as much as one now. For a task that never ends, such as
holding a setpoint or running a plant, the discount is not part of the
problem; it only shortens how far ahead the agent looks. An average-reward
agent maximises the reward per step over the long run instead, and has no
discount factor. Ajax has three, each following its paper:
[ASAC](../algorithms/asac.md) (Adamczyk et al., 2025) and
[RVI-SAC](../algorithms/rvisac.md) (Hisaki and Ono, 2024), SAC for this
criterion, and [APO](../algorithms/apo.md) (Ma et al., 2021), PPO for it.
ASAC and APO keep their estimate of the reward per step on the state:

```python
from ajax import APO, ASAC

# On-policy: PPO's clipped update on the reward minus the reward per step.
apo = APO("Pendulum-v1", n_envs=4)
state, _ = apo.train(seed=[0, 1], n_timesteps=4_000)
print(state.average_reward)  # one estimate per seed

# Off-policy: soft actor-critic on the same criterion.
asac = ASAC("Pendulum-v1", learning_starts=1_000)
state, _ = asac.train(seed=[0, 1], n_timesteps=2_000)
print(state.theta)  # its reward-rate estimate, entropy term included
```

These budgets show the calls; Pendulum needs far more steps to be solved.

## Learning from an expert or from data

Often a controller already exists for the task: a tuned PID, a model
predictive controller, an earlier policy. In Ajax this *expert* is a JAX
function from the environment's raw observations to actions in its units
(a stateful one, such as a PID with its integral, works too), and an agent
can learn from it instead of starting from scratch.

Most agents can clone the expert before training (`CloningPretrain`) or add
an imitation term to the actor's loss (`ImitationLoss`), and most replay
agents can start with its transitions in the replay (`ExpertPrefill`). SAC
takes nearly every such extension, among them residual policies, JSRL
hand-over and IBRL targets. Recorded transitions work too: prior data in the
replay, as in RLPD (`PriorData`), or `train_offline`. Given
`expert_policy=`, an agent logs the expert's return at each evaluation, so
the curves show when it overtakes the expert.
[Expert guidance](../guide/expert-guidance.md) gives each method and its
paper.

```python
import jax.numpy as jnp

from ajax import SAC, TD3
from ajax.extensions.expert import ImitationLoss
from ajax.extensions.pretrain import CloningPretrain, ExpertPrefill
from ajax.logging import LoggingConfig


def pd_expert(obs):
    """A PD controller on Pendulum's raw observation [cos θ, sin θ, θ̇],
    answering a torque in the environment's units, [-2, 2]."""
    theta = jnp.arctan2(obs[..., 1], obs[..., 0])
    return jnp.clip(-(2.0 * theta + 0.5 * obs[..., 2]), -2.0, 2.0)[..., None]


# SAC writes 1,000 expert steps to its replay buffer before training.
sac = SAC("Pendulum-v1", expert_policy=pd_expert, learning_starts=1_000,
          extensions=(ExpertPrefill(n_steps=1_000),))
logging = LoggingConfig(log_frequency=1_000, backends=())
state, evaluations = sac.train(seed=0, n_timesteps=2_000, logging_config=logging)
print(evaluations["Eval/expert bias"])  # the agent's return minus the expert's

# TD3 clones the expert on 1,000 of its steps before training, then keeps
# pulling its actor towards the expert's actions with an extra loss term.
td3 = TD3("Pendulum-v1", learning_starts=1_000, extensions=(
    CloningPretrain(pd_expert, n_steps=1_000, epochs=2),
    ImitationLoss(pd_expert),
))
state, _ = td3.train(seed=0, n_timesteps=2_000)
```

## World models

A world-model agent learns a model of the environment, a network that
predicts what happens next and what reward follows, and uses it to act.
[DreamerV3](../algorithms/dreamerv3.md) learns its policy inside the model,
from imagined trajectories; [TD-MPC2](../algorithms/tdmpc2.md) plans with
the model at every step, and `TDMPC2MultiTask` trains one model offline on
the data of several tasks. Each follows the code behind its paper's
results, and the [deviations register](../world_models/deviations.md) lists
where Ajax departs from it. [World models](../guide/world-models.md)
explains them in plain words.

```python
# doc: skip (about half a minute on a laptop CPU, compilation included)
from ajax import DreamerV3

# The smallest model and a low train ratio, for a first look on a CPU.
agent = DreamerV3("CartPole-v1", model_size="1m", n_envs=4, train_ratio=32,
                  batch_size=4, batch_length=16)
state, _ = agent.train(seed=0, n_timesteps=1_000)
```

Both are much heavier than the model-free agents: use a GPU for anything
beyond a check.

## Comparison

[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (SB3) is a
PyTorch library of agents, with more in sb3-contrib;
[SBX](https://github.com/araffin/sbx) is its JAX version, and
[PureJaxRL](https://github.com/luchris429/purejaxrl) a set of single-file
JAX implementations. Each cell about another library links to the page it
comes from, as read in October 2026; the Ajax cells link to its
documentation.

| | Ajax | SB3 with sb3-contrib | SBX | PureJaxRL |
| --- | --- | --- | --- | --- |
| Environment stepping inside the compiled program | Yes, for JAX environments ([Environments](../guide/environments.md)) | No: Gymnasium environments ([custom envs][sb3-env]), stepped from Python ([VecEnv][sb3-vec]) | No: Gymnasium environments, through SB3's API ([README][sbx]); its agents subclass SB3's, whose collection loop steps them from Python ([source][sbx-src]) | Yes: the whole pipeline, environment included, is JAX ([README][pjx]) |
| Vectorised environments | `n_envs` copies stepped as one operation, seeds with `jax.vmap` ([Performance](../guide/performance.md)) | `DummyVecEnv` (one process) and `SubprocVecEnv` (one process per copy) ([VecEnv][sb3-vec]) | Through the SB3 API it shares ([README][sbx]) | `vmap` over environment copies ([`ppo.py`][pjx-ppo]) and over seeds ([README][pjx]) |
| Average-reward agents | [ASAC](../algorithms/asac.md), [RVI-SAC](../algorithms/rvisac.md), [APO](../algorithms/apo.md) | None listed ([algorithms][sb3-algos], [contrib][contrib]) | None listed ([README][sbx]) | None listed ([files][pjx-files]) |
| Learning from an expert or from data | Cloning, imitation terms, prior data and hand-over on most agents, offline training, and the DAgger agent ([Expert guidance](../guide/expert-guidance.md)) | None listed in SB3 ([algorithms][sb3-algos]) or sb3-contrib ([contrib][contrib]); the guide points to the separate `imitation` package: behaviour cloning, DAgger, AIRL, GAIL, learning from human preferences ([imitation][sb3-imitation]) | None listed ([README][sbx]) | None listed ([README][pjx]) |
| World models | [DreamerV3](../algorithms/dreamerv3.md), [TD-MPC2 and its multi-task agent](../algorithms/tdmpc2.md) | None listed ([algorithms][sb3-algos], [contrib][contrib]) | None listed ([README][sbx]) | None listed ([files][pjx-files]) |
| Recurrent memory | GRU, LSTM, transformer or Mamba, on the agents [Memory](../guide/memory.md#which-agents) lists; DreamerV3 is recurrent by design | RecurrentPPO, an LSTM PPO, in sb3-contrib ([contrib][contrib]) | None listed ([README][sbx]) | A GRU PPO, `ppo_rnn.py`, and an experimental S5 PPO in `experimental/s5` ([files][pjx-files]) |
| Differentiable-simulation agent | [APG](../algorithms/apg.md) | None listed ([algorithms][sb3-algos], [contrib][contrib]) | None listed ([README][sbx]) | None listed ([files][pjx-files]) |
| Algorithms | Off-policy, on-policy, average-reward, offline, imitation and model-based agents ([Algorithms](../algorithms/index.md)) | 7 in SB3's core (A2C, DDPG, DQN, HER, PPO, SAC, TD3) ([algorithms][sb3-algos]) and 7 in sb3-contrib (ARS, CrossQ, MaskablePPO, QR-DQN, RecurrentPPO, TQC, TRPO) ([contrib][contrib]) | 9 entries: SAC and SAC-N, TQC, DroQ, PPO, DQN, TD3, DDPG, CrossQ, SimBa ([README][sbx]) | 3: PPO (`ppo.py`, `ppo_continuous_action.py`, `ppo_rnn.py`, `ppo_minigrid.py`, `experimental/s5/ppo_s5.py`), DQN (`dqn.py`), DPO (`dpo_continuous_action.py`) ([files][pjx-files]) |

"None listed" means the page cited does not list one; it is not a claim
about every fork or companion project.

### Other libraries

These also compile the environment and the training loop into one JAX
program:

- **[Brax's training agents][brax-agents]**: PPO, SAC, APG, ARS, ES and
  behaviour cloning (with DAgger from a teacher policy), trained across
  devices and hosts. Ajax's [APG](../algorithms/apg.md) adds recurrent
  memory and Busetto et al.'s contextual controller.
- **[Stoix][stoix]**: about 25 algorithms, among them R2D2, MPO and MuZero,
  trained across devices. Its Anakin architecture runs end to end, and
  Sebulba trains on environments not written in JAX.
- **[Rejax][rejax]**: PPO, SAC, DQN and its variants, PQN, IQN and TD3, each
  run jitted whole and vmapped over seeds and hyperparameters
  ([README][rejax]).

CleanRL's [QDagger][cleanrl-qdagger] also learns from a teacher policy, for
DQN. [TorchRL][torchrl] and [Tianshou][tianshou] take Gymnasium
environments, and between them their pages list IQL, CQL, GAIL, REDQ, CrossQ
and Dreamer, all of which Ajax has; TorchRL's Dreamer is written in PyTorch
and steps its environments from Python.

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
[brax-agents]: https://github.com/google/brax/tree/main/brax/training/agents
[stoix]: https://github.com/EdanToledo/Stoix
[rejax]: https://github.com/keraJLi/rejax
[cleanrl-qdagger]: https://docs.cleanrl.dev/rl-algorithms/qdagger/
[torchrl]: https://github.com/pytorch/rl
[tianshou]: https://github.com/thu-ml/tianshou

## When to use something else

- **Your environment is not written in JAX.** Ajax trains on a Gymnasium
  environment, but slowly, stepping it in Python
  ([Gymnasium environments](../guide/environments.md#gymnasium-environments)).
  SB3 and SBX are built around Gymnasium's interface, and SB3 ships a
  checker for it ([custom envs][sb3-env], [README][sbx]); TorchRL and
  Tianshou take such environments too ([TorchRL][torchrl],
  [Tianshou][tianshou]), and Stoix's Sebulba architecture trains on them at
  scale ([Stoix][stoix]).
- **You train one run on several devices.** Ajax splits a run's seeds over
  devices ([Performance](../guide/performance.md)), but each seed trains on
  one: it has no `pmap` or sharding. Brax's training agents and Stoix train
  across devices ([Brax][brax-agents], [Stoix][stoix]).
- **You want to sweep any hyperparameter of any agent in one program.**
  `train(sweep=...)` varies only the numbers an agent declares sweepable,
  and APG, DreamerV3 and TD-MPC2 declare none
  ([Hyperparameter sweeps](../guide/training.md#hyperparameter-sweeps)). Any
  other value compiles a program of its own. Rejax vmaps over
  hyperparameters as well as seeds ([README][rejax]).
- **You want a mature, stable library.** SB3 describes itself as stable,
  focused on bug fixes and maintenance, with documentation, high code
  coverage and type hints, and its RL Baselines3 Zoo adds tuned
  hyperparameters, pre-trained agents and training scripts ([README][sb3]).
  Ajax is younger, with no zoo or pre-trained agents.
- **Your observations are dictionaries.** SB3 supports single-level `Dict`
  observation spaces ([algorithms][sb3-algos]); Ajax's networks read one
  flat vector, and flatten an image into one
  ([Networks](../guide/networks.md#image-observations)).
- **You want SB3's interface with JAX networks.** SBX shares SB3's API and
  works with the RL Zoo; it calls itself a proof of concept ([README][sbx]).
- **You want an algorithm in one file, with nothing imported.** PureJaxRL's
  implementations are single files, not a library to import ([README][pjx]);
  an Ajax agent's file holds its algorithm, on a shared loop, networks and
  logging. PureJaxRL's README compares its speed with PyTorch
  implementations; [Ajax's speeds](../guide/performance.md#cpu-or-gpu)
  include no comparison with PureJaxRL.
- **You need an algorithm Ajax does not have,** such as A2C, TRPO or
  invalid-action masking: SB3, sb3-contrib or SBX have them
  ([algorithms][sb3-algos], [contrib][contrib], [README][sbx]).

Coming from Stable-Baselines3? [Coming from SB3](coming-from-sb3.md) puts
its calls and defaults beside Ajax's.
