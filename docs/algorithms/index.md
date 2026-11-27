# Algorithms

Ajax has twenty agents, all exported from `ajax`. This page lists them,
sorts them into families and helps you choose one; each agent's page gives
its paper, its defaults and where Ajax departs from the reference.

Every agent takes an environment (`TDMPC2MultiTask`: a dataset, and
environments to evaluate on), its own hyperparameters, `episode_length`
and `action_repeat` for Brax and Playground environments, and
`extensions=()`. `agent.train(seed=[...], n_timesteps=...)` trains one or
several seeds in a single compiled program.

## Overview

| Agent | Family | Paper | Actions | Memory | Update phases |
| --- | --- | --- | --- | --- | --- |
| [SAC](sac.md) | off-policy actor-critic | Haarnoja et al. 2018 ([arXiv:1801.01290](https://arxiv.org/abs/1801.01290)) | continuous | yes | `on_target`, `critic_loss`, `actor_loss` |
| [ASAC](asac.md) | off-policy actor-critic, average reward | Adamczyk et al. 2025 ([arXiv:2501.09080v2](https://arxiv.org/abs/2501.09080v2)) | continuous | yes | `on_target`, `critic_loss`, `actor_loss` |
| [RVISAC](rvisac.md) | off-policy actor-critic, average reward, RVI Q-learning (RVI-SAC) | Hisaki and Ono, ICML 2024 ([arXiv:2408.01972](https://arxiv.org/abs/2408.01972)) | continuous | yes | `on_target`, `critic_loss`, `actor_loss` |
| [REDQ](redq.md) | off-policy actor-critic, critic ensemble | Chen et al. 2021 ([arXiv:2101.05982](https://arxiv.org/abs/2101.05982)) | continuous | yes | `on_target`, `critic_loss`, `actor_loss` |
| [CrossQ](crossq.md) | off-policy actor-critic, batch renormalisation, no target networks | Bhatt et al., ICLR 2024 ([arXiv:1902.05605](https://arxiv.org/abs/1902.05605)) | continuous | no | `on_target`, `critic_loss`, `actor_loss` |
| [TQC](tqc.md) | off-policy actor-critic, distributional critics, truncated target | Kuznetsov et al., ICML 2020 ([arXiv:2005.04269](https://arxiv.org/abs/2005.04269)) | continuous | yes | `on_target`, `critic_loss`, `actor_loss` |
| [TD3](td3.md) | off-policy actor-critic, deterministic policy | Fujimoto et al. 2018 ([arXiv:1802.09477](https://arxiv.org/abs/1802.09477)) | continuous | yes | `on_target`, `critic_loss`, `actor_loss` |
| [IQL](iql.md) | offline actor-critic, expectile value | Kostrikov, Nair and Levine, ICLR 2022 ([arXiv:2110.06169](https://arxiv.org/abs/2110.06169)) | continuous | no | `on_target`, `critic_loss`, `actor_loss` |
| [AVG](avg.md) | incremental actor-critic, no replay | Vasan et al. 2024 ([arXiv:2411.15370](https://arxiv.org/abs/2411.15370)) | continuous | no | `on_target`, `critic_loss`, `actor_loss` |
| [DQN](dqn.md) | value-based, replay | Mnih et al., Nature 2015 | discrete | no | `on_target`, `critic_loss` |
| [PQN](pqn.md) | value-based, parallel environments, no replay | Gallici et al. 2024 | discrete | no | `on_target`, `critic_loss` |
| [PPO](ppo.md) | on-policy policy gradient | Schulman et al. 2017 ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347)) | both | yes | `reward`, `on_target`, `critic_loss`, `actor_loss` |
| [APO](apo.md) | on-policy policy gradient, average reward | Ma et al. 2021 ([arXiv:2106.03442](https://arxiv.org/abs/2106.03442)) | both | no | `on_target`, `critic_loss`, `actor_loss` |
| [REINFORCE](reinforce.md) | on-policy policy gradient, one step per rollout | Williams 1992 | both | no | none |
| [UDRL](udrl.md) | command-conditioned supervised learning | Schmidhuber 2019 ([arXiv:1912.02875](https://arxiv.org/abs/1912.02875)) | both | no | `actor_loss` |
| [DAgger](dagger.md) | imitation by dataset aggregation, no reward | Ross, Gordon and Bagnell, AISTATS 2011 ([arXiv:1011.0686](https://arxiv.org/abs/1011.0686)) | both | no | `actor_loss` |
| [APG](apg.md) | gradients through a differentiable simulator | Busetto et al. 2024 ([arXiv:2411.06482](https://arxiv.org/abs/2411.06482)) for `APG.contextual_controller` | continuous | yes | `actor_loss` |
| [DreamerV3](dreamerv3.md) | model-based, learns in imagination | Hafner et al. 2023, Nature 2025 ([arXiv:2301.04104v2](https://arxiv.org/abs/2301.04104v2)) | both | built in | none |
| [TDMPC2](tdmpc2.md) | model-based, plans at every step | Hansen, Su and Wang, ICLR 2024 ([arXiv:2310.16828](https://arxiv.org/abs/2310.16828)) | continuous | no | none |
| [TDMPC2MultiTask](tdmpc2.md#multi-task-tdmpc2multitask) | model-based, offline, several tasks | the same paper | continuous | no | none |

- **Actions:** *both* means discrete and continuous action spaces. Agents
  that take only one kind check it at construction.
- **Memory:** whether the agent accepts `memory=` (a GRU, LSTM, transformer or
  Mamba block; see [Memory](../guide/memory.md)). With memory, every agent
  but APG refuses an `expert_policy` and most extensions. DreamerV3's world
  model is recurrent by design.
- **Update phases:** the [extension](../guide/extensions.md) phases the
  agent's update applies: a change to its targets (`on_target`), a loss
  term (`critic_loss`, `actor_loss`), or new rewards (`reward`). Every
  agent also runs `pretrain`, `post_update` and `eval_metrics`, and most
  the collection phases (`observe`, `action` and others): each agent's page
  lists them all, and [Compatibility](../guide/compatibility.md) says which
  extension runs on which agent. An extension that needs a phase the agent
  does not run is refused at construction.
- **Continuous actions:** every continuous agent acts in `[-1, 1]`, squashed
  by `tanh` or clipped there, and Ajax maps that range onto the
  environment's bounds before each step: `centre + half_range * action`. On
  gymnax Pendulum, whose torque lies in `[-2, 2]`, the agents reach the whole
  range. Experts are written in the environment's units; Ajax reads them in
  the agent's. See [Environments](../guide/environments.md).
- **Evaluation:** with a `logging_config`, every agent evaluates its policy
  every `log_frequency` steps on `num_episode_test` episodes and `train`
  returns the evaluations. Each evaluation draws fresh random keys (new
  initial states), and a rerun of the same seed draws the same ones. The
  model-free agents evaluate their deterministic action (the mean, or the
  greedy action); DreamerV3 samples, as its reference does, and TD-MPC2
  plans in evaluation mode. The model-free agents with a discounted critic
  also log `Eval/mean bias`, the critic's estimate minus the discounted
  return actually obtained ([Logging](../guide/logging.md)).

## Agent families

An agent belongs to a family according to where its updates take their data
from. The other pages use these names, and a family's class in
`ajax.agents.families` builds its agents' training loop.

| Family | Agents | Its updates learn from |
| --- | --- | --- |
| Replay agents (`ReplayAgent`) | <!-- agents: replay -->SAC, ASAC, RVI-SAC, REDQ, CrossQ, TQC, TD3, IQL, DQN, DAgger<!-- /agents --> | rows sampled from a replay buffer of past transitions |
| On-policy agents (`OnPolicyAgent`) | <!-- agents: on-policy -->PPO, APO, REINFORCE, PQN<!-- /agents --> | the rollout just collected, then discarded |
| Incremental agent (`IncrementalAgent`) | <!-- agents: incremental -->AVG<!-- /agents --> | each step as it comes, with no replay |
| Agents with an iteration of their own (`OwnIterationAgent`) | <!-- agents: own-iteration -->UDRL, APG<!-- /agents --> | UDRL: a replay of episode segments; APG: gradients through the simulator |
| World models (`OwnReplayAgent`) | <!-- agents: own-replay -->DreamerV3, TD-MPC2<!-- /agents --> | a replay of their own, which trains the model and the policy |

`TDMPC2MultiTask` is outside the families: it learns from a dataset only.

Two properties follow the replay agents:

- **Training offline.** The replay agents
  (<!-- agents: offline -->SAC, ASAC, RVI-SAC, REDQ, CrossQ, TQC, TD3, IQL, DQN, DAgger<!-- /agents -->)
  accept a dataset in place of the environment
  ([`train_offline`](../guide/training.md#training-offline)).
- **Sequence replay.** Six of them
  (<!-- agents: sequence-replay -->SAC, ASAC, RVI-SAC, REDQ, TQC, TD3<!-- /agents -->)
  take `memory=` and then replay sequences
  ([Memory](../guide/memory.md#off-policy-sequence-replay)).

A test compares each of these lists with the classes
(`tests/test_docs_agent_sets.py`).

## Which agent should I use?

Start from your problem, not from the algorithm.

**Discrete actions.** [PPO](ppo.md) is the general choice; it also takes
memory and image observations. [DQN](dqn.md) learns from a replay buffer, so
it reuses every sample many times. [PQN](pqn.md) has no replay: it learns from
many environments at once, which suits fast JAX simulators.

**Continuous actions, and samples are cheap.** If the simulator runs many
copies in parallel, [PPO](ppo.md) learns from large fresh rollouts and needs
no replay buffer.

**Continuous actions, and every sample counts.** Use an off-policy agent with
a replay buffer. [SAC](sac.md) is the default; [TD3](td3.md) learns a
deterministic policy. [TQC](tqc.md) and [CrossQ](crossq.md) change SAC's
critics: distributional ones that drop their largest quantiles, or
batch-renormalised ones without target networks. [REDQ](redq.md) trains
an ensemble of ten critics with twenty critic updates per step, so it does
far more work per sample.

**A task that never ends,** such as holding a setpoint: discounting only
shortens the agent's view. [ASAC](asac.md) and [RVI-SAC](rvisac.md)
(off-policy) and [APO](apo.md) (on-policy) optimise the average reward per
step instead. A discounted agent can also centre its values on the reward
rate with [`ValueCentering`](../guide/extensions.md#catalogue).

**Only a dataset, no simulator to explore with** (logged runs, a plant's
history): [IQL](iql.md) trains on it offline (`train_offline`) and can
fine-tune online after. [TD3+BC](td3.md#td3bc-offline) is TD3 with a
cloning term, and [CQL](sac.md#cql-and-cal-ql-offline) SAC with a
conservative critic.

**Learning one transition at a time,** without a replay buffer or target
networks: [AVG](avg.md).

**The observation hides part of the state** (a velocity, a cue seen
earlier): give [PPO](ppo.md), [APG](apg.md) or a sequence-replay agent
(<!-- agents: sequence-replay -->SAC, ASAC, RVI-SAC, REDQ, TQC, TD3<!-- /agents -->)
a `memory=`. [DreamerV3](dreamerv3.md) is recurrent already.

**The observation is an image.** [PPO](ppo.md), [DQN](dqn.md),
[PQN](pqn.md) and [UDRL](udrl.md) read images through a convolutional
encoder ([Image observations](../guide/networks.md#image-observations)); all
four take discrete actions, and PPO and UDRL continuous ones too. The other
agents, DreamerV3 included, read vector observations only.

**You have an expert controller** (a PID, a hand-written policy): most
agents take an `expert_policy` and the extensions that learn from it, by
prefilling the replay, cloning the expert or letting it act. SAC runs the
most of them. See [Expert guidance](../guide/expert-guidance.md), and
[Compatibility](../guide/compatibility.md) for which agent runs which.
Most continuous agents can also learn a PID controller directly, with the
PID policy of [Networks](../guide/networks.md). To imitate the expert
without a reward, use [DAgger](dagger.md).

**Your simulator is differentiable** (a gymnax environment that exposes its
transition gradients) and you want one controller for a family of systems:
[APG](apg.md). Its contextual controller reads the tracking history, so one
controller can serve the whole family.

**You want a model of the environment.** [DreamerV3](dreamerv3.md) learns
inside its world model and handles discrete and continuous actions.
[TD-MPC2](tdmpc2.md) plans with its model at every step, for continuous
control with fixed-length episodes (terminating ones with `episodic=True`).
Both do much more computation per environment step than the model-free
agents. To train one model offline on data from several tasks, use
`TDMPC2MultiTask`.

**Supervised learning instead of value estimation:** [UDRL](udrl.md) learns
which action leads to which return, then asks for more.

## Speed

Speeds differ by orders of magnitude, with the work each update does. On a
CPU, PPO, PQN and DQN make tens of thousands of steps per second, SAC and
TD3 a few thousand, and REDQ, CrossQ, TQC and the world models a few
hundred or fewer: they update many or wide critics, or a model, at every
step. Those gain the most from a GPU. [Performance](../guide/performance.md)
gives every agent's measured speed, on a CPU and on a GPU.
