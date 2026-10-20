# Algorithms

Ajax has sixteen agents, all exported from `ajax`. Each follows a published
paper, and each page says where Ajax departs from it. Every agent takes an
environment (`TDMPC2MultiTask`: a dataset, and environments to evaluate
on), its own hyperparameters, `episode_length` and `action_repeat` (for
Brax and Playground environments) and `extensions=()`. It trains one or
several seeds in a single compiled program with
`agent.train(seed=[...], n_timesteps=...)`.

## Overview

| Agent | Family | Paper | Actions | Memory | Extra extension phases |
| --- | --- | --- | --- | --- | --- |
| [SAC](sac.md) | off-policy actor-critic | Haarnoja et al. 2018 ([arXiv:1801.01290](https://arxiv.org/abs/1801.01290)) | continuous | yes | `on_target`, `critic_loss`, `actor_loss`, `action` |
| [ASAC](asac.md) | off-policy actor-critic, average reward | Adamczyk et al. 2025 ([arXiv:2501.09080v2](https://arxiv.org/abs/2501.09080v2)) | continuous | yes | `on_target`, `critic_loss`, `actor_loss` |
| [REDQ](redq.md) | off-policy actor-critic, critic ensemble | Chen et al. 2021 ([arXiv:2101.05982](https://arxiv.org/abs/2101.05982)) | continuous | yes | `on_target`, `critic_loss`, `actor_loss` |
| [TD3](td3.md) | off-policy actor-critic, deterministic policy | Fujimoto et al. 2018 ([arXiv:1802.09477](https://arxiv.org/abs/1802.09477)) | continuous | yes | `on_target`, `critic_loss`, `actor_loss` |
| [IQL](iql.md) | offline actor-critic, expectile value | Kostrikov, Nair and Levine, ICLR 2022 ([arXiv:2110.06169](https://arxiv.org/abs/2110.06169)) | continuous | no | `on_target`, `critic_loss`, `actor_loss` |
| [AVG](avg.md) | incremental actor-critic, no replay | Vasan et al. 2024 ([arXiv:2411.15370](https://arxiv.org/abs/2411.15370)) | continuous | no | `on_target`, `critic_loss`, `actor_loss` |
| [DQN](dqn.md) | value-based, replay | Mnih et al., Nature 2015 | discrete | no | `on_target`, `critic_loss` |
| [PQN](pqn.md) | value-based, parallel environments, no replay | Gallici et al. 2024 | discrete | no | `on_target`, `critic_loss` |
| [PPO](ppo.md) | on-policy policy gradient | Schulman et al. 2017 ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347)) | both | yes | `on_target`, `critic_loss`, `actor_loss` |
| [APO](apo.md) | on-policy policy gradient, average reward | Ma et al. 2021 ([arXiv:2106.03442](https://arxiv.org/abs/2106.03442)) | both | no | `on_target`, `critic_loss`, `actor_loss` |
| [UDRL](udrl.md) | command-conditioned supervised learning | Schmidhuber 2019 ([arXiv:1912.02875](https://arxiv.org/abs/1912.02875)) | both | no | `actor_loss` |
| [DAgger](dagger.md) | imitation by dataset aggregation, no reward | Ross, Gordon and Bagnell, AISTATS 2011 ([arXiv:1011.0686](https://arxiv.org/abs/1011.0686)) | both | no | `actor_loss` |
| [APG](apg.md) | gradients through a differentiable simulator | Busetto et al. 2024 ([arXiv:2411.06482](https://arxiv.org/abs/2411.06482)) for `APG.contextual_controller` | continuous | yes | `actor_loss` |
| [DreamerV3](dreamerv3.md) | model-based, learns in imagination | Hafner et al. 2023, Nature 2025 ([arXiv:2301.04104v2](https://arxiv.org/abs/2301.04104v2)) | both | built in | none |
| [TDMPC2](tdmpc2.md) | model-based, plans at every step | Hansen, Su and Wang, ICLR 2024 ([arXiv:2310.16828](https://arxiv.org/abs/2310.16828)) | continuous | no | none |
| [TDMPC2MultiTask](tdmpc2.md#multi-task-tdmpc2multitask) | model-based, offline, several tasks | the same paper | continuous | no | none |

- **Actions:** *both* means discrete and continuous action spaces. Agents
  that take only one kind check it at construction.
- **Memory:** whether the agent accepts `memory=` (a GRU, LSTM, transformer or
  Mamba block; see [Memory](../guide/memory.md)). DreamerV3's world model is
  recurrent by design.
- **Extension phases:** every agent folds `pretrain`, `post_update` and
  `eval_metrics`; the column lists the others it folds. An extension that
  needs a phase the agent does not fold is refused at construction. See
  [Extensions](../guide/extensions.md).
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
  plans in evaluation mode.
  For SAC, TD3, REDQ, AVG, DQN, PQN and PPO, `Eval/mean bias` is the
  critic's estimate minus the discounted return actually obtained, averaged
  over the evaluated steps.

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
deterministic policy; [REDQ](redq.md) trains an ensemble of ten critics with
twenty critic updates per step, so it does far more work per sample.

**A task that never ends,** such as holding a setpoint: discounting only
shortens the agent's view. [ASAC](asac.md) (off-policy) and [APO](apo.md)
(on-policy) optimise the average reward per step instead.

**Only a dataset, no simulator to explore with** (logged runs, a plant's
history): [IQL](iql.md) trains on it offline (`train_offline`) and can
fine-tune online after; TD3+BC is TD3 with a cloning term
([TD3](td3.md#td3bc-offline)).

**Learning one transition at a time,** without a replay buffer or target
networks: [AVG](avg.md).

**The observation hides part of the state** (a velocity, a cue seen earlier):
give [PPO](ppo.md), [SAC](sac.md), [TD3](td3.md), [REDQ](redq.md),
[ASAC](asac.md) or [APG](apg.md) a `memory=`. [DreamerV3](dreamerv3.md) is
recurrent already.

**You have an expert controller** (a PID, a hand-written policy): SAC has the
most expert support, and it alone folds the `action` phase, which lets the
expert act during collection. SAC, TD3, REDQ and APO can clone the expert
into the actor before training (the `CloningPretrain` extension), and SAC,
TD3, REDQ, ASAC and APO accept the `ImitationLoss` term. See
[Expert guidance](../guide/expert-guidance.md). SAC, TD3, REDQ, ASAC, AVG,
PPO and APO can also learn a PID controller directly, with the PID policy of
[Networks](../guide/networks.md). To imitate the expert without a reward,
use [DAgger](dagger.md).

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

The CPU speed benchmark (`benchmarks/agent_bench.py`: 4 environments, 8,000
steps, Pendulum or CartPole by agent) last ran at main `a8a13c2`, with every
agent updating within the run. Counting execution alone, without
compilation: PPO, APO, DQN, PQN, AVG, UDRL and APG ran at 9,000 to 37,000
steps per second; SAC, TD3 and ASAC at 1,250 to 2,700; REDQ at 47, since it
makes 20 updates of 10 critics per step; DreamerV3 and TD-MPC2 at about 370
and 290, with smaller models than the papers'. The presets differ by agent,
so the figures compare runs of one agent over time better than agents with
each other. The benchmarks run on a CPU; Ajax claims no GPU speed-up. The
full table is in
[PERFORMANCE_REPORT.md](https://github.com/YannBerthelot/Ajax/blob/main/PERFORMANCE_REPORT.md);
see also [Performance](../guide/performance.md).
